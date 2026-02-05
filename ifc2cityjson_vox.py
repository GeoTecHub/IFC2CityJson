"""
IFC4 to CityJSON Converter (Voxel & Strict)
==============================================
Features:
- Strict JSON structure (fixes validation errors).
- Dynamic Origin centering.
- **NEW: Voxel-based Room Detection** (Runs if no IfcSpace is found).
"""

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
import json
import subprocess
import logging
import sys
import argparse
from uuid import uuid4

# Voxelization Imports
try:
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    VOXEL_AVAILABLE = True
except ImportError:
    VOXEL_AVAILABLE = False

# Setup Logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

setup_logging()
logger = logging.getLogger(__name__)

# --- Classes ---

class IFCReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.ifc_file = None
        self.settings = None
        self._shape_cache = {}

    def create_shape_cached(self, entity):
        """Create and cache geometry shape for an entity to avoid duplicate processing."""
        eid = entity.id()
        if eid not in self._shape_cache:
            self._shape_cache[eid] = ifcopenshell.geom.create_shape(self.settings, entity)
        return self._shape_cache[eid]

    def load(self):
        try:
            self.ifc_file = ifcopenshell.open(self.filepath)
            self.settings = ifcopenshell.geom.settings()
            self.settings.set(self.settings.USE_WORLD_COORDS, True)
            self.settings.set(self.settings.WELD_VERTICES, True)
            return True
        except Exception as e:
            logger.error(f"Failed to load IFC: {e}")
            return False

    def get_elements(self):
        types = ["IfcWall", "IfcSlab", "IfcRoof", "IfcWindow", "IfcDoor", "IfcColumn", "IfcBeam", "IfcStair"]
        elements = []
        if self.ifc_file:
            for t in types:
                elements.extend(self.ifc_file.by_type(t))
        return elements

    def get_spaces(self):
        return self.ifc_file.by_type("IfcSpace") if self.ifc_file else []

    def get_project(self):
        if not self.ifc_file:
            return None
        projects = self.ifc_file.by_type("IfcProject")
        return projects[0] if projects else None
    
    def get_buildings(self):
        return self.ifc_file.by_type("IfcBuilding") if self.ifc_file else []

# --- NEW: Voxel Space Detector Class ---
class VoxelSpaceDetector:
    def __init__(self, reader, voxel_size=0.5):
        self.reader = reader
        self.voxel_size = voxel_size

    def detect(self):
        if not VOXEL_AVAILABLE:
            logger.warning("Numpy/Scipy not installed. Skipping voxel detection.")
            return []
        
        logger.info(f"Starting Voxel Analysis (Size: {self.voxel_size}m)...")
        elements = self.reader.get_elements() # Walls, Slabs, etc.
        
        if not elements: return []

        # 1. Bounding Box & Grid Setup
        min_pt = np.array([float('inf')] * 3)
        max_pt = np.array([float('-inf')] * 3)
        mesh_cache = []

        # Pre-calculate bounds
        for elem in elements:
            try:
                shape = self.reader.create_shape_cached(elem)
                verts = np.array(shape.geometry.verts).reshape((-1, 3))
                if verts.size > 0:
                    min_pt = np.minimum(min_pt, verts.min(axis=0))
                    max_pt = np.maximum(max_pt, verts.max(axis=0))
                    mesh_cache.append(verts)
            except Exception:
                continue

        if not mesh_cache:
            return []

        # Padding
        min_pt -= 1.0
        max_pt += 1.0
        dims = np.ceil((max_pt - min_pt) / self.voxel_size).astype(int)

        MAX_GRID_CELLS = 50_000_000  # ~50 MB for bool grid
        total_cells = int(np.prod(dims))
        if total_cells > MAX_GRID_CELLS:
            logger.warning(f"Grid too large ({total_cells} cells). Skipping voxel detection.")
            return []

        logger.info(f"Grid Size: {dims}")

        grid = np.zeros(dims, dtype=bool)

        # 2. Rasterize Walls/Slabs
        def to_grid(p): 
            return np.floor((p - min_pt) / self.voxel_size).astype(int)

        for verts in mesh_cache:
            g_idx = to_grid(verts)
            g_min = np.maximum(0, g_idx.min(axis=0))
            g_max = np.minimum(dims, g_idx.max(axis=0) + 1)
            grid[g_min[0]:g_max[0], g_min[1]:g_max[1], g_min[2]:g_max[2]] = True

        # 3. Find Void Regions (Rooms)
        void_grid = ~grid
        labeled_array, num_features = ndimage.label(void_grid, structure=np.ones((3,3,3)))
        
        # Identify "Outside" — check all 8 grid corners for the most common void label
        corner_indices = [
            (0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
            (-1, -1, 0), (-1, 0, -1), (0, -1, -1), (-1, -1, -1),
        ]
        corner_labels = [int(labeled_array[c]) for c in corner_indices]
        corner_labels = [c for c in corner_labels if c > 0]

        if corner_labels:
            outside_id = int(np.bincount(corner_labels).argmax())
        else:
            # No corner is in a void region; fall back to largest void region
            region_counts = np.bincount(labeled_array.ravel())
            # Ignore label 0 (solid)
            region_counts[0] = 0
            outside_id = int(region_counts.argmax()) if region_counts.max() > 0 else 0

        rooms = []
        origin = min_pt
        
        for region_id in range(1, num_features + 1):
            if region_id == outside_id: continue
            
            mask = (labeled_array == region_id)
            if np.sum(mask) * (self.voxel_size**3) < 3.0: continue # Skip small noise (<3m3)

            # 4. Refine to Geometry (Convex Hull)
            indices = np.argwhere(mask)
            if len(indices) < 4: continue
            
            # Map grid index back to world coords
            points = indices * self.voxel_size + origin + (self.voxel_size/2)
            
            try:
                hull = ConvexHull(points)
                # Extract geometry data compatible with Processor
                verts = []
                faces = [] # [ [v1, v2, v3], ... ]
                
                # We need to map hull vertex indices to our local list
                hull_v_map = { hv_idx: i for i, hv_idx in enumerate(hull.vertices) }
                
                # Store vertices
                for hv_idx in hull.vertices:
                    verts.append(points[hv_idx].tolist())
                
                # Store faces (MultiSurface style: [ [idx1, idx2, idx3] ])
                for simplex in hull.simplices:
                    # simplex contains indices into the original 'points' array
                    # map them to 'verts' array
                    face_indices = [hull_v_map[si] for si in simplex]
                    faces.append([face_indices]) # Nested list for MultiSurface
                
                rooms.append({"vertices": verts, "boundaries": faces})
            except Exception:
                continue

        return rooms

class GeometryProcessor:
    def __init__(self, reader):
        self.reader = reader

    def extract(self, entity):
        try:
            shape = self.reader.create_shape_cached(entity)
            if not shape: return self.fallback_box(entity)
            
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            
            vertices = []
            for i in range(0, len(verts), 3):
                vertices.append([verts[i], verts[i+1], verts[i+2]])
            
            boundaries = []
            for i in range(0, len(faces), 3):
                idx = [faces[i], faces[i+1], faces[i+2]]
                boundaries.append([idx]) 
                
            return vertices, boundaries
        except Exception:
            return None, None

    def fallback_box(self, entity):
        if not entity.is_a("IfcSpace"):
            return None, None
        # Try to derive dimensions from the space's decomposed elements or use defaults
        dx, dy, dz = 4.0, 4.0, 3.0  # Reasonable default room size in meters
        try:
            m = ifcopenshell.util.placement.get_local_placement(entity.ObjectPlacement)
            tx, ty, tz = float(m[0][3]), float(m[1][3]), float(m[2][3])
        except Exception:
            tx, ty, tz = 0.0, 0.0, 0.0
            logger.warning(f"Could not read placement for {getattr(entity, 'Name', 'IfcSpace')}; using origin.")
        v = [
            [tx, ty, tz], [tx+dx, ty, tz], [tx+dx, ty+dy, tz], [tx, ty+dy, tz],
            [tx, ty, tz+dz], [tx+dx, ty, tz+dz], [tx+dx, ty+dy, tz+dz], [tx, ty+dy, tz+dz],
        ]
        b = [[[0,3,2,1]], [[4,5,6,7]], [[0,1,5,4]], [[1,2,6,5]], [[2,3,7,6]], [[3,0,4,7]]]
        return v, b

class Georeferencer:
    # SLD99 / Sri Lanka Grid 1999 (EPSG:5235)
    # Default origin: central Colombo area
    DEFAULT_CRS = "https://www.opengis.net/def/crs/EPSG/0/5235"
    DEFAULT_TRANSLATE = [399800.0, 492200.0, 10.0]  # Easting, Northing, Height in SLD99

    def __init__(self, reader):
        self.reader = reader
        self.translate = list(self.DEFAULT_TRANSLATE)
        self.reference_system = self.DEFAULT_CRS
        self.found_crs = False

    def solve(self):
        project = self.reader.get_project()
        if project:
            for ctx in project.RepresentationContexts or []:
                if not ctx.is_a("IfcGeometricRepresentationContext"):
                    continue
                coord_ops = getattr(ctx, "HasCoordinateOperation", None) or []
                for op in coord_ops:
                    if op.is_a("IfcMapConversion"):
                        self.translate = [float(op.Eastings), float(op.Northings), float(op.OrthogonalHeight)]
                        self.found_crs = True
                        return
        if not self.found_crs:
            # Offset building placement relative to SLD99 default origin
            buildings = self.reader.get_buildings()
            if buildings and buildings[0].ObjectPlacement:
                try:
                    m = ifcopenshell.util.placement.get_local_placement(buildings[0].ObjectPlacement)
                    self.translate = [
                        self.DEFAULT_TRANSLATE[0] + float(m[0][3]),
                        self.DEFAULT_TRANSLATE[1] + float(m[1][3]),
                        self.DEFAULT_TRANSLATE[2] + float(m[2][3]),
                    ]
                except Exception:
                    pass
            logger.info(f"Using SLD99 (EPSG:5235) georeference: {self.translate}")

class Converter:
    def __init__(self, ifc, out):
        self.ifc = ifc
        self.out = out
    
    IFC_TYPE_MAP = {
        "IfcSpace": "Room",
        "IfcWindow": "BuildingInstallation",
        "IfcDoor": "BuildingInstallation",
    }

    def map_type(self, entity):
        for ifc_type, cj_type in self.IFC_TYPE_MAP.items():
            if entity.is_a(ifc_type):
                return cj_type
        return "BuildingPart"

    def run(self):
        reader = IFCReader(self.ifc)
        if not reader.load(): return
        
        geo = Georeferencer(reader)
        geo.solve()
        proc = GeometryProcessor(reader)
        
        city_objects = {}
        all_vertices = []
        vertex_index = {}  # (x, y, z) -> index for deduplication

        def add_verts(raw_verts):
            indices = []
            for v in raw_verts:
                x = int(round((v[0] - geo.translate[0]) * 1000))
                y = int(round((v[1] - geo.translate[1]) * 1000))
                z = int(round((v[2] - geo.translate[2]) * 1000))
                key = (x, y, z)
                if key not in vertex_index:
                    vertex_index[key] = len(all_vertices)
                    all_vertices.append([x, y, z])
                indices.append(vertex_index[key])
            return indices

        def add_object(cj_type, verts, bounds, attrs=None):
            if not verts: return None

            idx_map = add_verts(verts)
            new_bounds = []
            for surf in bounds:
                new_surf = []
                for ring in surf:
                    new_surf.append([idx_map[idx] for idx in ring])
                new_bounds.append(new_surf)

            obj_id = str(uuid4())
            obj = {
                "type": cj_type,
                "geometry": [{
                    "type": "MultiSurface",
                    "lod": "2.2",
                    "boundaries": new_bounds
                }]
            }
            if attrs: obj["attributes"] = attrs
            city_objects[obj_id] = obj
            return obj_id

        child_ids = []

        # 1. Process Elements
        logger.info("Processing standard elements...")
        for elem in reader.get_elements():
            v, b = proc.extract(elem)
            oid = add_object(self.map_type(elem), v, b, {"ifc_type": elem.is_a(), "name": getattr(elem, "Name", "")})
            if oid: child_ids.append(oid)

        # 2. Process Spaces (Native)
        spaces = reader.get_spaces()
        if spaces:
            logger.info(f"Processing {len(spaces)} native IFC spaces...")
            for space in spaces:
                v, b = proc.extract(space)
                oid = add_object("Room", v, b, {"name": getattr(space, "Name", "Space")})
                if oid: child_ids.append(oid)
        else:
            # --- MODIFICATION: VOXEL FALLBACK ---
            logger.info("No IfcSpace found. Attempting Voxel Detection...")
            detector = VoxelSpaceDetector(reader, voxel_size=0.5)
            detected_rooms = detector.detect()
            logger.info(f"Voxelization detected {len(detected_rooms)} rooms.")

            for i, room in enumerate(detected_rooms):
                oid = add_object("Room", room["vertices"], room["boundaries"], {"name": f"Detected Room {i+1}"})
                if oid: child_ids.append(oid)
            # ------------------------------------

        # 3. Create parent Building and link children
        building_id = str(uuid4())
        city_objects[building_id] = {
            "type": "Building",
            "attributes": {},
            "geometry": [],
            "children": child_ids
        }
        for cid in child_ids:
            city_objects[cid]["parents"] = [building_id]

        # Final JSON
        cj = {
            "type": "CityJSON",
            "version": "2.0",
            "transform": {
                "scale": [0.001, 0.001, 0.001],
                "translate": geo.translate
            },
            "metadata": {
                "referenceSystem": geo.reference_system
            },
            "CityObjects": city_objects,
            "vertices": all_vertices
        }
        
        with open(self.out, "w") as f:
            json.dump(cj, f, separators=(',', ':'))
        
        logger.info(f"Saved to {self.out}")
        
        try:
            logger.info("Validating...")
            subprocess.run(["cjio", self.out, "validate"], check=False)
        except Exception: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    
    Converter(args.input, args.output).run()
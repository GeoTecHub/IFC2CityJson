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
import os
import subprocess
import math
import logging
import sys
import argparse
from uuid import uuid4
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)

# --- Classes ---

class IFCReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.ifc_file = None
        self.settings = None

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
                shape = ifcopenshell.geom.create_shape(self.reader.settings, elem)
                verts = np.array(shape.geometry.verts).reshape((-1, 3))
                if verts.size > 0:
                    min_pt = np.minimum(min_pt, verts.min(axis=0))
                    max_pt = np.maximum(max_pt, verts.max(axis=0))
                    mesh_cache.append(verts)
            except: continue

        # Padding
        min_pt -= 1.0
        max_pt += 1.0
        dims = np.ceil((max_pt - min_pt) / self.voxel_size).astype(int)
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
        
        # Identify "Outside" (assumed largest region touching corner)
        corners = [labeled_array[0,0,0], labeled_array[-1,-1,-1]]
        counts = np.bincount([c for c in corners if c > 0])
        outside_id = np.argmax(counts) if len(counts) > 0 else 0

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
            except: continue
            
        return rooms

class GeometryProcessor:
    def __init__(self, reader):
        self.reader = reader

    def extract(self, entity):
        try:
            shape = ifcopenshell.geom.create_shape(self.reader.settings, entity)
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
        except:
            return None, None

    def fallback_box(self, entity):
        if not entity.is_a("IfcSpace"): return None, None
        v = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,3], [1,0,3], [1,1,3], [0,1,3]]
        try:
            m = ifcopenshell.util.placement.get_local_placement(entity.ObjectPlacement)
            tx, ty, tz = m[0][3], m[1][3], m[2][3]
            v = [[p[0]+tx, p[1]+ty, p[2]+tz] for p in v]
        except: pass
        b = [[[0,3,2,1]], [[4,5,6,7]], [[0,1,5,4]], [[1,2,6,5]], [[2,3,7,6]], [[3,0,4,7]]]
        return v, b

class Georeferencer:
    def __init__(self, reader):
        self.reader = reader
        self.translate = [0.0, 0.0, 0.0]
        self.found_crs = False

    def solve(self):
        project = self.reader.get_project()
        if project:
            for ctx in project.RepresentationContexts or []:
                if ctx.is_a("IfcGeometricRepresentationContext") and getattr(ctx, "HasCoordinateOperation", None):
                    for op in ctx.HasCoordinateOperation:
                        if op.is_a("IfcMapConversion"):
                            self.translate = [float(op.Eastings), float(op.Northings), float(op.OrthogonalHeight)]
                            self.found_crs = True
                            return
        if not self.found_crs:
            buildings = self.reader.get_buildings()
            if buildings and buildings[0].ObjectPlacement:
                try:
                    m = ifcopenshell.util.placement.get_local_placement(buildings[0].ObjectPlacement)
                    self.translate = [float(m[0][3]), float(m[1][3]), float(m[2][3])]
                    logger.info(f"Dynamic center calculated: {self.translate}")
                except: pass

class Converter:
    def __init__(self, ifc, out):
        self.ifc = ifc
        self.out = out
    
    def map_type(self, entity):
        t = entity.is_a()
        if "Space" in t: return "Room"
        if "Window" in t or "Door" in t: return "BuildingInstallation"
        return "BuildingPart"

    def run(self):
        reader = IFCReader(self.ifc)
        if not reader.load(): return
        
        geo = Georeferencer(reader)
        geo.solve()
        proc = GeometryProcessor(reader)
        
        city_objects = {}
        all_vertices = []
        
        def add_verts(raw_verts):
            base_idx = len(all_vertices)
            for v in raw_verts:
                x = int(round((v[0] - geo.translate[0]) * 1000))
                y = int(round((v[1] - geo.translate[1]) * 1000))
                z = int(round((v[2] - geo.translate[2]) * 1000))
                all_vertices.append([x, y, z])
            return base_idx

        def add_object(cj_type, verts, bounds, attrs=None):
            if not verts: return
            
            base = add_verts(verts)
            new_bounds = []
            for surf in bounds:
                new_surf = []
                for ring in surf:
                    new_surf.append([idx + base for idx in ring])
                new_bounds.append(new_surf)
            
            obj = {
                "type": cj_type,
                "geometry": [{
                    "type": "MultiSurface",
                    "lod": "2.2", 
                    "boundaries": new_bounds
                }]
            }
            if attrs: obj["attributes"] = attrs
            city_objects[str(uuid4())] = obj

        # 1. Process Elements
        logger.info("Processing standard elements...")
        for elem in reader.get_elements():
            v, b = proc.extract(elem)
            add_object(self.map_type(elem), v, b, {"ifc_type": elem.is_a(), "name": getattr(elem, "Name", "")})

        # 2. Process Spaces (Native)
        spaces = reader.get_spaces()
        if spaces:
            logger.info(f"Processing {len(spaces)} native IFC spaces...")
            for space in spaces:
                v, b = proc.extract(space)
                add_object("Room", v, b, {"name": getattr(space, "Name", "Space")})
        else:
            # --- MODIFICATION: VOXEL FALLBACK ---
            logger.info("No IfcSpace found. Attempting Voxel Detection...")
            detector = VoxelSpaceDetector(reader, voxel_size=0.5)
            detected_rooms = detector.detect()
            logger.info(f"Voxelization detected {len(detected_rooms)} rooms.")
            
            for i, room in enumerate(detected_rooms):
                add_object("Room", room["vertices"], room["boundaries"], {"name": f"Detected Room {i+1}"})
            # ------------------------------------

        # Final JSON
        cj = {
            "type": "CityJSON",
            "version": "2.0",
            "transform": {
                "scale": [0.001, 0.001, 0.001],
                "translate": geo.translate
            },
            "metadata": {},
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
    
    setup_logging()
    Converter(args.input, args.output).run()
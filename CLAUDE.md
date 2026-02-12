# CLAUDE.md — IFC2CityJson

## Project Overview

IFC2CityJson is a Python converter that transforms IFC (Industry Foundation Classes) BIM models into CityJSON 2.0 format. It features voxel-based room detection as a fallback when IFC files lack explicit `IfcSpace` definitions, georeferencing (defaulting to SLD99/EPSG:5235), vertex deduplication, and CityJSON 2.0 validation via `cjio`.

## Repository Structure

```
IFC2CityJson/
├── ifc2cityjson_vox.py       # Single-file converter (all source code)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── LargeBuilding.ifc          # Sample IFC input (gitignored by default)
├── simple-model-spaces01.ifc  # Sample IFC input (gitignored by default)
└── largebuilding.json         # Sample CityJSON output (committed)
```

This is a **single-file project** — all source code lives in `ifc2cityjson_vox.py` (430 lines).

## Architecture

### Class Hierarchy

| Class | Lines | Purpose |
|-------|-------|---------|
| `IFCReader` | 42–85 | Loads IFC files via ifcopenshell, caches geometry shapes, retrieves elements by type |
| `VoxelSpaceDetector` | 88–209 | Voxel-based room detection fallback using numpy/scipy when no `IfcSpace` exists |
| `GeometryProcessor` | 211–252 | Extracts vertex/face data from IFC shapes, maps to CityJSON MultiSurface format |
| `Georeferencer` | 254–291 | Resolves CRS from IFC or defaults to SLD99 (EPSG:5235), applies coordinate translation |
| `Converter` | 293–422 | Main orchestrator: coordinates all classes, builds CityJSON output, validates result |

### Execution Flow

1. `Converter.run()` creates `IFCReader` and loads the IFC file
2. `Georeferencer.solve()` determines CRS and origin translation
3. `GeometryProcessor` extracts geometry from each building element
4. If no native `IfcSpace` objects exist, `VoxelSpaceDetector` runs as fallback
5. Vertices are deduplicated and scaled to integer mm precision (×1000)
6. CityJSON 2.0 structure is assembled with parent Building + children hierarchy
7. JSON is written and validated via `cjio validate`

### IFC Type to CityJSON Type Mapping

- `IfcSpace` → `"Room"`
- `IfcWindow`, `IfcDoor` → `"BuildingInstallation"`
- All other elements (Wall, Slab, Roof, Column, Beam, Stair) → `"BuildingPart"`

## Dependencies

Listed in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `ifcopenshell` | IFC file parsing and geometry processing |
| `numpy` | Numerical arrays for voxelization |
| `scipy` | Connected component labeling (`ndimage`), ConvexHull for room geometry |
| `cjio` | CityJSON validation (called as subprocess) |

### Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 ifc2cityjson_vox.py <input.ifc> <output.json>
```

Example:
```bash
python3 ifc2cityjson_vox.py LargeBuilding.ifc output.json
```

## Key Technical Details

### Georeferencing

- Default CRS: SLD99 / Sri Lanka Grid 1999 (EPSG:5235)
- Default origin: Easting 399800, Northing 492200, Height 10 (central Colombo area)
- If the IFC file contains an `IfcMapConversion`, those values are used instead
- Building placement offsets are added on top of the CRS origin when no conversion is found

### Voxel Space Detection Algorithm

Activated when no `IfcSpace` objects are found in the IFC file:

1. Compute bounding box of all building elements with 1m padding
2. Create 3D boolean grid at 0.5m voxel resolution
3. Rasterize element meshes into grid (mark occupied cells)
4. Find void regions using binary complement + `scipy.ndimage.label`
5. Identify "outside" region from 8 grid corners (majority vote)
6. Extract interior void regions as rooms (filter < 3m³ as noise)
7. Generate `ConvexHull` geometry for each detected room

Memory safeguard: grids exceeding 50M cells are rejected.

### Vertex Handling

- Vertices are translated relative to the georeferenced origin
- Scaled to integer millimeters (×1000) for CityJSON `transform.scale: [0.001, 0.001, 0.001]`
- Deduplicated by `(x, y, z)` key to reduce output size

### CityJSON Output Format

- Version: 2.0
- Geometry type: MultiSurface at LoD 2.2
- A parent `Building` object is always created with all elements as `children` (required for CityJSON 2.0 compliance)
- Object IDs are UUIDs

## Development Notes

### No Build System

There is no `setup.py`, `pyproject.toml`, or build tooling. The project is run directly as a Python script.

### No Test Suite

There are no automated tests. Manual testing is done using the included sample IFC files:
- `LargeBuilding.ifc` — complex multi-story building
- `simple-model-spaces01.ifc` — simpler model with explicit spaces

Validate output manually:
```bash
cjio output.json validate
```

### No Linter/Formatter Configuration

No `.pylintrc`, `.flake8`, `pyproject.toml`, or formatter config exists. When modifying code, follow the existing style:
- 4-space indentation
- f-string logging
- `logging` module (not print) for all output
- `except Exception` for error handling (not bare `except`)
- Type-prefixed class names (`IFCReader`, `GeometryProcessor`, etc.)

### Logging

All classes use the module-level `logger` (`logging.getLogger(__name__)`). Format:
```
%(asctime)s [%(levelname)-8s] %(name)s - %(message)s
```

### Error Handling Patterns

- Geometry extraction failures fall back to a 4×4×3m box for `IfcSpace` entities, `None` for others
- Voxel detection degrades gracefully if numpy/scipy are not installed (`VOXEL_AVAILABLE` flag)
- CityJSON validation failures are non-blocking (subprocess errors caught and ignored)

### .gitignore Notes

- `*.ifc` and `*.json` are gitignored by default
- `largebuilding.json` and `requirements.txt` are explicitly un-ignored with `!` prefix
- Virtual environments (`.venv/`, `venv/`) and compiled Python files are ignored

## Commit Message Conventions

Based on the git history, commit messages use imperative mood and describe the change directly:
- `"Fix CityJSON 2.0 validation by adding parent Building object"`
- `"Georeference output to SLD99 / Sri Lanka Grid 1999 (EPSG:5235)"`
- `"Fix bugs, improve robustness, and add project files from code review"`

## Common Tasks

### Adding a new IFC element type

1. Add the IFC type string to `IFCReader.get_elements()` types list (line 68)
2. If it needs a special CityJSON type mapping, add it to `Converter.IFC_TYPE_MAP` (line 298)

### Changing the default CRS

Edit `Georeferencer.DEFAULT_CRS` and `Georeferencer.DEFAULT_TRANSLATE` (lines 257–258).

### Adjusting voxel resolution

Pass a different `voxel_size` to `VoxelSpaceDetector.__init__()`. Currently hardcoded to 0.5m in `Converter.run()` (line 379).

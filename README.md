# Geospatial YOLO Pipeline (geospatial-yolo-pipeline)

[![Pixi](https://img.shields.io/badge/package--manager-pixi-blue.svg)](https://pixi.sh)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.18.0-green.svg)](http://www.open3d.org/)
[![Voxel51 FiftyOne](https://img.shields.io/badge/FiftyOne-Voxel51-orange.svg)](https://voxel51.com/fiftyone/)

A robust, multi-campaign pipeline for **slicing, raycasting, stratifying, and mapping YOLO datasets** derived from high-resolution aerial imagery. 

The pipeline ensures high-quality dataset preparation for object detection models by:
1. **Slicing** large orthophoto sensors/tiles into model-ready patches while preserving local coordinate bounds.
2. **Raycasting** image slice corners onto 3D surface meshes (e.g., photogrammetry digital twin models) to compute accurate real-world geospatial footprints.
3. **Stratifying** datasets using spatial partitions (e.g., North-South separation) to prevent geographic data leakage during validation and testing.
4. **Ingesting** georeferenced slices into **Voxel51 FiftyOne** for interactive visualization and spatial queries, and generating cartographic split maps.

---

## 🗺️ Pipeline Architecture & Workflow

```mermaid
graph TD
    %% Define inputs
    subgraph Raw Input Data
        A1[Combined Images & YOLO Labels]
        A2[Metashape Camera XMLs]
        A3[3D Surface Meshes .obj]
    end

    %% Pipeline Steps
    subgraph Step 1: Slicing
        B1[prepare_sliced_dataset_shiftback] --> B2[Sliced Images & Labels]
        B1 --> B3[slice_metadata.json]
    end

    subgraph Step 2: 3D Raycasting
        C1[Open3D Raycaster Engine] --> C2[image_footprints.geojson]
        C1 --> C3[image_edge_coordinates.csv]
    end

    subgraph Step 3: Spatial Stratification
        D1[Spatial Spatio-Temporal Splitter] --> D2[Train Split / Val Split / Test Split]
        D1 --> D3[YOLO data.yaml]
        D1 --> D4[dropped_images.txt]
    end

    subgraph Step 4: FiftyOne & Mapping
        E1[FiftyOne Dataset Ingestion] --> E2[Interactive Voxel51 App]
        E1 --> E3[geostrat_dataset_map.png]
    end

    %% Connections
    A1 --> B1
    B3 --> C1
    A2 & A3 --> C1
    C2 --> D1
    B2 --> D1
    D2 --> E1
    C2 --> E1
```

---

## 🛠️ Environment Setup & Dependencies

This project uses [Pixi](https://pixi.sh) for declarative dependency and task management. It integrates Conda packages for complex geospatial/3D graphics libraries with PyPI packages to prevent library collisions.

### Installation

To set up the environment, run:

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Initialize and install all dependencies
pixi install
```

### Dependencies Overview
- **Core Environment**: Python `3.10.*`, NumPy `<2.0.0`, `tqdm`.
- **Geospatial & Cartography**: `geopandas`, `shapely`, `pyproj`, `contextily`, `matplotlib`, `matplotlib-scalebar`, `seaborn`.
- **3D Graphics & ML Visuals**:
  - `open3d` (PyPI) - Tensor-based raycasting and mesh interface.
  - `fiftyone` (PyPI) - Dataset exploration, metadata curation, and GIS mapping.
  - `opencv-python` (PyPI) - Fast image operations (slicing and cropping).

---

## 🚀 Execution Tasks

The project configuration defines four sequential CLI tasks inside [pixi.toml](file:///C:/Users/emilb/_trainer_dataset_prep/pixi.toml). 

Currently, the implementation resides in the notebook [_rework_slice_stratify_import.ipynb](file:///C:/Users/emilb/_trainer_dataset_prep/_rework_slice_stratify_import.ipynb). You can run them via the CLI as soon as they are compiled to standalone scripts:

```bash
# Step 1: Slice the orthophoto dataset into patches
pixi run slice

# Step 2: Raycast local coordinates onto the 3D meshes to get GeoJSON footprints
pixi run raycast

# Step 3: Stratify the sliced images into geographically separated Train/Val/Test subsets
pixi run stratify

# Step 4: Ingest the final dataset into FiftyOne and draw static cartographic maps
pixi run visualize
```

---

## 📖 Pipeline Breakdown

The code is divided into four main logical modules inside [_rework_slice_stratify_import.ipynb](file:///C:/Users/emilb/_trainer_dataset_prep/_rework_slice_stratify_import.ipynb):

### 1. Dataset Slicing (Metadata Export)
* **Function**: `prepare_sliced_dataset_shiftback(...)`
* **Purpose**: Divides large, high-resolution source orthophotos and associated YOLO annotations into standard tile sizes (e.g., `1024x1024` pixels).
* **Key Innovation**: Tracks the exact pixel bounding box of every generated tile relative to the parent sensor and exports a `slice_metadata.json` mapping database. This preserves local spatial references so the raycaster knows exactly where each crop originated on the parent sensor.

### 2. 3D Raycasting (Georeferencing)
* **Function**: `run_multi_campaign_pipeline(...)`
* **Purpose**: Projects the local 2D pixel coordinates of each image slice corner back through the camera sensor model into 3D UTM coordinate space.
* **Mechanism**:
  - Parses camera projection orientation from Metashape XML files (`cameras.xml` / `cameras_33n.xml`).
  - Sets up an Open3D tensor-based raycaster scene using highly detailed decimate 3D meshes (`.obj`).
  - Casts rays from the cameras' optical centers through the slice boundary corners to find the exact intersection with the digital twin mesh surface.
  - Uses `pyproj` to transform the coordinates to standard EPSG zones (e.g., `32633` WGS 84 / UTM zone 33N).
* **Outputs**:
  - `image_footprints.geojson`: Detailed spatial polygons for all georeferenced slices.
  - `image_edge_coordinates.csv`: Tabular coordinate limits.

### 3. Spatial Stratification
* **Function**: `generate_spatial_splits(...)` and `build_dataset_structure(...)`
* **Purpose**: Implements a spatial train-test split that avoids spatial autocorrelation (data leakage). Simply splitting slices randomly causes neighboring (overlapping) slices to fall into both Train and Test, yielding overly optimistic evaluation metrics.
* **Mechanism**:
  - Automatically cleans bowtie or invalid geometries in the raycasted GeoJSON footprints.
  - Drops oversized polygons (e.g., raycaster misses or boundary noise exceeding `600%` of the average area).
  - Segregates data geographically: allocating the southern region (South) to **TEST**, northern region (North) to **VAL**, and central regions to **TRAIN**.
  - Creates a boundary buffer: overlapping or boundary-crossing slices are dropped to guarantee independent validation.
* **Outputs**:
  - Sliced dataset organized into YOLO-compliant folder structures (`images/train`, `labels/train`, etc.).
  - `data.yaml`: YOLOv8-compatible dataset configuration.
  - `dropped_images.txt`: Diagnostics of slices discarded due to size or boundary-clash.

### 4. FiftyOne Ingestion & Visualization
* **Function**: `load_and_visualize_dataset(...)`
* **Purpose**: Slicing images strips metadata (like EXIF coordinates) because OpenCV doesn't carry over tags. This module integrates the georeferenced database back into Voxel51 FiftyOne.
* **Key Features**:
  - Parses the `image_footprints.geojson` file to calculate the footprint centroids.
  - Ingests samples and attaches spatial GPS labels (`fo.GeoLocation`) to each dataset record.
  - Spawns the FiftyOne App showing the visual grids alongside an interactive map panel.
  - Calls a cartographic engine (`contextily` + `matplotlib` + `matplotlib-scalebar`) to render a publication-ready overview map (`geostrat_dataset_map.png`) showing spatial training groupings.

---

## 📈 Example Processing Outputs (Diagnostics)

```text
===Raycasting===
Processing Campaign 2023... cameras.xml... totareaAug23_decim.obj... Mapped 3922 slices.
Processing Campaign 2024... cameras_33n.xml... totarea_decim.obj... Mapped 3375 slices.
Finished processing all campaigns. 7297/7450 total slices mapped.

===Stratification===
[INFO] Loading spatial footprints from: image_footprints.geojson
[DIAGNOSTIC] Loaded 7297 image footprints.
[INFO] Checking for oversized polygons... Dropped 3 footprints.
[INFO] Forcing orientation: South -> North.
[INFO] Allocated 510 images to TEST (South).
[INFO] Allocated 510 images to VAL (North).
[INFO] Computing intersections (Max allowed overlap: 0%)...
[INFO] Dropped 749 boundary-overlap images.

[STATS] SPATIAL SPLIT SUMMARY
------------------------------
 Total Footprints Analyzed : 7297
 Retained (Safe Set)       : 6545
   -> Train Allocation     : 5525
   -> Val Allocation       : 510
   -> Test Allocation      : 510
============================================================
 Process Complete
============================================================
```

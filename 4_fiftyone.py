"""
4_fiftyone.py - FiftyOne Ingestion and Cartographic Overview Mapping
=====================================================================

This script integrates the georeferenced YOLO dataset into Voxel51 FiftyOne
and generates publication-ready cartographic maps of the spatial splits.

Key Capabilities:
-----------------
1. FiftyOne Dataset Curation:
   Loads YOLO bounding boxes, images, and split tags (`train`, `val`, `test`).
2. Geographic Geolocation Attachment:
   Reads polygon centroids and survey flight years from `image_footprints.geojson`
   and attaches `fo.GeoLocation` GPS coordinates to every sample.
3. Ghost / Buffer Sample Injection:
   Re-injects discarded boundary buffer images (`dropped_images.txt`) as tagged
   "ghost" points on the map so researchers can visually verify spatial isolation.
4. Publication-Ready Cartography:
   Generates a high-resolution overview map (`geostrat_dataset_map.png`) overlaid on
   an aerial satellite basemap (Esri World Imagery) with a scale bar, north arrow,
   and clean legend separating cohorts and flight years.
5. Interactive Web Server:
   Launches the FiftyOne web UI with dual-panel layout (Image Samples + Interactive Map)
   without automatic browser popups.
"""

import os
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import fiftyone as fo
import fiftyone.types as fot
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple


def _load_slice_data_from_geojson(geojson_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Extracts the polygon centroid (longitude, latitude) and survey campaign year
    for every image slice from a GeoJSON footprint file.

    Args:
        geojson_path (str): Path to `image_footprints.geojson`.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of `image_name` to:
            - `lon`: Centroid longitude (WGS84).
            - `lat`: Centroid latitude (WGS84).
            - `campaign`: Survey year (e.g. '2023', '2024').
    """
    slice_data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(geojson_path):
        print(f"[Warning] GeoJSON not found at: {geojson_path}. Map coordinates will be empty.")
        return slice_data

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for feature in data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})

        if geom.get('type') == 'Polygon' and 'image_name' in props:
            coords = geom['coordinates'][0]
            # Average polygon vertices (excluding the repeated closing coordinate)
            lons = [c[0] for c in coords[:-1]]
            lats = [c[1] for c in coords[:-1]]

            if lons and lats:
                centroid_lon = sum(lons) / len(lons)
                centroid_lat = sum(lats) / len(lats)

                slice_data[props['image_name']] = {
                    "lon": centroid_lon,
                    "lat": centroid_lat,
                    "campaign": str(props.get("campaign", "0"))
                }

    return slice_data

def generate_static_map(
    dataset: fo.Dataset,
    output_path: Path,
    show_grid: bool = False,
    base_marker_size: int = 15
) -> None:
    """
    Generates a publication-ready cartographic map of the dataset's geographic and temporal distribution.

    Visual Elements:
    ----------------
    - Overlaid on high-resolution satellite imagery (`Esri.WorldImagery` via Contextily).
    - Map groups color-coded:
      * Train: Red
      * Validation: Green
      * Test: Blue
      * Ghost (boundary buffer): Semi-transparent gray squares
      * Excluded orphans: Orange
    - Temporal dual-layering:
      * Newer flight campaign (e.g. 2024): Distinct circular marker with white 'x' center.
      * Older flight campaign (e.g. 2023): Solid circular marker.
    - True north arrow, metric scale bar, and external legend.

    Args:
        dataset (fo.Dataset): Ingested FiftyOne dataset containing geographic metadata.
        output_path (Path): Destination file path for the saved PNG map.
        show_grid (bool): If True, renders an EPSG:3857 coordinate grid over the map.
        base_marker_size (int): Baseline scatter point size for map symbols.
    """
    print("\n" + "=" * 50)
    print(" Executing Cartographic Mapping Engine")
    print("=" * 50)

    data = []
    for sample in dataset.iter_samples():
        if sample.has_field("location") and sample.location is not None:
            ts = sample.timestamp if sample.has_field("timestamp") else None
            year = datetime.fromtimestamp(ts).year if ts else 0

            data.append({
                "map_group": sample.map_group,
                "geometry": Point(sample.location.point),
                "year": year
            })

    if not data:
        print("[CRITICAL] Map generation aborted: No valid GPS data points found in dataset.")
        return

    # Convert WGS84 coordinates into Web Mercator (EPSG:3857) to align with online tile services
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326").to_crs(epsg=3857)

    print(f"[DIAGNOSTIC] Total geospatial coordinates extracted: {len(gdf)}")
    print("[DIAGNOSTIC] Distribution by Map Group:")
    for group_name, count in gdf['map_group'].value_counts().items():
        print(f"  -> {group_name.upper()}: {count} points")

    max_year = gdf[gdf['year'] > 0]['year'].max() if (gdf['year'] > 0).any() else 0
    old_years_series = gdf[(gdf['year'] < max_year) & (gdf['year'] > 0)]['year']
    old_year = old_years_series.max() if not old_years_series.empty else (max_year - 1 if max_year > 0 else 0)

    print(f"[DIAGNOSTIC] Temporal spans identified - Newest: {max_year}, Older layer: {old_year}")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.margins(0.15)

    colors = {
        "train": "red",
        "val": "green",
        "test": "blue",
        "ghost": "#7f8c8d",
        "no_footprint": "magenta",
        "excluded_orphan": "orange"
    }

    for map_group, group in gdf.groupby("map_group"):
        base_color = colors.get(map_group, "black")

        is_dropped = (map_group in ["ghost", "excluded_orphan"])
        alpha_val = 0.60 if is_dropped else 1.0
        edge_color = "grey" if is_dropped else "white"

        # Plot Newer Data (circle with 'x' overlay)
        new_group = group[group['year'] >= max_year]
        if not new_group.empty:
            new_group.plot(
                ax=ax,
                color=base_color,
                marker='o',
                markersize=base_marker_size * 1.25,
                edgecolor=edge_color,
                linewidth=0.8,
                alpha=alpha_val
            )
            new_group.plot(
                ax=ax,
                color="white" if not is_dropped else "#e0e0e0",
                marker='x',
                markersize=base_marker_size * 0.3,
                linewidth=0.6,
                alpha=alpha_val
            )

        # Plot Older Data (solid circle)
        old_group = group[(group['year'] < max_year) & (group['year'] > 0)]
        if not old_group.empty:
            old_group.plot(
                ax=ax,
                color=base_color,
                marker='o',
                markersize=base_marker_size,
                edgecolor=edge_color,
                linewidth=0.6,
                alpha=alpha_val
            )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - ((ymax - ymin) * 0.15), ymax)

    print("[INFO] Requesting aerial basemap from Esri World Imagery...")
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        print("[INFO] Basemap rendered successfully.")
    except Exception as e:
        print(f"[WARNING] Basemap download failed (check network connection): {e}")

    # True North Arrow Annotation
    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.87),
                arrowprops=dict(facecolor='white', edgecolor='black', width=4, headwidth=12),
                ha='center', va='center', fontsize=18, fontweight='bold', color='black',
                xycoords=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Metric Scale Bar
    scalebar = ScaleBar(1, "m", length_fraction=0.2, location="lower right",
                        box_alpha=0.9, bbox_to_anchor=(0.98, 0.06), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar)

    # Build Legend
    handles = []
    labels = []

    for map_group, color in colors.items():
        if map_group in gdf['map_group'].unique():
            if map_group == "ghost":
                handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, alpha=0.4))
                labels.append("DROPPED BUFFER")
            elif map_group == "no_footprint":
                handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10))
                labels.append("MISSING FOOTPRINT (TRAIN)")
            elif map_group == "excluded_orphan":
                handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, alpha=0.4))
                labels.append("EXCLUDED ORPHAN")
            else:
                handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10))
                labels.append(map_group.upper())

    handles.append(Line2D([0], [0], marker='', color='w', markerfacecolor='none', markersize=0))
    labels.append("---")

    # Legend symbols for flight campaigns
    layered_new_marker = (
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='white', markeredgewidth=1.5, markersize=8)
    )
    handles.append(layered_new_marker)
    labels.append(f"{int(max_year)} flight")

    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10))
    labels.append(f"{int(old_year)} flight")

    ax.legend(
        handles=handles, labels=labels, title="Dataset Legend",
        bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.9,
        fontsize=12, handler_map={tuple: HandlerTuple(ndivide=1)}
    )

    ax.set_title("GEOSPATIAL COHORT SEGREGATION MAP\n(Spatial Stratification Summary per Aerial Campaign Year)",
                 fontsize=11, fontweight='bold', family="sans-serif", color="#2c3e50")

    if show_grid:
        ax.grid(True, which='both', color='#bdc3c7', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Easting (Meters - EPSG:3857)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Northing (Meters - EPSG:3857)", fontsize=10, fontweight='bold')
    else:
        ax.set_axis_off()

    print("[INFO] Writing map image to disk...")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Publication-ready map saved to: {output_path}")


def load_and_visualize_dataset(
    folder_path: str,
    geojson_path: str,
    dataset_name: str = "yolo_dataset",
    show_grid: bool = False,
    show_excluded_orphans: bool = True
) -> None:
    """
    Ingests a stratified YOLO dataset into Voxel51 FiftyOne, attaches geospatial metadata,
    generates a static cohort overview map, and launches the FiftyOne interactive viewer.

    Args:
        folder_path (str): Path to stratified dataset containing `data.yaml`.
        geojson_path (str): Path to `image_footprints.geojson`.
        dataset_name (str): Identifier name for FiftyOne internal catalog.
        show_grid (bool): If True, draws metric coordinate grid on static map.
        show_excluded_orphans (bool): If True, shows un-georeferenced slices on map as orphans.
    """
    folder_path = Path(folder_path)
    yaml_path = folder_path / "data.yaml"

    source_images_dir = folder_path.parent / "sliced_dataset" / "images"
    dropped_log_path = folder_path / "dropped_images.txt"
    missing_footprints_path = folder_path / "missing_footprints.txt"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    
    print("="*50)
    print(" Starting FiftyOne Dataset Ingestion")
    print("="*50)
    
    if not yaml_path.exists():
        print(f"[Error] Configuration file missing: {yaml_path}")
        return
        
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        
    dataset = fo.Dataset(name=dataset_name)
    dataset.add_sample_field("split", fo.StringField)
    dataset.add_sample_field("map_group", fo.StringField)
    dataset.add_sample_field("timestamp", fo.FloatField)
    dataset.add_sample_field("campaign", fo.StringField)

    loaded_splits = []
    for split in ["train", "val", "test"]:
        try:
            dataset.add_dir(
                dataset_dir=str(folder_path),
                yaml_path=str(yaml_path),
                dataset_type=fot.YOLOv5Dataset,
                split=split,
                tags=[split],
            )
            loaded_splits.append(split)
        except Exception as e:
            print(f"  [Error] Split '{split}' failed to load: {e}")

    # Load spatial slice data and campaign years
    slice_metadata = _load_slice_data_from_geojson(geojson_path)

    missing_footprints = set()
    if missing_footprints_path.exists():
        with open(missing_footprints_path, "r") as f:
            missing_footprints = {line.strip() for line in f if line.strip()}

    if source_images_dir.exists():
        print("\n[Progress] Parsing logs for mapping injection...")
        
        dropped_stems = set()
        if dropped_log_path.exists():
            with open(dropped_log_path, "r") as f:
                dropped_stems = {line.strip() for line in f if line.strip()}
        
        loaded_stems = {Path(s.filepath).stem for s in dataset}
        injected_samples = []
        ghost_count = 0
        orphan_count = 0
        
        # Searching through all subdirectories in the sliced_dataset/images folder
        for root, _, files in os.walk(source_images_dir):
            for file_name in files:
                p = Path(root) / file_name
                if p.suffix not in IMAGE_EXTENSIONS:
                    continue
                    
                if p.stem in dropped_stems:
                    sample = fo.Sample(filepath=str(p), tags=["ghost"])
                    sample["split"] = "ghost"
                    injected_samples.append(sample)
                    ghost_count += 1
                    
                elif show_excluded_orphans and p.stem in missing_footprints and p.stem not in loaded_stems:
                    sample = fo.Sample(filepath=str(p), tags=["excluded_orphan", "no_footprint"])
                    sample["split"] = "excluded"
                    injected_samples.append(sample)
                    orphan_count += 1
                
        if injected_samples:
            dataset.add_samples(injected_samples)
            if ghost_count > 0:
                print(f"[INFO] Injected {ghost_count} boundary buffer samples for map visualization.")
            if orphan_count > 0:
                print(f"[INFO] Injected {orphan_count} excluded orphan samples for map visualization.")

    print("\n[Progress] Assigning Coordinates to metadata and calculating stats...")
    has_gps = False
    stats = defaultdict(lambda: {"bg": 0, "fg": 0})
    campaign_stats = defaultdict(int)
    campaign_split_stats = defaultdict(lambda: defaultdict(int)) # Nested counter for Splits + Campaigns
    
    for sample in dataset.iter_samples(progress=True):
        if sample.tags and "excluded_orphan" in sample.tags:
            split_name = "excluded"
        else:
            split_name = sample.tags[0] if sample.tags else "unknown"
            
        sample["split"] = split_name
        
        file_stem = Path(sample.filepath).stem
        if "excluded_orphan" in sample.tags:
            sample["map_group"] = "excluded_orphan"
        elif file_stem in missing_footprints:
            sample.tags.append("no_footprint")
            sample["map_group"] = "no_footprint"
        else:
            sample["map_group"] = split_name

        has_detections = (
            sample.has_field("ground_truth") and 
            sample.ground_truth is not None and 
            hasattr(sample.ground_truth, "detections") and 
            len(sample.ground_truth.detections) > 0
        )
        
        stat_key = "fg" if has_detections else "bg"
        stats["total"][stat_key] += 1
        stats[split_name][stat_key] += 1

        campaign_str = "Unknown" 
        
        # Use GeoJSON coordinates and campaign year instead of EXIF
        if file_stem in slice_metadata:
            lon = slice_metadata[file_stem]["lon"]
            lat = slice_metadata[file_stem]["lat"]
            sample["location"] = fo.GeoLocation(point=(lon, lat))
            has_gps = True
            
            # Reconstruct a valid timestamp from the campaign year
            campaign_str = str(slice_metadata[file_stem]["campaign"])
            if campaign_str.isdigit():
                # Creates a timestamp for Jan 1st of the specific campaign year
                sample["timestamp"] = datetime(int(campaign_str), 1, 1).timestamp()
                
        # Save campaign year to sample and update counters
        sample["campaign"] = campaign_str
        campaign_stats[campaign_str] += 1
        campaign_split_stats[split_name][campaign_str] += 1 # Increment split-specific counter
            
        sample.save()
    
    print("\n" + "="*50)
    print("                DATASET OVERVIEW")
    print("="*50)
    print(f"Total Samples Ingested: {len(dataset)}")
    
    print("\n--- Target Distribution (Foreground vs Background) ---")
    for s in loaded_splits:
        print(f"    [{s.upper()}] Foreground: {stats[s]['fg']} | Background: {stats[s]['bg']}")
        
    if "ghost" in stats:
        print(f"    [DROPPED BUFFER] Map-Only Points: {stats['ghost']['fg'] + stats['ghost']['bg']}")
    if "excluded" in stats:
        print(f"    [EXCLUDED ORPHANS] Map-Only Points: {stats['excluded']['fg'] + stats['excluded']['bg']}")

    print("\n--- Temporal Distribution (Overall) ---")
    for camp, count in sorted(campaign_stats.items()):
        print(f"    [{camp}] Total Slices: {count}")

    print("\n--- Split Breakdown by Campaign ---")
    for s in loaded_splits:
        print(f"    [{s.upper()}]")
        for camp, count in sorted(campaign_split_stats[s].items()):
            print(f"        -> {camp}: {count} slices")
            
    # Also print breakdown for ghosts/excluded if they exist
    for extra_split in ["ghost", "excluded"]:
        if extra_split in campaign_split_stats:
            print(f"    [{extra_split.upper()}]")
            for camp, count in sorted(campaign_split_stats[extra_split].items()):
                print(f"        -> {camp}: {count} slices")
                
    print("="*50 + "\n")

    if has_gps:
        map_output = folder_path / "geostrat_dataset_map.png"
        generate_static_map(dataset, output_path=map_output, show_grid=show_grid, base_marker_size=15)
    
    spaces = fo.Space(children=[
        fo.Panel(type="Samples"),
        fo.Panel(type="Map", state={"locationField": "location"})
    ]) if has_gps else None
    
    print("[Progress] Starting FiftyOne server...")
    session = fo.launch_app(dataset, auto=False, spaces=spaces)
    
    print("\n" + "="*50)
    print(" FIFTYONE WEB INTERFACE READY")
    print("="*50)
    print(f" Access the interface here: {session.url}")
    print(" (Copy and paste the URL into your browser to view)")
    print("="*50 + "\n")

    try:
        session.wait()
    except KeyboardInterrupt:
        print("\n[Progress] Shutting down FiftyOne server.")

def parse_args():
    """Parses command-line arguments for FiftyOne dataset ingestion and visualization."""
    import argparse
    import config

    parser = argparse.ArgumentParser(
        description="Ingest stratified YOLO dataset into Voxel51 FiftyOne, attach geolocations, and launch viewer."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--folder-path", "-i",
        type=str,
        default=str(config.FIFTYONE_FOLDER_PATH),
        help=f"Path to stratified dataset directory containing data.yaml (default from config.yaml: {config.FIFTYONE_FOLDER_PATH})."
    )
    parser.add_argument(
        "--geojson-path", "-g",
        type=str,
        default=str(config.FIFTYONE_GEOJSON_PATH),
        help=f"Path to image_footprints.geojson file (default from config.yaml: {config.FIFTYONE_GEOJSON_PATH})."
    )
    parser.add_argument(
        "--dataset-name", "-n",
        type=str,
        default=config.FIFTYONE_DATASET_NAME,
        help=f"Name for FiftyOne internal catalog dataset (default from config.yaml: '{config.FIFTYONE_DATASET_NAME}')."
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        default=False,
        help="Render coordinate grid on static satellite overview map."
    )
    parser.add_argument(
        "--no-orphans",
        action="store_true",
        default=False,
        help="Do not display un-georeferenced slices as orphans on the map."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_and_visualize_dataset(
        folder_path=args.folder_path,
        geojson_path=args.geojson_path,
        dataset_name=args.dataset_name,
        show_grid=args.show_grid,
        show_excluded_orphans=(not args.no_orphans)
    )

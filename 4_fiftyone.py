import os
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import fiftyone as fo
import fiftyone.types as fot
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def _load_slice_data_from_geojson(geojson_path):
    """Calculates the center coordinate and extracts the campaign year for each slice."""
    slice_data = {}
    if not os.path.exists(geojson_path):
        print(f"[Warning] GeoJSON not found at {geojson_path}. Map will be blank.")
        return slice_data
        
    with open(geojson_path, 'r') as f:
        data = json.load(f)
        
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        if geom.get('type') == 'Polygon' and 'image_name' in props:
            coords = geom['coordinates'][0]
            # Simple average for centroid of bounding box coordinates
            lons = [c[0] for c in coords[:-1]] # exclude closing point
            lats = [c[1] for c in coords[:-1]]
            
            centroid_lon = sum(lons) / len(lons)
            centroid_lat = sum(lats) / len(lats)
            
            slice_data[props['image_name']] = {
                "lon": centroid_lon,
                "lat": centroid_lat,
                "campaign": props.get("campaign", "0") # Extract the campaign year
            }
            
    return slice_data

def generate_static_map(dataset, output_path: Path, show_grid=False, base_marker_size=15):
    """
    Generates a publication-ready cartographic map of the dataset's geographic and temporal distribution.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The ingested FiftyOne dataset containing geographic metadata.
        output_path (Path): Destination file path for the saved PNG map.
        show_grid (bool): If True, renders an EPSG:3857 coordinate grid over the map.
        base_marker_size (int): The baseline scatter point size. Decrease to shrink the dots on the map.

    Returns:
        None: Saves the plot directly to disk.
    """
    print("\n" + "="*50)
    print(" Executing Cartographic Mapping Engine")
    print("="*50)
    
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
        print("[CRITICAL] Map generation aborted: No valid GPS data points found in the dataset.")
        return
        
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326").to_crs(epsg=3857)
    
    print(f"[DIAGNOSTIC] Total geospatial coordinates extracted: {len(gdf)}")
    print(f"[DIAGNOSTIC] Distribution by Map Group:")
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
        
        # Plot Newer Data (Now with the 'x' overlay)
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
            
        # Plot Older Data (Now just the plain dot)
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

    print("[INFO] Requesting basemap from Mapbox/Esri servers...")
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        print("[INFO] Basemap rendered successfully.")
    except Exception as e:
        print(f"[WARNING] Basemap download failed. Check network or EPSG limits. Error: {e}")
        
    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.87),
                arrowprops=dict(facecolor='white', edgecolor='black', width=4, headwidth=12),
                ha='center', va='center', fontsize=18, fontweight='bold', color='black',
                xycoords=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    scalebar = ScaleBar(1, "m", length_fraction=0.2, location="lower right", 
                        box_alpha=0.9, bbox_to_anchor=(0.98, 0.06), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar)
    
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
    
    # Legend update: Max year (2024) gets the cross
    layered_new_marker = (
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='white', markeredgewidth=1.5, markersize=8)
    )
    handles.append(layered_new_marker)
    labels.append(f"{int(max_year)} flight")
    
    # Legend update: Old year (2023) gets the dot
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10))
    labels.append(f"{int(old_year)} flight")

    # Show double legend for temporal layers (placed outside the plot)
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

def load_and_visualize_dataset(folder_path: str, geojson_path: str, dataset_name: str = "yolo_dataset", show_grid=False, show_excluded_orphans=True):
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

if __name__ == "__main__":
    load_and_visualize_dataset(
        folder_path=r"C:\Users\emilb\_data\dataset_fin\sliced_split_dataset", 
        geojson_path=r"C:\Users\emilb\_data\dataset_fin\sliced_dataset_edge_coordinates\image_footprints.geojson",
        show_grid=True,
        show_excluded_orphans=True
    )

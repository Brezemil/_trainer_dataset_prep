"""
3_stratification.py - Spatial Dataset Stratification and Leakage Prevention
===========================================================================

This script splits image slices geographically into Train, Validation, and Test
subsets, while enforcing a spatial boundary buffer to eliminate data leakage.

Why Spatial Stratification is Critical:
---------------------------------------
In aerial and satellite imagery, standard random train/test splitting causes severe
data leakage (spatial autocorrelation). Overlapping tiles or neighboring views of the
exact same tree would end up in both the training and test sets, artificially
inflating evaluation metrics (making models appear much better than they actually are).

Stratification Solution:
------------------------
1. Geographical Partitioning:
   Orders all georeferenced slices along the South-to-North axis (Northing).
   - Southernmost region -> Allocated to TEST.
   - Northernmost region -> Allocated to VALIDATION.
   - Central region      -> Allocated to TRAIN.
2. Boundary Buffer Isolation:
   Calculates spatial intersections between split boundaries. Any candidate training
   slice that touches or overlaps a test or validation slice is discarded into a
   buffer log (`dropped_images.txt`), creating an empty spatial moat between splits.
3. Geometric Cleansing:
   Automatically repairs bowtie/self-intersecting polygons using `buffer(0)` and
   discards oversized outlier polygons (e.g. raycasting misses).
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import geopandas as gpd

import config

# --- DEFAULT CONFIGURATION (Loaded from config.py / .env) ---
DATASET_DIR = config.STRAT_DATASET_DIR
OUTPUT_DIR = config.STRAT_OUTPUT_DIR
GEOJSON_PATH = config.STRAT_GEOJSON_PATH

SPLIT_RATIO = config.STRAT_SPLIT_RATIO
MAX_ALLOWED_OVERLAP_PCT = config.STRAT_MAX_OVERLAP_PCT
MAX_OVERSIZED_PCT = config.STRAT_MAX_OVERSIZED_PCT
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
CLASSES = config.STRAT_CLASSES
UTM_EPSG = config.STRAT_UTM_EPSG


def generate_spatial_splits(
    geojson_path: Path,
    split_ratio: Tuple[float, float, float],
    utm_epsg: int = 32633,
    output_dir: Optional[Path] = None,
    max_oversized_pct: float = MAX_OVERSIZED_PCT,
    max_allowed_overlap_pct: float = MAX_ALLOWED_OVERLAP_PCT
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Computes spatial train/val/test allocations and boundary buffer exclusions.

    Process Outline:
    ----------------
    1. Loads georeferenced slice polygons from GeoJSON.
    2. Reprojects to local metric CRS (e.g. UTM Zone 33N) for distortion-free metric geometry.
    3. Repairs self-intersecting / bowtie polygons using `.buffer(0)`.
    4. Filters out oversized outlier polygons (raycaster terrain misses).
    5. Sorts slices geographically from South to North using polygon centroid Y (Northing).
    6. Allocates the southern slice cohort to TEST and the northern slice cohort to VAL.
    7. Detects all cross-boundary polygon intersections between TRAIN and TEST/VAL.
    8. Discards boundary-crossing training slices into `dropped_images.txt` to guarantee
       spatial independence.

    Args:
        geojson_path (Path): Path to `image_footprints.geojson`.
        split_ratio (Tuple[float, float, float]): Allocation fractions (train, val, test).
        utm_epsg (int): EPSG zone for metric calculations.

    Returns:
        Tuple[Dict[str, str], Dict[str, Any]]:
            - `allocation_dict`: Mapping of image name to assigned split ('train', 'val', 'test').
            - `stats`: Diagnostic summary dictionary with counts and percentages.
    """
    print(f"[INFO] Loading spatial footprints from: {geojson_path.name}")
    gdf = gpd.read_file(geojson_path)
    initial_image_count = len(gdf)
    print(f"[DIAGNOSTIC] Loaded {initial_image_count} image footprints.")

    # Reproject to metric UTM projection for accurate square-meter calculations
    try:
        gdf_metric = gdf.to_crs(epsg=utm_epsg)
    except Exception as e:
        print(f"[WARNING] Could not re-project to EPSG:{utm_epsg}. Falling back to source CRS: {e}")
        gdf_metric = gdf.copy()

    # Repair topological self-intersections (bowties) created by 3D raycasting
    print("[INFO] Cleaning invalid geometries (fixing bowties with buffer(0))...")
    gdf_metric['geometry'] = gdf_metric['geometry'].buffer(0)

    # --- OVERSIZED POLYGON SAFEGUARD ---
    print(f"[INFO] Checking for oversized polygons (Threshold: +{max_oversized_pct}% of mean)...")
    average_area = gdf_metric.area.mean()
    area_multiplier = 1.0 + (max_oversized_pct / 100.0)
    max_allowed_area = average_area * area_multiplier

    oversized_mask = gdf_metric.area > max_allowed_area
    oversized_count = int(oversized_mask.sum())

    if oversized_count > 0:
        print(f"[WARNING] Dropped {oversized_count} footprint(s) exceeding {max_oversized_pct}% of average area.")
        print(f"           -> Average Area: {average_area:.2f} m² | Max Allowed: {max_allowed_area:.2f} m²")
        gdf_metric = gdf_metric[~oversized_mask].reset_index(drop=True)
    else:
        print(f"[INFO] All footprints within size threshold (Avg Area: {average_area:.2f} m²).")

    # --- SOUTH-TO-NORTH GEOGRAPHIC SORTING ---
    print("[INFO] Forcing geographic orientation: South -> North.")
    gdf_metric['sort_val'] = gdf_metric.geometry.centroid.y
    gdf_metric = gdf_metric.sort_values('sort_val').reset_index(drop=True)

    n_total = len(gdf_metric)
    n_test = int(n_total * split_ratio[2])
    n_val = int(n_total * split_ratio[1])

    # Assign initial cohorts
    gdf_metric['split'] = 'train'
    gdf_metric.loc[:n_test - 1, 'split'] = 'test'
    print(f"[INFO] Allocated {n_test} images to TEST (Southern cohort).")

    gdf_metric.loc[n_total - n_val:, 'split'] = 'val'
    print(f"[INFO] Allocated {n_val} images to VAL (Northern cohort).")

    # --- BOUNDARY BUFFER: SPATIAL INTERSECTION CHECK ---
    print(f"[INFO] Computing cross-split intersections (Max allowed overlap: {max_allowed_overlap_pct}%)...")
    overlaps = gpd.sjoin(gdf_metric, gdf_metric, how='inner', predicate='intersects')
    cross_boundary = overlaps[overlaps['split_left'] != overlaps['split_right']]
    leakage_images: Set[str] = set()

    for idx_left, row in cross_boundary.iterrows():
        idx_right = row['index_right']
        split_left = row['split_left']
        split_right = row['split_right']

        # Only drop from TRAIN when it borders TEST or VAL (preserve test/val evaluations intact)
        if 'train' not in (split_left, split_right):
            continue

        train_img_name = row['image_name_left'] if split_left == 'train' else row['image_name_right']
        if train_img_name in leakage_images:
            continue

        poly_left = gdf_metric.loc[idx_left, 'geometry']
        poly_right = gdf_metric.loc[idx_right, 'geometry']

        intersection_area = poly_left.intersection(poly_right).area
        min_area = min(poly_left.area, poly_right.area)

        if min_area > 0:
            overlap_pct = (intersection_area / min_area) * 100.0
            if overlap_pct > max_allowed_overlap_pct:
                leakage_images.add(train_img_name)

    boundary_dropped_count = len(leakage_images)
    safe_gdf = gdf_metric[~gdf_metric['image_name'].isin(leakage_images)]
    allocation_dict = dict(zip(safe_gdf['image_name'], safe_gdf['split']))

    split_stats = safe_gdf['split'].value_counts().to_dict()
    stats = {
        "initial": initial_image_count,
        "oversized_dropped": oversized_count,
        "boundary_dropped": boundary_dropped_count,
        "retained": len(allocation_dict),
        "train": split_stats.get("train", 0),
        "val": split_stats.get("val", 0),
        "test": split_stats.get("test", 0)
    }

    # Log boundary buffer dropped slices for diagnostics and FiftyOne map rendering
    target_out_dir = output_dir or OUTPUT_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)
    dropped_log_path = target_out_dir / "dropped_images.txt"
    with open(dropped_log_path, "w", encoding="utf-8") as f:
        for img in sorted(leakage_images):
            f.write(f"{img}\n")
    print(f"[INFO] Logged {len(leakage_images)} boundary-buffer images to: {dropped_log_path.name}")

    return allocation_dict, stats


def create_yolo_yaml(output_dir: Path, classes: List[str]) -> None:
    """
    Generates a YOLOv8/v11/v12 compatible `data.yaml` configuration file.

    Args:
        output_dir (Path): Destination root folder of the dataset.
        classes (List[str]): List of target class labels.
    """
    yaml_path = output_dir / "data.yaml"
    formatted_classes = "[" + ", ".join([f"'{c}'" for c in classes]) + "]"

    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(classes)}
names: {formatted_classes}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def build_dataset_structure(
    allocation_dict: Dict[str, str],
    dataset_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Physically organizes images and labels into standard YOLO split folders:
    - `<OUTPUT_DIR>/train/images/` and `labels/`
    - `<OUTPUT_DIR>/val/images/` and `labels/`
    - `<OUTPUT_DIR>/test/images/` and `labels/`

    Args:
        allocation_dict (Dict[str, str]): Mapping of slice stem to assigned split name.
        dataset_dir (Optional[Path]): Source directory containing sliced images and labels.
        output_dir (Optional[Path]): Target root directory for split dataset.
    """
    src_dir = dataset_dir or DATASET_DIR
    target_dir = output_dir or OUTPUT_DIR

    all_files = list(src_dir.rglob("*"))
    all_images = [f for f in all_files if f.is_file() and f.suffix in IMAGE_EXTENSIONS]
    all_labels = [f for f in all_files if f.is_file() and f.suffix.lower() == ".txt"]

    img_map = {f.stem.strip(): f for f in all_images}
    lbl_map = {f.stem.strip(): f for f in all_labels}

    print("\n" + "=" * 40)
    print("[DEBUG] DIRECTORY SCAN RESULTS")
    print(f"  -> Searched inside: {src_dir}")
    print(f"  -> Valid images found: {len(img_map):,}")
    print(f"  -> Valid .txt labels found: {len(lbl_map):,}")
    print("=" * 40)

    if len(img_map) == 0:
        print(f"\n[CRITICAL ERROR] 0 images found. Verify dataset_dir path: {src_dir}")
        return

    # Create destination directories
    for split_name in ["train", "val", "test"]:
        (target_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Copying stratified files to {target_dir}...")
    missing_from_disk = 0
    missing_labels = 0
    copied_files = 0

    for raw_img_stem, assigned_split in allocation_dict.items():
        clean_stem = Path(raw_img_stem).stem.strip()

        if clean_stem not in img_map:
            missing_from_disk += 1
            continue

        img_path = img_map[clean_stem]
        label_path = lbl_map.get(clean_stem)

        if not label_path:
            missing_labels += 1
            continue

        img_target = target_dir / assigned_split / "images" / img_path.name
        lbl_target = target_dir / assigned_split / "labels" / label_path.name

        shutil.copy2(img_path, img_target)
        shutil.copy2(label_path, lbl_target)
        copied_files += 1

    print("\n" + "-" * 30)
    print("[DIAGNOSTIC] FILE COPY SUMMARY")
    print("-" * 30)
    print(f" Allocated Footprints   : {len(allocation_dict):,}")
    print(f" Missing from Disk      : {missing_from_disk}")
    print(f" Missing Label Files    : {missing_labels}")
    print(f" Successfully Copied    : {copied_files:,}")
    print("-" * 30)


def parse_args():
    """Parses command-line arguments for spatial dataset stratification."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Stratify aerial YOLO dataset geographically into Train/Val/Test splits with boundary buffer isolation."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--dataset-dir", "-i",
        type=Path,
        default=DATASET_DIR,
        help="Source directory containing sliced images and labels."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination directory for stratified dataset splits."
    )
    parser.add_argument(
        "--geojson-path", "-g",
        type=Path,
        default=GEOJSON_PATH,
        help="Path to image_footprints.geojson file."
    )
    parser.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=list(SPLIT_RATIO),
        help="Fractions for Train, Val, Test splits (e.g. 0.86 0.07 0.07)."
    )
    parser.add_argument(
        "--max-overlap-pct",
        type=float,
        default=MAX_ALLOWED_OVERLAP_PCT,
        help="Maximum allowed percentage overlap between Train and Val/Test before dropping to buffer (default: 0.0)."
    )
    parser.add_argument(
        "--max-oversized-pct",
        type=float,
        default=MAX_OVERSIZED_PCT,
        help="Drop footprint if area exceeds the mean by this percentage (default: 600.0)."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=CLASSES,
        help="Class names list for YOLO data.yaml."
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=UTM_EPSG,
        help="EPSG code for metric calculation (default: 32633 for UTM Zone 33N)."
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point for spatial stratification."""
    args = parse_args()
    print("=" * 60)
    print(" Executing Spatial Spatio-Temporal Splitter")
    print("=" * 60)

    allocation_dict, stats = generate_spatial_splits(
        geojson_path=args.geojson_path,
        split_ratio=tuple(args.split_ratio),
        utm_epsg=args.utm_epsg,
        output_dir=args.output_dir,
        max_oversized_pct=args.max_oversized_pct,
        max_allowed_overlap_pct=args.max_overlap_pct
    )

    print("\n" + "-" * 30)
    print("[STATS] SPATIAL SPLIT SUMMARY")
    print("-" * 30)
    print(f" Total Footprints Analyzed : {stats['initial']:,}")
    print(f" Dropped (Oversized)       : {stats['oversized_dropped']} ({(stats['oversized_dropped']/stats['initial'])*100:.2f}%)")
    print(f" Dropped (Boundary Buffer) : {stats['boundary_dropped']} ({(stats['boundary_dropped']/stats['initial'])*100:.2f}%)")
    print(f" Retained (Safe Set)       : {stats['retained']:,}")
    print(f"   -> Train Allocation     : {stats['train']:,}")
    print(f"   -> Val Allocation       : {stats['val']:,}")
    print(f"   -> Test Allocation      : {stats['test']:,}")
    print("-" * 30)

    if stats['retained'] == 0:
        print("\n[ERROR] All images were dropped. Adjust SPLIT_RATIO configuration.")
        return

    build_dataset_structure(allocation_dict, dataset_dir=args.dataset_dir, output_dir=args.output_dir)
    create_yolo_yaml(args.output_dir, args.classes)
    print(f"[INFO] Generated YOLO configuration at: {args.output_dir / 'data.yaml'}")

    print("\n" + "=" * 60)
    print(" Process Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()


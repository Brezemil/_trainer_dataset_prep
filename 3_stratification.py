import os
import shutil
import geopandas as gpd
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = Path(r"C:\Users\emilb\_data\dataset_fin\sliced_dataset") 
OUTPUT_DIR = Path(r"C:\Users\emilb\_data\dataset_fin\sliced_split_dataset")
GEOJSON_PATH = Path(r"C:\Users\emilb\_data\dataset_fin\sliced_dataset_edge_coordinates\image_footprints.geojson")

SPLIT_RATIO = (0.86, 0.07, 0.07) 
MAX_ALLOWED_OVERLAP_PCT = 0  
MAX_OVERSIZED_PCT = 600.0     # Drop footprint if its area exceeds the average by this percentage
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
CLASSES = ["A. altissima"]
UTM_EPSG = 32633  # Coordinate reference system zone for local metric calculations

def generate_spatial_splits(geojson_path, split_ratio, utm_epsg=32633):
    print(f"[INFO] Loading spatial footprints from: {geojson_path.name}")
    gdf = gpd.read_file(geojson_path)
    initial_image_count = len(gdf)
    print(f"[DIAGNOSTIC] Loaded {initial_image_count} image footprints.")

    try:
        gdf_metric = gdf.to_crs(epsg=utm_epsg)
    except Exception as e:
        print("[WARNING] Could not re-project. Falling back to geographic CRS.")
        gdf_metric = gdf.copy()

    print("[INFO] Cleaning invalid geometries (fixing bowties)...")
    gdf_metric['geometry'] = gdf_metric['geometry'].buffer(0)

    # --- OVERSIZED POLYGON SAFEGUARD ---
    print(f"[INFO] Checking for oversized polygons (Threshold: +{MAX_OVERSIZED_PCT}%)...")
    average_area = gdf_metric.area.mean()
    
    area_multiplier = 1.0 + (MAX_OVERSIZED_PCT / 100.0)
    max_allowed_area = average_area * area_multiplier  
    
    oversized_mask = gdf_metric.area > max_allowed_area
    oversized_count = oversized_mask.sum()
    
    if oversized_count > 0:
        print(f"[WARNING] Dropped {oversized_count} footprint(s) exceeding {MAX_OVERSIZED_PCT}% of average area.")
        print(f"           -> Average Area: {average_area:.2f} m^2 | Max Allowed: {max_allowed_area:.2f} m^2")
        gdf_metric = gdf_metric[~oversized_mask].reset_index(drop=True)
    else:
        print(f"[INFO] All footprints are within the size threshold (Avg Area: {average_area:.2f} m^2).")
    # -----------------------------------------

    print("[INFO] Forcing orientation: South -> North.")
    gdf_metric['sort_val'] = gdf_metric.geometry.centroid.y
    gdf_metric = gdf_metric.sort_values('sort_val').reset_index(drop=True)

    n_total = len(gdf_metric)
    n_test = int(n_total * split_ratio[2]) 
    n_val = int(n_total * split_ratio[1])  

    gdf_metric['split'] = 'train' 
    gdf_metric.loc[:n_test-1, 'split'] = 'test'
    print(f"[INFO] Allocated {n_test} images to TEST (South).")
    
    gdf_metric.loc[n_total-n_val:, 'split'] = 'val'
    print(f"[INFO] Allocated {n_val} images to VAL (North).")
    
    print(f"[INFO] Computing intersections (Max allowed overlap: {MAX_ALLOWED_OVERLAP_PCT}%)...")
    overlaps = gpd.sjoin(gdf_metric, gdf_metric, how='inner', predicate='intersects')
    cross_boundary = overlaps[overlaps['split_left'] != overlaps['split_right']]
    leakage_images = set()

    for idx_left, row in cross_boundary.iterrows():
        idx_right = row['index_right']
        split_left = row['split_left']
        split_right = row['split_right']
        
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
            overlap_pct = (intersection_area / min_area) * 100
            if overlap_pct > MAX_ALLOWED_OVERLAP_PCT:
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dropped_log_path = OUTPUT_DIR / "dropped_images.txt"
    with open(dropped_log_path, "w") as f:
        for img in leakage_images:
            f.write(f"{img}\n")
    print(f"[INFO] Logged boundary-dropped images to {dropped_log_path}")
    
    return allocation_dict, stats

def create_yolo_yaml(output_dir, classes):
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
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

def build_dataset_structure(allocation_dict):
    all_files = list(DATASET_DIR.rglob("*"))
    
    all_images = [f for f in all_files if f.is_file() and f.suffix in IMAGE_EXTENSIONS]
    all_labels = [f for f in all_files if f.is_file() and f.suffix.lower() == ".txt"]

    img_map = {f.stem.strip(): f for f in all_images}
    lbl_map = {f.stem.strip(): f for f in all_labels}

    print("\n" + "="*40)
    print("[DEBUG] DIRECTORY SCAN RESULTS")
    print(f"  -> Searched everywhere inside: {DATASET_DIR}")
    print(f"  -> Valid images found: {len(img_map)}")
    print(f"  -> Valid .txt labels found: {len(lbl_map)}")
    print("="*40)

    if len(img_map) == 0:
        print("\n[CRITICAL ERROR] 0 images found. Double check that DATASET_DIR points to the exact folder containing the images.")
        return

    for split_name in ["train", "val", "test"]:
        (OUTPUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Copying valid files to {OUTPUT_DIR}...")
    
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

        img_target = OUTPUT_DIR / assigned_split / "images" / img_path.name
        lbl_target = OUTPUT_DIR / assigned_split / "labels" / label_path.name

        shutil.copy2(img_path, img_target)
        shutil.copy2(label_path, lbl_target)
        copied_files += 1

    print("\n" + "-"*30)
    print("[DIAGNOSTIC] FILE COPY SUMMARY")
    print("-" * 30)
    print(f" Allocated Footprints   : {len(allocation_dict)}")
    print(f" Missing from Disk      : {missing_from_disk}")
    print(f" Missing Label Files    : {missing_labels}")
    print(f" Successfully Copied    : {copied_files}")
    print("-" * 30)

    if copied_files == 0 and len(allocation_dict) > 0 and len(img_map) > 0:
        sample_alloc = list(allocation_dict.keys())[0]
        sample_disk = list(img_map.keys())[0]
        print("\n[CRITICAL ERROR] Zero files copied. Complete mismatch detected between GeoJSON and Disk.")
        print(f" -> Example GeoJSON Key : '{sample_alloc}'")
        print(f" -> Example Disk Key    : '{sample_disk}'")

def main():
    print("="*60)
    print(" Executing Spatial Spatio-Temporal Splitter")
    print("="*60)

    allocation_dict, stats = generate_spatial_splits(GEOJSON_PATH, SPLIT_RATIO, UTM_EPSG)

    print("\n" + "-"*30)
    print("[STATS] SPATIAL SPLIT SUMMARY")
    print("-" * 30)
    print(f" Total Footprints Analyzed : {stats['initial']}")
    print(f" Dropped (Oversized)       : {stats['oversized_dropped']} ({(stats['oversized_dropped']/stats['initial'])*100:.2f}%)")
    print(f" Dropped (Boundary Buffer) : {stats['boundary_dropped']} ({(stats['boundary_dropped']/stats['initial'])*100:.2f}%)")
    print(f" Retained (Safe Set)       : {stats['retained']}")
    print(f"   -> Train Allocation     : {stats['train']}")
    print(f"   -> Val Allocation       : {stats['val']}")
    print(f"   -> Test Allocation      : {stats['test']}")
    print("-" * 30)

    if stats['retained'] == 0:
        print("\n[ERROR] All images were dropped. Adjust SPLIT_RATIO configuration.")
        return

    build_dataset_structure(allocation_dict)

    create_yolo_yaml(OUTPUT_DIR, CLASSES)
    print(f"[INFO] Generated YOLO configuration at {OUTPUT_DIR / 'data.yaml'}")

    print("\n" + "="*60)
    print(" Process Complete")
    print("="*60)

if __name__ == "__main__":
    main()

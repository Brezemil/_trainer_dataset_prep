"""
5_build_testset_combined.py - Strict Zero-Leakage Combined Testset Builder
========================================================================

This script generates a comprehensive, pure test dataset (`testset_unlabeled_combined`)
under Option A (Strict Zero-Leakage).

Why This Step is Needed:
------------------------
When evaluating object detection models (e.g., YOLO), testing solely on image tiles
that contain positive tree annotations gives an incomplete and overly optimistic
assessment of performance. In real-world aerial surveys, most land area does not
contain the target species. To accurately measure False Positive rates and ensure
model reliability, the evaluation test set must include realistic negative
(unlabeled / background) image tiles from the exact test flight zone.

What Option A Guarantees (Zero Data Leakage):
--------------------------------------------
1. Pure Geographic Separation:
   All included test slices fall strictly within the geographic footprint of the
   designated Test area (Southern region).
2. Boundary Buffer Isolation:
   Any slice that touches or overlaps the South boundary buffer zone is dropped.
3. Strict Train Set Independence:
   Any slice that touches or overlaps any retained training image footprint is dropped.
4. Clean Negative Annotation:
   Unlabeled slices are paired with empty `.txt` annotation files, adhering to
   standard YOLO format conventions for negative background images.

Outputs:
--------
- `<OUTPUT_DIR>/images/`: Pure test images (labeled + unlabeled background).
- `<OUTPUT_DIR>/labels/`: Corresponding YOLO label files (empty for background).
- `<OUTPUT_DIR>/data.yaml`: YOLO configuration file for evaluation.
- `<OUTPUT_DIR>/testset_unlabeled_combined_footprints.geojson`: Real-world polygons.
- `<OUTPUT_DIR>/testset_combined_spatial_map.png`: Publication-ready satellite map.
- `<OUTPUT_DIR>/testset_combined_breakdown.png`: Composition and campaign charts.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D


def read_image_robust(file_path: str) -> Optional[np.ndarray]:
    """
    Reads an image from disk safely handling Unicode characters (e.g. 'ö' in Windows paths).

    Args:
        file_path (str): Filesystem path to image.

    Returns:
        Optional[np.ndarray]: Decoded image array or None if read fails.
    """
    try:
        buffer = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as err:
        print(f"[Warning] Could not read image at {file_path}: {err}")
        return None


def write_image_robust(file_path: str, image: np.ndarray, quality: int = 95) -> bool:
    """
    Writes an image to disk safely handling Unicode characters in Windows paths.

    Args:
        file_path (str): Destination path.
        image (np.ndarray): Image array to write.
        quality (int): JPEG quality setting (1-100).

    Returns:
        bool: True on success, False on failure.
    """
    try:
        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            buffer.tofile(file_path)
            return True
        return False
    except Exception as err:
        print(f"[Warning] Could not write image to {file_path}: {err}")
        return False


def build_pure_testset(
    export_dir: Path,
    raw_images_dir: Path,
    plots_dir: Optional[Path] = None,
    utm_epsg: int = 32633,
    max_workers: int = 8
) -> None:
    """
    Constructs the Option A strict zero-leakage combined test set.

    Args:
        export_dir (Path): Base directory containing split datasets and footprint GeoJSONs.
        raw_images_dir (Path): Root directory containing raw un-sliced drone imagery.
        plots_dir (Optional[Path]): Dedicated folder where all generated plots are saved.
        utm_epsg (int): EPSG code for local metric projection (default 32633).
        max_workers (int): ThreadPool concurrency limit for cropping images.
    """
    start_time = time.time()
    print("=" * 65)
    print(" EXECUTING STRICT ZERO-LEAKAGE TESTSET BUILDER (OPTION A)")
    print("=" * 65)

    if plots_dir is None:
        import config
        plots_dir = config.TESTSET_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_dir = export_dir / "testset_unlabeled_combined"
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load Footprints and Establish Geographic Boundaries
    # -------------------------------------------------------------
    print("\n[Step 1] Loading footprint geometries and spatial boundaries...")
    labeled_geojson_path = export_dir / "sliced_dataset_edge_coordinates" / "image_footprints.geojson"
    gdf_labeled = gpd.read_file(labeled_geojson_path).to_crs(epsg=utm_epsg)
    gdf_labeled["clean_stem"] = gdf_labeled["image_name"].apply(lambda x: Path(x).stem.strip())
    gdf_labeled["geometry"] = gdf_labeled["geometry"].buffer(0)

    # Training stems (to check for overlap)
    train_dir = export_dir / "sliced_split_dataset" / "train" / "images"
    train_stems = {p.stem for p in train_dir.iterdir() if p.is_file()}
    gdf_train = gdf_labeled[gdf_labeled["clean_stem"].isin(train_stems)].copy()

    # Labeled test stems
    test_src_dir = export_dir / "sliced_split_dataset" / "test" / "images"
    test_lbl_src_dir = export_dir / "sliced_split_dataset" / "test" / "labels"
    test_stems = {p.stem for p in test_src_dir.iterdir() if p.is_file()}
    gdf_test = gdf_labeled[gdf_labeled["clean_stem"].isin(test_stems)].copy()

    # Boundary buffer slices (dropped between Train and Test/Val)
    dropped_file = export_dir / "sliced_split_dataset" / "dropped_images.txt"
    dropped_stems = {Path(line.strip()).stem for line in dropped_file.read_text().splitlines() if line.strip()}
    gdf_dropped = gdf_labeled[gdf_labeled["clean_stem"].isin(dropped_stems)].copy()
    # Southern buffer separating Train from Test
    gdf_south_buffer = gdf_dropped[gdf_dropped.geometry.centroid.y < 5332400].copy()

    print(f"  -> Labeled test slices in baseline split: {len(gdf_test)}")
    print(f"  -> Retained training slices: {len(gdf_train)}")
    print(f"  -> South boundary buffer slices: {len(gdf_south_buffer)}")

    # Candidate unlabeled slices from raycaster
    unlabeled_geojson_path = export_dir / "sliced_unlabeled" / "edge_coordinates" / "image_footprints.geojson"
    gdf_cand = gpd.read_file(unlabeled_geojson_path).to_crs(epsg=utm_epsg)
    gdf_cand["clean_stem"] = gdf_cand["image_name"].apply(lambda x: Path(x).stem.strip())
    gdf_cand["geometry"] = gdf_cand["geometry"].buffer(0)

    # Candidate selection: fall inside the convex hull or spatial union of labeled test slices
    test_hull = gdf_test.union_all().convex_hull
    test_union = gdf_test.union_all()
    s_hull = set(gdf_cand[gdf_cand.geometry.centroid.within(test_hull)]["clean_stem"])
    s_union = set(gdf_cand[gdf_cand.geometry.intersects(test_union)]["clean_stem"])
    candidate_unlabeled_stems = s_hull.union(s_union)
    gdf_cand_in_test = gdf_cand[gdf_cand["clean_stem"].isin(candidate_unlabeled_stems)].copy()

    # -------------------------------------------------------------
    # Step 2: Enforce Option A (Eliminate Buffer and Train Overlap)
    # -------------------------------------------------------------
    print("\n[Step 2] Enforcing Option A spatial isolation...")
    # Form candidate pool: labeled test slices + candidate unlabeled slices
    gdf_test_labeled_subset = gdf_test.copy()
    gdf_test_labeled_subset["slice_type"] = "labeled_test"

    gdf_cand_in_test_subset = gdf_cand_in_test.copy()
    gdf_cand_in_test_subset["slice_type"] = "unlabeled_test"

    combined_df_list = [gdf_test_labeled_subset, gdf_cand_in_test_subset]
    gdf_combined_candidates = gpd.GeoDataFrame(
        pd.concat(combined_df_list, ignore_index=True),
        crs=gdf_test.crs
    )

    # Detect overlaps with retained train set
    join_train = gpd.sjoin(gdf_combined_candidates, gdf_train, how="inner", predicate="intersects")
    stems_touching_train = set(join_train["clean_stem_left"].unique())

    # Detect overlaps with southern buffer zone
    join_buffer = gpd.sjoin(gdf_combined_candidates, gdf_south_buffer, how="inner", predicate="intersects")
    stems_touching_buffer = set(join_buffer["clean_stem_left"].unique())

    # Option A pure test cohort: exclude both buffer and training overlaps
    pure_stems = set(gdf_combined_candidates["clean_stem"]) - stems_touching_buffer - stems_touching_train
    gdf_pure = gdf_combined_candidates[gdf_combined_candidates["clean_stem"].isin(pure_stems)].copy()

    labeled_pure_stems = pure_stems.intersection(test_stems)
    unlabeled_pure_stems = pure_stems - labeled_pure_stems

    print(f"  -> Candidate slices in test footprint: {len(gdf_combined_candidates)}")
    print(f"  -> Excluded touching South buffer: {len(stems_touching_buffer)}")
    print(f"  -> Excluded touching Train set: {len(stems_touching_train)}")
    print(f"  -> Retained Pure Test Slices: {len(gdf_pure)} "
          f"({len(labeled_pure_stems)} labeled + {len(unlabeled_pure_stems)} unlabeled background)")

    # -------------------------------------------------------------
    # Step 3: Copy / Crop Images and Labels to Output Directory
    # -------------------------------------------------------------
    print("\n[Step 3] Populating pure testset files on disk...")
    # 1. Copy labeled pure test images & labels
    for stem in labeled_pure_stems:
        for ext in (".jpg", ".png", ".jpeg"):
            img_file = test_src_dir / f"{stem}{ext}"
            if img_file.exists():
                shutil.copy2(img_file, out_img_dir / f"{stem}.jpg")
                break
        lbl_file = test_lbl_src_dir / f"{stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, out_lbl_dir / f"{stem}.txt")
        else:
            open(out_lbl_dir / f"{stem}.txt", "w", encoding="utf-8").close()

    # 2. Check which unlabeled pure slices are already cropped vs need slicing
    unlabeled_src_dir = export_dir / "sliced_unlabeled" / "images"
    existing_unlabeled = {p.stem: p for p in unlabeled_src_dir.iterdir() if p.is_file()}

    already_on_disk = unlabeled_pure_stems.intersection(set(existing_unlabeled.keys()))
    needing_reslice = unlabeled_pure_stems - set(existing_unlabeled.keys())

    for stem in already_on_disk:
        shutil.copy2(existing_unlabeled[stem], out_img_dir / f"{stem}.jpg")
        open(out_lbl_dir / f"{stem}.txt", "w", encoding="utf-8").close()

    if needing_reslice:
        print(f"  -> Reslicing {len(needing_reslice)} background tiles from raw drone images...")
        with open(export_dir / "sliced_unlabeled" / "slice_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_index: Dict[str, Path] = {}
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.tif", "*.TIF"):
            for p in raw_images_dir.rglob(ext):
                if "_export_data" not in str(p):
                    raw_index[p.stem] = p

        parent_groups: Dict[str, List[Tuple[str, int, int, int, int]]] = {}
        for stem in needing_reslice:
            info = meta[stem]
            parent = info["parent_image"]
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append((stem, info["x_start"], info["y_start"], info["x_end"], info["y_end"]))

        def crop_parent(parent_name: str, crops: List[Tuple[str, int, int, int, int]]) -> int:
            if parent_name not in raw_index:
                return 0
            img = read_image_robust(str(raw_index[parent_name]))
            if img is None:
                return 0
            count = 0
            for s_stem, xs, ys, xe, ye in crops:
                tile = img[ys:ye, xs:xe]
                write_image_robust(str(out_img_dir / f"{s_stem}.jpg"), tile)
                open(out_lbl_dir / f"{s_stem}.txt", "w", encoding="utf-8").close()
                count += 1
            return count

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(crop_parent, p, c) for p, c in parent_groups.items()]
            resliced_count = sum(f.result() for f in futures)
        print(f"  -> Resliced {resliced_count} crops from raw drone photos.")

    # -------------------------------------------------------------
    # Step 4: Write YOLO data.yaml Configuration
    # -------------------------------------------------------------
    print("\n[Step 4] Writing data.yaml configuration...")
    yaml_content = f"""# YOLO Test Dataset Configuration (Pure Option A)
path: {output_dir.absolute()}
test: images

# Classes
nc: 1
names: ['A. altissima']
"""
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # -------------------------------------------------------------
    # Step 5: Export GeoJSON Footprints
    # -------------------------------------------------------------
    print("\n[Step 5] Exporting testset footprints GeoJSON...")
    geojson_out = output_dir / "testset_unlabeled_combined_footprints.geojson"
    gdf_pure_wgs84 = gdf_pure.to_crs(epsg=4326)
    gdf_pure_wgs84.to_file(geojson_out, driver="GeoJSON")
    print(f"  -> Saved {len(gdf_pure)} footprints to {geojson_out.name}")

    # -------------------------------------------------------------
    # Step 6: Generate Publication-Ready Maps and Charts
    # -------------------------------------------------------------
    render_all_figures(gdf_pure, output_dir, labels_dir=output_dir / "labels", plots_dir=plots_dir)

    print("\n" + "=" * 65)
    print(f" COMPLETED IN {time.time() - start_time:.1f}s | TOTAL PURE TEST SLICES: {len(gdf_pure):,}")
    print("=" * 65)


def render_all_figures(
    gdf_pure: gpd.GeoDataFrame,
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None
) -> None:
    """
    Renders all publication-ready cartographic maps and statistical charts for the pure testset:
    1. testset_combined_points_map.png: 4-way centroid point scatter map (overview + zoom inset + stats card)
    2. testset_combined_spatial_map.png: Geographic polygon footprints overlay on satellite imagery
    3. testset_combined_breakdown.png: Cohort composition bar chart and flight campaign pie chart
    4. extended_testset_centroids_multitemporal_map.png: Multi-temporal 5-panel centroid map
    """
    if plots_dir is None:
        import config
        plots_dir = config.TESTSET_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_plots_dir = output_dir / "plots"
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 6] Rendering cartographic maps and figures into {plots_dir}...")

    # 1. Minimalist Publication Centroid Map (Composite 2023 vs 2024)
    try:
        from plot_testset_centroids import render_minimalist_composite_map, render_minimalist_by_campaign_map
        m1 = render_minimalist_composite_map(
            gdf_pure=gdf_pure,
            output_plots_dir=plots_dir,
            use_basemap=True
        )
        shutil.copy2(m1, dataset_plots_dir / m1.name)
        shutil.copy2(m1, plots_dir / "testset_combined_points_map.png")
        shutil.copy2(m1, dataset_plots_dir / "testset_combined_points_map.png")

        m2 = render_minimalist_by_campaign_map(
            gdf_pure=gdf_pure,
            output_plots_dir=plots_dir,
            use_basemap=True
        )
        shutil.copy2(m2, dataset_plots_dir / m2.name)
    except Exception as e:
        print(f"  [Warning] Minimalist centroid map generation skipped: {e}")

    # 2. Spatial Polygon Footprint Map
    gdf_pure_3857 = gdf_pure.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 8))
    gdf_unl_plot = gdf_pure_3857[gdf_pure_3857["slice_type"] != "labeled_test"]
    gdf_unl_plot.plot(
        ax=ax, color="#f39c12", alpha=0.5, markersize=15, marker="o",
        label=f"Unlabeled Pure Test (n={len(gdf_unl_plot):,})"
    )

    gdf_lab_plot = gdf_pure_3857[gdf_pure_3857["slice_type"] == "labeled_test"]
    gdf_lab_plot.plot(
        ax=ax, color="#00ffff", edgecolor="#004488", linewidth=1.0, alpha=0.9,
        markersize=30, marker="s", label=f"Labeled Pure Test (n={len(gdf_lab_plot):,})"
    )

    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    except Exception as e:
        print(f"  [Notice] Basemap download skipped: {e}")

    scalebar = ScaleBar(1, "m", length_fraction=0.2, location="lower right",
                        box_alpha=0.85, bbox_to_anchor=(0.98, 0.05), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar)

    ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.87),
                arrowprops=dict(facecolor='white', edgecolor='black', width=4, headwidth=12),
                ha='center', va='center', fontsize=18, fontweight='bold', color='black',
                xycoords=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#00ffff', markeredgecolor='#004488', markersize=12),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, alpha=0.7),
        Line2D([0], [0], marker='', color='w', markersize=0),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2c3e50', markersize=10)
    ]
    labels = [
        f"Labeled Pure Test Slices ({len(gdf_lab_plot):,})",
        f"Unlabeled Pure Test Slices ({len(gdf_unl_plot):,})",
        "---",
        f"Total Zero-Leakage Test Slices ({len(gdf_pure):,})"
    ]

    ax.legend(handles=handles, labels=labels, title="Testset Composition",
              bbox_to_anchor=(1.02, 1.0), loc="upper left", framealpha=0.92, fontsize=11, title_fontsize=12)
    ax.set_title("STRICT ZERO-LEAKAGE COMBINED TESTSET GEOSPATIAL FOOTPRINT\n(Pure Testset: Buffer & Training Overlap Excluded)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_axis_off()

    def safe_copy(src: Path, dst: Path, max_retries: int = 5, delay: float = 0.3):
        for _ in range(max_retries):
            try:
                if dst.exists():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                shutil.copy2(src, dst)
                return
            except OSError:
                time.sleep(delay)

    def safe_savefig(fig, out_path: Path, dpi: int = 300):
        out_path = Path(out_path).resolve()
        for _ in range(5):
            try:
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
                return
            except OSError:
                time.sleep(0.3)

    map_out_path = plots_dir / "testset_combined_spatial_map.png"
    plt.tight_layout()
    safe_savefig(fig, map_out_path, dpi=300)
    plt.close(fig)
    safe_copy(map_out_path, output_dir / "testset_combined_spatial_map.png")
    safe_copy(map_out_path, dataset_plots_dir / "testset_combined_spatial_map.png")
    print(f"  -> Saved overview footprint map to: {map_out_path.name}")

    # 4. Summary Breakdown Charts (Distribution and Campaign)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    types = ["Labeled Pure Test", "Unlabeled Pure Test"]
    counts = [len(gdf_lab_plot), len(gdf_unl_plot)]
    bars = ax1.bar(types, counts, color=["#2980b9", "#e67e22"], edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Number of Slices", fontweight="bold")
    ax1.set_title("Pure Testset Cohort Composition", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    camp_counts = gdf_pure["campaign"].value_counts()
    ax2.pie(camp_counts, labels=camp_counts.index, autopct='%1.1f%%',
            colors=["#3498db", "#e74c3c"], startangle=90, textprops=dict(fontweight='bold'))
    ax2.set_title("Campaign Breakdown (2023 vs 2024)", fontweight="bold")

    plt.suptitle(f"Strict Zero-Leakage Testset Summary (Total: {len(gdf_pure):,} Slices)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    breakdown_out_path = plots_dir / "testset_combined_breakdown.png"
    safe_savefig(fig2, breakdown_out_path, dpi=200)
    plt.close(fig2)
    safe_copy(breakdown_out_path, output_dir / "testset_combined_breakdown.png")
    safe_copy(breakdown_out_path, dataset_plots_dir / "testset_combined_breakdown.png")
    print(f"  -> Saved breakdown charts to: {breakdown_out_path.name}")


def render_testset_points_map(
    gdf_pure: gpd.GeoDataFrame,
    output_dir: Path,
    labels_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None
) -> Path:
    """
    Renders a 4-way publication-ready geospatial point scatter map of slice centroids
    over high-resolution satellite imagery, distinguishing 2023 vs 2024 survey flights
    and labeled ground-truth slices vs new unlabeled slices.

    Includes:
    - Main corridor overview (600m)
    - Detail zoom inset (130m x 75m)
    - Comprehensive Foreground (FG) vs Background (BG) metrics and survey distribution table.
    """
    if plots_dir is None:
        import config
        plots_dir = config.TESTSET_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    gdf_3857 = gdf_pure.to_crs(epsg=3857).copy()
    gdf_3857["centroid"] = gdf_3857.geometry.centroid
    pts = gdf_3857.set_geometry("centroid")

    # Group subsets
    u_2023 = pts[(pts["slice_type"] == "unlabeled_resliced") & (pts["campaign"].astype(str) == "2023")]
    u_2024 = pts[(pts["slice_type"] == "unlabeled_resliced") & (pts["campaign"].astype(str) == "2024")]
    l_2023 = pts[(pts["slice_type"] == "labeled_test") & (pts["campaign"].astype(str) == "2023")]
    l_2024 = pts[(pts["slice_type"] == "labeled_test") & (pts["campaign"].astype(str) == "2024")]

    if labels_dir is None:
        labels_dir = output_dir / "labels"

    # Count FG / BG dynamically
    lab_fg, lab_bg, lab_boxes = 0, 0, 0
    labeled_rows = pts[pts["slice_type"] == "labeled_test"]
    for _, row in labeled_rows.iterrows():
        stem = Path(row["image_name"]).stem.strip()
        lbl_file = labels_dir / f"{stem}.txt"
        if lbl_file.exists():
            lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                lab_fg += 1
                lab_boxes += len(lines)
            else:
                lab_bg += 1
        else:
            lab_bg += 1

    unl_count = len(u_2023) + len(u_2024)
    total_count = len(pts)
    total_fg = lab_fg
    total_bg = lab_bg + unl_count

    fig = plt.figure(figsize=(20, 11), facecolor='#0b1329')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.25, 0.75], hspace=0.22, wspace=0.12)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_stat = fig.add_subplot(gs[1, 1])

    for ax in [ax_main, ax_zoom]:
        ax.set_facecolor('#070d1e')

    ax_stat.set_facecolor('#0f172a')
    ax_stat.set_xticks([])
    ax_stat.set_yticks([])
    for spine in ax_stat.spines.values():
        spine.set_edgecolor('#334155')
        spine.set_linewidth(1.5)

    def plot_scatter(ax, unl_s, unl_a, lab_s):
        # 1. Unlabeled 2023: Cyan / Sky Blue circle
        ax.scatter(u_2023.geometry.x, u_2023.geometry.y,
                   c='#38bdf8', marker='o', s=unl_s, alpha=unl_a, edgecolors='none', zorder=2)
        # 2. Unlabeled 2024: Amber / Warm Gold circle
        ax.scatter(u_2024.geometry.x, u_2024.geometry.y,
                   c='#fb923c', marker='o', s=unl_s, alpha=unl_a, edgecolors='none', zorder=3)
        # 3. Labeled 2023: Neon Electric Turquoise Diamond with dark edge
        ax.scatter(l_2023.geometry.x, l_2023.geometry.y,
                   c='#00f5d4', marker='D', s=lab_s, alpha=0.98, edgecolors='#022c22', linewidth=1.4, zorder=5)
        # 4. Labeled 2024: Neon Coral / Magenta Diamond with dark edge
        ax.scatter(l_2024.geometry.x, l_2024.geometry.y,
                   c='#ff4757', marker='D', s=lab_s, alpha=0.98, edgecolors='#3b0712', linewidth=1.4, zorder=6)

    # 1. MAIN OVERVIEW
    plot_scatter(ax_main, unl_s=14, unl_a=0.45, lab_s=50)
    minx, miny, maxx, maxy = pts.total_bounds
    pad_x, pad_y = 35, 20
    ax_main.set_xlim(minx - pad_x, maxx + pad_x)
    ax_main.set_ylim(miny - pad_y, maxy + pad_y)
    ax_main.set_aspect('equal')

    # Zoom window coords
    zx0 = minx + (maxx - minx) * 0.28
    zx1 = zx0 + 130
    zy0 = miny - 5
    zy1 = zy0 + 75

    rect = patches.Rectangle((zx0, zy0), 130, 75, linewidth=2.0,
                             edgecolor='#facc15', facecolor='#facc15', alpha=0.15, linestyle='--', zorder=8)
    ax_main.add_patch(rect)
    ax_main.text(zx0 + 4, zy1 - 10, 'DETAIL INSET (130m × 75m)',
                 color='#facc15', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.25", fc="#0b1329", ec="#facc15", lw=1.2, alpha=0.9), zorder=9)

    # 2. ZOOM INSET
    plot_scatter(ax_zoom, unl_s=38, unl_a=0.6, lab_s=100)
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy0, zy1)
    ax_zoom.set_aspect('equal')

    try:
        ctx.add_basemap(ax_main, source=ctx.providers.Esri.WorldImagery, attribution="")
        ctx.add_basemap(ax_zoom, source=ctx.providers.Esri.WorldImagery, attribution="")
    except Exception as e:
        print("  [Notice] Basemap download notice:", e)

    sb_main = ScaleBar(1, "m", length_fraction=0.15, location="lower right",
                       box_alpha=0.85, box_color='#0b1329', color='white',
                       bbox_to_anchor=(0.98, 0.08), bbox_transform=ax_main.transAxes)
    ax_main.add_artist(sb_main)

    sb_zoom = ScaleBar(1, "m", length_fraction=0.22, location="lower right",
                       box_alpha=0.85, box_color='#0b1329', color='white',
                       bbox_to_anchor=(0.98, 0.08), bbox_transform=ax_zoom.transAxes)
    ax_zoom.add_artist(sb_zoom)

    for ax in [ax_main, ax_zoom]:
        ax.annotate('N', xy=(0.025, 0.88), xytext=(0.025, 0.70),
                    arrowprops=dict(facecolor='white', edgecolor='black', width=3, headwidth=8),
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                    xycoords=ax.transAxes, bbox=dict(boxstyle="round,pad=0.2", fc="#0b1329", ec="white", alpha=0.85))
        ax.set_xticks([])
        ax.set_yticks([])

    ax_main.set_title("OVERVIEW: Combined Testset Slice Centroids (Full Railway Corridor, 600m)",
                      color="white", fontsize=13, fontweight='bold', pad=8)
    ax_zoom.set_title("ZOOM DETAIL: 130m × 75m Sub-corridor showing Dense Centroid Grid",
                      color="#facc15", fontsize=12, fontweight='bold', pad=8)

    # 3. STATISTICS PANEL
    ax_stat.text(0.5, 0.94, "TESTSET COMPOSITION & FG/BG METRICS",
                 color="#38bdf8", fontsize=13, fontweight='bold', ha='center', va='top', transform=ax_stat.transAxes)
    ax_stat.text(0.5, 0.87, "Option A Strict Zero-Leakage Test Cohort",
                 color="#94a3b8", fontsize=10, fontstyle='italic', ha='center', va='top', transform=ax_stat.transAxes)

    headers = ["Cohort / Dataset", "Total", "FG (Trees)", "BG (Empty)", "% FG"]
    rows = [
        ["Original Testset (Split)", "532", "331", "201", "62.2%"],
        ["Retained Labeled", f"{len(labeled_rows):,}", f"{lab_fg:,}", f"{lab_bg:,}", f"{(lab_fg/max(1, len(labeled_rows))*100):.1f}%"],
        ["New Unlabeled Slices", f"{unl_count:,}", "0", f"{unl_count:,}", "0.0%"],
        ["Combined Total", f"{total_count:,}", f"{total_fg:,}", f"{total_bg:,}", f"{(total_fg/max(1, total_count)*100):.1f}%"]
    ]

    col_widths = [0.38, 0.15, 0.18, 0.18, 0.13]
    start_y = 0.78
    row_height = 0.065

    cur_x = 0.04
    for w, h in zip(col_widths, headers):
        ax_stat.text(cur_x, start_y, h, color="#e2e8f0", fontsize=9.5, fontweight='bold', transform=ax_stat.transAxes)
        cur_x += w

    ax_stat.plot([0.03, 0.97], [start_y - 0.018, start_y - 0.018], color="#475569", lw=1.2, transform=ax_stat.transAxes)

    y = start_y - 0.05
    for row_idx, r in enumerate(rows):
        cur_x = 0.04
        is_bold = (row_idx in [0, 3])
        font_c = "#f8fafc" if is_bold else "#cbd5e1"
        if row_idx == 3:
            font_c = "#38bdf8"
            ax_stat.plot([0.03, 0.97], [y + 0.015, y + 0.015], color="#334155", lw=1.0, linestyle='--', transform=ax_stat.transAxes)

        for w, val in zip(col_widths, r):
            ax_stat.text(cur_x, y, val, color=font_c, fontsize=9,
                         fontweight='bold' if is_bold else 'normal', transform=ax_stat.transAxes)
            cur_x += w
        y -= row_height

    y += 0.01
    ax_stat.plot([0.03, 0.97], [y, y], color="#475569", lw=1.2, transform=ax_stat.transAxes)

    y -= 0.05
    ax_stat.text(0.04, y, "SURVEY FLIGHT CAMPAIGN DISTRIBUTION",
                 color="#facc15", fontsize=10.5, fontweight='bold', transform=ax_stat.transAxes)

    tot_2023 = len(l_2023) + len(u_2023)
    tot_2024 = len(l_2024) + len(u_2024)
    camp_headers = ["Campaign", "Labeled", "Unlabeled", "Total", "% Share"]
    camp_rows = [
        ["Flight 2023", f"{len(l_2023):,}", f"{len(u_2023):,}", f"{tot_2023:,}", f"{(tot_2023/max(1, total_count)*100):.1f}%"],
        ["Flight 2024", f"{len(l_2024):,}", f"{len(u_2024):,}", f"{tot_2024:,}", f"{(tot_2024/max(1, total_count)*100):.1f}%"],
        ["Combined", f"{len(labeled_rows):,}", f"{unl_count:,}", f"{total_count:,}", "100.0%"]
    ]
    camp_widths = [0.28, 0.18, 0.20, 0.18, 0.16]

    y -= 0.045
    cur_x = 0.04
    for w, h in zip(camp_widths, camp_headers):
        ax_stat.text(cur_x, y, h, color="#e2e8f0", fontsize=9, fontweight='bold', transform=ax_stat.transAxes)
        cur_x += w

    ax_stat.plot([0.03, 0.97], [y - 0.015, y - 0.015], color="#475569", lw=1.0, transform=ax_stat.transAxes)

    y -= 0.045
    for row_idx, r in enumerate(camp_rows):
        cur_x = 0.04
        is_bold = (row_idx == 2)
        font_c = "#facc15" if is_bold else "#cbd5e1"
        for w, val in zip(camp_widths, r):
            ax_stat.text(cur_x, y, val, color=font_c, fontsize=8.5,
                         fontweight='bold' if is_bold else 'normal', transform=ax_stat.transAxes)
            cur_x += w
        y -= 0.045

    y -= 0.01
    ax_stat.plot([0.03, 0.97], [y, y], color="#475569", lw=1.0, transform=ax_stat.transAxes)
    y -= 0.04
    ax_stat.text(0.04, y, "• Option A Guarantee: 0 train slices & 0 buffer slices touch testset.",
                 color="#10b981", fontsize=8.5, fontweight='bold', transform=ax_stat.transAxes)
    y -= 0.035
    ax_stat.text(0.04, y, "• High-precision spatial tiling: 1024x1024 px slices with 0.2 overlap.",
                 color="#94a3b8", fontsize=8.2, transform=ax_stat.transAxes)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#38bdf8', markersize=9, alpha=0.7, linestyle=''),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fb923c', markersize=9, alpha=0.7, linestyle=''),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#00f5d4', markeredgecolor='#022c22', markeredgewidth=1.3, markersize=10, linestyle=''),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff4757', markeredgecolor='#3b0712', markeredgewidth=1.3, markersize=10, linestyle=''),
    ]
    labels = [
        f"Unlabeled Slices 2023 (n={len(u_2023):,})",
        f"Unlabeled Slices 2024 (n={len(u_2024):,})",
        f"Original Labeled 2023 (n={len(l_2023):,})",
        f"Original Labeled 2024 (n={len(l_2024):,})",
    ]

    fig.legend(handles, labels, loc='upper center', ncol=4, framealpha=0.92, facecolor='#0b1329',
               edgecolor='#475569', labelcolor='white', fontsize=11, bbox_to_anchor=(0.5, 0.985))

    out_path = plots_dir / "testset_combined_points_map.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    shutil.copy2(out_path, output_dir / "testset_combined_points_map.png")
    dataset_plots = output_dir / "plots"
    dataset_plots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, dataset_plots / "testset_combined_points_map.png")
    print(f"  -> Saved centroid point map to: {out_path.name}")
    return out_path


def parse_args():
    """Parses command-line arguments for building the pure testset."""
    import argparse
    import config

    parser = argparse.ArgumentParser(
        description="Build strict zero-leakage combined test dataset (Option A) with realistic background slices."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--export-dir", "-e",
        type=Path,
        default=config.TESTSET_EXPORT_DIR,
        help=f"Base export directory containing sliced_split_dataset and edge coordinates (default from config.yaml: {config.TESTSET_EXPORT_DIR})."
    )
    parser.add_argument(
        "--raw-images-dir", "-r",
        type=Path,
        default=config.TESTSET_RAW_IMAGES_DIR,
        help=f"Root directory containing raw un-sliced drone photography (default from config.yaml: {config.TESTSET_RAW_IMAGES_DIR})."
    )
    parser.add_argument(
        "--plots-dir", "-p",
        type=Path,
        default=config.TESTSET_PLOTS_DIR,
        help=f"Dedicated folder where all plots and figures are saved (default: {config.TESTSET_PLOTS_DIR})."
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=config.TESTSET_UTM_EPSG,
        help=f"EPSG code for local metric projection (default from config.yaml: {config.TESTSET_UTM_EPSG})."
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=config.TESTSET_MAX_WORKERS,
        help=f"Thread pool workers for multi-threaded image cropping (default from config.yaml: {config.TESTSET_MAX_WORKERS})."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip image cropping and only re-generate cartographic maps and figures from existing footprints GeoJSON."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.export_dir / "testset_unlabeled_combined"
    if args.plot_only:
        geojson_path = output_dir / "testset_unlabeled_combined_footprints.geojson"
        if not geojson_path.exists():
            raise FileNotFoundError(f"Cannot run --plot-only: {geojson_path} does not exist. Run full builder first.")
        print(f"Loading existing testset footprints from {geojson_path.name}...")
        gdf_pure = gpd.read_file(geojson_path).to_crs(epsg=args.utm_epsg)
        render_all_figures(gdf_pure, output_dir, labels_dir=output_dir / "labels", plots_dir=args.plots_dir)
        print("Done re-generating all plots and figures!")
    else:
        build_pure_testset(
            export_dir=args.export_dir,
            raw_images_dir=args.raw_images_dir,
            plots_dir=args.plots_dir,
            utm_epsg=args.utm_epsg,
            max_workers=args.max_workers
        )


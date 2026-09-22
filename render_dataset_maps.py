"""
render_dataset_maps.py - Minimalist Cartographic Suite for Entire Dataset & Splits
==================================================================================

Generates minimalist, publication-ready cartographic maps across all dataset cohorts
with consistent formatting:
- All font sizes >= 16 pt (axis labels 18 pt, ticks 16 pt, legends 16 pt, scalebar 16 pt).
- High-precision metric coordinate system (EPSG:32633, UTM Zone 33N).
- Metric Maßstab (ScaleBar).
- Discrete North arrow.
- Embedded satellite tile source attribution: "Satellite Imagery: (c) Esri, Maxar, Earthstar Geographics".
- Zero speech bubbles, zero callouts, zero text blocks across the map canvas.

Figures Produced:
-----------------
1. `whole_dataset_splits_with_buffer_map.png`:
   Full dataset splits (Train = Red, Val = Green, Test = Blue, Dropped Buffer = Gray opaque ghost points).
   2023 flight shown as solid circles; 2024 flight shown with a cross inside the circle.
2. `distillation_dataset_map.png`:
   Distillation training set, validation set, and test set without the buffer zone.
3. `extended_testset_centroids_map.png`:
   Extended testset centroids color-coded in shades of blue.
4. `extended_testset_centroids_by_campaign.png`:
   Extended testset comparison by campaign (2023 vs 2024 in shades of blue).
5. `whole_dataset_images_2023_map.png` & `whole_dataset_images_2024_map.png`:
   Locations of all raw flight survey images for 2023 and 2024.
6. `selected_images_for_labelling_map.png`:
   Combined map showing the location of parent images selected for labelling (2023 vs 2024).
"""

import os
import sys
import json
import time
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx

# Import central typed configuration
try:
    import config
except ImportError:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    import config


# ------------------------------------------------------------------------------
# Universal Cartographic Theme & Typography (Fonts >= 16 pt for primary elements)
# ------------------------------------------------------------------------------
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 16
FONT_SCALE = 16
FONT_ATTR = 11

ATTR_TEXT = "Satellite Imagery: © Esri, Maxar, Earthstar Geographics"

# Color Codes for Splits
COLOR_TRAIN = "#ff2222"      # Vibrant, brighter red
COLOR_VAL = "#00e676"        # Vibrant, bright neon/emerald green
COLOR_TEST = "#2563eb"       # Blue
COLOR_GHOST = "#64748b"      # Gray opaque ghost points

# Survey Campaign Colors (consistent across 2023, 2024, and combined survey maps)
COLOR_SURVEY_2023 = "#0284c7"  # Deep Sky / Steel Blue
COLOR_SURVEY_2024 = "#ea580c"  # Vibrant Coral / Orange

# Shades of Blue for Extended Test Set
BLUE_2023_UNL = "#60a5fa"    # Light Sky Blue (2023 Unlabeled)
BLUE_2023_LAB = "#1d4ed8"    # Royal Blue (2023 Labeled)
BLUE_2024_UNL = "#0284c7"    # Steel Blue (2024 Unlabeled)
BLUE_2024_LAB = "#0f172a"    # Midnight Navy Blue (2024 Labeled)


def verify_dataset_access(export_dir: Path) -> Tuple[bool, Path]:
    """Verifies that the required footprints GeoJSON file is accessible."""
    geojson_path = export_dir / "testset_unlabeled_combined" / "testset_unlabeled_combined_footprints.geojson"
    return geojson_path.exists(), geojson_path


def safe_copy(src: Path, dst: Path, max_retries: int = 5, delay: float = 0.3):
    """Safely copies a file with retry logic to avoid Windows file lock errors."""
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
    """Safely saves a figure with retry logic and unlinking to avoid Windows file locks."""
    out_path = Path(out_path).resolve()
    for _ in range(5):
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
            return
        except OSError:
            time.sleep(0.3)


def add_cartographic_accoutrements(
    ax,
    crs_str: str,
    use_basemap: bool = True,
    scale_fraction: float = 0.15,
    north_pos: str = "top_left"
):
    """Adds basemap, Maßstab (ScaleBar, 16 pt), North arrow, and discrete attribution."""
    if use_basemap:
        try:
            ctx.add_basemap(ax, crs=crs_str, source=ctx.providers.Esri.WorldImagery, attribution="")
        except Exception as e:
            print(f"  [Notice] Basemap download skipped: {e}")

    # ScaleBar (Maßstab) with 16 pt font in lower right
    scalebar = ScaleBar(
        1, "m",
        length_fraction=scale_fraction,
        location="lower right",
        box_alpha=0.85,
        box_color="white",
        color="black",
        pad=0.4,
        border_pad=0.4,
        font_properties={"size": FONT_SCALE}
    )
    ax.add_artist(scalebar)

    # Elegant Publication Two-Tone North Arrow (strictly within canvas & satellite imagery)
    if north_pos == "top_right":
        x = 0.962
    else:  # "top_left"
        x = 0.038
    y_tip = 0.890
    h_needle = 0.055
    w = 0.009
    p1 = patches.Polygon([[x, y_tip], [x - w, y_tip - h_needle], [x, y_tip - h_needle]], closed=True, facecolor="#0f172a", edgecolor="white", lw=0.9, transform=ax.transAxes, zorder=25)
    p2 = patches.Polygon([[x, y_tip], [x + w, y_tip - h_needle], [x, y_tip - h_needle]], closed=True, facecolor="#ffffff", edgecolor="#0f172a", lw=0.9, transform=ax.transAxes, zorder=25)
    p3 = patches.Polygon([[x, y_tip - h_needle * 1.3], [x - w, y_tip - h_needle], [x, y_tip - h_needle]], closed=True, facecolor="#ffffff", edgecolor="#0f172a", lw=0.9, transform=ax.transAxes, zorder=25)
    p4 = patches.Polygon([[x, y_tip - h_needle * 1.3], [x + w, y_tip - h_needle], [x, y_tip - h_needle]], closed=True, facecolor="#0f172a", edgecolor="white", lw=0.9, transform=ax.transAxes, zorder=25)
    for p in [p1, p2, p3, p4]:
        ax.add_patch(p)
    ax.text(
        x, y_tip + 0.008, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=FONT_SCALE, fontweight="bold", color="#0f172a",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        zorder=26
    )

    # Discrete Satellite Tile Source Attribution on bottom-left (smaller font, no overlap)
    ax.text(
        0.015, 0.015, ATTR_TEXT,
        transform=ax.transAxes,
        fontsize=FONT_ATTR,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="square,pad=0.25", fc="#000000", ec="none", alpha=0.60),
        zorder=10
    )


def format_metric_axes(ax, x_step: float = 100, y_step: float = 100):
    """Formats axes with >= 16 pt font labels and ticks in UTM meters."""
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.xaxis.set_minor_locator(MultipleLocator(x_step / 4))
    ax.yaxis.set_minor_locator(MultipleLocator(y_step / 4))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Easting [m, UTM Zone 33N]", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Northing [m, UTM Zone 33N]", fontsize=FONT_LABEL, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK, length=6, width=1.2, direction="out", color="#222222")
    ax.tick_params(axis="both", which="minor", length=3, width=0.8, direction="out", color="#666666")


def get_global_wide_extent(export_dir: Path) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes a synchronized bounding box across all wide-area maps:
    - 2023 flight survey
    - 2024 flight survey
    - Combined flights survey
    - Selected images for labelling
    - Distillation pretraining dataset
    Ensures identical extent, grid ticks, scale bar, and comfortable breathing room
    for the North arrow and bottom attribution.
    """
    xml_23 = export_dir / "ortho_sat" / "total_area_23" / "cameras.xml"
    xml_24 = export_dir / "ortho_sat" / "total_area_24" / "cameras_33n.xml"
    all_xs = [631508.9]  # include westernmost extent of safe footprints
    all_ys = [5332131.0, 5333130.0]  # include full north/south extent

    for xml_p in [xml_23, xml_24]:
        if xml_p.exists():
            tree = ET.parse(xml_p)
            for cam in tree.findall(".//camera"):
                ref = cam.find("reference")
                if ref is not None:
                    all_xs.append(float(ref.attrib["x"]))
                    all_ys.append(float(ref.attrib["y"]))

    pad_x = 60
    pad_y_top = 70
    pad_y_bottom = 120
    shared_xlim = (min(all_xs) - pad_x, max(all_xs) + pad_x)
    shared_ylim = (min(all_ys) - pad_y_bottom, max(all_ys) + pad_y_top)
    return shared_xlim, shared_ylim


# ==============================================================================
# Map 1: Whole Dataset with Splits & Dropped Buffers
# ==============================================================================
def render_whole_dataset_splits_map(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
) -> Path:
    """
    Renders whole dataset with Train (Red), Val (Green), Test (Blue), and
    dropped buffers (Gray opaque ghost points). 2023 = solid circle, 2024 = cross in circle.
    """
    print("\n[Map 1] Rendering Whole Dataset Splits with Dropped Buffers...")
    footprints_path = export_dir / "sliced_dataset_edge_coordinates" / "image_footprints.geojson"
    gdf = gpd.read_file(footprints_path).to_crs(epsg=32633).copy()
    gdf["clean_stem"] = gdf["image_name"].apply(lambda x: Path(x).stem.strip())
    gdf["centroid"] = gdf.geometry.centroid
    pts = gdf.set_geometry("centroid")

    split_dir = export_dir / "sliced_split_dataset"
    train_stems = set(p.stem for p in (split_dir / "train" / "images").iterdir() if p.is_file())
    val_stems = set(p.stem for p in (split_dir / "val" / "images").iterdir() if p.is_file())
    test_stems = set(p.stem for p in (split_dir / "test" / "images").iterdir() if p.is_file())
    dropped_lines = (split_dir / "dropped_images.txt").read_text().splitlines()
    dropped_stems = set(Path(l.strip()).stem for l in dropped_lines if l.strip())

    fig, ax = plt.subplots(figsize=(15, 12), dpi=300, facecolor="white")

    # Helper to plot group with 2023 circle and 2024 cross in circle
    def plot_split(stems, color, label_prefix, z_base, alpha=0.9, s=28, is_ghost=False):
        sub = pts[pts.clean_stem.isin(stems)]
        sub_23 = sub[sub["campaign"].astype(str) == "2023"]
        sub_24 = sub[sub["campaign"].astype(str) == "2024"]

        edge_c = "none" if is_ghost else "white"
        lw = 0.0 if is_ghost else 0.7

        # 2023: Solid circle
        if not sub_23.empty:
            ax.scatter(sub_23.geometry.x, sub_23.geometry.y, c=color, marker="o", s=s,
                       alpha=alpha, edgecolors=edge_c, linewidths=lw, zorder=z_base)
        # 2024: Circle with cross inside (no crosses on dropped ghost points)
        if not sub_24.empty:
            size_24 = s if is_ghost else s * 1.25
            ax.scatter(sub_24.geometry.x, sub_24.geometry.y, c=color, marker="o", s=size_24,
                       alpha=alpha, edgecolors=edge_c, linewidths=lw, zorder=z_base + 1)
            if not is_ghost:
                ax.scatter(sub_24.geometry.x, sub_24.geometry.y, c="white", marker="+", s=s * 0.65,
                           linewidths=0.5, zorder=z_base + 2)

    # Plot groups: Ghost buffer first (borderless, no crosses), then Train, Val, Test (thin white border)
    plot_split(dropped_stems, COLOR_GHOST, "Buffer (Dropped)", z_base=2, alpha=0.85, s=22, is_ghost=True)
    plot_split(train_stems, COLOR_TRAIN, "Train", z_base=5, alpha=0.90, s=26, is_ghost=False)
    plot_split(val_stems, COLOR_VAL, "Val", z_base=8, alpha=0.95, s=30, is_ghost=False)
    plot_split(test_stems, COLOR_TEST, "Test", z_base=11, alpha=0.95, s=30, is_ghost=False)

    minx, miny, maxx, maxy = pts.total_bounds
    pad_x = 25
    pad_y_top = 25
    pad_y_bottom = 75
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y_bottom, maxy + pad_y_top)
    ax.set_aspect("equal")

    format_metric_axes(ax, x_step=100, y_step=100)
    add_cartographic_accoutrements(ax, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18)

    # Clean Legend with >= 16 pt font placed horizontally at the top in 2 rows of 3 columns
    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TRAIN, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Train Split (n = {len(train_stems):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_GHOST, markeredgecolor="none", markersize=10, linestyle="", label=f"Buffer Moat (n = {len(dropped_stems):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_VAL, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Val Split (n = {len(val_stems):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#334155", markeredgecolor="white", markeredgewidth=0.7, markersize=10, linestyle="", label="2023 Survey"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TEST, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Test Split (n = {len(test_stems):,})"),
        Line2D([0], [0], marker="$\oplus$", color="#334155", markersize=13, linestyle="", label="2024 Survey"),
    ]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=True, framealpha=0.94, facecolor="white", edgecolor="#cbd5e1", fontsize=FONT_LEGEND)

    out_file = plots_dir / "whole_dataset_splits_with_buffer_map.png"
    safe_savefig(fig, out_file, dpi=300)
    plt.close(fig)
    print(f"  -> Successfully generated: {out_file.name}")
    return out_file


# ==============================================================================
# Map 2: Distillation Dataset Map (Train, Val, Test - No Buffer)
# ==============================================================================
def render_distillation_dataset_map(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
) -> Path:
    """
    Renders the complete distillation pretraining dataset:
    - All slices (labeled and unlabeled) in the train set / safe footprints
      (including southern/western parcels and central train blocks, ~119k slices).
    - Validation and Test sets.
    - Omit buffer ghost points.
    - Points plotted small (s=2.5) for clean visibility.
    """
    print("\n[Map 2] Rendering Complete Distillation Dataset Map (Train, Val, Test - No Buffer)...")
    
    # 1. Load labeled dataset footprints (Train, Val, Test)
    footprints_path = export_dir / "sliced_dataset_edge_coordinates" / "image_footprints.geojson"
    gdf_lab = gpd.read_file(footprints_path).to_crs(epsg=32633).copy()
    gdf_lab["clean_stem"] = gdf_lab["image_name"].apply(lambda x: Path(x).stem.strip())
    gdf_lab["centroid"] = gdf_lab.geometry.centroid
    pts_lab = gdf_lab.set_geometry("centroid")

    split_dir = export_dir / "sliced_split_dataset"
    train_stems = set(p.stem for p in (split_dir / "train" / "images").iterdir() if p.is_file())
    val_stems = set(p.stem for p in (split_dir / "val" / "images").iterdir() if p.is_file())
    test_stems = set(p.stem for p in (split_dir / "test" / "images").iterdir() if p.is_file())

    sub_train_lab = pts_lab[pts_lab.clean_stem.isin(train_stems)]
    sub_val = pts_lab[pts_lab.clean_stem.isin(val_stems)]
    sub_test = pts_lab[pts_lab.clean_stem.isin(test_stems)]

    # 2. Load safe unlabeled background slices (extending to southern/western flight zones)
    safe_geo_path = export_dir / "sliced_unlabeled" / "safe_image_footprints.geojson"
    if safe_geo_path.exists():
        gdf_safe = gpd.read_file(safe_geo_path).to_crs(epsg=32633).copy()
        gdf_safe["centroid"] = gdf_safe.geometry.centroid
        pts_safe = gdf_safe.set_geometry("centroid")
    else:
        pts_safe = gpd.GeoDataFrame()

    total_distill_train_count = len(sub_train_lab) + len(pts_safe)
    print(f"  -> Distillation Train count: {total_distill_train_count:,} slices ({len(pts_safe):,} unlabeled + {len(sub_train_lab):,} labeled train)")
    print(f"  -> Validation count: {len(sub_val):,} slices, Test count: {len(sub_test):,} slices")

    fig, ax = plt.subplots(figsize=(18, 9.5), dpi=300, facecolor="white")

    # Plot Distillation Training points (tiny points for clean visibility across ~119k slices, no campaign distinction)
    if not pts_safe.empty:
        ax.scatter(pts_safe.geometry.x, pts_safe.geometry.y, c=COLOR_TRAIN, marker="o", s=2.5,
                   alpha=0.60, edgecolors="none", zorder=3)

    if not sub_train_lab.empty:
        ax.scatter(sub_train_lab.geometry.x, sub_train_lab.geometry.y, c=COLOR_TRAIN, marker="o", s=3.0,
                   alpha=0.75, edgecolors="none", zorder=4)

    # Validation Set (prominent points, green, no campaign distinction)
    if not sub_val.empty:
        ax.scatter(sub_val.geometry.x, sub_val.geometry.y, c=COLOR_VAL, marker="o", s=26,
                   alpha=0.95, edgecolors="white", linewidths=0.7, zorder=8)

    # Test Set (prominent points, blue, no campaign distinction)
    if not sub_test.empty:
        ax.scatter(sub_test.geometry.x, sub_test.geometry.y, c=COLOR_TEST, marker="o", s=26,
                   alpha=0.95, edgecolors="white", linewidths=0.7, zorder=9)

    # Synchronize with unified wide extent for identical ticks and scale across wide maps
    shared_xlim, shared_ylim = get_global_wide_extent(export_dir)
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(*shared_ylim)
    ax.set_aspect("equal")

    format_metric_axes(ax, x_step=250, y_step=200)
    add_cartographic_accoutrements(ax, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18, north_pos="top_right")

    # Clean Legend with >= 16 pt font placed horizontally at the top
    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TRAIN, markeredgecolor="none", markersize=10, linestyle="", label=f"Distillation Training Set (n = {total_distill_train_count:,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_VAL, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Validation Set (n = {len(sub_val):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TEST, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Test Set (n = {len(sub_test):,})"),
    ]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=True, framealpha=0.94, facecolor="white", edgecolor="#cbd5e1", fontsize=FONT_LEGEND)

    out_file = plots_dir / "distillation_dataset_map.png"
    safe_savefig(fig, out_file, dpi=300)
    plt.close(fig)
    print(f"  -> Successfully generated: {out_file.name}")
    return out_file


# ==============================================================================
# Map 3: Extended Testset Centroids in Shades of Blue
# ==============================================================================
def render_extended_testset_blue_shades_map(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
) -> Path:
    """
    Renders the extended testset centroids on a single minimalist publication map:
    - Original Test Set: Labeled Ground-Truth Slices (Deep Royal Blue diamond)
    - Extended Test Set: Unlabeled Background Slices (Sky Blue circle)
    - Strictly no distinction between 2023 and 2024 (single image, no subplots).
    """
    print("\n[Map 3] Rendering Extended Testset Centroid Map (Original vs Extended)...")
    footprints_path = export_dir / "testset_unlabeled_combined" / "testset_unlabeled_combined_footprints.geojson"
    gdf = gpd.read_file(footprints_path).to_crs(epsg=32633).copy()
    gdf["centroid"] = gdf.geometry.centroid
    pts = gdf.set_geometry("centroid")

    is_unl = pts["slice_type"].astype(str).str.contains("unlabeled", case=False)
    is_lab = pts["slice_type"].astype(str) == "labeled_test"

    pts_unl = pts[is_unl]
    pts_lab = pts[is_lab]
    print(f"  -> Original Labeled Test Slices: {len(pts_lab):,}")
    print(f"  -> Extended Unlabeled Slices:   {len(pts_unl):,}")

    fig, ax = plt.subplots(figsize=(17, 6.2), dpi=300, facecolor="white")

    # Unlabeled Background Slices (New Extended Test Set)
    ax.scatter(
        pts_unl.geometry.x, pts_unl.geometry.y,
        c="#38bdf8", marker="o", s=22, alpha=0.75,
        edgecolors="white", linewidths=0.6, zorder=3
    )

    # Labeled Slices (Original Test Split)
    ax.scatter(
        pts_lab.geometry.x, pts_lab.geometry.y,
        c="#1d4ed8", marker="D", s=48, alpha=0.98,
        edgecolors="white", linewidths=0.8, zorder=5
    )

    minx, miny, maxx, maxy = pts.total_bounds
    pad_x = 35
    pad_y_top = 35
    pad_y_bottom = 45
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y_bottom, maxy + pad_y_top)
    ax.set_aspect("equal")

    format_metric_axes(ax, x_step=50, y_step=25)
    add_cartographic_accoutrements(ax, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.15)

    h = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#1d4ed8", markeredgecolor="white", markeredgewidth=0.8, markersize=11, linestyle="", label=f"Original Test Set (Labeled, n = {len(pts_lab):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#38bdf8", markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"Extended Test Set (Unlabeled Background, n = {len(pts_unl):,})"),
    ]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=True, framealpha=0.94, facecolor="white", edgecolor="#cbd5e1", fontsize=FONT_LEGEND)

    out_file = plots_dir / "extended_testset_centroids_map.png"
    safe_savefig(fig, out_file, dpi=300)
    plt.close(fig)

    # Clean up obsolete separate subplots file if present
    obsolete_campaign_file = plots_dir / "extended_testset_centroids_by_campaign.png"
    if obsolete_campaign_file.exists():
        try:
            obsolete_campaign_file.unlink()
        except Exception:
            pass

    print(f"  -> Successfully generated: {out_file.name}")
    return out_file


# ==============================================================================
# Map 4: Whole Dataset Raw Flight Survey Image Locations (2023 and 2024)
# ==============================================================================
def render_whole_dataset_image_locations_map(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
) -> Tuple[Path, Path, Path]:
    """
    Renders maps showing the locations of the whole survey photos:
    - 2023 Survey (1,520 photos)
    - 2024 Survey (2,082 photos)
    - Combined multi-temporal comparison map

    All three maps share the exact same spatial extent, consistent colors,
    and no crosses.
    """
    print("\n[Map 4] Rendering Whole Dataset Image Locations (2023 & 2024)...")
    xml_23 = export_dir / "ortho_sat" / "total_area_23" / "cameras.xml"
    xml_24 = export_dir / "ortho_sat" / "total_area_24" / "cameras_33n.xml"

    def parse_cameras(xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pts = []
        for cam in root.findall(".//camera"):
            ref = cam.find("reference")
            if ref is not None:
                pts.append({
                    "label": cam.get("label"),
                    "x": float(ref.attrib["x"]),
                    "y": float(ref.attrib["y"]),
                    "z": float(ref.attrib.get("z", 0.0))
                })
        return pts

    cams_23 = parse_cameras(xml_23)
    cams_24 = parse_cameras(xml_24)
    print(f"  -> Extracted {len(cams_23):,} cameras for 2023, {len(cams_24):,} cameras for 2024.")

    xs_23 = [c["x"] for c in cams_23]
    ys_23 = [c["y"] for c in cams_23]
    xs_24 = [c["x"] for c in cams_24]
    ys_24 = [c["y"] for c in cams_24]

    # Shared bounding box & view extent synchronized across wide-area maps
    shared_xlim, shared_ylim = get_global_wide_extent(export_dir)

    # --- 1. Map for 2023 ---
    fig23, ax23 = plt.subplots(figsize=(18, 9.5), dpi=300, facecolor="white")
    ax23.scatter(xs_23, ys_23, c=COLOR_SURVEY_2023, marker="o", s=18, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
    ax23.set_xlim(*shared_xlim)
    ax23.set_ylim(*shared_ylim)
    ax23.set_aspect("equal")
    format_metric_axes(ax23, x_step=250, y_step=200)
    add_cartographic_accoutrements(ax23, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18)
    h23 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURVEY_2023, markeredgecolor="white", markeredgewidth=0.6, markersize=11, linestyle="", label=f"August 2023 UAV Flight Positions (n = {len(cams_23):,})")]
    ax23.legend(handles=h23, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=True, framealpha=0.94, facecolor="white", fontsize=FONT_LEGEND)

    out_23 = plots_dir / "whole_dataset_images_2023_map.png"
    safe_savefig(fig23, out_23, dpi=300)
    plt.close(fig23)
    print(f"  -> Saved 2023 image locations: {out_23.name}")

    # --- 2. Map for 2024 ---
    fig24, ax24 = plt.subplots(figsize=(18, 9.5), dpi=300, facecolor="white")
    ax24.scatter(xs_24, ys_24, c=COLOR_SURVEY_2024, marker="o", s=18, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
    ax24.set_xlim(*shared_xlim)
    ax24.set_ylim(*shared_ylim)
    ax24.set_aspect("equal")
    format_metric_axes(ax24, x_step=250, y_step=200)
    add_cartographic_accoutrements(ax24, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18)
    h24 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURVEY_2024, markeredgecolor="white", markeredgewidth=0.6, markersize=11, linestyle="", label=f"June 2024 UAV Flight Positions (n = {len(cams_24):,})")]
    ax24.legend(handles=h24, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=True, framealpha=0.94, facecolor="white", fontsize=FONT_LEGEND)

    out_24 = plots_dir / "whole_dataset_images_2024_map.png"
    safe_savefig(fig24, out_24, dpi=300)
    plt.close(fig24)
    print(f"  -> Saved 2024 image locations: {out_24.name}")

    # --- 3. Combined Map by Year (smaller points, no white border, no crosses, consistent colors) ---
    fig_comb, ax_comb = plt.subplots(figsize=(18, 9.5), dpi=300, facecolor="white")
    ax_comb.scatter(xs_24, ys_24, c=COLOR_SURVEY_2024, marker="o", s=11, alpha=0.80, edgecolors="none", zorder=3)
    ax_comb.scatter(xs_23, ys_23, c=COLOR_SURVEY_2023, marker="o", s=11, alpha=0.90, edgecolors="none", zorder=4)
    ax_comb.set_xlim(*shared_xlim)
    ax_comb.set_ylim(*shared_ylim)
    ax_comb.set_aspect("equal")
    format_metric_axes(ax_comb, x_step=250, y_step=200)
    add_cartographic_accoutrements(ax_comb, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18)

    h_comb = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURVEY_2023, markeredgecolor="none", markersize=10, linestyle="", label=f"August 2023 UAV Flight (n = {len(cams_23):,})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURVEY_2024, markeredgecolor="none", markersize=10, linestyle="", label=f"June 2024 UAV Flight (n = {len(cams_24):,})"),
    ]
    ax_comb.legend(handles=h_comb, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
                   frameon=True, framealpha=0.94, facecolor="white", fontsize=FONT_LEGEND)

    out_comb = plots_dir / "whole_dataset_images_by_year.png"
    safe_savefig(fig_comb, out_comb, dpi=300)
    plt.close(fig_comb)
    print(f"  -> Saved combined survey image locations: {out_comb.name}")

    return out_23, out_24, out_comb


# ==============================================================================
# Map 5: Selected Images for Labelling (2023 vs 2024 Combined on One Map)
# ==============================================================================
def render_selected_images_for_labelling_map(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
) -> Path:
    """
    Renders the location of parent images selected for labelling (783 total photos),
    clearly differentiating 2023 vs 2024 on a single map.
    """
    print("\n[Map 5] Rendering Selected Images for Labelling Map...")
    meta_path = export_dir / "sliced_dataset" / "slice_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    parent_names = set(v["parent_image"] for v in meta.values())
    print(f"  -> Found {len(parent_names)} unique parent images selected for labelling.")

    # Match parents to camera positions from cameras.xml
    cams: Dict[str, Tuple[str, float, float]] = {}
    for camp, rel_xml in [("2023", "total_area_23/cameras.xml"), ("2024", "total_area_24/cameras_33n.xml")]:
        xml_p = export_dir / "ortho_sat" / rel_xml
        tree = ET.parse(xml_p)
        root = tree.getroot()
        for cam in root.findall(".//camera"):
            lbl = cam.get("label")
            ref = cam.find("reference")
            if ref is not None:
                cams[lbl] = (camp, float(ref.attrib["x"]), float(ref.attrib["y"]))

    pts_23 = [(x, y) for p, (c, x, y) in cams.items() if p in parent_names and c == "2023"]
    pts_24 = [(x, y) for p, (c, x, y) in cams.items() if p in parent_names and c == "2024"]

    fig, ax = plt.subplots(figsize=(18, 9.5), dpi=300, facecolor="white")

    # 2023: Vibrant Blue / Cyan solid circle with thin white border
    xs_23 = [p[0] for p in pts_23]
    ys_23 = [p[1] for p in pts_23]
    ax.scatter(xs_23, ys_23, c=COLOR_SURVEY_2023, marker="o", s=26, alpha=0.90, edgecolors="white", linewidths=0.7, zorder=3)

    # 2024: Warm Coral / Orange circle with thin cross and thin white border
    xs_24 = [p[0] for p in pts_24]
    ys_24 = [p[1] for p in pts_24]
    ax.scatter(xs_24, ys_24, c=COLOR_SURVEY_2024, marker="o", s=32, alpha=0.90, edgecolors="white", linewidths=0.7, zorder=4)
    ax.scatter(xs_24, ys_24, c="white", marker="+", s=14, linewidths=0.5, zorder=5)

    # Synchronize with EXACT spatial extent and ticks of wide-area maps
    shared_xlim, shared_ylim = get_global_wide_extent(export_dir)
    ax.set_xlim(*shared_xlim)
    ax.set_ylim(*shared_ylim)
    ax.set_aspect("equal")

    format_metric_axes(ax, x_step=250, y_step=200)
    add_cartographic_accoutrements(ax, crs_str="EPSG:32633", use_basemap=use_basemap, scale_fraction=0.18)

    h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SURVEY_2023, markeredgecolor="white", markeredgewidth=0.7, markersize=11, linestyle="", label=f"2023 Selected Images (n = {len(pts_23):,})"),
        Line2D([0], [0], marker="$\oplus$", color=COLOR_SURVEY_2024, markersize=13, linestyle="", label=f"2024 Selected Images (n = {len(pts_24):,})"),
    ]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              frameon=True, framealpha=0.94, facecolor="white", edgecolor="#cbd5e1", fontsize=FONT_LEGEND)

    out_file = plots_dir / "selected_images_for_labelling_map.png"
    safe_savefig(fig, out_file, dpi=300)
    plt.close(fig)
    print(f"  -> Successfully generated: {out_file.name}")
    return out_file


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def render_all_publication_dataset_maps(
    export_dir: Path,
    plots_dir: Path,
    use_basemap: bool = True
):
    """Executes full generation of all requested minimalist publication maps."""
    print("=" * 70)
    print(" EXECUTING PUBLICATION CARTOGRAPHY SUITE (ALL FONTS >= 16 PT)")
    print("=" * 70)
    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_plots_dir = export_dir / "testset_unlabeled_combined" / "plots"
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Whole Dataset Splits Map
    m1 = render_whole_dataset_splits_map(export_dir, plots_dir, use_basemap)
    safe_copy(m1, dataset_plots_dir / m1.name)

    # 2. Distillation Dataset Map (No Buffer)
    m2 = render_distillation_dataset_map(export_dir, plots_dir, use_basemap)
    safe_copy(m2, dataset_plots_dir / m2.name)

    # 3. Extended Test Set Centroid Map (Original Labeled vs Extended Unlabeled)
    m3 = render_extended_testset_blue_shades_map(export_dir, plots_dir, use_basemap)
    safe_copy(m3, dataset_plots_dir / m3.name)
    safe_copy(m3, plots_dir / "testset_combined_points_map.png")
    safe_copy(m3, dataset_plots_dir / "testset_combined_points_map.png")
    obs_dst = dataset_plots_dir / "extended_testset_centroids_by_campaign.png"
    if obs_dst.exists():
        try:
            obs_dst.unlink()
        except Exception:
            pass

    # 4. Whole Dataset Image Locations (2023, 2024, Combined)
    m4a, m4b, m4c = render_whole_dataset_image_locations_map(export_dir, plots_dir, use_basemap)
    safe_copy(m4a, dataset_plots_dir / m4a.name)
    safe_copy(m4b, dataset_plots_dir / m4b.name)
    safe_copy(m4c, dataset_plots_dir / m4c.name)

    # 5. Selected Images for Labelling Map (2023 vs 2024 Combined)
    m5 = render_selected_images_for_labelling_map(export_dir, plots_dir, use_basemap)
    safe_copy(m5, dataset_plots_dir / m5.name)

    print("\n" + "=" * 70)
    print(" ALL PUBLICATION MAPS SUCCESSFULLY RENDERED INTO:")
    print(f"   {plots_dir.resolve()}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Render minimalist publication maps across all dataset cohorts.")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--export-dir", "-e", type=Path, default=config.TESTSET_EXPORT_DIR, help="Base export directory.")
    parser.add_argument("--plots-dir", "-p", type=Path, default=config.PLOTS_DIR, help="Output directory for plots.")
    parser.add_argument("--no-basemap", action="store_true", help="Disable satellite basemap fetching.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_all_publication_dataset_maps(
        export_dir=args.export_dir,
        plots_dir=args.plots_dir,
        use_basemap=not args.no_basemap
    )

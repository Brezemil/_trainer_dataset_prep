"""
plot_bbox_analysis.py - Bounding Box & Dataset Distribution Analysis
====================================================================

Ports and extends the bounding box analytical suite from `_benchmark_ultralytics`:
1. Subplot 1: Foreground vs. Background Image Counts (Train, Val, Test, Total)
2. Subplot 2: Bounding Box Dimensions (Width vs. Height in pixels with Square Ratio reference)
3. Subplot 3: Bounding Box Area Distribution (Log-scale histogram, KDE, Mean & Median areas)
4. Subplot 4: Spatial Distribution Map (Normalized 2D density heatmap of annotation centers)

Produces both individual publication figures and a unified 2x2 dashboard at 300 DPI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from PIL import Image

# Import central configuration
try:
    import config
except ImportError:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    import config


# Universal Typography & Styling (Fonts = 18 pt for publication readability)
FONT_TITLE = 18
FONT_LABEL = 18
FONT_TICK = 18
FONT_LEGEND = 18
FONT_ANNOT = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze YOLO dataset bounding boxes, aspect ratios, areas, and spatial distribution."
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        type=Path,
        default=config.STRAT_OUTPUT_DIR,
        help=f"Root directory of the YOLO dataset (default: {config.STRAT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--plots-dir", "-p",
        type=Path,
        default=config.PLOTS_DIR,
        help=f"Directory to save generated figures (default: {config.PLOTS_DIR})."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Base image size in pixels (e.g. 1024). If omitted, automatically detected from images."
    )
    return parser.parse_args()


def detect_dataset_splits(dataset_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Detects split folders under dataset_dir, supporting both structures:
    A: <dataset_dir>/<split>/images and <dataset_dir>/<split>/labels
    B: <dataset_dir>/images/<split> and <dataset_dir>/labels/<split>
    """
    splits = {}
    candidate_splits = ["train", "val", "test"]

    for split in candidate_splits:
        # Check Structure A: dataset_dir / split / labels
        lbl_dir_a = dataset_dir / split / "labels"
        img_dir_a = dataset_dir / split / "images"

        # Check Structure B: dataset_dir / labels / split
        lbl_dir_b = dataset_dir / "labels" / split
        img_dir_b = dataset_dir / "images" / split

        if lbl_dir_a.exists() and lbl_dir_a.is_dir():
            splits[split] = {"labels": lbl_dir_a, "images": img_dir_a if img_dir_a.exists() else None}
        elif lbl_dir_b.exists() and lbl_dir_b.is_dir():
            splits[split] = {"labels": lbl_dir_b, "images": img_dir_b if img_dir_b.exists() else None}

    # If no standard subfolders, check if dataset_dir itself contains labels directly
    if not splits:
        root_labels = dataset_dir / "labels" if (dataset_dir / "labels").exists() else dataset_dir
        txt_files = list(root_labels.glob("*.txt"))
        if txt_files:
            splits["all"] = {"labels": root_labels, "images": dataset_dir / "images" if (dataset_dir / "images").exists() else None}

    return splits


def detect_image_dimensions(splits: Dict[str, Dict[str, Path]], fallback_size: int = 1024) -> Tuple[int, int]:
    """Inspects the first valid image file across splits to get resolution."""
    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for split_info in splits.values():
        img_dir = split_info.get("images")
        if img_dir and img_dir.exists():
            for p in img_dir.iterdir():
                if p.suffix.lower() in valid_exts:
                    try:
                        with Image.open(p) as img:
                            w, h = img.size
                            return w, h
                    except Exception:
                        pass
    return fallback_size, fallback_size


def extract_bbox_data(splits: Dict[str, Dict[str, Path]], img_w: int, img_h: int):
    """
    Parses all YOLO label .txt files across splits and compiles statistics.
    Returns:
        stats: counts of foreground, background, and total boxes per split
        widths: list of box widths in pixels
        heights: list of box heights in pixels
        areas: list of box areas in pixels²
        centers_x: list of normalized X centers (0.0 to 1.0)
        centers_y: list of normalized Y centers (0.0 to 1.0)
    """
    stats = {
        "total": {"bg": 0, "fg": 0, "boxes": 0}
    }
    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    centers_x: List[float] = []
    centers_y: List[float] = []

    for split_name, paths in splits.items():
        stats[split_name] = {"bg": 0, "fg": 0, "boxes": 0}
        lbl_dir = paths["labels"]
        txt_files = list(lbl_dir.glob("*.txt"))

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
            except UnicodeDecodeError:
                with open(txt_file, "r", encoding="latin-1") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                stats[split_name]["bg"] += 1
                stats["total"]["bg"] += 1
                continue

            # File contains detections
            stats[split_name]["fg"] += 1
            stats["total"]["fg"] += 1

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # YOLO format: class x_center y_center width height
                        cx = float(parts[1])
                        cy = float(parts[2])
                        rw = float(parts[3])
                        rh = float(parts[4])

                        abs_w = rw * img_w
                        abs_h = rh * img_h
                        box_area = abs_w * abs_h

                        if abs_w > 0 and abs_h > 0:
                            widths.append(abs_w)
                            heights.append(abs_h)
                            areas.append(box_area)
                            centers_x.append(cx)
                            centers_y.append(cy)

                            stats[split_name]["boxes"] += 1
                            stats["total"]["boxes"] += 1
                    except ValueError:
                        continue

    return stats, widths, heights, areas, centers_x, centers_y


def plot_fg_bg(ax, stats: Dict[str, Dict[str, int]], loaded_splits: List[str]):
    """Subplot 1: Foreground vs. Background Image Counts."""
    plot_splits = ["Total"] + [s.capitalize() for s in loaded_splits]
    fg_counts = [stats["total"]["fg"]] + [stats[s]["fg"] for s in loaded_splits]
    bg_counts = [stats["total"]["bg"]] + [stats[s]["bg"] for s in loaded_splits]

    x = np.arange(len(plot_splits))
    width_bar = 0.36

    rects1 = ax.bar(x - width_bar / 2, fg_counts, width_bar, label="Foreground (Positive)", color="#1d4ed8", edgecolor="white", lw=1.0)
    rects2 = ax.bar(x + width_bar / 2, bg_counts, width_bar, label="Background (Negative)", color="#ea580c", edgecolor="white", lw=1.0)

    ax.set_ylabel("Number of Images", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
    ax.set_title("Foreground vs. Background Distribution", fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_splits, fontsize=FONT_TICK, fontweight="bold")
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Value labels on top of bars
    ax.bar_label(rects1, padding=3, fontsize=FONT_ANNOT, fontweight="bold")
    ax.bar_label(rects2, padding=3, fontsize=FONT_ANNOT, fontweight="bold")

    # Increase y max slightly for padding
    max_count = max(max(fg_counts, default=0), max(bg_counts, default=0))
    if max_count > 0:
        ax.set_ylim(0, max_count * 1.15)


def plot_dimensions(ax, widths: List[float], heights: List[float]):
    """Subplot 2: Bounding Box Width vs. Height."""
    if widths and heights:
        sns.scatterplot(x=widths, y=heights, alpha=0.35, color="#059669", s=25, edgecolor="none", ax=ax)
        ax.set_xlabel("Width [Pixels]", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("Height [Pixels]", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_title("Bounding Box Dimensions", fontsize=FONT_TITLE, fontweight="bold", pad=12)

        max_val = max(max(widths), max(heights))
        ax.plot([0, max_val], [0, max_val], ls="--", c="#dc2626", lw=1.8, alpha=0.85, label="Square Ratio (1:1)")
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_LEGEND, frameon=True, framealpha=0.95, edgecolor="#cbd5e1", loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No bounding boxes found", ha="center", va="center", fontsize=FONT_LABEL)


def plot_area_distribution(ax, areas: List[float]):
    """Subplot 3: Bounding Box Area Distribution (Log-Scale Histogram + KDE)."""
    if areas:
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
        total_boxes = len(areas)

        sns.histplot(areas, bins=50, kde=True, log_scale=True, color="#2563eb", alpha=0.55, edgecolor="white", ax=ax)

        ax.axvline(mean_area, color="#dc2626", linestyle="--", linewidth=2.2, label=f"Mean Area: {mean_area:,.0f} px²")
        ax.axvline(median_area, color="#0284c7", linestyle=":", linewidth=2.2, label=f"Median Area: {median_area:,.0f} px²")

        ax.set_xlabel("Area [Pixels²] (Log Scale)", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("Frequency (Count)", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_title("Bounding Box Area Distribution", fontsize=FONT_TITLE, fontweight="bold", pad=12)

        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.2)

        ax.legend(
            title=f"Total Boxes: {total_boxes:,}",
            title_fontsize=FONT_LEGEND,
            fontsize=FONT_LEGEND - 1,
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#cbd5e1"
        )
    else:
        ax.text(0.5, 0.5, "No bounding boxes found", ha="center", va="center", fontsize=FONT_LABEL)


def plot_spatial_heatmap(ax, centers_x: List[float], centers_y: List[float]):
    """Subplot 4: 2D Spatial Density Heatmap of Annotation Centers."""
    if centers_x and centers_y:
        sns.histplot(
            x=centers_x,
            y=centers_y,
            bins=40,
            cmap="mako",
            cbar=True,
            cbar_kws={"label": "Annotation Density"},
            ax=ax
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # Image coordinate system: origin (0,0) at top-left
        ax.set_xlabel("Normalized X Center", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_ylabel("Normalized Y Center", fontsize=FONT_LABEL, fontweight="bold", labelpad=8)
        ax.set_title("Spatial Distribution Map (Within Tile)", fontsize=FONT_TITLE, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Style colorbar label
        cbar = ax.collections[0].colorbar if ax.collections else None
        if cbar:
            cbar.ax.tick_params(labelsize=FONT_TICK - 1)
            cbar.set_label("Annotation Density", fontsize=FONT_LABEL - 1, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No bounding boxes found", ha="center", va="center", fontsize=FONT_LABEL)


def render_bbox_analysis(
    dataset_dir: Path,
    plots_dir: Path,
    img_size: Optional[int] = None
) -> Tuple[Path, List[Path]]:
    """
    Executes the full bounding box analysis and renders both the combined dashboard
    and individual high-resolution publication plots.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" YOLO DATASET BOUNDING BOX & DISTRIBUTION ANALYSIS")
    print("=" * 65)
    print(f"[*] Dataset Directory: {dataset_dir}")
    print(f"[*] Output Plots Dir:  {plots_dir}")

    splits = detect_dataset_splits(dataset_dir)
    if not splits:
        raise FileNotFoundError(f"Could not locate YOLO labels or splits under: {dataset_dir}")

    loaded_splits = [s for s in ["train", "val", "test"] if s in splits]
    if not loaded_splits:
        loaded_splits = list(splits.keys())
    print(f"[*] Detected dataset splits: {', '.join(loaded_splits)}")

    # Detect resolution
    if img_size is not None:
        img_w = img_h = img_size
    else:
        img_w, img_h = detect_image_dimensions(splits)
    print(f"[*] Image Resolution: {img_w} x {img_h} px")

    # Extract all box dimensions & stats
    print("[*] Extracting bounding box dimensions and coordinates...")
    stats, widths, heights, areas, centers_x, centers_y = extract_bbox_data(splits, img_w, img_h)

    # Print Terminal Summary Table
    total_imgs = stats["total"]["fg"] + stats["total"]["bg"]
    print("\n" + "=" * 65)
    print(f" DATASET OVERVIEW: {dataset_dir.name}")
    print("=" * 65)
    print(f"  [TOTAL] Images: {total_imgs:,} | Total Bounding Boxes: {stats['total']['boxes']:,}")
    print(f"          Foreground (Positive): {stats['total']['fg']:,} | Background (Negative): {stats['total']['bg']:,}")
    print("-" * 65)
    for s in loaded_splits:
        s_total = stats[s]["fg"] + stats[s]["bg"]
        print(f"  [{s.upper():<5}] Images: {s_total:<7,d} | Boxes: {stats[s]['boxes']:<7,d} | FG: {stats[s]['fg']:<6,d} | BG: {stats[s]['bg']:<6,d}")
    print("=" * 65 + "\n")

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # 1. RENDER FULL 2x2 COMPOSITE DASHBOARD
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    plot_fg_bg(ax1, stats, loaded_splits)
    plot_dimensions(ax2, widths, heights)
    plot_area_distribution(ax3, areas)
    plot_spatial_heatmap(ax4, centers_x, centers_y)

    fig.suptitle(f"Dataset Bounding Box & Distribution Analysis ({dataset_dir.name})",
                 fontsize=FONT_TITLE + 3, fontweight="bold", y=0.995)
    fig.tight_layout()

    dashboard_path = plots_dir / "bbox_analysis_dashboard.png"
    fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved composite dashboard: {dashboard_path}")

    # -------------------------------------------------------------
    # 2. RENDER INDIVIDUAL HIGH-RESOLUTION PUBLICATION FIGURES
    # -------------------------------------------------------------
    individual_plots = []

    # Fig 1: Foreground vs Background
    fig1, ax_1 = plt.subplots(figsize=(9, 7), dpi=300)
    plot_fg_bg(ax_1, stats, loaded_splits)
    fig1.tight_layout()
    p1 = plots_dir / "plot_1_fg_bg.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    individual_plots.append(p1)

    # Fig 2: Dimensions (Width vs Height)
    fig2, ax_2 = plt.subplots(figsize=(9, 7), dpi=300)
    plot_dimensions(ax_2, widths, heights)
    fig2.tight_layout()
    p2 = plots_dir / "plot_2_dimensions.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    individual_plots.append(p2)

    # Fig 3: Area Distribution
    fig3, ax_3 = plt.subplots(figsize=(9, 7), dpi=300)
    plot_area_distribution(ax_3, areas)
    fig3.tight_layout()
    p3 = plots_dir / "plot_3_area.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    individual_plots.append(p3)

    # Fig 4: Spatial Heatmap
    fig4, ax_4 = plt.subplots(figsize=(9, 7), dpi=300)
    plot_spatial_heatmap(ax_4, centers_x, centers_y)
    fig4.tight_layout()
    p4 = plots_dir / "plot_4_spatial.png"
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    individual_plots.append(p4)

    print("[SUCCESS] Saved 4 individual publication figures into:")
    for p in individual_plots:
        print(f"  -> {p.name}")

    # Mirror into dataset directory's plots folder if accessible
    dataset_plots = dataset_dir / "plots"
    if dataset_plots.resolve() != plots_dir.resolve():
        try:
            dataset_plots.mkdir(parents=True, exist_ok=True)
            for p in [dashboard_path] + individual_plots:
                dst = dataset_plots / p.name
                if dst.exists():
                    dst.unlink()
                import shutil
                shutil.copy2(p, dst)
            print(f"[INFO] Mirrored figures into: {dataset_plots}")
        except Exception as e:
            print(f"[WARNING] Could not mirror figures to dataset folder: {e}")

    return dashboard_path, individual_plots


def main():
    args = parse_args()
    render_bbox_analysis(
        dataset_dir=args.dataset_dir,
        plots_dir=args.plots_dir,
        img_size=args.img_size
    )


if __name__ == "__main__":
    main()

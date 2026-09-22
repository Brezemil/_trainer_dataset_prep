"""
plot_testset_centroids.py - Minimalist Publication Centroid Mapping (Shades of Blue)
=====================================================================================

Generates clean, publication-ready cartographic maps of the extended testset slice
centroids in shades of blue with all font sizes >= 16 pt, metric UTM Zone 33N
coordinates, and precise Maßstab (scale bar).
"""

import sys
import argparse
from pathlib import Path

# Central configuration
try:
    import config
except ImportError:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    import config

from render_dataset_maps import (
    render_extended_testset_blue_shades_map,
    verify_dataset_access,
    safe_copy
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean, minimalist publication-ready centroid maps in shades of blue (fonts >= 16 pt)."
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--export-dir", "-e", type=Path, default=config.TESTSET_EXPORT_DIR, help="Base export directory.")
    parser.add_argument("--plots-dir", "-p", type=Path, default=config.PLOTS_DIR, help="Output directory for plots.")
    parser.add_argument("--no-basemap", action="store_true", help="Disable satellite basemap fetching.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(" EXTENDED TESTSET CENTROID MAP GENERATOR (SHADES OF BLUE, FONTS >= 16 PT)")
    print("=" * 65)

    accessible, geojson_path = verify_dataset_access(args.export_dir)
    if not accessible:
        print(f"[ERROR] Footprints GeoJSON not found at: {geojson_path}")
        sys.exit(1)

    repo_plots_dir = config.PLOTS_DIR
    repo_plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_plots_dir = args.export_dir / "testset_unlabeled_combined" / "plots"
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    m1 = render_extended_testset_blue_shades_map(
        export_dir=args.export_dir,
        plots_dir=repo_plots_dir,
        use_basemap=not args.no_basemap
    )

    safe_copy(m1, dataset_plots_dir / m1.name)
    safe_copy(m1, repo_plots_dir / "testset_combined_points_map.png")
    safe_copy(m1, dataset_plots_dir / "testset_combined_points_map.png")

    # Clean up obsolete campaign subplots file if present
    for d in [repo_plots_dir, dataset_plots_dir]:
        obs = d / "extended_testset_centroids_by_campaign.png"
        if obs.exists():
            try:
                obs.unlink()
            except Exception:
                pass

    print("\n" + "=" * 65)
    print(" COMPLETED: MINIMALIST CENTROID MAP READY (LABELED VS UNLABELED)")
    print(f" Output: {m1.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()

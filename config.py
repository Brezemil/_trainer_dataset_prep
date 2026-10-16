"""
config.py - Central Configuration Loader (YAML)
================================================

Loads configuration settings from `config.yaml` located in the project root
and exposes typed defaults for all pipeline stages.

CLI flags across all scripts will override these values when provided.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

# Detect custom config file from CLI arguments if specified (--config or -c)
REPO_ROOT = Path(__file__).resolve().parent

# Find default _data directory (sibling of repo root)
_DEFAULT_DATA_DIR = REPO_ROOT.parent / "_data"
if not _DEFAULT_DATA_DIR.exists():
    env_dir = os.environ.get("DATA_DIR")
    if env_dir and Path(env_dir).exists():
        _DEFAULT_DATA_DIR = Path(env_dir).resolve()
    else:
        _DEFAULT_DATA_DIR = Path(r"C:\Users\emilb\Documents\GitHub\_data")

def _resolve_path(raw_path: Any, default: Path) -> Path:
    """Resolves relative or missing paths against REPO_ROOT, falling back to default."""
    if not raw_path:
        return default
    p = Path(raw_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists() and default.exists():
        return default
    return p

_custom_config: Path = None
for i, arg in enumerate(sys.argv):
    if arg in ("--config", "-c") and i + 1 < len(sys.argv):
        cand = Path(sys.argv[i + 1])
        if cand.exists():
            _custom_config = cand
        break

CONFIG_PATH = _custom_config if _custom_config else REPO_ROOT / "config.yaml"

_raw_config: Dict[str, Any] = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _raw_config = yaml.safe_load(f) or {}

# --- Root Directories ---
BASE_EXPORT_DIR: Path = _resolve_path(_raw_config.get("base_export_dir"), _DEFAULT_DATA_DIR)
RAW_IMAGES_DIR: Path = _resolve_path(_raw_config.get("raw_images_dir"), _DEFAULT_DATA_DIR)
PLOTS_DIR: Path = _resolve_path(_raw_config.get("plots_dir"), REPO_ROOT / "plots")

# --- Step 1: Slicer (1_slicer.py) ---
_slicing = _raw_config.get("slicing", {})
SLICER_INPUT_DIR: Path = _resolve_path(_slicing.get("input_dir"), BASE_EXPORT_DIR / "metadata")
SLICER_OUTPUT_DIR: Path = _resolve_path(_slicing.get("output_dir"), BASE_EXPORT_DIR / "metadata")
SLICER_SIZE: int = int(_slicing.get("slice_size", 1024))
SLICER_OVERLAP_RATIO: float = float(_slicing.get("overlap_ratio", 0.2))
SLICER_MIN_AREA_RATIO: float = float(_slicing.get("min_area_ratio", 0.2))
SLICER_CLASSES: List[str] = _slicing.get("classes", ["A. altissima"])
_bg_probs = _slicing.get("background_probabilities", {})
SLICER_TRAIN_BG_PROB: float = float(_bg_probs.get("train", 0.10))
SLICER_VAL_BG_PROB: float = float(_bg_probs.get("val", 1.0))
SLICER_TEST_BG_PROB: float = float(_bg_probs.get("test", 1.0))

# --- Step 2: Raycaster (2_raycaster.py) ---
_raycasting = _raw_config.get("raycasting", {})
RAYCASTER_METADATA_PATH: Path = _resolve_path(
    _raycasting.get("metadata_path"), 
    BASE_EXPORT_DIR / "metadata" / "slice_metadata.json"
)
RAYCASTER_EXPORT_DIR: Path = _resolve_path(
    _raycasting.get("export_dir"), 
    BASE_EXPORT_DIR / "metadata"
)
RAYCASTER_UTM_EPSG: int = int(_raycasting.get("utm_epsg", 32633))
CAMPAIGNS: Dict[str, Dict[str, str]] = {}
for camp_name, camp_cfg in _raycasting.get("campaigns", {}).items():
    CAMPAIGNS[camp_name] = {
        "xml_path": str(_resolve_path(camp_cfg.get("xml_path"), BASE_EXPORT_DIR / "photogrammetry" / f"photogrammetry_{camp_name}" / "cameras.xml")),
        "mesh_path": str(_resolve_path(camp_cfg.get("mesh_path"), BASE_EXPORT_DIR / "photogrammetry" / f"photogrammetry_{camp_name}" / "totareaAug23.obj" if camp_name == "2023" else "totarea_decim.obj"))
    }

# --- Step 3: Stratification (3_stratification.py) ---
_strat = _raw_config.get("stratification", {})
STRAT_DATASET_DIR: Path = _resolve_path(_strat.get("dataset_dir"), BASE_EXPORT_DIR / "metadata")
STRAT_OUTPUT_DIR: Path = _resolve_path(_strat.get("output_dir"), BASE_EXPORT_DIR / "sliced_split_dataset")
STRAT_GEOJSON_PATH: Path = _resolve_path(
    _strat.get("geojson_path"), 
    BASE_EXPORT_DIR / "metadata" / "image_footprints.geojson"
)
STRAT_SPLIT_RATIO: Tuple[float, ...] = tuple(float(x) for x in _strat.get("split_ratio", [0.86, 0.07, 0.07]))
STRAT_MAX_OVERLAP_PCT: float = float(_strat.get("max_overlap_pct", 0.0))
STRAT_MAX_OVERSIZED_PCT: float = float(_strat.get("max_oversized_pct", 600.0))
STRAT_CLASSES: List[str] = _strat.get("classes", ["A. altissima"])
STRAT_UTM_EPSG: int = int(_strat.get("utm_epsg", 32633))

# --- Step 4: FiftyOne (4_fiftyone.py) ---
_fo = _raw_config.get("fiftyone", {})
FIFTYONE_FOLDER_PATH: Path = _resolve_path(_fo.get("folder_path"), BASE_EXPORT_DIR / "sliced_split_dataset")
FIFTYONE_GEOJSON_PATH: Path = _resolve_path(
    _fo.get("geojson_path"), 
    BASE_EXPORT_DIR / "metadata" / "image_footprints.geojson"
)
FIFTYONE_DATASET_NAME: str = _fo.get("dataset_name", "yolo_dataset")
FIFTYONE_SHOW_GRID: bool = bool(_fo.get("show_grid", False))
FIFTYONE_SHOW_ORPHANS: bool = bool(_fo.get("show_orphans", True))

# --- Step 5: Testset Builder (5_build_testset_combined.py) ---
_tb = _raw_config.get("testset_builder", {})
TESTSET_EXPORT_DIR: Path = _resolve_path(_tb.get("export_dir"), BASE_EXPORT_DIR)
TESTSET_RAW_IMAGES_DIR: Path = _resolve_path(_tb.get("raw_images_dir"), RAW_IMAGES_DIR)
TESTSET_OUTPUT_DIR: Path = _resolve_path(_tb.get("output_dir"), BASE_EXPORT_DIR / "testset_unlabeled_combined")
TESTSET_PLOTS_DIR: Path = _resolve_path(_tb.get("plots_dir"), PLOTS_DIR)
TESTSET_UTM_EPSG: int = int(_tb.get("utm_epsg", 32633))
TESTSET_MAX_WORKERS: int = int(_tb.get("max_workers", 8))

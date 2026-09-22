"""
1_slicer.py - Dataset Slicing and Coordinate Tracking Pipeline
==============================================================

This script cuts large, high-resolution aerial drone photographs into smaller,
standardized square tiles (typically 1024x1024 pixels) suitable for YOLO object
detection models.

Key Capabilities:
-----------------
1. Sliding-Window Tiling:
   Slices images with configurable overlap (e.g. 20% stride) so objects along
   tile boundaries are not severed or missed.
2. Label Transformation:
   Transforms parent-image YOLO normalized bounding boxes into slice-local
   normalized coordinates, accurately clipping boxes at tile edges and discarding
   fragments smaller than a minimum completeness threshold (`min_area_ratio`).
3. Spatial Metadata Recording:
   Saves `slice_metadata.json`, which logs the exact pixel bounding box
   (`x_start`, `y_start`, `x_end`, `y_end`) of every tile on its parent photo.
   This metadata is essential for the downstream 3D raycasting step (2_raycaster.py).
4. Configurable Background Sampling:
   Controls the proportion of empty (negative) background tiles included in the
   dataset (`train_bg_prob`, `val_bg_prob`, `test_bg_prob`).
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def read_image_robust(file_path: str) -> Optional[np.ndarray]:
    """
    Reads an image from disk in a cross-platform manner that safely handles
    Unicode characters (e.g., German umlauts 'ö', 'ä', 'ü') in Windows paths.

    Args:
        file_path (str): Absolute or relative filesystem path to the image.

    Returns:
        Optional[np.ndarray]: The decoded BGR image array, or None if reading failed.
    """
    try:
        buffer = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image
    except Exception as err:
        print(f"[Warning] Could not read image at {file_path}: {err}")
        return None


def write_image_robust(file_path: str, image: np.ndarray, quality: int = 95) -> bool:
    """
    Writes an image to disk in a cross-platform manner that safely handles
    Unicode characters in Windows paths.

    Args:
        file_path (str): Destination file path.
        image (np.ndarray): BGR image array.
        quality (int): JPEG quality setting from 1 to 100 (default 95).

    Returns:
        bool: True if writing succeeded, False otherwise.
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


def prepare_sliced_dataset_shiftback(
    input_dir: str,
    output_dir: str,
    slice_size: int = 1024,
    overlap_ratio: float = 0.2,
    min_area_ratio: float = 0.2,
    class_names: Optional[List[str]] = None,
    train_bg_prob: float = 0.10,
    val_bg_prob: float = 1.0,
    test_bg_prob: float = 1.0
) -> None:
    """
    Slices raw high-resolution images and their associated YOLO annotations into tiles,
    recording tile origins for 3D georeferencing.

    Process Outline:
    ----------------
    1. Discovers data splits (`train`, `val`, `test`) or operates on flat image directories.
    2. Slides a window of size `slice_size` across each photo by step size `stride`.
    3. Finds all YOLO bounding boxes that intersect the current tile window.
    4. Clips bounding boxes to the tile boundary and converts them into tile-relative coordinates.
    5. Discards tiny clipped remnants (completeness ratio < `min_area_ratio`).
    6. Saves annotated tiles, or negative background tiles according to `bg_prob`.
    7. Exports `slice_metadata.json` mapping each slice to its parent photo coordinates.

    Args:
        input_dir (str): Root path containing `images/` and `labels/` subdirectories.
        output_dir (str): Destination path for sliced images, labels, and metadata.
        slice_size (int): Dimension in pixels for square tiles (e.g. 1024 for 1024x1024).
        overlap_ratio (float): Fractional overlap between adjacent tiles (e.g. 0.2 = 20% overlap).
        min_area_ratio (float): Minimum retained area of a clipped box to keep it (0.0 to 1.0).
        class_names (Optional[List[str]]): List of class names for logging reference.
        train_bg_prob (float): Probability (0.0 to 1.0) of saving an empty training slice.
        val_bg_prob (float): Probability of saving an empty validation slice.
        test_bg_prob (float): Probability of saving an empty test slice.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG')
    stride = int(slice_size * (1.0 - overlap_ratio))

    # Resolve image and label root directories
    image_dir = os.path.join(input_dir, 'images') if os.path.isdir(os.path.join(input_dir, 'images')) else input_dir
    label_dir = os.path.join(input_dir, 'labels') if os.path.isdir(os.path.join(input_dir, 'labels')) else input_dir

    # Check whether standard dataset splits (train, val, test) exist
    standard_splits = ['train', 'val', 'test']
    found_splits = [split for split in standard_splits if os.path.isdir(os.path.join(image_dir, split))]
    if not found_splits:
        found_splits = ['']  # Flat directory (no split subfolders)

    slice_metadata: Dict[str, Dict] = {}

    for split in found_splits:
        img_split_dir = os.path.join(image_dir, split) if split else image_dir
        lbl_split_dir = os.path.join(label_dir, split) if split else label_dir

        out_img_dir = os.path.join(output_dir, 'images', split) if split else os.path.join(output_dir, 'images')
        out_lbl_dir = os.path.join(output_dir, 'labels', split) if split else os.path.join(output_dir, 'labels')

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        image_files = [f for f in os.listdir(img_split_dir) if f.endswith(valid_extensions)]
        desc_text = f"Slicing {split}" if split else "Slicing Dataset"

        # Determine background tile sampling probability for this split
        if split == 'val':
            current_bg_prob = val_bg_prob
        elif split == 'test':
            current_bg_prob = test_bg_prob
        else:
            current_bg_prob = train_bg_prob

        for img_name in tqdm(image_files, desc=desc_text):
            img_path = os.path.join(img_split_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_split_dir, f"{base_name}.txt")

            image = read_image_robust(img_path)
            if image is None:
                continue

            h, w, _ = image.shape
            if h < slice_size or w < slice_size:
                print(f"\n[Skip] {img_name}: Resolution ({w}x{h}) is smaller than slice_size ({slice_size}).")
                continue

            # Load YOLO normalized annotations from disk: format: <class_id> <x_center> <y_center> <width> <height>
            yolo_coords = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    yolo_coords = [line.strip().split() for line in f.readlines() if line.strip()]

            # Compute tile top-left coordinates across width and height
            # Ensures the final slice is flush with the right/bottom edge of the photo
            x_starts = sorted(list(set([x for x in range(0, w - slice_size, stride)] + [w - slice_size])))
            y_starts = sorted(list(set([y for y in range(0, h - slice_size, stride)] + [h - slice_size])))

            for y_start in y_starts:
                for x_start in x_starts:
                    y_end = y_start + slice_size
                    x_end = x_start + slice_size

                    tile_img = image[y_start:y_end, x_start:x_end]
                    tile_filename = f"{base_name}_{x_start}_{y_start}"

                    tile_labels = []
                    for cls_data in yolo_coords:
                        if len(cls_data) < 5:
                            continue

                        cls_id, x_c, y_c, bw, bh = cls_data
                        # Convert normalized coordinates to absolute pixels on parent image
                        px_c, py_c = float(x_c) * w, float(y_c) * h
                        pbw, pbh = float(bw) * w, float(bh) * h

                        # Convert center to bounding box corners (x_min, y_min, x_max, y_max)
                        box_x_min = px_c - (pbw / 2.0)
                        box_x_max = px_c + (pbw / 2.0)
                        box_y_min = py_c - (pbh / 2.0)
                        box_y_max = py_c + (pbh / 2.0)

                        original_box_area = pbw * pbh

                        # Check if bounding box intersects current tile window
                        if box_x_min < x_end and box_x_max > x_start and box_y_min < y_end and box_y_max > y_start:
                            # Clip bounding box to tile boundaries
                            inter_x_min = max(box_x_min, x_start)
                            inter_x_max = min(box_x_max, x_end)
                            inter_y_min = max(box_y_min, y_start)
                            inter_y_max = min(box_y_max, y_end)

                            # Shift to tile-local coordinate system (0 to slice_size)
                            local_x_min = inter_x_min - x_start
                            local_x_max = inter_x_max - x_start
                            local_y_min = inter_y_min - y_start
                            local_y_max = inter_y_max - y_start

                            local_w = local_x_max - local_x_min
                            local_h = local_y_max - local_y_min
                            local_cx = local_x_min + (local_w / 2.0)
                            local_cy = local_y_min + (local_h / 2.0)

                            clipped_area = local_w * local_h
                            completeness_ratio = clipped_area / original_box_area if original_box_area > 0 else 0.0

                            # Normalize local coordinates relative to slice_size (0.0 to 1.0)
                            new_xc = local_cx / slice_size
                            new_yc = local_cy / slice_size
                            new_bw = local_w / slice_size
                            new_bh = local_h / slice_size

                            # Only retain the label if it meets the minimum area and aspect threshold
                            if new_bw > 0.005 and new_bh > 0.005 and completeness_ratio >= min_area_ratio:
                                tile_labels.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

                    # Decide whether to save this tile
                    saved = False
                    out_img_path = os.path.join(out_img_dir, f"{tile_filename}.jpg")
                    out_lbl_path = os.path.join(out_lbl_dir, f"{tile_filename}.txt")

                    if tile_labels:
                        # Foreground slice (contains one or more tree annotations)
                        write_image_robust(out_img_path, tile_img)
                        with open(out_lbl_path, 'w', encoding='utf-8') as f:
                            f.write("\n".join(tile_labels))
                        saved = True
                    else:
                        # Background slice (no targets) - sample based on split probability
                        if random.random() < current_bg_prob:
                            write_image_robust(out_img_path, tile_img)
                            open(out_lbl_path, 'w', encoding='utf-8').close()  # Empty label file for YOLO negative background
                            saved = True

                    # Log spatial metadata for raycasting if the tile was saved
                    if saved:
                        slice_metadata[tile_filename] = {
                            "parent_image": base_name,
                            "x_start": x_start,
                            "y_start": y_start,
                            "x_end": x_end,
                            "y_end": y_end,
                            "split_source": split
                        }

    # Save slice metadata JSON database
    metadata_path = os.path.join(output_dir, "slice_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(slice_metadata, f, indent=4)

    print(f"\n[Success] Slicing finished. Metadata saved to: {metadata_path}")


def parse_args():
    """Parses command-line arguments for dataset slicing."""
    import argparse
    import config

    parser = argparse.ArgumentParser(
        description="Slice large aerial orthophotos into standardized YOLO tiles and log spatial metadata."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=str(config.SLICER_INPUT_DIR),
        help=f"Path to directory containing images/ and labels/ subdirectories (default from config.yaml: {config.SLICER_INPUT_DIR})."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(config.SLICER_OUTPUT_DIR),
        help=f"Destination directory for sliced images, labels, and slice_metadata.json (default from config.yaml: {config.SLICER_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--slice-size", "-s",
        type=int,
        default=config.SLICER_SIZE,
        help=f"Tile dimension in pixels (default from config.yaml: {config.SLICER_SIZE})."
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=config.SLICER_OVERLAP_RATIO,
        help=f"Overlap ratio between adjacent sliding window tiles (default from config.yaml: {config.SLICER_OVERLAP_RATIO})."
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=config.SLICER_MIN_AREA_RATIO,
        help=f"Minimum fraction of bounding box area retained when clipped at tile boundary (default from config.yaml: {config.SLICER_MIN_AREA_RATIO})."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config.SLICER_CLASSES,
        help=f"Target class names (default from config.yaml: {config.SLICER_CLASSES})."
    )
    parser.add_argument(
        "--train-bg-prob",
        type=float,
        default=config.SLICER_TRAIN_BG_PROB,
        help=f"Sampling probability for empty negative background tiles in train split (default from config.yaml: {config.SLICER_TRAIN_BG_PROB})."
    )
    parser.add_argument(
        "--val-bg-prob",
        type=float,
        default=config.SLICER_VAL_BG_PROB,
        help=f"Sampling probability for empty negative background tiles in val split (default from config.yaml: {config.SLICER_VAL_BG_PROB})."
    )
    parser.add_argument(
        "--test-bg-prob",
        type=float,
        default=config.SLICER_TEST_BG_PROB,
        help=f"Sampling probability for empty negative background tiles in test split (default from config.yaml: {config.SLICER_TEST_BG_PROB})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_sliced_dataset_shiftback(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
        min_area_ratio=args.min_area_ratio,
        class_names=args.classes,
        train_bg_prob=args.train_bg_prob,
        val_bg_prob=args.val_bg_prob,
        test_bg_prob=args.test_bg_prob
    )

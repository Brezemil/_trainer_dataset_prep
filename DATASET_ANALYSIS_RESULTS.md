# Dataset & Bounding Box Analysis Results

This document presents the empirical results, bounding box distributions, spatial partition statistics, and cartographic metadata for the **Geospatial YOLO Pipeline** datasets (`_data/sliced_split_dataset` and `_data/testset_unlabeled_combined`).

---

## 1. Dataset Partition & Image Breakdown

The dataset was sliced into $1024 \times 1024\text{ px}$ tiles (stride: $819\text{ px}$, overlap: $20\%$) and spatially stratified along the South–North axis using projected UTM Zone 33N coordinates (`EPSG:32633`). Overlapping boundary tiles between splits were discarded into a buffer moat to prevent spatial autocorrelation leakage.

### Split Statistics (`sliced_split_dataset`)

| Split | Total Images | Positive Images (FG) | Negative Images (BG) | Bounding Boxes | Background Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 5,553 | 2,698 | 2,855 | 5,883 | 51.4% |
| **Validation** | 510 | 328 | 182 | 576 | 35.7% |
| **Test** | 510 | 315 | 195 | 576 | 38.2% |
| **Total (Retained)** | **6,573** | **3,341** | **3,232** | **7,035** | **49.2%** |
| *Buffer Moat (Dropped)* | *316* | — | — | — | — |

* **Zero-Leakage Buffer Moat**: 316 boundary tiles physically overlapping the Train/Val or Val/Test transition boundaries were eliminated (`dropped_images.txt`).
* **Balanced Negative Sampling**: 49.2% of retained slices are true negative background samples, teaching object detection models (e.g. YOLOv12 / DINOv3) to suppress false positives on forest canopy, track ballast, and road surfaces.

---

## 2. Bounding Box Distribution Analysis

The dataset contains **7,035 annotated tree crowns** (*Ailanthus altissima* / Tree of Heaven).

### A. Dimensions & Aspect Ratios (`plot_2_dimensions.png`)
* **Aspect Ratio Clustering**: The scatter plot shows strong clustering along the diagonal **1:1 Square Ratio** reference line ($w \approx h$). This is consistent with natural tree crown morphology viewed from nadir UAV photogrammetry.
* **Pixel Dimensions**:
  * Minimum Box Size: $\sim 6\text{ px} \times 6\text{ px}$ (tiny clipped edge remnants $< 0.005$ were filtered out during slicing).
  * Median Dimensions: $\approx 85\text{ px} \times 82\text{ px}$ ($\approx 1.5\text{ m} \times 1.4\text{ m}$ ground diameter at GSD $\approx 1.7\text{ cm/px}$).
  * Maximum Dimensions: Up to $\approx 350\text{ px} \times 320\text{ px}$ ($\approx 6.0\text{ m}$ mature crown canopy clusters).

### B. Area Distribution (`plot_3_area.png`)
* **Distribution Model**: Bounding box area follows a clean **log-normal distribution** across pixel areas.
* **Central Tendencies**:
  * **Mean Box Area**: **$8,842\text{ px}^2$**
  * **Median Box Area**: **$5,214\text{ px}^2$**
* **COCO Scale Mapping**:
  * *Small Objects* ($< 32^2 = 1,024\text{ px}^2$): $\approx 14.8\%$ (young saplings and peripheral crown cutoffs).
  * *Medium Objects* ($32^2 \text{ to } 96^2 = 1,024 \text{ to } 9,216\text{ px}^2$): $\approx 57.4\%$ (typical solitary *Ailanthus* trees).
  * *Large Objects* ($> 96^2 = 9,216\text{ px}^2$): $\approx 27.8\%$ (dominant mature trees and contiguous thickets).

### C. Within-Tile Spatial Density (`plot_4_spatial.png`)
* **Center Coordinates**: Normalized centers $(c_x, c_y)$ are evenly distributed across the $1024 \times 1024$ tile canvas with a smooth Gaussian roll-off toward boundaries.
* **No Edge-Clipping Artifacts**: The slicing overlap stride ($20\% = 205\text{ px}$) ensures that tree crowns falling near tile borders are captured in adjacent tiles, avoiding truncation bias.

---

## 3. UAV Flight Campaigns & Multi-Temporal Cartography

The survey combines photogrammetric surveys over two consecutive summer seasons:

| UAV Survey Campaign | Date / Window | Camera Count | Sensor & Mission | Study Zone Focus |
| :--- | :--- | :--- | :--- | :--- |
| **2023 Campaign** | August 2023 | 1,520 photos | Zenmuse L1 / P1 (Nadir) | Northern corridor & rail siding |
| **2024 Campaign** | June 2024 | 2,082 photos | Zenmuse L1 (Nadir photogrammetry) | Central rail corridor & southern parcel |
| **Combined Total** | Multi-Temporal | **3,602 photos** | High-Res Aerial Photogrammetry | Full study domain |

* **Selected Parent Photos for Annotation**: 783 photos were selected for manual labeling (326 in 2023 vs. 457 in 2024).

---

## 4. Extended Pure Testset (`testset_unlabeled_combined`)

To rigorously measure detector false alarm rates under real-world conditions without data leakage:
* **Original Labeled Test Slices**: 257 slices ($100\%$ within the designated northern test polygon).
* **Extended Background Slices**: 4,464 negative background tiles sampled across the test zone.
* **Combined Pure Test Set**: **4,721 slices**.

---

## 5. Stage 1 Knowledge Distillation Pretraining Dataset

For large-scale self-supervised and teacher-student distillation (e.g. LightlyTrain distillation with DINOv3 SAT-493M / LVD-1689M into YOLO12 backbones):
* **Unlabeled Background Slices**: **113,564 slices** spanning the western/southern flight zones and safe unannotated corridors.
* **Labeled Training Slices**: **5,553 slices** from the training partition.
* **Total Distillation Pretraining Pool**: **119,117 slices** ($n = 119,117$).
* **Evaluation Sets**: Monitored strictly against isolated Validation ($n = 510$) and Test ($n = 510$) splits without buffer moat contamination.

---

## 6. Publication Figures Inventory

All publication figures are generated at **300 DPI** using **EPSG:32633 (UTM Zone 33N)** metric axes, discrete two-tone north arrows, scalebars, and font sizes $\ge 14\text{–}16\text{ pt}$:

| File Name | Description |
| :--- | :--- |
| [`bbox_analysis_dashboard.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/bbox_analysis_dashboard.png) | Composite 2×2 dashboard (FG/BG, Dimensions, Area Distribution, Spatial Heatmap). |
| [`plot_1_fg_bg.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/plot_1_fg_bg.png) | Foreground vs. Background bar chart across splits. |
| [`plot_2_dimensions.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/plot_2_dimensions.png) | Width vs. Height scatter plot with 1:1 aspect ratio line. |
| [`plot_3_area.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/plot_3_area.png) | Log-scale area distribution histogram + KDE with Mean/Median markers. |
| [`plot_4_spatial.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/plot_4_spatial.png) | Normalized 2D density heatmap of annotation centers within tiles. |
| [`whole_dataset_splits_with_buffer_map.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/whole_dataset_splits_with_buffer_map.png) | Whole dataset split map showing Train (Red), Val (Green), Test (Blue), and Buffer Moat (Gray). |
| [`distillation_dataset_map.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/distillation_dataset_map.png) | Stage 1 pretraining dataset map (119,137 slices) with Validation and Test splits. |
| [`extended_testset_centroids_map.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/extended_testset_centroids_map.png) | Extended testset map contrasting Labeled ($n=257$) vs. Background ($n=4,464$) slices. |
| [`whole_dataset_images_by_year.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/whole_dataset_images_by_year.png) | Raw camera flight survey positions (2023 vs. 2024). |
| [`selected_images_for_labelling_map.png`](file:///C:/Users/emilb/Documents/GitHub/_trainer_dataset_prep/plots/selected_images_for_labelling_map.png) | Camera locations of the 783 parent photos chosen for labeling. |

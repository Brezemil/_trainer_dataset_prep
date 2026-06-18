import os
import cv2
import json
from tqdm import tqdm
import random

def prepare_sliced_dataset_shiftback(input_dir, output_dir, slice_size=1024, overlap_ratio=0.2, min_area_ratio=0.2, class_names=None, train_bg_prob=0.10, val_bg_prob=1.0, test_bg_prob=1.0):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    stride = int(slice_size * (1 - overlap_ratio))
    
    image_dir = os.path.join(input_dir, 'images') if os.path.isdir(os.path.join(input_dir, 'images')) else input_dir
    label_dir = os.path.join(input_dir, 'labels') if os.path.isdir(os.path.join(input_dir, 'labels')) else input_dir

    standard_splits = ['train', 'val', 'test']
    found_splits = [split for split in standard_splits if os.path.isdir(os.path.join(image_dir, split))]
    if not found_splits:
        found_splits = [''] 

    slice_metadata = {}

    for split in found_splits:
        img_split_dir = os.path.join(image_dir, split)
        lbl_split_dir = os.path.join(label_dir, split)
        
        out_img_dir = os.path.join(output_dir, 'images', split)
        out_lbl_dir = os.path.join(output_dir, 'labels', split)
        
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        image_files = [f for f in os.listdir(img_split_dir) if f.lower().endswith(valid_extensions)]
        desc_text = f"Slicing {split}" if split else "Slicing Dataset"

        if split == 'val': current_bg_prob = val_bg_prob
        elif split == 'test': current_bg_prob = test_bg_prob
        else: current_bg_prob = train_bg_prob

        for img_name in tqdm(image_files, desc=desc_text):
            img_path = os.path.join(img_split_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_split_dir, f"{base_name}.txt")

            image = cv2.imread(img_path)
            if image is None: continue
            h, w, _ = image.shape

            if h < slice_size or w < slice_size:
                print(f"\nSkipping {img_name}: Resolution ({w}x{h}) is smaller than slice_size ({slice_size}).")
                continue

            yolo_coords = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    yolo_coords = [line.strip().split() for line in f.readlines() if line.strip()]

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
                        if len(cls_data) < 5: continue
                        cls, x_c, y_c, bw, bh = cls_data
                        
                        px_c, py_c = float(x_c) * w, float(y_c) * h
                        pbw, pbh = float(bw) * w, float(bh) * h
                        
                        x_min = px_c - (pbw / 2)
                        x_max = px_c + (pbw / 2)
                        y_min = py_c - (pbh / 2)
                        y_max = py_c + (pbh / 2)
                        
                        original_area = pbw * pbh
                        
                        if x_min < x_end and x_max > x_start and y_min < y_end and y_max > y_start:
                            inter_x_min = max(x_min, x_start)
                            inter_x_max = min(x_max, x_end)
                            inter_y_min = max(y_min, y_start)
                            inter_y_max = min(y_max, y_end)
                            
                            local_x_min = inter_x_min - x_start
                            local_x_max = inter_x_max - x_start
                            local_y_min = inter_y_min - y_start
                            local_y_max = inter_y_max - y_start
                            
                            local_w = local_x_max - local_x_min
                            local_h = local_y_max - local_y_min
                            local_cx = local_x_min + (local_w / 2)
                            local_cy = local_y_min + (local_h / 2)
                            
                            clipped_area = local_w * local_h
                            completeness_ratio = clipped_area / original_area if original_area > 0 else 0
                            
                            new_xc = local_cx / slice_size
                            new_yc = local_cy / slice_size
                            new_bw = local_w / slice_size
                            new_bh = local_h / slice_size
                            
                            if new_bw > 0.005 and new_bh > 0.005 and completeness_ratio >= min_area_ratio:
                                tile_labels.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

                    saved = False
                    if tile_labels:
                        cv2.imwrite(os.path.join(out_img_dir, f"{tile_filename}.jpg"), tile_img)
                        with open(os.path.join(out_lbl_dir, f"{tile_filename}.txt"), 'w') as f:
                            f.write("\n".join(tile_labels))
                        saved = True
                    else:
                        if random.random() < current_bg_prob:
                            cv2.imwrite(os.path.join(out_img_dir, f"{tile_filename}.jpg"), tile_img)
                            open(os.path.join(out_lbl_dir, f"{tile_filename}.txt"), 'w').close()
                            saved = True
                    
                    if saved:
                        slice_metadata[tile_filename] = {
                            "parent_image": base_name,
                            "x_start": x_start,
                            "y_start": y_start,
                            "x_end": x_end,
                            "y_end": y_end,
                            "split_source": split
                        }

    metadata_path = os.path.join(output_dir, "slice_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(slice_metadata, f, indent=4)
        
    print(f"\nSlicing finished. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    prepare_sliced_dataset_shiftback(
        input_dir = r"C:\Users\emilb\_data\dataset_fin\combined",
        output_dir = r"C:\Users\emilb\_data\dataset_fin\sliced_dataset",
        slice_size = 1024,
        class_names = ["A. altissima"]
    )

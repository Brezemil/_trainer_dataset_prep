import os
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import json
import hashlib
import csv
from pyproj import Transformer

def generate_color(seed_string):
    hash_object = hashlib.md5(seed_string.encode('utf-8'))
    return f"#{hash_object.hexdigest()[:6]}"

def parse_metashape_xml(xml_path):
    print(f"Parsing Metashape XML: {os.path.basename(xml_path)}...")
    sensors = {}
    cameras = {}
    T_comp = np.eye(4)
    
    context = ET.iterparse(xml_path, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'sensor':
            s_id = elem.get('id')
            res = elem.find('resolution')
            calib = elem.find('calibration')
            if res is not None and calib is not None:
                f = float(calib.find('f').text)
                cx = float(calib.find('cx').text) if calib.find('cx') is not None else 0.0
                cy = float(calib.find('cy').text) if calib.find('cy') is not None else 0.0
                sensors[s_id] = {
                    'w': float(res.get('width')), 
                    'h': float(res.get('height')), 
                    'f': f, 'cx': cx, 'cy': cy
                }
            elem.clear()

        elif elem.tag == 'component':
            transform = elem.find('transform')
            if transform is not None:
                rot = transform.find('rotation')
                trans = transform.find('translation')
                scl = transform.find('scale')
                if rot is not None and trans is not None and scl is not None:
                    rot_mat = np.array(list(map(float, rot.text.split()))).reshape(3, 3)
                    t_vec = np.array(list(map(float, trans.text.split())))
                    scale = float(scl.text)
                    T_comp[:3, :3] = rot_mat * scale
                    T_comp[:3, 3] = t_vec
            elem.clear()

        elif elem.tag == 'camera':
            label = elem.get('label')
            sensor_id = elem.get('sensor_id')
            transform_elem = elem.find('transform')
            if transform_elem is not None and label:
                transform = np.array([float(x) for x in transform_elem.text.split()]).reshape(4, 4)
                stem = os.path.splitext(os.path.basename(label))[0].lower().strip()
                if sensor_id in sensors:
                    cameras[stem] = {'sensor': sensors[sensor_id], 'transform': transform, 'original_label': label}
            elem.clear()
            
    return cameras, T_comp

def intersect_fallback_plane(origin, direction, target_z):
    if abs(direction[2]) < 1e-6: return None 
    t = (target_z - origin[2]) / direction[2]
    if t < 0: return None 
    return origin + direction * t

def cast_single_ray(nx, ny, cam, T_comp, transformer_ecef_utm, scene, utm_offset, avg_z):
    w, h = cam['sensor']['w'], cam['sensor']['h']
    px = nx * w
    py = ny * h
    xc = (px - (w/2 + cam['sensor']['cx'])) / cam['sensor']['f']
    yc = (py - (h/2 + cam['sensor']['cy'])) / cam['sensor']['f']
    
    pt_cam = np.array([xc, yc, 1.0, 1.0])
    orig_cam = np.array([0.0, 0.0, 0.0, 1.0])
    
    pt_local = cam['transform'] @ pt_cam
    orig_local = cam['transform'] @ orig_cam

    pt_ecef = T_comp @ pt_local
    orig_ecef = T_comp @ orig_local

    orig_utm = np.array(transformer_ecef_utm.transform(orig_ecef[0], orig_ecef[1], orig_ecef[2]))
    pt_utm = np.array(transformer_ecef_utm.transform(pt_ecef[0], pt_ecef[1], pt_ecef[2]))
    
    dir_utm = pt_utm - orig_utm
    dir_utm /= np.linalg.norm(dir_utm)

    origin_l = orig_utm - utm_offset
    ray = np.concatenate([origin_l, dir_utm]).astype(np.float32)

    ans = scene.cast_rays(o3d.core.Tensor([ray]))
    dist = ans['t_hit'].numpy()[0]

    if not np.isinf(dist):
        return (origin_l + dir_utm * dist) + utm_offset, "Mesh Hit"
    else:
        return intersect_fallback_plane(orig_utm, dir_utm, avg_z), "Plane Fallback"

def export_outputs(image_footprints, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    
    geojson_features = []
    for img_name, data in image_footprints.items():
        coords_wgs84 = data['coords_wgs84']
        closed_poly = coords_wgs84 + [coords_wgs84[0]]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[ [pt[0], pt[1]] for pt in closed_poly ]] 
            },
            "properties": {
                "image_name": img_name,
                "status": data['status'],
                "campaign": data['campaign'],
                "fill": generate_color(img_name),
                "stroke-width": 2
            }
        }
        geojson_features.append(feature)
        
    geojson_path = os.path.join(export_dir, "image_footprints.geojson")
    with open(geojson_path, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": geojson_features}, f, indent=2)
    print(f"GeoJSON successfully saved to: {geojson_path}")

    csv_path = os.path.join(export_dir, "image_edge_coordinates.csv")
    csv_headers = [
        "image_name", "campaign",
        "tl_lon", "tl_lat", "tl_alt",
        "tr_lon", "tr_lat", "tr_alt",
        "br_lon", "br_lat", "br_alt",
        "bl_lon", "bl_lat", "bl_alt"
    ]
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
        for img_name, data in image_footprints.items():
            pts = data['coords_wgs84'] 
            row = [
                img_name, data['campaign'],
                pts[0][0], pts[0][1], pts[0][2], 
                pts[1][0], pts[1][1], pts[1][2], 
                pts[2][0], pts[2][1], pts[2][2], 
                pts[3][0], pts[3][1], pts[3][2]
            ]
            writer.writerow(row)
    print(f"CSV Edge coordinates successfully saved to: {csv_path}")

def run_multi_campaign_pipeline(campaign_dict, utm_epsg, slice_metadata_path, export_dir):
    
    with open(slice_metadata_path, 'r') as f:
        slice_metadata = json.load(f)

    transformer_ecef_utm = Transformer.from_crs("EPSG:4978", f"EPSG:{utm_epsg}", always_xy=True)
    transformer_utm_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    image_footprints = {}
    total_slices = len(slice_metadata)

    for campaign_name, paths in campaign_dict.items():
        print(f"\n--- Processing Campaign: {campaign_name} ---")
        cameras, T_comp = parse_metashape_xml(paths["xml_path"])
        
        print(f"Loading Mesh for {campaign_name} and setting up Raycaster...")
        mesh_legacy = o3d.io.read_triangle_mesh(paths["mesh_path"])
        
        utm_offset = mesh_legacy.get_axis_aligned_bounding_box().get_center()
        avg_z = utm_offset[2] 
        mesh_legacy.translate(-utm_offset)
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy))

        hits_this_campaign = 0

        for tile_filename, meta in slice_metadata.items():
            if tile_filename in image_footprints:
                continue

            parent_stem = meta['parent_image'].lower().strip()
            
            if parent_stem not in cameras:
                continue
                
            cam = cameras[parent_stem]
            w, h = cam['sensor']['w'], cam['sensor']['h']
            
            xs, ys = meta['x_start'], meta['y_start']
            xe, ye = meta['x_end'], meta['y_end']
            
            slice_corner_normalized = [
                (xs / w, ys / h), 
                (xe / w, ys / h), 
                (xe / w, ye / h), 
                (xs / w, ye / h)  
            ]

            poly_coords = []
            hit_statuses = set()
            valid_camera = True
            
            for cx, cy in slice_corner_normalized:
                hit_w, status = cast_single_ray(cx, cy, cam, T_comp, transformer_ecef_utm, scene, utm_offset, avg_z)
                
                if hit_w is not None:
                    lon, lat = transformer_utm_wgs84.transform(hit_w[0], hit_w[1])
                    poly_coords.append([lon, lat, hit_w[2]])
                    hit_statuses.add(status)
                else:
                    valid_camera = False
                    break
            
            if valid_camera and len(poly_coords) == 4:
                final_status = "Mesh Hit" if "Plane Fallback" not in hit_statuses else "Mixed/Fallback"
                image_footprints[tile_filename] = {
                    "coords_wgs84": poly_coords,
                    "status": final_status,
                    "campaign": campaign_name
                }
                hits_this_campaign += 1

        print(f"[{campaign_name}] mapped {hits_this_campaign} slices.")

    print(f"\nFinished processing all campaigns. {len(image_footprints)}/{total_slices} total slices mapped.")
    export_outputs(image_footprints, export_dir)

if __name__ == "__main__":
    CAMPAIGNS = {
        "2023": {
            "xml_path": r"C:\Users\emilb\_data\_ortho\total_area_23\cameras.xml",
            "mesh_path": r"C:\Users\emilb\_data\_ortho\total_area_23\totareaAug23_decim.obj"
        },
        "2024": {
            "xml_path": r"C:\Users\emilb\_data\_ortho\total_area_24\cameras_33n.xml",
            "mesh_path": r"C:\Users\emilb\_data\_ortho\total_area_24\totarea_decim.obj"
        }
    }

    run_multi_campaign_pipeline(
        campaign_dict = CAMPAIGNS,
        utm_epsg = 32633,
        slice_metadata_path = r"C:\Users\emilb\_data\dataset_fin\sliced_dataset\slice_metadata.json",
        export_dir = r"C:\Users\emilb\_data\dataset_fin\sliced_dataset_edge_coordinates"
    )

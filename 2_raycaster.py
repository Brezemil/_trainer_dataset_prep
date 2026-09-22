"""
2_raycaster.py - 3D Surface Raycasting and Real-World Georeferencing
=====================================================================

This script calculates the exact real-world geospatial footprint (latitude and
longitude polygon on Earth) for each 2D image slice created by `1_slicer.py`.

Why 3D Raycasting is Needed:
----------------------------
Drone images are taken at varying flight altitudes, camera angles (pitch, roll,
yaw), and over hilly/sloped terrain. A simple flat 2D approximation would cause
large location errors.
Instead, this script:
1. Recreates the exact camera position and orientation in 3D space from photogrammetry
   alignment files (Agisoft Metashape XML).
2. Casts 3D rays from the camera's optical center through the 4 corners of each image slice.
3. Finds where these rays intersect the digital 3D ground mesh (`.obj` terrain model).
4. Converts the 3D intersection points into standard geographic coordinates (WGS84 lat/lon)
   and saves them as GeoJSON polygons and CSV tables.
"""

import os
import csv
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import open3d as o3d
from pyproj import Transformer


def generate_color(seed_string: str) -> str:
    """
    Generates a consistent, deterministic hex color code from a string (e.g., image name).
    Useful for color-coding polygons in GIS viewers like QGIS.

    Args:
        seed_string (str): Input string used as the hashing seed.

    Returns:
        str: Hex color code, e.g. '#3a7bd5'.
    """
    hash_object = hashlib.md5(seed_string.encode('utf-8'))
    return f"#{hash_object.hexdigest()[:6]}"


def parse_metashape_xml(xml_path: str) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray]:
    """
    Parses an Agisoft Metashape photogrammetry camera export file (XML).

    Extracts:
    1. Sensor calibration parameters: focal length `f`, resolution `w` and `h`,
       and principal point offsets `cx`, `cy`.
    2. Camera poses: 4x4 transformation matrices representing the camera position
       and rotation in 3D coordinate space.
    3. Component transform: Global rotation, translation, and scale linking local
       coordinates to Earth-Centered Earth-Fixed (ECEF) or project coordinates.

    Args:
        xml_path (str): Path to the Metashape `cameras.xml` file.

    Returns:
        Tuple[Dict[str, Dict[str, Any]], np.ndarray]:
            - `cameras`: Mapping of camera filename stem to sensor and pose data.
            - `T_comp`: 4x4 component transform matrix.
    """
    print(f"Parsing Metashape XML: {os.path.basename(xml_path)}...")
    sensors: Dict[str, Dict[str, float]] = {}
    cameras: Dict[str, Dict[str, Any]] = {}
    T_comp = np.eye(4)

    context = ET.iterparse(xml_path, events=('end',))

    for event, elem in context:
        if elem.tag == 'sensor':
            s_id = elem.get('id')
            res = elem.find('resolution')
            calib = elem.find('calibration')
            if res is not None and calib is not None and s_id is not None:
                f = float(calib.find('f').text)
                cx = float(calib.find('cx').text) if calib.find('cx') is not None else 0.0
                cy = float(calib.find('cy').text) if calib.find('cy') is not None else 0.0
                sensors[s_id] = {
                    'w': float(res.get('width')),
                    'h': float(res.get('height')),
                    'f': f,
                    'cx': cx,
                    'cy': cy
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
                    cameras[stem] = {
                        'sensor': sensors[sensor_id],
                        'transform': transform,
                        'original_label': label
                    }
            elem.clear()

    return cameras, T_comp


def intersect_fallback_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    target_z: float
) -> Optional[np.ndarray]:
    """
    Fallback mathematical intersection with a horizontal elevation plane.
    Used if a ray misses the 3D surface mesh (e.g. near the boundary of the photogrammetry model).

    Args:
        origin (np.ndarray): 3D ray starting position [x, y, z].
        direction (np.ndarray): Normalized 3D ray direction vector [dx, dy, dz].
        target_z (float): Average ground elevation altitude.

    Returns:
        Optional[np.ndarray]: 3D coordinates of intersection, or None if parallel/facing away.
    """
    if abs(direction[2]) < 1e-6:
        return None  # Ray is horizontal, never intersects horizontal plane
    t = (target_z - origin[2]) / direction[2]
    if t < 0:
        return None  # Intersection is behind the camera
    return origin + direction * t


def cast_single_ray(
    nx: float,
    ny: float,
    cam: Dict[str, Any],
    T_comp: np.ndarray,
    transformer_ecef_utm: Transformer,
    scene: o3d.t.geometry.RaycastingScene,
    utm_offset: np.ndarray,
    avg_z: float
) -> Tuple[Optional[np.ndarray], str]:
    """
    Projects a normalized 2D image coordinate (0.0 to 1.0) through the camera optics
    and casts a ray onto the Open3D surface mesh.

    Math Outline:
    -------------
    1. 2D image pixels -> normalized camera coordinates (pinhole camera inversion).
    2. Camera coordinates -> local photogrammetry 3D space via camera transform matrix.
    3. Local 3D space -> Earth ECEF coordinates via component transform matrix `T_comp`.
    4. ECEF -> projected metric UTM coordinates via PyProj.
    5. Raycast in Open3D scene to find triangle intersection distance `dist`.
    6. If no mesh hit, fallback to horizontal plane at average ground elevation.

    Args:
        nx (float): Normalized horizontal image coordinate (0.0=left, 1.0=right).
        ny (float): Normalized vertical image coordinate (0.0=top, 1.0=bottom).
        cam (Dict[str, Any]): Camera sensor calibration and pose dictionary.
        T_comp (np.ndarray): 4x4 component transform matrix.
        transformer_ecef_utm (Transformer): PyProj transformer from ECEF to UTM.
        scene (o3d.t.geometry.RaycastingScene): Open3D BVH raycaster scene.
        utm_offset (np.ndarray): Translation offset applied to mesh for numerical precision.
        avg_z (float): Average ground elevation for fallback.

    Returns:
        Tuple[Optional[np.ndarray], str]:
            - [x, y, z] intersection coordinate in UTM space.
            - Status string: "Mesh Hit" or "Plane Fallback".
    """
    w, h = cam['sensor']['w'], cam['sensor']['h']
    px = nx * w
    py = ny * h

    # Inverse pinhole camera model
    xc = (px - (w / 2.0 + cam['sensor']['cx'])) / cam['sensor']['f']
    yc = (py - (h / 2.0 + cam['sensor']['cy'])) / cam['sensor']['f']

    pt_cam = np.array([xc, yc, 1.0, 1.0])
    orig_cam = np.array([0.0, 0.0, 0.0, 1.0])

    # Transform into photogrammetry local space
    pt_local = cam['transform'] @ pt_cam
    orig_local = cam['transform'] @ orig_cam

    # Transform into ECEF coordinates
    pt_ecef = T_comp @ pt_local
    orig_ecef = T_comp @ orig_local

    # Transform into UTM metric coordinates
    orig_utm = np.array(transformer_ecef_utm.transform(orig_ecef[0], orig_ecef[1], orig_ecef[2]))
    pt_utm = np.array(transformer_ecef_utm.transform(pt_ecef[0], pt_ecef[1], pt_ecef[2]))

    # Compute normalized ray direction in UTM space
    dir_utm = pt_utm - orig_utm
    dir_utm /= np.linalg.norm(dir_utm)

    # Offset ray origin to match the mesh's centered coordinate system (prevents float32 precision loss)
    origin_l = orig_utm - utm_offset
    ray = np.concatenate([origin_l, dir_utm]).astype(np.float32)

    # Query the 3D mesh via Open3D accelerated raycaster
    ans = scene.cast_rays(o3d.core.Tensor([ray]))
    dist = ans['t_hit'].numpy()[0]

    if not np.isinf(dist):
        return (origin_l + dir_utm * dist) + utm_offset, "Mesh Hit"
    else:
        return intersect_fallback_plane(orig_utm, dir_utm, avg_z), "Plane Fallback"


def export_outputs(image_footprints: Dict[str, Dict[str, Any]], export_dir: str) -> None:
    """
    Exports computed footprints into standard GIS formats:
    - GeoJSON: Closed multi-coordinate polygons with properties (`status`, `campaign`).
    - CSV: Tabular table of the 4 corner edge coordinates (top-left, top-right, bottom-right, bottom-left).

    Args:
        image_footprints (Dict[str, Dict[str, Any]]): Mapping of slice name to footprint data.
        export_dir (str): Directory where output files will be created.
    """
    os.makedirs(export_dir, exist_ok=True)

    geojson_features = []
    for img_name, data in image_footprints.items():
        coords_wgs84 = data['coords_wgs84']
        # Polygons in GeoJSON must be explicitly closed (first point repeated at end)
        closed_poly = coords_wgs84 + [coords_wgs84[0]]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[pt[0], pt[1]] for pt in closed_poly]]
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
    with open(geojson_path, 'w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": geojson_features}, f, indent=2)
    print(f"[Success] GeoJSON footprints saved to: {geojson_path}")

    csv_path = os.path.join(export_dir, "image_edge_coordinates.csv")
    csv_headers = [
        "image_name", "campaign",
        "tl_lon", "tl_lat", "tl_alt",
        "tr_lon", "tr_lat", "tr_alt",
        "br_lon", "br_lat", "br_alt",
        "bl_lon", "bl_lat", "bl_alt"
    ]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
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
    print(f"[Success] CSV Edge coordinates saved to: {csv_path}")


def run_multi_campaign_pipeline(
    campaign_dict: Dict[str, Dict[str, str]],
    utm_epsg: int,
    slice_metadata_path: str,
    export_dir: str
) -> None:
    """
    Executes 3D raycasting across multiple aerial survey campaigns (e.g. 2023, 2024).

    Workflow:
    ---------
    1. Loads `slice_metadata.json` generated during slicing.
    2. Initializes PyProj transformers (ECEF -> UTM and UTM -> WGS84).
    3. For each survey campaign:
       a. Parses Metashape XML for camera calibration and 3D positions.
       b. Loads the 3D surface mesh (`.obj`) into an Open3D RaycastingScene.
       c. Casts rays through the 4 corners of every slice originating from that campaign.
       d. Transforms 3D ground hit points into WGS84 latitude/longitude.
    4. Exports `image_footprints.geojson` and `image_edge_coordinates.csv`.

    Args:
        campaign_dict (Dict[str, Dict[str, str]]): Campaign definitions with `xml_path` and `mesh_path`.
        utm_epsg (int): EPSG zone number for metric projection (e.g. 32633 for UTM 33N).
        slice_metadata_path (str): Path to `slice_metadata.json`.
        export_dir (str): Output destination directory.
    """
    with open(slice_metadata_path, 'r', encoding='utf-8') as f:
        slice_metadata = json.load(f)

    # Coordinate transformation pipelines
    transformer_ecef_utm = Transformer.from_crs("EPSG:4978", f"EPSG:{utm_epsg}", always_xy=True)
    transformer_utm_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    image_footprints: Dict[str, Dict[str, Any]] = {}
    total_slices = len(slice_metadata)

    for campaign_name, paths in campaign_dict.items():
        print(f"\n--- Processing Campaign: {campaign_name} ---")
        cameras, T_comp = parse_metashape_xml(paths["xml_path"])

        print(f"Loading Mesh for {campaign_name} and setting up Raycaster...")
        mesh_legacy = o3d.io.read_triangle_mesh(paths["mesh_path"])

        # Center mesh around origin to maintain numerical precision in single-precision GPU/CPU raycasting
        utm_offset = mesh_legacy.get_axis_aligned_bounding_box().get_center()
        avg_z = utm_offset[2]
        mesh_legacy.translate(-utm_offset)

        # Build tensor-accelerated RaycastingScene
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

            # Corner points ordered: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            slice_corner_normalized = [
                (xs / w, ys / h),
                (xe / w, ys / h),
                (xe / w, ye / h),
                (xs / w, ye / h)
            ]

            poly_coords = []
            hit_statuses: Set[str] = set()
            valid_camera = True

            for cx, cy in slice_corner_normalized:
                hit_w, status = cast_single_ray(
                    cx, cy, cam, T_comp, transformer_ecef_utm, scene, utm_offset, avg_z
                )

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

        print(f"[{campaign_name}] Mapped {hits_this_campaign} slices.")

    print(f"\nFinished processing all campaigns: {len(image_footprints)}/{total_slices} total slices mapped.")
    export_outputs(image_footprints, export_dir)


def parse_args():
    """Parses command-line arguments for 3D raycasting."""
    import argparse
    import config

    parser = argparse.ArgumentParser(
        description="Raycast 2D image slice corners onto 3D photogrammetry surface meshes to compute real-world GPS footprints."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)."
    )
    parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=str(config.RAYCASTER_METADATA_PATH),
        help=f"Path to slice_metadata.json produced by 1_slicer.py (default from config.yaml: {config.RAYCASTER_METADATA_PATH})."
    )
    parser.add_argument(
        "--export-dir", "-o",
        type=str,
        default=str(config.RAYCASTER_EXPORT_DIR),
        help=f"Destination directory for image_footprints.geojson and image_edge_coordinates.csv (default from config.yaml: {config.RAYCASTER_EXPORT_DIR})."
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=config.RAYCASTER_UTM_EPSG,
        help=f"EPSG code for local metric projection (default from config.yaml: {config.RAYCASTER_UTM_EPSG})."
    )
    parser.add_argument(
        "--campaign-config",
        type=str,
        default=None,
        help="Optional path to a JSON file defining campaigns with 'xml_path' and 'mesh_path' keys."
    )
    parser.add_argument(
        "--xml-2023",
        type=str,
        default=str(config.CAMPAIGN_2023_XML),
        help=f"Path to Metashape camera XML for 2023 flight campaign (default from config.yaml: {config.CAMPAIGN_2023_XML})."
    )
    parser.add_argument(
        "--mesh-2023",
        type=str,
        default=str(config.CAMPAIGN_2023_MESH),
        help=f"Path to 3D surface mesh (.obj) for 2023 flight campaign (default from config.yaml: {config.CAMPAIGN_2023_MESH})."
    )
    parser.add_argument(
        "--xml-2024",
        type=str,
        default=str(config.CAMPAIGN_2024_XML),
        help=f"Path to Metashape camera XML for 2024 flight campaign (default from config.yaml: {config.CAMPAIGN_2024_XML})."
    )
    parser.add_argument(
        "--mesh-2024",
        type=str,
        default=str(config.CAMPAIGN_2024_MESH),
        help=f"Path to 3D surface mesh (.obj) for 2024 flight campaign (default from config.yaml: {config.CAMPAIGN_2024_MESH})."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.campaign_config and os.path.exists(args.campaign_config):
        with open(args.campaign_config, 'r', encoding='utf-8') as f:
            campaigns = json.load(f)
    elif config.CAMPAIGNS:
        campaigns = config.CAMPAIGNS
    else:
        campaigns = {
            "2023": {
                "xml_path": args.xml_2023,
                "mesh_path": args.mesh_2023
            },
            "2024": {
                "xml_path": args.xml_2024,
                "mesh_path": args.mesh_2024
            }
        }

    run_multi_campaign_pipeline(
        campaign_dict=campaigns,
        utm_epsg=args.utm_epsg,
        slice_metadata_path=args.metadata,
        export_dir=args.export_dir
    )

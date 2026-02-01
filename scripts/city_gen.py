import os
import math
import random

import cv2
import numpy as np
from PIL import Image


PALETTE_ORDER = ["avenue", "park", "water", "main_street", "side_street", "building"]
PALETTE_RGB = {
    "avenue": (253, 221, 1),
    "park": (11, 79, 17),
    "water": (19, 32, 90),
    "main_street": (254, 254, 254),
    "side_street": (155, 156, 155),
    "building": (0, 0, 0),
}
IDX = {name: i for i, name in enumerate(PALETTE_ORDER)}

ROLE_COLORS = {
    "vertiport": (1.0, 0.0, 0.0, 1.0),
    "supplier": (0.0, 1.0, 0.0, 1.0),
    "client": (1.0, 0.5, 0.0, 1.0),
    "charging": (0.0, 0.3, 1.0, 1.0),
    "building": (0.6, 0.6, 0.6, 1.0),
}

BUILDING_H_MIN = 8.0
BUILDING_H_MAX = 35.0


def next_heightmap_size(n):
    """Return the next 2^k+1 size used by Gazebo heightmaps."""
    k = 0
    while (2 ** k + 1) < n:
        k += 1
    return 2 ** k + 1


def pad_to_square_pow2p1(gray_u8):
    """Pad a grayscale image to a square with size (2^k+1)."""
    h, w = gray_u8.shape[:2]
    n = max(h, w)
    s = next_heightmap_size(n)
    out = np.zeros((s, s), dtype=np.uint8)
    out[:h, :w] = gray_u8
    return out


def segment_by_palette(img_bgr):
    """Assign each pixel to the nearest palette color and return (labels, min_dist)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
    colors = np.array([PALETTE_RGB[name] for name in PALETTE_ORDER], dtype=np.int32)
    diff = img_rgb[:, :, None, :] - colors[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=3, dtype=np.int32)
    label_map = np.argmin(dist2, axis=2).astype(np.uint8)
    min_dist = np.sqrt(np.min(dist2, axis=2).astype(np.float32))
    return label_map, min_dist


def build_masks(label_map, min_dist, building_tol=12.0, road_tol=18.0):
    """Build building and road masks from palette labels and distance thresholds."""
    building_idx = IDX["building"]
    road_indices = [IDX["avenue"], IDX["main_street"], IDX["side_street"]]
    mask_building = ((label_map == building_idx) & (min_dist <= building_tol)).astype(np.uint8) * 255
    mask_roads = (np.isin(label_map, road_indices) & (min_dist <= road_tol)).astype(np.uint8) * 255
    return mask_building, mask_roads


def compute_integral_image(mask_binary):
    """Compute integral image for fast rectangle sum queries."""
    return cv2.integral(mask_binary)


def rectangle_sum(integral_img, x, y, w, h):
    """Return sum over rectangle [x:x+w, y:y+h] using an integral image."""
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return int(integral_img[y2, x2] - integral_img[y1, x2] - integral_img[y2, x1] + integral_img[y1, x1])


def extract_building_boxes(
    building_mask_255=None,
    roads_mask_255=None,
    *,
    img_bgr=None,
    building_tol=12.0,
    road_tol=18.0,
    min_area=800,
    min_side=20,
    max_side=300,
    max_aspect_ratio=3.5,
    max_road_fraction=0.02,
    debug_dir=None,
    sort_boxes=True,
):
    """Extract building bounding boxes from palette masks with geometric filtering."""
    if img_bgr is None:
        if isinstance(building_mask_255, np.ndarray) and building_mask_255.ndim == 3 and roads_mask_255 is None:
            img_bgr = building_mask_255
            building_mask_255 = None

    if img_bgr is not None:
        if img_bgr.ndim != 3 or img_bgr.shape[2] not in (3, 4):
            raise ValueError("img_bgr must be BGR/RGBA (H,W,3) or (H,W,4).")
        if img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        label_map, min_dist = segment_by_palette(img_bgr)
        building_mask_255, roads_mask_255 = build_masks(
            label_map, min_dist, building_tol=building_tol, road_tol=road_tol
        )

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "mask_building.png"), building_mask_255)
            cv2.imwrite(os.path.join(debug_dir, "mask_roads.png"), roads_mask_255)

    if building_mask_255 is None or roads_mask_255 is None:
        raise ValueError("Provide (building_mask_255, roads_mask_255) OR img_bgr.")

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(building_mask_255, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    roads_binary = (roads_mask_255 > 0).astype(np.uint8)
    roads_integral = compute_integral_image(roads_binary)

    valid = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)

        if area < min_area:
            continue
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue

        ar = max(w / h, h / w)
        if ar > max_aspect_ratio:
            continue

        road_pixels = rectangle_sum(roads_integral, x, y, w, h)
        if (road_pixels / float(w * h)) > max_road_fraction:
            continue

        valid.append((area, int(x), int(y), int(w), int(h)))

    if sort_boxes:
        valid.sort(key=lambda b: (b[2], b[1], -b[0]))

    return valid


def write_ogre_material(out_dir, texture_filename="finalmap.png"):
    """Write OGRE material script for a ground texture."""
    scripts_dir = os.path.join(out_dir, "materials", "scripts")
    textures_dir = os.path.join(out_dir, "materials", "textures")
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(textures_dir, exist_ok=True)

    material_path = os.path.join(scripts_dir, "utm.material")
    material_txt = f"""material UTM/FinalMap
{{
  technique
  {{
    pass
    {{
      lighting on
      ambient 1 1 1 1
      diffuse 1 1 1 1
      specular 0 0 0 1

      texture_unit
      {{
        texture {texture_filename}
        scale 1 1
      }}
    }}
  }}
}}
"""
    with open(material_path, "w", encoding="utf-8") as f:
        f.write(material_txt)


def write_stub_model_config(model_dir, model_name="utm_materials_stub"):
    """Create stub model.config and model.sdf to avoid Gazebo folder scan warnings."""
    os.makedirs(model_dir, exist_ok=True)
    cfg_path = os.path.join(model_dir, "model.config")
    sdf_path = os.path.join(model_dir, "model.sdf")

    if not os.path.exists(cfg_path):
        cfg = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>auto</name><email>auto@local</email></author>
  <description>Stub model to silence Gazebo folder scan warnings.</description>
</model>
"""
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(cfg)

    if not os.path.exists(sdf_path):
        sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link"/>
  </model>
</sdf>
"""
        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(sdf)


def box_center_px(box):
    """Return bbox center in pixel space."""
    _, x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def point_in_rect(px, py, rect_xywh):
    """Check if a point lies inside an axis-aligned rectangle."""
    x, y, w, h = rect_xywh
    return (x <= px <= x + w) and (y <= py <= y + h)


def point_in_any_box(pt, boxes, ignore_indices=None):
    """Check if a point lies inside any building bbox."""
    if ignore_indices is None:
        ignore_indices = set()
    px, py = pt
    for i, b in enumerate(boxes):
        if i in ignore_indices:
            continue
        _, x, y, w, h = b
        if point_in_rect(px, py, (x, y, w, h)):
            return True
    return False


def seg_hits_rect(pt1, pt2, rect_xywh):
    """Check if a segment intersects a rectangle using OpenCV clipLine."""
    x, y, w, h = rect_xywh
    rect = (int(x), int(y), int(w), int(h))
    ok, _, _ = cv2.clipLine(rect, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])))
    return bool(ok)


def seg_hits_any_box(pt1, pt2, boxes, ignore_indices=None):
    """Check if a segment intersects any building bbox."""
    if ignore_indices is None:
        ignore_indices = set()
    for i, b in enumerate(boxes):
        if i in ignore_indices:
            continue
        _, x, y, w, h = b
        if seg_hits_rect(pt1, pt2, (x, y, w, h)):
            return True
    return False


def px_to_world(px, py, W_px, H_px, res):
    """Convert pixel coordinates to world coordinates with map center as origin."""
    wx = (px - W_px / 2.0) * res
    wy = (H_px / 2.0 - py) * res
    return (wx, wy)


def farthest_point_sampling(centers, k, seed=42):
    """Select k indices using farthest point sampling."""
    if k <= 0:
        return []
    rnd = random.Random(seed)
    idx0 = rnd.randrange(len(centers))
    chosen = [idx0]
    dmin = np.full(len(centers), np.inf, dtype=np.float64)

    c0 = np.array(centers[idx0], dtype=np.float64)
    for i, c in enumerate(centers):
        dmin[i] = np.sum((np.array(c, dtype=np.float64) - c0) ** 2)

    while len(chosen) < k:
        j = int(np.argmax(dmin))
        chosen.append(j)
        cj = np.array(centers[j], dtype=np.float64)
        for i, c in enumerate(centers):
            dij = np.sum((np.array(c, dtype=np.float64) - cj) ** 2)
            if dij < dmin[i]:
                dmin[i] = dij
    return chosen


def sample_building_heights(n, seed, random_height=True, height_fixed=20.0):
    """Sample building heights either fixed or uniform random in [BUILDING_H_MIN, BUILDING_H_MAX]."""
    rnd = random.Random(seed + 12345)
    if not random_height:
        return [float(height_fixed)] * n
    return [rnd.uniform(BUILDING_H_MIN, BUILDING_H_MAX) for _ in range(n)]


def distribute_vehicles_over_vertiports(num_vehicles, vertiports, z_vehicle_above_vertiport):
    """Distribute vehicles among vertiports."""
    V = len(vertiports)
    if V <= 0 or num_vehicles <= 0:
        return []

    base = num_vehicles // V
    rem = num_vehicles % V
    vehicles = []
    vid = 0

    for i, vp in enumerate(vertiports):
        k = base + (1 if i < rem else 0)
        if k <= 0:
            continue
        for j in range(k):
            if k == 1:
                dx, dy = 0.0, 0.0
            else:
                ang = 2.0 * math.pi * (j / k)
                r = 1.5
                dx = r * math.cos(ang)
                dy = r * math.sin(ang)

            vehicles.append({
                "id": f"VEHICLE_{vid:03d}",
                "x": vp["x"] + dx,
                "y": vp["y"] + dy,
                "z": float(vp["z"]) + float(z_vehicle_above_vertiport),
            })
            vid += 1

    return vehicles


def make_world_sdf(
    W_px,
    H_px,
    boxes,
    roles_by_index,
    building_heights,
    resolution_m_per_px=0.2,
    seed=42,
    logical_markers=None,
    vehicle_markers=None,
):
    """Generate a Gazebo SDF world with buildings, overlays, and optional markers."""
    width_m = W_px * resolution_m_per_px
    height_m = H_px * resolution_m_per_px
    hm_uri = "file://heightmap.png"

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<sdf version="1.7">')
    lines.append('  <world name="utm_world">')
    lines.append('    <gravity>0 0 -9.81</gravity>')
    lines.append('    <physics name="default_physics" type="ode">')
    lines.append('      <max_step_size>0.002</max_step_size>')
    lines.append('      <real_time_factor>1.0</real_time_factor>')
    lines.append('      <real_time_update_rate>500</real_time_update_rate>')
    lines.append('    </physics>')

    lines.append('    <scene>')
    lines.append('      <ambient>0.8 0.8 0.8 1</ambient>')
    lines.append('      <background>0.9 0.9 0.9 1</background>')
    lines.append('    </scene>')

    lines.append('    <model name="map_ground">')
    lines.append('      <static>true</static>')
    lines.append('      <pose>0 0 0 0 0 0</pose>')
    lines.append('      <link name="link">')

    lines.append('        <collision name="collision">')
    lines.append('          <geometry>')
    lines.append('            <heightmap>')
    lines.append(f'              <uri>{hm_uri}</uri>')
    lines.append(f'              <size>{width_m:.6f} {height_m:.6f} 1.0</size>')
    lines.append('              <pos>0 0 0</pos>')
    lines.append('            </heightmap>')
    lines.append('          </geometry>')
    lines.append('        </collision>')

    lines.append('        <visual name="visual">')
    lines.append('          <pose>0 0 0.001 0 0 0</pose>')
    lines.append('          <geometry>')
    lines.append('            <plane>')
    lines.append('              <normal>0 0 1</normal>')
    lines.append(f'              <size>{width_m:.6f} {height_m:.6f}</size>')
    lines.append('            </plane>')
    lines.append('          </geometry>')
    lines.append('          <material>')
    lines.append('            <script>')
    lines.append('              <uri>file://materials/scripts</uri>')
    lines.append('              <uri>file://materials/textures</uri>')
    lines.append('              <name>UTM/FinalMap</name>')
    lines.append('            </script>')
    lines.append('          </material>')
    lines.append('        </visual>')

    lines.append('      </link>')
    lines.append('    </model>')

    overlay_xy_scale = 1.03
    overlay_z_extra = 0.20

    for i, (_area, x, y, w, h) in enumerate(boxes):
        cx = x + w / 2.0
        cy = y + h / 2.0
        wx = (cx - W_px / 2.0) * resolution_m_per_px
        wy = (H_px / 2.0 - cy) * resolution_m_per_px

        sx = w * resolution_m_per_px
        sy = h * resolution_m_per_px
        hz = float(building_heights[i])
        wz = hz / 2.0

        br, bg, bb, ba = ROLE_COLORS["building"]
        base_name = f"building_{i:04d}"

        lines.append(f'    <model name="{base_name}">')
        lines.append('      <static>true</static>')
        lines.append(f'      <pose>{wx:.3f} {wy:.3f} {wz:.3f} 0 0 0</pose>')
        lines.append('      <link name="link">')
        lines.append('        <collision name="collision">')
        lines.append('          <geometry>')
        lines.append(f'            <box><size>{sx:.3f} {sy:.3f} {hz:.3f}</size></box>')
        lines.append('          </geometry>')
        lines.append('        </collision>')
        lines.append('        <visual name="visual">')
        lines.append('          <geometry>')
        lines.append(f'            <box><size>{sx:.3f} {sy:.3f} {hz:.3f}</size></box>')
        lines.append('          </geometry>')
        lines.append('          <material>')
        lines.append(f'            <ambient>{br:.3f} {bg:.3f} {bb:.3f} {ba:.3f}</ambient>')
        lines.append(f'            <diffuse>{br:.3f} {bg:.3f} {bb:.3f} {ba:.3f}</diffuse>')
        lines.append(f'            <emissive>{br:.3f} {bg:.3f} {bb:.3f} {ba:.3f}</emissive>')
        lines.append('          </material>')
        lines.append('        </visual>')
        lines.append('      </link>')
        lines.append('    </model>')

        if i in roles_by_index:
            role = roles_by_index[i]
            rr, rg, rb, ra = ROLE_COLORS.get(role, ROLE_COLORS["building"])
            overlay_name = f"{role}_{i:04d}"

            sx2 = sx * overlay_xy_scale
            sy2 = sy * overlay_xy_scale
            hz2 = hz + overlay_z_extra
            wz2 = hz2 / 2.0

            lines.append(f'    <model name="{overlay_name}">')
            lines.append('      <static>true</static>')
            lines.append(f'      <pose>{wx:.3f} {wy:.3f} {wz2:.3f} 0 0 0</pose>')
            lines.append('      <link name="link">')
            lines.append('        <visual name="visual">')
            lines.append('          <geometry>')
            lines.append(f'            <box><size>{sx2:.3f} {sy2:.3f} {hz2:.3f}</size></box>')
            lines.append('          </geometry>')
            lines.append('          <material>')
            lines.append(f'            <ambient>{rr:.3f} {rg:.3f} {rb:.3f} {ra:.3f}</ambient>')
            lines.append(f'            <diffuse>{rr:.3f} {rg:.3f} {rb:.3f} {ra:.3f}</diffuse>')
            lines.append(f'            <emissive>{rr:.3f} {rg:.3f} {rb:.3f} {ra:.3f}</emissive>')
            lines.append('          </material>')
            lines.append('        </visual>')
            lines.append('      </link>')
            lines.append('    </model>')

    def _sphere_model(model_name, x, y, z, rgba):
        """Add a simple sphere marker model."""
        rr, gg, bb, aa = rgba
        lines.append(f'    <model name="{model_name}">')
        lines.append('      <static>true</static>')
        lines.append(f'      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>')
        lines.append('      <link name="link">')
        lines.append('        <collision name="collision">')
        lines.append('          <geometry><sphere><radius>0.4</radius></sphere></geometry>')
        lines.append('        </collision>')
        lines.append('        <visual name="visual">')
        lines.append('          <geometry><sphere><radius>0.4</radius></sphere></geometry>')
        lines.append('          <material>')
        lines.append(f'            <ambient>{rr:.3f} {gg:.3f} {bb:.3f} {aa:.3f}</ambient>')
        lines.append(f'            <diffuse>{rr:.3f} {gg:.3f} {bb:.3f} {aa:.3f}</diffuse>')
        lines.append('          </material>')
        lines.append('        </visual>')
        lines.append('      </link>')
        lines.append('    </model>')

    if logical_markers:
        for name, x, y, z in logical_markers:
            _sphere_model(name, x, y, z, (1.0, 1.0, 0.0, 1.0))

    if vehicle_markers:
        for name, x, y, z in vehicle_markers:
            _sphere_model(name, x, y, z, (1.0, 0.0, 1.0, 1.0))

    lines.append('  </world>')
    lines.append('</sdf>')
    return "\n".join(lines)

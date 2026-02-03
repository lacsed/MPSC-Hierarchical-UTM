import os
import math
import random

import cv2
import numpy as np


PALETTE_ORDER = ["avenue", "park", "water", "main_street", "side_street", "building"]
PALETTE_RGB = {
    "avenue": (253, 221, 1),
    "park": (11, 79, 17),
    "water": (19, 32, 90),
    "main_street": (254, 254, 254),
    "side_street": (155, 156, 155),
    "building": (0, 0, 0),
}
IDX = dict((name, i) for i, name in enumerate(PALETTE_ORDER))

ROLE_COLORS = {
    "vertiport": (1.0, 0.0, 0.0, 1.0),
    "supplier": (0.0, 1.0, 0.0, 1.0),
    "client": (1.0, 0.5, 0.0, 1.0),
    "charging": (0.0, 0.3, 1.0, 1.0),
    "building": (0.05, 0.05, 0.05, 1.0),
}

BUILDING_H_MIN = 8.0
BUILDING_H_MAX = 35.0
SPECIAL_ABOVE_ROOF_Z = 0.1 

def find_box_index_for_world_xy(x, y, boxes, W_px, H_px, resolution_m_per_px):
    """
    Retorna o índice 'bi' do prédio (box) que contém o ponto (x,y) em coordenadas do mundo.
    Se não cair em nenhum prédio, retorna None.
    """
    px, py = world_to_px(float(x), float(y), W_px, H_px, resolution_m_per_px)
    for j, (_a, bx, by, bw, bh) in enumerate(boxes or []):
        if point_in_rect(px, py, (bx, by, bw, bh)):
            return j
    return None


def roof_spawn_z_for_world_xy(
    x, y,
    boxes,
    building_heights,
    W_px, H_px,
    resolution_m_per_px,
    ground_z=0.0,
    above_roof=SPECIAL_ABOVE_ROOF_Z,
    min_vehicle_alt=1.0,
):
    """
    Se (x,y) estiver sobre um prédio: z = ground_z + altura_do_prédio + above_roof
    Caso contrário: z = ground_z + min_vehicle_alt
    """
    bi = find_box_index_for_world_xy(x, y, boxes, W_px, H_px, resolution_m_per_px)
    if bi is not None:
        return float(ground_z) + float(building_heights[bi]) + float(above_roof), bi
    return float(ground_z) + float(min_vehicle_alt), None




def seg_hits_rect(pt1, pt2, rect_xywh):
    x, y, w, h = rect_xywh
    rect = (int(x), int(y), int(w), int(h))
    ok, _, _ = cv2.clipLine(rect, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])))
    return bool(ok)


def seg_hits_any_box(pt1, pt2, boxes, ignore_indices=None):
    if ignore_indices is None:
        ignore_indices = set()
    for i, b in enumerate(boxes):
        if i in ignore_indices:
            continue
        _, x, y, w, h = b
        if seg_hits_rect(pt1, pt2, (x, y, w, h)):
            return True
    return False


def next_heightmap_size(n):
    k = 0
    while (2 ** k + 1) < n:
        k += 1
    return 2 ** k + 1


def pad_to_square_pow2p1(gray_u8):
    h, w = gray_u8.shape[:2]
    n = max(h, w)
    s = next_heightmap_size(n)
    out = np.zeros((s, s), dtype=np.uint8)
    out[:h, :w] = gray_u8
    return out


def segment_by_palette(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
    colors = np.array([PALETTE_RGB[name] for name in PALETTE_ORDER], dtype=np.int32)
    diff = img_rgb[:, :, None, :] - colors[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=3, dtype=np.int32)
    label_map = np.argmin(dist2, axis=2).astype(np.uint8)
    min_dist = np.sqrt(np.min(dist2, axis=2).astype(np.float32))
    return label_map, min_dist


def build_masks(label_map, min_dist, building_tol=12.0, road_tol=18.0):
    building_idx = IDX["building"]
    road_indices = [IDX["avenue"], IDX["main_street"], IDX["side_street"]]
    mask_building = ((label_map == building_idx) & (min_dist <= building_tol)).astype(np.uint8) * 255
    mask_roads = (np.isin(label_map, road_indices) & (min_dist <= road_tol)).astype(np.uint8) * 255
    return mask_building, mask_roads


def build_park_mask(label_map, min_dist, park_tol=18.0):
    park_idx = IDX["park"]
    mask_park = ((label_map == park_idx) & (min_dist <= park_tol)).astype(np.uint8) * 255
    return mask_park


def largest_component_centroid_px(mask_255, min_area=2000):
    if mask_255 is None:
        return None

    bin_u8 = (mask_255 > 0).astype(np.uint8) * 255
    cnts = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    if not contours:
        return None

    best = None
    best_area = 0.0
    for c in contours:
        a = float(cv2.contourArea(c))
        if a >= float(min_area) and a > best_area:
            best_area = a
            best = c

    if best is None:
        return None

    m = cv2.moments(best)
    if abs(m["m00"]) < 1e-9:
        return None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (float(cx), float(cy))


def sample_points_in_mask_px(mask_255, n, seed=42, min_dist_px=10, max_tries=200000):
    if mask_255 is None or n <= 0:
        return []

    rnd = random.Random(seed)
    H, W = mask_255.shape[:2]

    pts = []
    min_d2 = float(min_dist_px * min_dist_px)
    tries = 0

    while len(pts) < n and tries < max_tries:
        x = rnd.randrange(W)
        y = rnd.randrange(H)
        tries += 1

        if mask_255[y, x] == 0:
            continue

        ok = True
        for (px, py) in pts:
            dx = x - px
            dy = y - py
            if (dx * dx + dy * dy) < min_d2:
                ok = False
                break

        if ok:
            pts.append((float(x), float(y)))

    return pts


def sample_points_in_mask_rect_px(mask_255, x0, y0, x1, y1, n, rnd, min_dist_px=6, max_tries=20000):
    if mask_255 is None or n <= 0:
        return []

    x0 = int(max(0, x0))
    y0 = int(max(0, y0))
    x1 = int(min(mask_255.shape[1], x1))
    y1 = int(min(mask_255.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return []

    sub = mask_255[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return []

    pts = []
    min_d2 = float(min_dist_px * min_dist_px)
    tries = 0

    while len(pts) < n and tries < max_tries:
        k = rnd.randrange(len(xs))
        x = x0 + int(xs[k])
        y = y0 + int(ys[k])
        tries += 1

        ok = True
        for (px, py) in pts:
            dx = x - px
            dy = y - py
            if (dx * dx + dy * dy) < min_d2:
                ok = False
                break

        if ok:
            pts.append((float(x), float(y)))

    if len(pts) < n:
        tries2 = 0
        while len(pts) < n and tries2 < max_tries:
            k = rnd.randrange(len(xs))
            x = x0 + int(xs[k])
            y = y0 + int(ys[k])
            tries2 += 1

            dup = False
            for (px, py) in pts:
                if int(px) == int(x) and int(py) == int(y):
                    dup = True
                    break
            if not dup:
                pts.append((float(x), float(y)))

    return pts


def plan_park_models(
    park_mask_255,
    W_px,
    H_px,
    resolution_m_per_px,
    seed=42,
    trees_per_square=5,
    square_px=24,
    min_fill=0.55,
):
    if park_mask_255 is None:
        return None

    bin_u8 = (park_mask_255 > 0).astype(np.uint8)
    if int(np.sum(bin_u8)) == 0:
        return None

    rnd = random.Random(seed + 555)

    ys, xs = np.where(bin_u8 > 0)
    gazebo = None
    if len(xs) > 0:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        gx, gy = px_to_world(cx, cy, W_px, H_px, resolution_m_per_px)
        gazebo = {"name": "PARK_GAZEBO", "x": float(gx), "y": float(gy), "z": 0.0, "yaw": 0.0}

    trees = []
    tid = 0
    step = int(max(4, square_px))

    for y0 in range(0, H_px, step):
        for x0 in range(0, W_px, step):
            x1 = min(W_px, x0 + step)
            y1 = min(H_px, y0 + step)

            cell = bin_u8[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            fill = float(np.count_nonzero(cell)) / float(cell.size)
            if fill < float(min_fill):
                continue

            pts = sample_points_in_mask_rect_px(
                park_mask_255,
                x0, y0, x1, y1,
                n=int(trees_per_square),
                rnd=rnd,
                min_dist_px=max(3, int(step * 0.25)),
                max_tries=20000,
            )

            for (tx, ty) in pts:
                xw, yw = px_to_world(tx, ty, W_px, H_px, resolution_m_per_px)
                yaw = rnd.uniform(-math.pi, math.pi)
                trees.append({
                    "name": "PINE_TREE_%05d" % tid,
                    "x": float(xw),
                    "y": float(yw),
                    "z": 0.0,
                    "yaw": float(yaw),
                })
                tid += 1

    return {"gazebo": gazebo, "trees": trees}


def compute_integral_image(mask_binary):
    return cv2.integral(mask_binary)


def rectangle_sum(integral_img, x, y, w, h):
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return int(integral_img[y2, x2] - integral_img[y1, x2] - integral_img[y2, x1] + integral_img[y1, x1])


def extract_building_boxes(
    building_mask_255=None,
    roads_mask_255=None,
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
        building_mask_255, roads_mask_255 = build_masks(label_map, min_dist, building_tol=building_tol, road_tol=road_tol)

        if debug_dir:
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
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

        ar = max(w / float(h), h / float(w))
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
    scripts_dir = os.path.join(out_dir, "materials", "scripts")
    textures_dir = os.path.join(out_dir, "materials", "textures")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    if not os.path.exists(textures_dir):
        os.makedirs(textures_dir)

    material_path = os.path.join(scripts_dir, "utm.material")
    material_txt = """material UTM/FinalMap
{
  technique
  {
    pass
    {
      lighting on
      ambient 1 1 1 1
      diffuse 1 1 1 1
      specular 0 0 0 1

      texture_unit
      {
        texture %s
        scale 1 1
      }
    }
  }
}
""" % (texture_filename)

    with open(material_path, "w", encoding="utf-8") as f:
        f.write(material_txt)


def write_stub_model_config(model_dir, model_name="utm_materials_stub"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cfg_path = os.path.join(model_dir, "model.config")
    sdf_path = os.path.join(model_dir, "model.sdf")

    if not os.path.exists(cfg_path):
        cfg = """<?xml version="1.0"?>
<model>
  <name>%s</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>auto</name><email>auto@local</email></author>
  <description>Stub model to silence Gazebo folder scan warnings.</description>
</model>
""" % (model_name)
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(cfg)

    if not os.path.exists(sdf_path):
        sdf = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="%s">
    <static>true</static>
    <link name="link"/>
  </model>
</sdf>
""" % (model_name)
        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(sdf)


def box_center_px(box):
    _, x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def point_in_rect(px, py, rect_xywh):
    x, y, w, h = rect_xywh
    return (x <= px <= x + w) and (y <= py <= y + h)


def point_in_any_box(pt, boxes, ignore_indices=None):
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


def px_to_world(px, py, W_px, H_px, res):
    wx = (px - W_px / 2.0) * res
    wy = (H_px / 2.0 - py) * res
    return (wx, wy)


def farthest_point_sampling(centers, k, seed=42):
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

def world_to_px(wx, wy, W_px, H_px, res):
    # Inverse of px_to_world (map center is origin)
    px = (wx / res) + (W_px / 2.0)
    py = (H_px / 2.0) - (wy / res)
    return (px, py)


def sample_building_heights(n, seed, random_height=True, height_fixed=20.0):
    rnd = random.Random(seed + 12345)
    if not random_height:
        return [float(height_fixed)] * n
    return [rnd.uniform(BUILDING_H_MIN, BUILDING_H_MAX) for _ in range(n)]

def distribute_vehicles_over_vertiports(
    num_vehicles,
    vertiports,
    z_vehicle_above_vertiport,
    building_heights=None,
    ground_z=0.0,
):
    V = len(vertiports)
    if V <= 0 or num_vehicles <= 0:
        return []

    base = num_vehicles // V
    rem = num_vehicles % V

    vehicles = []
    vid = 0
    dz = float(z_vehicle_above_vertiport)

    for i, vp in enumerate(vertiports):
        k = base + (1 if i < rem else 0)
        if k <= 0:
            continue

        # Choose spawn base altitude (roof-aware if available)
        if "roof_z" in vp:
            z_base = float(vp["roof_z"])
        elif ("bi" in vp) and (building_heights is not None):
            z_base = float(ground_z) + float(building_heights[int(vp["bi"])])
        elif "height" in vp:
            z_base = float(ground_z) + float(vp["height"])
        else:
            z_base = float(vp.get("z", 0.0))

        for j in range(k):
            if k == 1:
                dx, dy = 0.0, 0.0
                yaw = 0.0
            else:
                ang = 2.0 * math.pi * (j / float(k))
                r = 1.5
                dx = r * math.cos(ang)
                dy = r * math.sin(ang)
                yaw = math.atan2(dy, dx)

            vehicles.append({
                "id": "VEHICLE_%03d" % vid,
                "x": float(vp["x"]) + dx,
                "y": float(vp["y"]) + dy,
                "z": float(z_base) + dz,
                "yaw": float(yaw),
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
    park_models=None,
    vehicle_model_uri="model://quadrotor",
    ground_z=0.0,
    min_vehicle_alt=1.0,
):
    lines = []

    def _include_model(model_uri, model_name, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, static=None):
        lines.append("    <include>")
        lines.append("      <uri>%s</uri>" % model_uri)
        lines.append("      <name>%s</name>" % model_name)
        lines.append("      <pose>%.3f %.3f %.3f %.3f %.3f %.3f</pose>" % (x, y, z, roll, pitch, yaw))
        if static is not None:
            lines.append("      <static>%s</static>" % ("true" if static else "false"))
        lines.append("    </include>")

    width_m = W_px * resolution_m_per_px
    height_m = H_px * resolution_m_per_px
    hm_uri = "file://heightmap.png"
    rnd = random.Random(seed + 123)

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
    lines.append('      <pose>0 0 %.3f 0 0 0</pose>' % ground_z)
    lines.append('      <link name="link">')

    lines.append('        <collision name="collision">')
    lines.append('          <geometry>')
    lines.append('            <heightmap>')
    lines.append('              <uri>%s</uri>' % hm_uri)
    lines.append('              <size>%.6f %.6f 1.0</size>' % (width_m, height_m))
    lines.append('              <pos>0 0 0</pos>')
    lines.append('            </heightmap>')
    lines.append('          </geometry>')
    lines.append('        </collision>')

    lines.append('        <visual name="visual">')
    lines.append('          <pose>0 0 %.3f 0 0 0</pose>' % (ground_z + 0.001))
    lines.append('          <geometry>')
    lines.append('            <plane>')
    lines.append('              <normal>0 0 1</normal>')
    lines.append('              <size>%.6f %.6f</size>' % (width_m, height_m))
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

    if park_models is not None:
        gz = park_models.get("gazebo")
        if gz is not None:
            gz_z = float(gz.get("z", 0.0)) + (ground_z + 0.02)
            gz_yaw = float(gz.get("yaw", 0.0))
            _include_model("model://gazebo", gz["name"], gz["x"], gz["y"], gz_z, yaw=gz_yaw, static=True)

        for t in park_models.get("trees", []):
            tz = float(t.get("z", 0.0)) + (ground_z + 0.02)
            tyaw = float(t.get("yaw", rnd.uniform(-math.pi, math.pi)))
            _include_model("model://pine_tree", t["name"], t["x"], t["y"], tz, yaw=tyaw, static=True)

    overlay_xy_scale = 1.03
    overlay_z_extra = 0.20

    for i, (_area, x, y, w, h) in enumerate(boxes):
        cx = x + w / 2.0
        cy = y + h / 2.0
        wx, wy = px_to_world(cx, cy, W_px, H_px, resolution_m_per_px)

        sx = w * resolution_m_per_px
        sy = h * resolution_m_per_px
        hz = float(building_heights[i])
        wz = (ground_z + hz / 2.0)

        br, bg, bb, ba = ROLE_COLORS["building"]
        base_name = "building_%04d" % i

        lines.append('    <model name="%s">' % base_name)
        lines.append('      <static>true</static>')
        lines.append('      <pose>%.3f %.3f %.3f 0 0 0</pose>' % (wx, wy, wz))
        lines.append('      <link name="link">')
        lines.append('        <collision name="collision">')
        lines.append('          <geometry>')
        lines.append('            <box><size>%.3f %.3f %.3f</size></box>' % (sx, sy, hz))
        lines.append('          </geometry>')
        lines.append('        </collision>')
        lines.append('        <visual name="visual">')
        lines.append('          <geometry>')
        lines.append('            <box><size>%.3f %.3f %.3f</size></box>' % (sx, sy, hz))
        lines.append('          </geometry>')
        lines.append('          <material>')
        lines.append('            <ambient>%.3f %.3f %.3f %.3f</ambient>' % (br, bg, bb, ba))
        lines.append('            <diffuse>%.3f %.3f %.3f %.3f</diffuse>' % (br, bg, bb, ba))
        lines.append('            <emissive>%.3f %.3f %.3f %.3f</emissive>' % (br, bg, bb, ba))
        lines.append('          </material>')
        lines.append('        </visual>')
        lines.append('      </link>')
        lines.append('    </model>')

        if i in roles_by_index:
            role = roles_by_index[i]
            rr, rg, rb, ra = ROLE_COLORS.get(role, ROLE_COLORS["building"])
            overlay_name = "%s_%04d" % (role, i)

            sx2 = sx * overlay_xy_scale
            sy2 = sy * overlay_xy_scale
            hz2 = hz + overlay_z_extra
            wz2 = (ground_z + hz2 / 2.0)

            lines.append('    <model name="%s">' % overlay_name)
            lines.append('      <static>true</static>')
            lines.append('      <pose>%.3f %.3f %.3f 0 0 0</pose>' % (wx, wy, wz2))
            lines.append('      <link name="link">')
            lines.append('        <visual name="visual">')
            lines.append('          <geometry>')
            lines.append('            <box><size>%.3f %.3f %.3f</size></box>' % (sx2, sy2, hz2))
            lines.append('          </geometry>')
            lines.append('          <material>')
            lines.append('            <ambient>%.3f %.3f %.3f %.3f</ambient>' % (rr, rg, rb, ra))
            lines.append('            <diffuse>%.3f %.3f %.3f %.3f</diffuse>' % (rr, rg, rb, ra))
            lines.append('            <emissive>%.3f %.3f %.3f %.3f</emissive>' % (rr, rg, rb, ra))
            lines.append('          </material>')
            lines.append('        </visual>')
            lines.append('      </link>')
            lines.append('    </model>')

    def _sphere_model(model_name, x, y, z, rgba, static="true"):
        rr, gg, bb, aa = rgba
        lines.append('    <model name="%s">' % model_name)
        lines.append('      <static>%s</static>' % static)
        lines.append('      <pose>%.3f %.3f %.3f 0 0 0</pose>' % (x, y, z))
        lines.append('      <link name="link">')
        lines.append('        <collision name="collision">')
        lines.append('          <geometry><sphere><radius>0.4</radius></sphere></geometry>')
        lines.append('        </collision>')
        lines.append('        <visual name="visual">')
        lines.append('          <geometry><sphere><radius>0.4</radius></sphere></geometry>')
        lines.append('          <material>')
        lines.append('            <ambient>%.3f %.3f %.3f %.3f</ambient>' % (rr, gg, bb, aa))
        lines.append('            <diffuse>%.3f %.3f %.3f %.3f</diffuse>' % (rr, gg, bb, aa))
        lines.append('          </material>')
        lines.append('        </visual>')
        lines.append('      </link>')
        lines.append('    </model>')

    if logical_markers:
        for name, x, y, z in logical_markers:
            _sphere_model(name, x, y, z, (1.0, 1.0, 0.0, 1.0))

    if vehicle_markers:
        for v in vehicle_markers:
            if isinstance(v, dict):
                name = v.get("id") or v.get("name")
                x = float(v.get("x", 0.0))
                y = float(v.get("y", 0.0))
                z = float(v.get("z", 0.0))
                yaw = float(v.get("yaw", 0.0))
            else:
                name, x, y, z = v
                x = float(x)
                y = float(y)
                z = float(z)
                yaw = 0.0

            z_roof, bi = roof_spawn_z_for_world_xy(
                x, y,
                boxes=boxes,
                building_heights=building_heights,
                W_px=W_px, H_px=H_px,
                resolution_m_per_px=resolution_m_per_px,
                ground_z=ground_z,
                above_roof=SPECIAL_ABOVE_ROOF_Z,
                min_vehicle_alt=min_vehicle_alt,
            )

            if bi is not None:
                safe_z = float(z_roof)  
            else:
                safe_z = max(float(z), float(z_roof))  

            _include_model(vehicle_model_uri, name, x, y, safe_z, roll=0.0, pitch=0.0, yaw=yaw, static=False)


    lines.append('    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">')
    lines.append('      <ros>')
    lines.append('        <namespace>gazebo</namespace>')
    lines.append('      </ros>')
    lines.append('    </plugin>')

    lines.append('  </world>')
    lines.append('</sdf>')
    return "\n".join(lines)

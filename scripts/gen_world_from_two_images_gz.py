#!/usr/bin/env python3

import os
import math
import shutil
import argparse
import logging
import itertools

import cv2
import numpy as np
from PIL import Image

import city_gen as city
import graph2d as g2d


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gen_world_from_two_images_gz")

OVERFLIGHT_MARGIN_M = 2.0
LOGICAL_ABOVE_TALLEST = 5.0
LOGICAL_LAYER_STEP_M = 5.0
DRONES_PER_EXTRA_LOGICAL_LAYER = 3

UAV_SCALE = 2.0
CHANNEL_ALPHA = 0.18
CHANNEL_RADIUS = 0.08

UAV_COLORS_RGBA = [
    (1.0, 0.0, 0.0, 1.0),      # red
    (0.0, 0.2, 1.0, 1.0),      # blue
    (0.0, 0.8, 0.1, 1.0),      # green
    (1.0, 0.55, 0.0, 1.0),     # orange
    (0.65, 0.0, 1.0, 1.0),     # purple
    (0.0, 0.85, 0.85, 1.0),    # cyan
    (1.0, 0.0, 0.65, 1.0),     # pink
    (0.85, 0.85, 0.0, 1.0),    # yellow
]


def write_nodes_csv(path, node_rows):
    """Write graph nodes CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("id,type,x,y,z\n")
        for nid, typ, x, y, z in node_rows:
            f.write(f"{nid},{typ},{x:.6f},{y:.6f},{z:.6f}\n")


def write_edges_csv(path, edge_rows):
    """Write graph edges CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("src,dst\n")
        for a, b in edge_rows:
            f.write(f"{a},{b}\n")


def rgba_text(rgba):
    r, g, b, a = rgba
    return f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}"


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(name))


def logical_extra_layer_count(num_vehicles, drones_per_layer=DRONES_PER_EXTRA_LOGICAL_LAYER):
    """
    Return the number of extra logical layers required by the fleet size.

    The base logical layer always exists. For every complete group of
    `drones_per_layer` UAVs, one extra logical layer is created above it.
    Examples with the default value 3:
        1--2 UAVs -> 0 extra layers
        3--5 UAVs -> 1 extra layer
        6--8 UAVs -> 2 extra layers
    """
    if drones_per_layer <= 0:
        raise ValueError("drones_per_layer must be positive")
    return max(0, int(num_vehicles) // int(drones_per_layer))


def logical_layer_count(num_vehicles):
    """Return total logical layers, including the base layer."""
    return 1 + logical_extra_layer_count(num_vehicles)


def logical_node_id(layer_idx, node_idx, nodes_per_layer):
    """
    Build a globally sequential logical-node identifier.

    All logical nodes use the same naming pattern LOGICAL_000, LOGICAL_001, ... .
    The global index is computed in layer-major order, so layer 0 occupies
    indices 0..M-1, layer 1 occupies M..2M-1, and so on.
    """
    if int(nodes_per_layer) <= 0:
        raise ValueError("nodes_per_layer must be positive")
    global_idx = int(layer_idx) * int(nodes_per_layer) + int(node_idx)
    return f"LOGICAL_{global_idx:03d}"


def cylinder_pose_between_points(x1, y1, z1, x2, y2, z2):
    """
    Return pose and length for a cylinder connecting two 3D points.

    Gazebo cylinders are aligned with the local z-axis.
    """
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    dz = float(z2 - z1)

    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        return None

    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    mz = 0.5 * (z1 + z2)

    horizontal = math.sqrt(dx * dx + dy * dy)

    yaw = math.atan2(dy, dx)
    pitch = math.atan2(horizontal, dz)
    roll = 0.0

    return mx, my, mz, roll, pitch, yaw, length


def make_channel_sdf(name, x1, y1, z1, x2, y2, z2, radius=CHANNEL_RADIUS, alpha=CHANNEL_ALPHA):
    """
    Create a transparent channel as a Gazebo cylinder.
    """
    pose = cylinder_pose_between_points(x1, y1, z1, x2, y2, z2)
    if pose is None:
        return ""

    mx, my, mz, roll, pitch, yaw, length = pose

    name = safe_name(name)
    alpha = max(0.0, min(1.0, float(alpha)))
    transparency = 1.0 - alpha

    rgba = rgba_text((0.10, 0.55, 1.00, alpha))

    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{mx:.6f} {my:.6f} {mz:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>
      <link name="link">
        <visual name="visual">
          <cast_shadows>false</cast_shadows>
          <transparency>{transparency:.6f}</transparency>
          <geometry>
            <cylinder>
              <radius>{radius:.6f}</radius>
              <length>{length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{rgba}</ambient>
            <diffuse>{rgba}</diffuse>
            <specular>0.05 0.05 0.05 {alpha:.6f}</specular>
          </material>
        </visual>
      </link>
    </model>
""".rstrip()


def make_uav_sdf(vehicle_id, x, y, z, rgba, scale=UAV_SCALE):
    """
    Create a simple movable UAV model.

    IMPORTANT:
    The model name is exactly vehicle_id, e.g. VEHICLE_000.
    This is required because the ROS/Gazebo controller calls SetEntityState
    using this exact entity name.
    """
    vehicle_id = safe_name(vehicle_id)
    color = rgba_text(rgba)

    s = float(scale)

    body_radius = 0.18 * s
    body_length = 0.12 * s

    arm_radius = 0.025 * s
    arm_length = 0.95 * s

    rotor_radius = 0.13 * s
    rotor_length = 0.025 * s
    rotor_offset = 0.48 * s

    collision_radius = 0.14 * s
    collision_length = 0.09 * s

    mass = 1.0
    ixx = 0.02 * s
    iyy = 0.02 * s
    izz = 0.04 * s

    return f"""
    <model name="{vehicle_id}">
      <static>false</static>
      <allow_auto_disable>false</allow_auto_disable>
      <pose>{float(x):.6f} {float(y):.6f} {float(z):.6f} 0 0 0</pose>

      <link name="base_link">
        <gravity>false</gravity>
        <self_collide>false</self_collide>

        <inertial>
          <mass>{mass:.6f}</mass>
          <inertia>
            <ixx>{ixx:.6f}</ixx>
            <iyy>{iyy:.6f}</iyy>
            <izz>{izz:.6f}</izz>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyz>0</iyz>
          </inertia>
        </inertial>

        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{collision_radius:.6f}</radius>
              <length>{collision_length:.6f}</length>
            </cylinder>
          </geometry>
        </collision>

        <visual name="body">
          <pose>0 0 0 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{body_radius:.6f}</radius>
              <length>{body_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
            <specular>0.25 0.25 0.25 1.0</specular>
          </material>
        </visual>

        <visual name="arm_x">
          <pose>0 0 0 0 {math.pi / 2:.6f} 0</pose>
          <geometry>
            <cylinder>
              <radius>{arm_radius:.6f}</radius>
              <length>{arm_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
            <specular>0.25 0.25 0.25 1.0</specular>
          </material>
        </visual>

        <visual name="arm_y">
          <pose>0 0 0 {math.pi / 2:.6f} 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{arm_radius:.6f}</radius>
              <length>{arm_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
            <specular>0.25 0.25 0.25 1.0</specular>
          </material>
        </visual>

        <visual name="rotor_front">
          <cast_shadows>false</cast_shadows>
          <pose>{rotor_offset:.6f} 0 0.03 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{rotor_radius:.6f}</radius>
              <length>{rotor_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.02 0.02 0.02 0.9</ambient>
            <diffuse>0.02 0.02 0.02 0.9</diffuse>
          </material>
        </visual>

        <visual name="rotor_back">
          <cast_shadows>false</cast_shadows>
          <pose>{-rotor_offset:.6f} 0 0.03 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{rotor_radius:.6f}</radius>
              <length>{rotor_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.02 0.02 0.02 0.9</ambient>
            <diffuse>0.02 0.02 0.02 0.9</diffuse>
          </material>
        </visual>

        <visual name="rotor_left">
          <cast_shadows>false</cast_shadows>
          <pose>0 {rotor_offset:.6f} 0.03 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{rotor_radius:.6f}</radius>
              <length>{rotor_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.02 0.02 0.02 0.9</ambient>
            <diffuse>0.02 0.02 0.02 0.9</diffuse>
          </material>
        </visual>

        <visual name="rotor_right">
          <cast_shadows>false</cast_shadows>
          <pose>0 {-rotor_offset:.6f} 0.03 0 0 0</pose>
          <geometry>
            <cylinder>
              <radius>{rotor_radius:.6f}</radius>
              <length>{rotor_length:.6f}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.02 0.02 0.02 0.9</ambient>
            <diffuse>0.02 0.02 0.02 0.9</diffuse>
          </material>
        </visual>

      </link>
    </model>
""".rstrip()


def inject_models_before_world_close(sdf, model_blocks):
    """
    Insert custom models before </world>.
    """
    model_blocks = [m for m in model_blocks if m and m.strip()]
    if not model_blocks:
        return sdf

    idx = sdf.rfind("</world>")
    if idx < 0:
        raise RuntimeError("Could not find </world> in generated SDF.")

    insertion = "\n\n" + "\n\n".join(model_blocks) + "\n"
    return sdf[:idx] + insertion + sdf[idx:]




# -----------------------------------------------------------------------------
# Two-image map processing following the paper's map-to-graph algorithm
# -----------------------------------------------------------------------------

ROLE_BGR = {
    "vertiport": (0, 0, 255),
    "supplier": (0, 200, 0),
    "client": (0, 165, 255),
    "charging": (255, 80, 0),
}

TRI_EDGE_BGR = (120, 120, 120)
LOGICAL_NODE_BGR = (170, 170, 170)
FINAL_EDGE_BGR = (255, 255, 0)


def load_two_aligned_images(mask_png, color_png):
    gray = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read grayscale building mask: {mask_png}")

    color = cv2.imread(str(color_png), cv2.IMREAD_UNCHANGED)
    if color is None:
        raise RuntimeError(f"Failed to read color map: {color_png}")

    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    elif color.ndim == 3 and color.shape[2] == 4:
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
    elif color.ndim != 3 or color.shape[2] != 3:
        raise RuntimeError(f"Unsupported color image shape: {color.shape}")

    if gray.shape[:2] != color.shape[:2]:
        raise RuntimeError(
            "The grayscale and color images must be pixel-aligned and have the "
            f"same dimensions. mask={gray.shape[::-1]}, color={color.shape[1::-1]}"
        )
    return gray, color


def build_binary_building_mask(gray, threshold=-1, close_kernel=1, open_kernel=0):
    """
    Building mask for the two-image version.

    threshold < 0 => Otsu automatic threshold. This is intentionally the default.
    The previous fixed threshold=20 classified the dark background as building in
    maps whose non-building palette contains values around 20--30.
    """
    if int(threshold) < 0:
        used, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        used = int(round(float(used)))
    else:
        used = int(threshold)
        if used < 0 or used > 255:
            raise ValueError("building threshold must be -1 (auto) or in [0,255]")
        binary = np.where(gray > used, 255, 0).astype(np.uint8)

    if int(close_kernel) > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)
        )

    if int(open_kernel) > 1:
        k = int(open_kernel)
        if k % 2 == 0:
            k += 1
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)
        )

    return binary, used


def extract_building_boxes_from_mask(
    building_mask,
    min_area=800,
    min_side=20,
    max_side=300,
    max_aspect_ratio=3.5,
):
    nlabels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (building_mask > 0).astype(np.uint8), connectivity=8
    )

    raw = []
    for label in range(1, nlabels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < int(min_area):
            continue
        if min(w, h) < int(min_side):
            continue
        if int(max_side) > 0 and max(w, h) > int(max_side):
            continue

        aspect = max(
            float(w) / max(1.0, float(h)),
            float(h) / max(1.0, float(w)),
        )
        if aspect > float(max_aspect_ratio):
            continue
        raw.append((x, y, w, h, area))

    raw.sort(key=lambda t: (t[1], t[0], t[2], t[3]))
    return [(idx, x, y, w, h) for idx, (x, y, w, h, _a) in enumerate(raw)]


def box_center_px(box):
    _idx, x, y, w, h = box
    return float(x) + 0.5 * float(w), float(y) + 0.5 * float(h)



def build_road_and_park_masks(
    color_bgr,
    building_mask,
    road_source="geometry",
    resolution_m_per_px=0.2,
    road_max_open_radius_m=12.0,
    road_min_clearance_m=0.5,
):
    """
    Build the traversable 2-D corridor mask.

    IMPORTANT FOR THE TWO-IMAGE FORMULATION
    ---------------------------------------
    The grayscale image contains ONLY two geometric classes:
        building     -> gray/white
        non-building -> black

    Therefore roads, water, parks and other open areas are all black and CANNOT
    be distinguished exactly from the grayscale image alone.

    Modes
    -----
    geometry (default):
        Infer street-like corridors from the building geometry itself.  A free
        pixel is considered part of the road/corridor mask only when its
        distance to the closest building lies in

            road_min_clearance_m <= d <= road_max_open_radius_m.

        This removes the interiors of wide open black regions (river, sea,
        large parks, large plazas) while preserving the narrow urban corridors
        between buildings.  It does not depend on the color-image palette.

    palette:
        Legacy mode for color maps that really follow city_gen's semantic
        palette.  In this mode water/parks can be excluded semantically.

    auto:
        Try palette first; if it is empty/suspicious, use geometry.

    complement:
        Pure complement of the building mask.  Kept only for debugging because
        it treats every black region, including water and parks, as traversable.
    """
    road_source = str(road_source).strip().lower()
    if road_source not in {"geometry", "palette", "auto", "complement"}:
        raise ValueError(
            "road_source must be one of: geometry, palette, auto, complement"
        )

    def _geometry_masks():
        res = float(resolution_m_per_px)
        if res <= 0:
            raise ValueError("resolution_m_per_px must be > 0")

        free01 = (building_mask == 0).astype(np.uint8)
        dist_px = cv2.distanceTransform(free01, cv2.DIST_L2, 5)

        dmin_px = max(0.0, float(road_min_clearance_m) / res)
        dmax_px = max(dmin_px + 1e-9, float(road_max_open_radius_m) / res)

        roads = np.where(
            (free01 > 0) &
            (dist_px >= dmin_px) &
            (dist_px <= dmax_px),
            255,
            0,
        ).astype(np.uint8)

        # Remove isolated single-pixel noise but preserve narrow streets.
        roads = cv2.morphologyEx(
            roads,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        # No semantic park mask is available in the grayscale geometry.
        park = np.zeros_like(building_mask)

        frac = float(np.count_nonzero(roads)) / float(roads.size)
        logger.info(
            "Geometry road mask: occupancy=%.4f, min_clearance=%.2fm, "
            "max_open_radius=%.2fm",
            frac,
            float(road_min_clearance_m),
            float(road_max_open_radius_m),
        )
        return roads, park

    def _palette_masks():
        label_map, min_dist = city.segment_by_palette(color_bgr)
        _mask_building_color, mask_roads = city.build_masks(
            label_map,
            min_dist,
            building_tol=12.0,
            road_tol=18.0,
        )
        try:
            mask_park = city.build_park_mask(
                label_map,
                min_dist,
                park_tol=18.0,
            )
        except Exception:
            mask_park = np.zeros_like(building_mask)

        mask_roads = np.where(
            (mask_roads > 0) & (building_mask == 0),
            255,
            0,
        ).astype(np.uint8)
        mask_park = np.where(mask_park > 0, 255, 0).astype(np.uint8)
        return mask_roads, mask_park

    if road_source == "geometry":
        return _geometry_masks()

    if road_source == "palette":
        roads, park = _palette_masks()
        if not np.any(roads > 0):
            raise RuntimeError(
                "Palette segmentation produced an empty road mask. "
                "Use --road-source geometry for a color image that is not "
                "palette-coded."
            )
        return roads, park

    if road_source == "auto":
        try:
            roads, park = _palette_masks()
            frac = float(np.count_nonzero(roads)) / float(roads.size)
            if 0.005 <= frac <= 0.95:
                logger.info("Using palette-derived road mask (occupancy %.4f)", frac)
                return roads, park
            logger.warning(
                "Palette road mask occupancy %.4f is suspicious; using geometry.",
                frac,
            )
        except Exception as exc:
            logger.warning("Palette road extraction failed: %s", exc)
        return _geometry_masks()

    logger.warning(
        "Using PURE COMPLEMENT of building mask as roads. "
        "Water/sea/parks will also be considered traversable."
    )
    roads = np.where(building_mask == 0, 255, 0).astype(np.uint8)
    park = np.zeros_like(building_mask)
    return roads, park

def expanded_box_slice(box, margin_px, W, H):
    _idx, x, y, w, h = box
    m = max(0, int(math.ceil(float(margin_px))))
    return (
        max(0, x - m),
        max(0, y - m),
        min(W, x + w + m),
        min(H, y + h + m),
    )


def box_has_nearby_road(box, road_mask, search_margin_px):
    H, W = road_mask.shape[:2]
    x0, y0, x1, y1 = expanded_box_slice(box, search_margin_px, W, H)
    _idx, x, y, w, h = box
    roi = road_mask[y0:y1, x0:x1].copy()
    bx0 = max(0, x - x0)
    by0 = max(0, y - y0)
    bx1 = min(roi.shape[1], x + w - x0)
    by1 = min(roi.shape[0], y + h - y0)
    if bx1 > bx0 and by1 > by0:
        roi[by0:by1, bx0:bx1] = 0
    return bool(np.any(roi > 0))



def building_anchor_px(building_mask, box):
    """
    Return a pixel guaranteed to lie INSIDE the detected building component.

    Using the bounding-box center directly is unsafe for irregular footprints:
    the center of the rectangle may fall on a black courtyard/street/water
    pixel.  We instead choose the deepest white pixel inside the box using a
    distance transform.
    """
    _idx, x, y, w, h = box
    roi = (building_mask[y:y + h, x:x + w] > 0).astype(np.uint8)
    if roi.size == 0 or not np.any(roi):
        raise RuntimeError(f"Building box has no white pixels: {box}")

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    yy, xx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    px = float(x + xx)
    py = float(y + yy)

    # Defensive assertion: a special node must NEVER lie on a black pixel.
    if building_mask[int(round(py)), int(round(px))] == 0:
        raise RuntimeError("Internal error: selected special anchor is not on a building")
    return px, py


def make_convex_hull_mask(shape_hw, points):
    """
    Build the exact convex polygon generated by the selected special nodes.

    Every logical node is later required to lie in this mask.  Because the
    polygon is convex, all SL and LL straight segments between accepted nodes
    also remain inside it.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        raise RuntimeError("At least three special nodes are required for a convex hull")

    hull = cv2.convexHull(pts)
    if hull is None or len(hull) < 3:
        raise RuntimeError("Selected special nodes are collinear; convex hull is degenerate")

    hull_i = np.rint(hull).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_i.reshape(-1, 2), 255)

    if cv2.countNonZero(mask) == 0:
        raise RuntimeError("Convex hull mask is empty")
    return mask, hull_i.reshape(-1, 2)


def point_is_inside_mask(mask, p):
    x = int(round(float(p[0])))
    y = int(round(float(p[1])))
    H, W = mask.shape[:2]
    return 0 <= x < W and 0 <= y < H and mask[y, x] > 0



def choose_random_specials(
    boxes,
    building_mask,
    road_mask,
    num_vertiports,
    num_charging,
    num_suppliers,
    num_clients,
    seed,
    road_search_margin_px,
):
    """
    Randomly select SPECIAL BUILDINGS.

    A special node is never sampled from free/black space.  Its coordinate is
    an interior white pixel of a selected connected building component, so it
    cannot be on a street, sea, river, park or any other black region.

    Buildings with a nearby navigable corridor are preferred because they make
    the subsequent SL construction better conditioned.
    """
    total = (
        int(num_vertiports)
        + int(num_charging)
        + int(num_suppliers)
        + int(num_clients)
    )
    if total <= 0:
        raise RuntimeError("At least one special node is required")

    eligible = [
        bi
        for bi, box in enumerate(boxes)
        if box_has_nearby_road(box, road_mask, road_search_margin_px)
    ]

    if len(eligible) < total:
        logger.warning(
            "Only %d/%d buildings have a nearby inferred road corridor; "
            "falling back to all detected BUILDINGS (never to black pixels).",
            len(eligible),
            total,
        )
        eligible = list(range(len(boxes)))

    if len(eligible) < total:
        raise RuntimeError(
            f"Requested {total} special buildings but only {len(eligible)} "
            "building components exist."
        )

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(eligible, size=total, replace=False).tolist()

    role_sequence = (
        ["vertiport"] * int(num_vertiports)
        + ["charging"] * int(num_charging)
        + ["supplier"] * int(num_suppliers)
        + ["client"] * int(num_clients)
    )
    rng.shuffle(role_sequence)

    counters = {
        "vertiport": 0,
        "charging": 0,
        "supplier": 0,
        "client": 0,
    }
    specials = []
    roles_by_index = {}

    for bi, role in zip(chosen, role_sequence):
        bi = int(bi)
        idx = counters[role]
        counters[role] += 1

        # Guaranteed interior-building coordinate.
        px, py = building_anchor_px(building_mask, boxes[bi])

        sid = f"{role.upper()}_{idx:03d}"
        roles_by_index[bi] = role
        specials.append(
            {
                "id": sid,
                "role": role,
                "bi": bi,
                "px": float(px),
                "py": float(py),
            }
        )

    # This is the deterministic construction order required by the manuscript.
    specials.sort(key=lambda s: (float(s["px"]), float(s["py"]), s["id"]))

    # Final defensive validation.
    for s in specials:
        if building_mask[int(round(s["py"])), int(round(s["px"]))] == 0:
            raise RuntimeError(
                f"Special node {s['id']} was not placed on a building"
            )

    return specials, roles_by_index

def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_on_segment(p, a, b, eps=1e-9):
    if abs(orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segments_properly_cross(a, b, c, d, eps=1e-9):
    # Shared endpoints are allowed by graph construction and are not crossings.
    for p in (a, b):
        for q in (c, d):
            if math.hypot(p[0] - q[0], p[1] - q[1]) <= eps:
                return False

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    # Collinear/touching at a non-shared interior point is also rejected.
    if abs(o1) <= eps and point_on_segment(c, a, b):
        return True
    if abs(o2) <= eps and point_on_segment(d, a, b):
        return True
    if abs(o3) <= eps and point_on_segment(a, c, d):
        return True
    if abs(o4) <= eps and point_on_segment(b, c, d):
        return True
    return False


def triangle_area_px2(a, b, c):
    return abs(orient(a, b, c)) * 0.5


def point_in_triangle(p, a, b, c, include_boundary=True, eps=1e-9):
    o1, o2, o3 = orient(a, b, p), orient(b, c, p), orient(c, a, p)
    if include_boundary:
        has_neg = (o1 < -eps) or (o2 < -eps) or (o3 < -eps)
        has_pos = (o1 > eps) or (o2 > eps) or (o3 > eps)
        return not (has_neg and has_pos)
    return (
        (o1 > eps and o2 > eps and o3 > eps)
        or (o1 < -eps and o2 < -eps and o3 < -eps)
    )


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / vv
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def segment_segment_distance(a, b, c, d):
    if segments_properly_cross(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def segment_rect_interval(p0, p1, rect):
    """Liang-Barsky intersection interval t in [0,1] with axis-aligned rectangle."""
    x0, y0 = p0
    x1, y1 = p1
    rx0, ry0, rx1, ry1 = rect
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0

    for p, q in (
        (-dx, x0 - rx0),
        ( dx, rx1 - x0),
        (-dy, y0 - ry0),
        ( dy, ry1 - y0),
    ):
        if abs(p) <= 1e-15:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t0 > t1:
        return None
    return max(0.0, t0), min(1.0, t1)


def segment_clear_of_building_prisms(
    p0,
    z0,
    p1,
    z1,
    boxes,
    building_heights,
    ignore_indices=None,
    vertical_clearance_m=0.0,
):
    ignore = set(ignore_indices or [])
    for bi, box in enumerate(boxes):
        if bi in ignore:
            continue
        _idx, x, y, w, h = box
        interval = segment_rect_interval(
            p0, p1, (float(x), float(y), float(x + w), float(y + h))
        )
        if interval is None:
            continue
        ta, tb = interval
        za = float(z0) + (float(z1) - float(z0)) * ta
        zb = float(z0) + (float(z1) - float(z0)) * tb
        minimum_z = min(za, zb)
        if minimum_z <= float(building_heights[bi]) + float(vertical_clearance_m):
            return False
    return True


def edge_respects_node_clearance(p0, p1, all_points, endpoint_keys, min_clearance_px):
    if float(min_clearance_px) <= 0:
        return True
    excluded = set(endpoint_keys)
    for key, p in all_points.items():
        if key in excluded:
            continue
        if point_segment_distance(p, p0, p1) < float(min_clearance_px):
            return False
    return True


def edge_respects_existing_edges(p0, p1, new_endpoints, existing_edges, min_edge_clearance_px):
    for e in existing_edges:
        q0, q1 = e["p0"], e["p1"]
        shared = set(new_endpoints) & set(e["endpoints"])
        if shared:
            # Graph edges are allowed to meet at a common graph node.
            continue
        if segments_properly_cross(p0, p1, q0, q1):
            return False
        if float(min_edge_clearance_px) > 0:
            if segment_segment_distance(p0, p1, q0, q1) < float(min_edge_clearance_px):
                return False
    return True


# -----------------------------------------------------------------------------
# Phase 2: constrained Delaunay-like triangulation and logical-node placement
# -----------------------------------------------------------------------------

def build_constrained_triangulation(
    specials,
    boxes,
    building_heights,
    z_special_by_id,
    min_triangle_area_px2,
    vertical_clearance_m=0.0,
):
    """
    Implements the four construction rules stated in the manuscript:
      1. special nodes sorted by increasing x;
      2. mutually visible, non-crossing edges;
      3. triangles with minimum area and no other special inside;
      4. logical nodes are created later at triangle centroids/free points.
    """
    sp = sorted(specials, key=lambda s: (float(s["px"]), float(s["py"]), s["id"]))
    n = len(sp)
    if n < 3:
        raise RuntimeError("At least three special nodes are required for triangulation")

    points = [(float(s["px"]), float(s["py"])) for s in sp]

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            p0, p1 = points[i], points[j]
            clear = segment_clear_of_building_prisms(
                p0,
                z_special_by_id[sp[i]["id"]],
                p1,
                z_special_by_id[sp[j]["id"]],
                boxes,
                building_heights,
                ignore_indices={sp[i]["bi"], sp[j]["bi"]},
                vertical_clearance_m=vertical_clearance_m,
            )
            if not clear:
                continue
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            candidates.append((d, i, j))

    # Shortest visible edges first produces a Delaunay-like planar structure.
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    edges = []
    edge_set = set()
    for _d, i, j in candidates:
        p0, p1 = points[i], points[j]
        if any(
            segments_properly_cross(p0, p1, points[a], points[b])
            for a, b in edges
        ):
            continue
        edges.append((i, j))
        edge_set.add((i, j))

    triangles = []
    for i, j, k in itertools.combinations(range(n), 3):
        ij = (min(i, j), max(i, j))
        ik = (min(i, k), max(i, k))
        jk = (min(j, k), max(j, k))
        if ij not in edge_set or ik not in edge_set or jk not in edge_set:
            continue

        a, b, c = points[i], points[j], points[k]
        area = triangle_area_px2(a, b, c)
        if area < float(min_triangle_area_px2):
            continue

        contains_other = False
        for m, p in enumerate(points):
            if m in (i, j, k):
                continue
            if point_in_triangle(p, a, b, c, include_boundary=True):
                contains_other = True
                break
        if contains_other:
            continue
        triangles.append((i, j, k))

    if not triangles:
        raise RuntimeError(
            "Constrained triangulation produced no valid triangles. Try another seed, "
            "reduce --min-triangle-area-m2, or increase special-node count."
        )

    return sp, points, edges, triangles



def nearest_skeleton_point_inside_triangle(
    centroid,
    tri_pts,
    pos_skel,
    building_mask,
    road_mask,
    hull_mask,
):
    a, b, c = tri_pts
    best = None
    best_d2 = float("inf")
    H, W = building_mask.shape[:2]

    for _sid, (x, y) in pos_skel.items():
        x = float(x)
        y = float(y)
        xi, yi = int(round(x)), int(round(y))

        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            continue
        if hull_mask[yi, xi] == 0:
            continue
        if building_mask[yi, xi] != 0:
            continue
        if road_mask[yi, xi] == 0:
            continue
        if not point_in_triangle((x, y), a, b, c, include_boundary=True):
            continue

        d2 = (x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = (x, y)

    return best


def sample_free_point_inside_triangle(
    tri_pts,
    centroid,
    building_mask,
    road_mask,
    hull_mask,
    rng,
    attempts=5000,
):
    """
    Rejection sampling fallback.

    A point is accepted only if it is:
      * inside the source triangle;
      * inside the convex polygon of ALL special nodes;
      * outside buildings;
      * inside the inferred/semantic road corridor.
    """
    a, b, c = tri_pts
    H, W = building_mask.shape[:2]

    minx = max(0, int(math.floor(min(a[0], b[0], c[0]))))
    maxx = min(W - 1, int(math.ceil(max(a[0], b[0], c[0]))))
    miny = max(0, int(math.floor(min(a[1], b[1], c[1]))))
    maxy = min(H - 1, int(math.ceil(max(a[1], b[1], c[1]))))

    best = None
    best_d2 = float("inf")

    for _ in range(int(attempts)):
        x = float(rng.uniform(minx, maxx + 1e-9))
        y = float(rng.uniform(miny, maxy + 1e-9))

        if not point_in_triangle((x, y), a, b, c, include_boundary=True):
            continue

        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            continue
        if hull_mask[yi, xi] == 0:
            continue
        if building_mask[yi, xi] != 0:
            continue
        if road_mask[yi, xi] == 0:
            continue

        d2 = (x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
        if d2 < best_d2:
            best = (x, y)
            best_d2 = d2

    return best


def place_logical_nodes(
    sp,
    sp_points,
    triangles,
    building_mask,
    road_mask,
    hull_mask,
    pos_skel,
    seed,
    rejection_attempts=5000,
    min_separation_px=0.0,
):
    """
    Place one logical node per valid triangle.

    HARD CONSTRAINT:
        every logical node must lie inside the convex polygon formed by all
        selected special nodes.

    Placement order follows the manuscript:
      1) triangle centroid if it is a valid navigable free point;
      2) nearest valid road-skeleton point inside the same triangle;
      3) rejection sampling inside the same triangle.
    """
    rng = np.random.default_rng(int(seed))
    logical = []
    triangle_records = []
    H, W = building_mask.shape[:2]

    for tri_idx, (i, j, k) in enumerate(triangles):
        a, b, c = sp_points[i], sp_points[j], sp_points[k]
        centroid = (
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
        )

        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        p = None

        if 0 <= cx < W and 0 <= cy < H:
            if (
                hull_mask[cy, cx] > 0
                and building_mask[cy, cx] == 0
                and road_mask[cy, cx] > 0
            ):
                p = centroid

        if p is None:
            p = nearest_skeleton_point_inside_triangle(
                centroid,
                (a, b, c),
                pos_skel,
                building_mask,
                road_mask,
                hull_mask,
            )

        if p is None:
            p = sample_free_point_inside_triangle(
                (a, b, c),
                centroid,
                building_mask,
                road_mask,
                hull_mask,
                rng,
                attempts=rejection_attempts,
            )

        if p is None:
            logger.warning(
                "Triangle %d has no valid logical point inside the convex hull; skipping.",
                tri_idx,
            )
            continue

        # Explicit hard assertion requested by the experiment definition.
        if not point_is_inside_mask(hull_mask, p):
            raise RuntimeError(
                f"Internal error: logical node for triangle {tri_idx} lies outside convex hull"
            )

        if float(min_separation_px) > 0:
            if any(
                math.hypot(p[0] - q[0], p[1] - q[1])
                < float(min_separation_px)
                for q in logical
            ):
                logger.warning(
                    "Triangle %d logical node is too close to another logical; skipping.",
                    tri_idx,
                )
                continue

        li = len(logical)
        logical.append((float(p[0]), float(p[1])))
        triangle_records.append(
            {
                "triangle": (i, j, k),
                "logical": li,
                "centroid": centroid,
            }
        )

    if not logical:
        raise RuntimeError(
            "No logical nodes could be placed inside the special-node convex hull"
        )

    # Every special must participate in at least one usable triangle.
    covered_specials = set()
    for rec in triangle_records:
        covered_specials.update(rec["triangle"])

    missing = [
        sp[i]["id"]
        for i in range(len(sp))
        if i not in covered_specials
    ]
    if missing:
        raise RuntimeError(
            "Some special nodes do not participate in any usable triangle: "
            + ", ".join(missing)
        )

    # Final hull validation over the complete logical set.
    outside = [
        li
        for li, p in enumerate(logical)
        if not point_is_inside_mask(hull_mask, p)
    ]
    if outside:
        raise RuntimeError(
            "Logical nodes outside convex hull: "
            + ", ".join(str(i) for i in outside)
        )

    return logical, triangle_records

def construct_sl_edges(
    sp,
    sp_points,
    logical_points,
    triangle_records,
    boxes,
    building_heights,
    z_special_by_id,
    z_logical,
    node_clearance_px,
    edge_clearance_px,
    vertical_clearance_m,
):
    # All graph points for clearance checks.
    all_points = {}
    for i, p in enumerate(sp_points):
        all_points[("S", i)] = p
    for li, p in enumerate(logical_points):
        all_points[("L", li)] = p

    candidates = []
    seen = set()
    for rec in triangle_records:
        li = rec["logical"]
        for si in rec["triangle"]:
            key = (si, li)
            if key in seen:
                continue
            seen.add(key)
            d = math.hypot(
                sp_points[si][0] - logical_points[li][0],
                sp_points[si][1] - logical_points[li][1],
            )
            candidates.append((d, si, li))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    accepted = []
    degree_special = [0] * len(sp)
    for _d, si, li in candidates:
        p0, p1 = sp_points[si], logical_points[li]
        if not segment_clear_of_building_prisms(
            p0,
            z_special_by_id[sp[si]["id"]],
            p1,
            z_logical,
            boxes,
            building_heights,
            ignore_indices={sp[si]["bi"]},
            vertical_clearance_m=vertical_clearance_m,
        ):
            continue
        endpoints = {("S", si), ("L", li)}
        if not edge_respects_node_clearance(
            p0, p1, all_points, endpoints, node_clearance_px
        ):
            continue
        if not edge_respects_existing_edges(
            p0, p1, endpoints, accepted, edge_clearance_px
        ):
            continue
        accepted.append({
            "kind": "SL",
            "a": si,
            "b": li,
            "p0": p0,
            "p1": p1,
            "endpoints": endpoints,
        })
        degree_special[si] += 1

    missing = [sp[i]["id"] for i, d in enumerate(degree_special) if d == 0]
    if missing:
        raise RuntimeError(
            "SL construction left special nodes disconnected: " + ", ".join(missing)
        )
    return accepted



class UnionFind:
    """
    Disjoint-set / Union-Find structure used by the greedy LL edge
    construction to guarantee that the logical-logical subgraph becomes
    connected without introducing cycles during the spanning-tree phase.
    """

    def __init__(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("UnionFind size must be non-negative")
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        x = int(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent

        return root

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        self.components -= 1
        return True

def construct_ll_edges(
    logical_points,
    existing_edges,
    boxes,
    building_heights,
    z_logical,
    all_special_points,
    node_clearance_px,
    edge_clearance_px,
    vertical_clearance_m,
):
    M = len(logical_points)
    if M == 1:
        return []

    all_points = {}
    for key, p in all_special_points.items():
        all_points[key] = p
    for li, p in enumerate(logical_points):
        all_points[("L", li)] = p

    candidates = []
    for i in range(M):
        for j in range(i + 1, M):
            p0, p1 = logical_points[i], logical_points[j]
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            candidates.append((d, i, j))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))

    uf = UnionFind(M)
    accepted = []
    all_edges = list(existing_edges)
    for _d, i, j in candidates:
        if uf.find(i) == uf.find(j):
            continue
        p0, p1 = logical_points[i], logical_points[j]
        if not segment_clear_of_building_prisms(
            p0,
            z_logical,
            p1,
            z_logical,
            boxes,
            building_heights,
            ignore_indices=None,
            vertical_clearance_m=vertical_clearance_m,
        ):
            continue
        endpoints = {("L", i), ("L", j)}
        if not edge_respects_node_clearance(
            p0, p1, all_points, endpoints, node_clearance_px
        ):
            continue
        if not edge_respects_existing_edges(
            p0, p1, endpoints, all_edges, edge_clearance_px
        ):
            continue

        e = {
            "kind": "LL",
            "a": i,
            "b": j,
            "p0": p0,
            "p1": p1,
            "endpoints": endpoints,
        }
        accepted.append(e)
        all_edges.append(e)
        uf.union(i, j)
        if uf.components == 1:
            break

    if uf.components != 1:
        raise RuntimeError(
            "Could not create a connected LL network under building/clearance/crossing constraints. "
            "Reduce --node-clearance-m/--edge-clearance-m or try another seed."
        )
    return accepted


def draw_special_nodes(base_bgr, boxes, specials):
    out = base_bgr.copy()
    for _bi, x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (70, 70, 70), 1)
    for sp in specials:
        pt = (int(round(sp["px"])), int(round(sp["py"])))
        color = ROLE_BGR.get(sp["role"], (0, 0, 255))
        cv2.circle(out, pt, 9, color, -1, cv2.LINE_AA)
        cv2.putText(out, sp["id"], (pt[0] + 10, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_triangulation(base_bgr, sp, sp_points, tri_edges, triangles, logical_points, triangle_records):
    out = base_bgr.copy()
    for i, j in tri_edges:
        a = tuple(int(round(v)) for v in sp_points[i])
        b = tuple(int(round(v)) for v in sp_points[j])
        cv2.line(out, a, b, TRI_EDGE_BGR, 1, cv2.LINE_AA)
    for rec in triangle_records:
        li = rec["logical"]
        p = tuple(int(round(v)) for v in logical_points[li])
        cv2.circle(out, p, 5, LOGICAL_NODE_BGR, -1, cv2.LINE_AA)
    for i, s in enumerate(sp):
        p = tuple(int(round(v)) for v in sp_points[i])
        cv2.circle(out, p, 7, ROLE_BGR.get(s["role"], (0, 0, 255)), -1, cv2.LINE_AA)
    return out


def draw_final_graph(base_bgr, sp, sp_points, logical_points, sl_edges, ll_edges):
    out = base_bgr.copy()
    for e in ll_edges + sl_edges:
        a = tuple(int(round(v)) for v in e["p0"])
        b = tuple(int(round(v)) for v in e["p1"])
        cv2.line(out, a, b, FINAL_EDGE_BGR, 2, cv2.LINE_AA)
    for li, p in enumerate(logical_points):
        q = tuple(int(round(v)) for v in p)
        cv2.circle(out, q, 5, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(out, q, 3, LOGICAL_NODE_BGR, -1, cv2.LINE_AA)
        cv2.putText(out, f"L{li}", (q[0] + 5, q[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    for i, s in enumerate(sp):
        q = tuple(int(round(v)) for v in sp_points[i])
        cv2.circle(out, q, 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, q, 6, ROLE_BGR.get(s["role"], (0, 0, 255)), -1, cv2.LINE_AA)
    return out


def main(
    num_vehicles,
    num_vertiports,
    num_charging,
    num_suppliers,
    num_clients,
    mask_png,
    color_png,
    out_dir="gz_world_out",
    resolution_m_per_px=0.2,
    seed=42,
    z_special=2.0,
    z_vehicle=1.0,
    building_threshold=-1,
    mask_close_kernel=1,
    mask_open_kernel=0,
    min_building_area=800,
    min_building_side=20,
    max_building_side=300,
    max_building_aspect=3.5,
    road_source="geometry",
    road_max_open_radius_m=12.0,
    road_min_clearance_m=0.5,
    special_road_search_m=8.0,
    min_triangle_area_m2=25.0,
    logical_rejection_attempts=5000,
    logical_min_separation_m=1.0,
    node_clearance_m=0.5,
    edge_clearance_m=0.0,
    vertical_clearance_m=1.0,
    special_selection_restarts=100,
    spawn_markers=True,
):
    logger.info("Starting TWO-IMAGE map-to-graph conversion")
    logger.info("  mask : %s", mask_png)
    logger.info("  color: %s", color_png)
    logger.info("  seed : %d", int(seed))
    os.makedirs(out_dir, exist_ok=True)

    gray, color = load_two_aligned_images(mask_png, color_png)
    H, W = gray.shape[:2]
    logger.info("Aligned image dimensions: %dx%d pixels", W, H)

    # ------------------------------------------------------------------
    # Phase 1: segmentation and road skeleton
    # ------------------------------------------------------------------
    building_mask, threshold_used = build_binary_building_mask(
        gray,
        threshold=building_threshold,
        close_kernel=mask_close_kernel,
        open_kernel=mask_open_kernel,
    )
    logger.info("Building threshold used: %d%s", threshold_used,
                " (Otsu auto)" if int(building_threshold) < 0 else "")
    cv2.imwrite(os.path.join(out_dir, "mask_building.png"), building_mask)

    boxes = extract_building_boxes_from_mask(
        building_mask,
        min_area=min_building_area,
        min_side=min_building_side,
        max_side=max_building_side,
        max_aspect_ratio=max_building_aspect,
    )
    logger.info("Detected %d valid building components", len(boxes))
    if not boxes:
        raise RuntimeError(
            "No buildings were detected after filtering. Use --building-threshold -1 "
            "(automatic) or inspect --min-building-area/--max-building-side."
        )

    road_mask, park_mask = build_road_and_park_masks(
        color,
        building_mask,
        road_source=road_source,
        resolution_m_per_px=resolution_m_per_px,
        road_max_open_radius_m=road_max_open_radius_m,
        road_min_clearance_m=road_min_clearance_m,
    )
    cv2.imwrite(os.path.join(out_dir, "mask_roads.png"), road_mask)
    cv2.imwrite(os.path.join(out_dir, "mask_park.png"), park_mask)

    skel01 = g2d.skeletonize_roads(road_mask)
    cv2.imwrite(
        os.path.join(out_dir, "roads_skeleton.png"),
        (skel01 * 255).astype(np.uint8),
    )
    _adj_skel, pos_skel = g2d.build_skeleton_graph(skel01)
    if len(pos_skel) == 0:
        raise RuntimeError("Road skeleton graph is empty. Check --road-source / color palette.")
    logger.info("Road skeleton contains %d pixels/nodes", len(pos_skel))

    special_road_search_px = float(special_road_search_m) / float(resolution_m_per_px)
    min_triangle_area_px2 = float(min_triangle_area_m2) / (float(resolution_m_per_px) ** 2)
    logical_min_sep_px = float(logical_min_separation_m) / float(resolution_m_per_px)
    node_clearance_px = float(node_clearance_m) / float(resolution_m_per_px)
    edge_clearance_px = float(edge_clearance_m) / float(resolution_m_per_px)

    # ------------------------------------------------------------------
    # Randomly choose special buildings, then run EXACT phases 2 and 3.
    # Retry only the random special-node selection if a selected set cannot
    # satisfy the constrained triangulation/edge conditions.
    # ------------------------------------------------------------------
    solution = None
    last_error = None

    for attempt in range(int(special_selection_restarts)):
        local_seed = int(seed) + 100003 * attempt
        try:
            specials, roles_by_index = choose_random_specials(
                boxes,
                building_mask,
                road_mask,
                num_vertiports,
                num_charging,
                num_suppliers,
                num_clients,
                seed=local_seed,
                road_search_margin_px=special_road_search_px,
            )

            # Convex polygon generated by the selected special nodes.  All
            # logical-node candidates and skeleton points are hard-clipped to it.
            raw_special_points = [
                (float(s["px"]), float(s["py"])) for s in specials
            ]
            hull_mask, hull_points = make_convex_hull_mask(
                building_mask.shape[:2],
                raw_special_points,
            )

            pos_skel_hull = {
                sid: (float(px), float(py))
                for sid, (px, py) in pos_skel.items()
                if point_is_inside_mask(hull_mask, (px, py))
            }
            if not pos_skel_hull:
                raise RuntimeError(
                    "No road-skeleton points lie inside the special-node convex hull"
                )

            building_heights = city.sample_building_heights(
                len(boxes),
                seed=local_seed,
                random_height=True,
                special_indices=roles_by_index.keys(),
                base_min=8.0,
                base_max=25.0,
                special_min=26.0,
                special_max=35.0,
            )
            tallest = float(max(building_heights))
            z_logical_final = tallest + float(LOGICAL_ABOVE_TALLEST)

            z_special_by_id = {}
            for s in specials:
                z_special_by_id[s["id"]] = float(building_heights[s["bi"]]) + float(z_special)

            sp, sp_points, tri_edges, triangles = build_constrained_triangulation(
                specials,
                boxes,
                building_heights,
                z_special_by_id,
                min_triangle_area_px2=min_triangle_area_px2,
                vertical_clearance_m=vertical_clearance_m,
            )

            logical_points, triangle_records = place_logical_nodes(
                sp,
                sp_points,
                triangles,
                building_mask,
                road_mask,
                hull_mask,
                pos_skel_hull,
                seed=local_seed + 17,
                rejection_attempts=logical_rejection_attempts,
                min_separation_px=logical_min_sep_px,
            )

            sl_edges = construct_sl_edges(
                sp,
                sp_points,
                logical_points,
                triangle_records,
                boxes,
                building_heights,
                z_special_by_id,
                z_logical=z_logical_final,
                node_clearance_px=node_clearance_px,
                edge_clearance_px=edge_clearance_px,
                vertical_clearance_m=vertical_clearance_m,
            )

            special_points_dict = {("S", i): p for i, p in enumerate(sp_points)}
            ll_edges = construct_ll_edges(
                logical_points,
                sl_edges,
                boxes,
                building_heights,
                z_logical=z_logical_final,
                all_special_points=special_points_dict,
                node_clearance_px=node_clearance_px,
                edge_clearance_px=edge_clearance_px,
                vertical_clearance_m=vertical_clearance_m,
            )

            solution = {
                "seed": local_seed,
                "specials": sp,
                "roles_by_index": roles_by_index,
                "building_heights": building_heights,
                "tallest": tallest,
                "z_logical_final": z_logical_final,
                "z_special_by_id": z_special_by_id,
                "sp_points": sp_points,
                "tri_edges": tri_edges,
                "triangles": triangles,
                "logical_points": logical_points,
                "triangle_records": triangle_records,
                "sl_edges": sl_edges,
                "ll_edges": ll_edges,
                "hull_mask": hull_mask,
                "hull_points": hull_points,
            }
            logger.info("Feasible map-to-graph solution found at special-selection attempt %d", attempt)
            break
        except RuntimeError as exc:
            last_error = exc
            logger.warning("Special-selection attempt %d failed: %s", attempt, exc)

    if solution is None:
        raise RuntimeError(
            f"Could not construct a graph after {special_selection_restarts} special-node selections: {last_error}"
        )

    specials = solution["specials"]
    roles_by_index = solution["roles_by_index"]
    building_heights = solution["building_heights"]
    tallest = solution["tallest"]
    z_logical_final = solution["z_logical_final"]
    z_special_by_id = solution["z_special_by_id"]
    sp_points = solution["sp_points"]
    tri_edges = solution["tri_edges"]
    triangles = solution["triangles"]
    logical_points = solution["logical_points"]
    triangle_records = solution["triangle_records"]
    sl_edges = solution["sl_edges"]
    ll_edges = solution["ll_edges"]
    hull_mask = solution["hull_mask"]
    hull_points = solution["hull_points"]

    # Hard validation: special nodes must be on buildings and logical nodes
    # must be inside the special-node convex polygon.
    for s in specials:
        sx, sy = int(round(s["px"])), int(round(s["py"]))
        if building_mask[sy, sx] == 0:
            raise RuntimeError(
                f"Special node {s['id']} is not on a building pixel"
            )
    for li, p in enumerate(logical_points):
        if not point_is_inside_mask(hull_mask, p):
            raise RuntimeError(
                f"Logical node L{li} lies outside the special-node convex hull"
            )

    cv2.imwrite(os.path.join(out_dir, "convex_hull_mask.png"), hull_mask)

    road_in_hull = cv2.bitwise_and(road_mask, hull_mask)
    cv2.imwrite(os.path.join(out_dir, "mask_roads_in_hull.png"), road_in_hull)

    # ------------------------------------------------------------------
    # Diagnostics: same geometry over grayscale and color images.
    # ------------------------------------------------------------------
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Visual proof of the admissible logical-node region.
    hull_debug = gray_bgr.copy()
    cv2.polylines(
        hull_debug,
        [hull_points.astype(np.int32)],
        True,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(
        os.path.join(out_dir, "special_convex_hull.png"),
        hull_debug,
    )

    special_img = draw_special_nodes(color, boxes, specials)
    cv2.imwrite(os.path.join(out_dir, "special_nodes_colored.png"), special_img)

    tri_img = draw_triangulation(
        gray_bgr, specials, sp_points, tri_edges, triangles, logical_points, triangle_records
    )
    cv2.imwrite(os.path.join(out_dir, "graph_debug_triangles.png"), tri_img)

    graph_gray = draw_final_graph(
        gray_bgr, specials, sp_points, logical_points, sl_edges, ll_edges
    )
    graph_color = draw_final_graph(
        color, specials, sp_points, logical_points, sl_edges, ll_edges
    )
    graph_gray_path = os.path.join(out_dir, "graph_debug.png")
    graph_color_path = os.path.join(out_dir, "graph_on_color.png")
    cv2.imwrite(graph_gray_path, graph_gray)
    cv2.imwrite(graph_color_path, graph_color)

    # ------------------------------------------------------------------
    # Convert base graph into the original multi-layer 3D graph.
    # ------------------------------------------------------------------
    node_rows = []
    edge_rows = []

    num_logical_layers = logical_layer_count(num_vehicles)
    num_extra_logical_layers = num_logical_layers - 1
    logical_layer_z = [
        float(z_logical_final) + layer_idx * float(LOGICAL_LAYER_STEP_M)
        for layer_idx in range(num_logical_layers)
    ]
    M = len(logical_points)

    logical_world = []
    for li, (px, py) in enumerate(logical_points):
        xw, yw = city.px_to_world(px, py, W, H, resolution_m_per_px)
        logical_world.append((li, xw, yw))
        for layer_idx, z_layer in enumerate(logical_layer_z):
            node_rows.append((
                logical_node_id(layer_idx, li, M),
                "logical",
                xw,
                yw,
                float(z_layer),
            ))

    vertiports_world = []
    special_index_by_id = {s["id"]: i for i, s in enumerate(specials)}
    for s in specials:
        xw, yw = city.px_to_world(s["px"], s["py"], W, H, resolution_m_per_px)
        z = z_special_by_id[s["id"]]
        node_rows.append((s["id"], s["role"], xw, yw, z))
        if s["role"] == "vertiport":
            vertiports_world.append({"id": s["id"], "x": xw, "y": yw, "z": z})

    for e in sl_edges:
        sid = specials[e["a"]]["id"]
        edge_rows.append((sid, logical_node_id(0, e["b"], M)))

    for layer_idx in range(num_logical_layers):
        for e in ll_edges:
            edge_rows.append((
                logical_node_id(layer_idx, e["a"], M),
                logical_node_id(layer_idx, e["b"], M),
            ))

    for layer_idx in range(1, num_logical_layers):
        for li in range(M):
            edge_rows.append((
                logical_node_id(layer_idx, li, M),
                logical_node_id(layer_idx - 1, li, M),
            ))

    vehicles = city.distribute_vehicles_over_vertiports(
        num_vehicles,
        vertiports_world,
        z_vehicle_above_vertiport=float(z_vehicle),
    )
    for v in vehicles:
        node_rows.append((v["id"], "vehicle", v["x"], v["y"], float(v["z"])))

    nodes_csv = os.path.join(out_dir, "graph_nodes.csv")
    edges_csv = os.path.join(out_dir, "graph_edges.csv")
    write_nodes_csv(nodes_csv, node_rows)
    write_edges_csv(edges_csv, edge_rows)

    # ------------------------------------------------------------------
    # Gazebo world: buildings from grayscale geometry, ground texture from color.
    # ------------------------------------------------------------------
    hm_raw = np.zeros((H, W), dtype=np.uint8)
    hm_fixed = city.pad_to_square_pow2p1(hm_raw)
    heightmap_path = os.path.join(out_dir, "heightmap.png")
    Image.fromarray(hm_fixed, mode="L").save(heightmap_path)

    texture_root_path = os.path.join(out_dir, "finalmap.png")
    Image.open(color_png).convert("RGB").save(texture_root_path)
    city.write_ogre_material(out_dir=out_dir, texture_filename="finalmap.png")
    textures_dir = os.path.join(out_dir, "materials", "textures")
    os.makedirs(textures_dir, exist_ok=True)
    shutil.copyfile(texture_root_path, os.path.join(textures_dir, "finalmap.png"))
    city.write_stub_model_config(
        os.path.join(out_dir, "materials"), model_name="utm_materials_stub"
    )

    park_models = {}
    if np.any(park_mask > 0):
        try:
            park_models = city.plan_park_models(
                park_mask,
                W_px=W,
                H_px=H,
                resolution_m_per_px=resolution_m_per_px,
                seed=solution["seed"],
            )
        except Exception as exc:
            logger.warning("Could not create park models: %s", exc)

    node_pos = {}
    for li, _xw, _yw in logical_world:
        xw, yw = _xw, _yw
        for layer_idx, z_layer in enumerate(logical_layer_z):
            node_pos[logical_node_id(layer_idx, li, M)] = (float(xw), float(yw), float(z_layer))
    for s in specials:
        xw, yw = city.px_to_world(s["px"], s["py"], W, H, resolution_m_per_px)
        node_pos[s["id"]] = (float(xw), float(yw), float(z_special_by_id[s["id"]]))

    custom_models = []
    if spawn_markers:
        edge_id = 0
        for a, b in edge_rows:
            if a not in node_pos or b not in node_pos:
                continue
            x1, y1, z1 = node_pos[a]
            x2, y2, z2 = node_pos[b]
            custom_models.append(
                make_channel_sdf(
                    name=f"EDGE_{edge_id:05d}",
                    x1=x1, y1=y1, z1=z1,
                    x2=x2, y2=y2, z2=z2,
                    radius=CHANNEL_RADIUS,
                    alpha=CHANNEL_ALPHA,
                )
            )
            edge_id += 1

        for i, v in enumerate(vehicles):
            custom_models.append(
                make_uav_sdf(
                    vehicle_id=v["id"],
                    x=v["x"], y=v["y"], z=float(v["z"]),
                    rgba=UAV_COLORS_RGBA[i % len(UAV_COLORS_RGBA)],
                    scale=UAV_SCALE,
                )
            )

    sdf = city.make_world_sdf(
        W_px=W,
        H_px=H,
        boxes=boxes,
        roles_by_index=roles_by_index,
        building_heights=building_heights,
        resolution_m_per_px=resolution_m_per_px,
        seed=solution["seed"],
        logical_markers=None,
        vehicle_markers=None,
        park_models=park_models,
        vehicle_model_uri="model://quadrotor",
        edge_markers=None,
    )
    sdf = inject_models_before_world_close(sdf, custom_models)
    sdf_path = os.path.join(out_dir, "utm_world.sdf")
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf)

    with open(os.path.join(out_dir, "graph_seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"master_seed={int(seed)}\n")
        f.write(f"selected_seed={int(solution['seed'])}\n")
        f.write(f"building_threshold={threshold_used}\n")
        f.write(f"road_source={road_source}\n")
        f.write(f"road_max_open_radius_m={road_max_open_radius_m}\n")
        f.write(f"road_min_clearance_m={road_min_clearance_m}\n")
        f.write(f"special_nodes_on_buildings=true\n")
        f.write(f"logical_nodes_inside_special_convex_hull=true\n")
        f.write(f"buildings={len(boxes)}\n")
        f.write(f"specials={len(specials)}\n")
        f.write(f"triangles={len(triangles)}\n")
        f.write(f"logical_nodes={len(logical_points)}\n")
        f.write(f"sl_edges={len(sl_edges)}\n")
        f.write(f"ll_edges={len(ll_edges)}\n")

    print("\n[SUCCESS] Map-to-Graph conversion completed")
    print(f"  • Building threshold: {threshold_used}")
    print(f"  • Buildings: {len(boxes)}")
    print(f"  • Special nodes: {len(specials)}")
    print(f"  • Triangulation edges: {len(tri_edges)}")
    print(f"  • Valid triangles: {len(triangles)}")
    print(f"  • Logical nodes: {len(logical_points)}")
    print(f"  • Special nodes on buildings: YES")
    print(f"  • Logical nodes inside special convex hull: YES")
    print(f"  • Road source: {road_source}")
    print(f"  • SL edges: {len(sl_edges)}")
    print(f"  • LL edges: {len(ll_edges)}")
    print(f"  • Logical layers: {num_logical_layers} = 1 base + {num_extra_logical_layers} extra")
    print(f"  • Graph on grayscale: {graph_gray_path}")
    print(f"  • Same graph on color: {graph_color_path}")
    print(f"  • Nodes CSV: {nodes_csv}")
    print(f"  • Edges CSV: {edges_csv}")
    print(f"  • World SDF: {sdf_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Convert an aligned grayscale-building map + color map to a UTM graph "
            "using the constrained-triangulation pipeline described in the manuscript."
        )
    )
    ap.add_argument("num_vehicles", type=int)
    ap.add_argument("num_vertiports", type=int)
    ap.add_argument("num_charging", type=int)
    ap.add_argument("num_suppliers", type=int)
    ap.add_argument("num_clients", type=int)

    ap.add_argument("--mask", "--gray", dest="mask_png", required=True)
    ap.add_argument("--color", dest="color_png", required=True)
    ap.add_argument("--out", dest="out_dir", default="gz_world_out")
    ap.add_argument("--res", dest="resolution", type=float, default=0.2)
    ap.add_argument("--seed", dest="seed", type=int, default=42)

    ap.add_argument("--z-special", dest="z_special", type=float, default=2.0)
    ap.add_argument("--z-vehicle", dest="z_vehicle", type=float, default=1.0)

    ap.add_argument(
        "--building-threshold",
        type=int,
        default=-1,
        help="-1 = Otsu automatic threshold (recommended).",
    )
    ap.add_argument("--mask-close-kernel", type=int, default=1)
    ap.add_argument("--mask-open-kernel", type=int, default=0)
    ap.add_argument("--min-building-area", type=int, default=800)
    ap.add_argument("--min-building-side", type=int, default=20)
    ap.add_argument("--max-building-side", type=int, default=300)
    ap.add_argument("--max-building-aspect", type=float, default=3.5)

    ap.add_argument(
        "--road-source",
        choices=["geometry", "palette", "auto", "complement"],
        default="geometry",
        help=(
            "geometry (recommended) infers street-like corridors from the "
            "gray/white building geometry and suppresses wide black regions "
            "such as sea/river/large parks; palette is only for legacy "
            "palette-coded color maps; complement treats every black pixel "
            "as traversable."
        ),
    )
    ap.add_argument(
        "--road-max-open-radius-m",
        type=float,
        default=12.0,
        help=(
            "geometry mode: free-space points farther than this from every "
            "building are treated as wide open non-road regions."
        ),
    )
    ap.add_argument(
        "--road-min-clearance-m",
        type=float,
        default=0.5,
        help="geometry mode: minimum clearance from building pixels.",
    )
    ap.add_argument("--special-road-search-m", type=float, default=8.0)
    ap.add_argument("--min-triangle-area-m2", type=float, default=25.0)
    ap.add_argument("--logical-rejection-attempts", type=int, default=5000)
    ap.add_argument("--logical-min-separation-m", type=float, default=1.0)
    ap.add_argument("--node-clearance-m", type=float, default=0.5)
    ap.add_argument("--edge-clearance-m", type=float, default=0.0)
    ap.add_argument("--vertical-clearance-m", type=float, default=1.0)
    ap.add_argument("--special-selection-restarts", type=int, default=100)
    ap.add_argument("--no-markers", dest="no_markers", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.num_vehicles,
        args.num_vertiports,
        args.num_charging,
        args.num_suppliers,
        args.num_clients,
        mask_png=args.mask_png,
        color_png=args.color_png,
        out_dir=args.out_dir,
        resolution_m_per_px=args.resolution,
        seed=args.seed,
        z_special=args.z_special,
        z_vehicle=args.z_vehicle,
        building_threshold=args.building_threshold,
        mask_close_kernel=args.mask_close_kernel,
        mask_open_kernel=args.mask_open_kernel,
        min_building_area=args.min_building_area,
        min_building_side=args.min_building_side,
        max_building_side=args.max_building_side,
        max_building_aspect=args.max_building_aspect,
        road_source=args.road_source,
        road_max_open_radius_m=args.road_max_open_radius_m,
        road_min_clearance_m=args.road_min_clearance_m,
        special_road_search_m=args.special_road_search_m,
        min_triangle_area_m2=args.min_triangle_area_m2,
        logical_rejection_attempts=args.logical_rejection_attempts,
        logical_min_separation_m=args.logical_min_separation_m,
        node_clearance_m=args.node_clearance_m,
        edge_clearance_m=args.edge_clearance_m,
        vertical_clearance_m=args.vertical_clearance_m,
        special_selection_restarts=args.special_selection_restarts,
        spawn_markers=(not args.no_markers),
    )
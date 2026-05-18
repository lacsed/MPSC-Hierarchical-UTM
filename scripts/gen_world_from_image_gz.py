#!/usr/bin/env python3

import os
import math
import shutil
import argparse
import logging

import cv2
import numpy as np
from PIL import Image

import city_gen as city
import graph2d as g2d


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gen_world_from_image_gz")

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


def main(
    num_vehicles,
    num_vertiports,
    num_charging,
    num_suppliers,
    num_clients,
    map_png="./assets/finalmap.png",
    out_dir="gz_world_out",
    resolution_m_per_px=0.2,
    seed=42,
    z_special=2.0,
    z_vehicle=1.0,
    max_candidates_per_special=40,
    max_deg_logical=4,
    spawn_markers=True,
    restarts=500,
    no_overflight=False,
):
    """Generate Gazebo world and a constrained 2D graph from a palette-coded map image."""
    logger.info(f"Starting world generation from: {map_png}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"K_SPECIAL = {g2d.K_SPECIAL}")
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(map_png, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {map_png}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    H, W = img.shape[:2]
    logger.info(f"Image dimensions: {W}x{H} pixels")

    label_map, min_dist = city.segment_by_palette(img)
    mask_building, mask_roads = city.build_masks(label_map, min_dist, building_tol=12.0, road_tol=18.0)
    mask_park = city.build_park_mask(label_map, min_dist, park_tol=18.0)
    cv2.imwrite(os.path.join(out_dir, "mask_park.png"), mask_park)
    cv2.imwrite(os.path.join(out_dir, "mask_building.png"), mask_building)
    cv2.imwrite(os.path.join(out_dir, "mask_roads.png"), mask_roads)

    park_models = city.plan_park_models(
        mask_park,
        W_px=W,
        H_px=H,
        resolution_m_per_px=resolution_m_per_px,
        seed=seed,
    )

    boxes = city.extract_building_boxes(
        img_bgr=img,
        building_tol=12.0,
        road_tol=18.0,
        min_area=800,
        min_side=20,
        max_side=300,
        max_aspect_ratio=3.5,
        max_road_fraction=0.02,
        debug_dir=out_dir,
    )

    logger.info(f"Detected {len(boxes)} building boxes")
    if len(boxes) == 0:
        raise RuntimeError("No building boxes detected (check palette tolerances / map colors).")

    total_special = num_vertiports + num_charging + num_suppliers + num_clients
    if total_special <= 0:
        raise RuntimeError("total_special == 0 (configure num_vertiports/charging/suppliers/clients).")
    if total_special > len(boxes):
        logger.warning(f"Requested {total_special} special buildings but only {len(boxes)} boxes exist. Capping.")
        total_special = len(boxes)

    centers = [city.box_center_px(b) for b in boxes]
    chosen = city.farthest_point_sampling(centers, total_special, seed=seed)

    roles_by_index = {}
    k = 0
    for _ in range(min(num_vertiports, total_special - k)):
        roles_by_index[chosen[k]] = "vertiport"
        k += 1
    for _ in range(min(num_charging, total_special - k)):
        roles_by_index[chosen[k]] = "charging"
        k += 1
    for _ in range(min(num_suppliers, total_special - k)):
        roles_by_index[chosen[k]] = "supplier"
        k += 1
    for _ in range(min(num_clients, total_special - k)):
        roles_by_index[chosen[k]] = "client"
        k += 1

    building_heights = city.sample_building_heights(
        len(boxes),
        seed=seed,
        random_height=True,
        special_indices=roles_by_index.keys(),
        base_min=8.0,
        base_max=25.0,
        special_min=26.0,
        special_max=35.0,
    )

    tallest = float(max(building_heights))
    z_logical_final = tallest + float(LOGICAL_ABOVE_TALLEST)
    allow_overflight = (not no_overflight) and (z_logical_final >= tallest + OVERFLIGHT_MARGIN_M)

    logger.info(
        f"Height stats: tallest={tallest:.2f}m | z_logical={z_logical_final:.2f}m | "
        f"z_special={z_special:.2f}m | z_vehicle={z_vehicle:.2f}m | overflight={'ON' if allow_overflight else 'OFF'}"
    )

    skel01 = g2d.skeletonize_roads(mask_roads)
    cv2.imwrite(os.path.join(out_dir, "roads_skeleton.png"), (skel01 * 255).astype(np.uint8))

    _adj_skel, pos_skel = g2d.build_skeleton_graph(skel01)
    if len(pos_skel) == 0:
        raise RuntimeError("Road skeleton graph is empty (check road mask / palette tolerances).")

    g2d.set_logical_globals(pos_skel, (W / 2.0, H / 2.0))

    candidate_logical_sids = [
        sid for sid, (px, py) in pos_skel.items()
        if not city.point_in_any_box((px, py), boxes, ignore_indices=set())
    ]
    if not candidate_logical_sids:
        raise RuntimeError("No candidate logical nodes outside buildings were found.")

    specials = []
    counters = {"vertiport": 0, "supplier": 0, "client": 0, "charging": 0}
    for bi, role in roles_by_index.items():
        cx, cy = centers[bi]
        idx = counters[role]
        counters[role] += 1
        s_id = f"{role.upper()}_{idx:03d}"
        specials.append({"id": s_id, "role": role, "bi": bi, "px": cx, "py": cy})

    special_ids = [s["id"] for s in specials]
    Ns = len(specials)
    logger.info(
        f"Special nodes: {Ns} (vertiports={num_vertiports}, charging={num_charging}, suppliers={num_suppliers}, clients={num_clients})"
    )

    allowed_by_special = g2d.build_visibility_candidates_for_specials(
        specials=specials,
        candidate_logical_sids=candidate_logical_sids,
        pos_skel=pos_skel,
        boxes=boxes,
        allow_overflight=allow_overflight,
    )
    for s in special_ids:
        if len(allowed_by_special.get(s, [])) < g2d.K_SPECIAL:
            raise RuntimeError(
                f"Special {s} has <{g2d.K_SPECIAL} visible logical candidates. "
                f"Try --max-candidates (e.g. 120) or adjust palette tolerances."
            )

    reserve_ll = 1
    if max_deg_logical <= 2:
        raise RuntimeError("max_deg_logical must be >= 3 to allow LL connectivity.")

    lower = max(2, (Ns + 1))
    max_M_default = min(len(candidate_logical_sids), max(lower, 4 * Ns))
    upper = 2 * max_M_default

    best_selected = None
    best_assignment = None
    best_ll_tree = None

    logger.info(f"Searching minimal logical nodes: M in [{lower}, {upper}] (deg_max_total={max_deg_logical})")

    for M in range(lower, upper + 1):
        ok_for_this_M = False
        for r in range(restarts):
            selected = g2d.greedy_select_logical_nodes_randomized(
                special_ids=special_ids,
                allowed_by_special=allowed_by_special,
                max_nodes=M,
                seed_local=seed + 1000 * M + r,
            )
            if selected is None:
                continue

            res = g2d.assign_special_edges_balanced(
                specials=specials,
                logical_sids_selected=selected,
                allowed_by_special_sid=allowed_by_special,
                max_deg_total=max_deg_logical,
                reserve_ll=reserve_ll,
                seed=seed + 777 * r,
                tries=250,
            )
            if res is None:
                continue

            assignment, deg_special_local = res

            ll_caps = [max_deg_logical - d for d in deg_special_local]
            ll_tree = g2d.build_ll_tree_from_caps(ll_caps, seed=seed + 999 * r)
            if ll_tree is None:
                continue

            best_selected = selected
            best_assignment = assignment
            best_ll_tree = ll_tree
            ok_for_this_M = True
            logger.info(f"Found feasible graph with M={M} logical nodes")
            break

        if ok_for_this_M:
            break

    if best_selected is None or best_assignment is None or best_ll_tree is None:
        mins = {s: len(allowed_by_special.get(s, [])) for s in special_ids}
        logger.error(f"Min candidates per special (smallest first): {sorted(mins.items(), key=lambda kv: kv[1])[:8]}")
        raise RuntimeError(
            "Failed to build a feasible graph under constraints. Try increasing --max-candidates or --max-deg-logical."
        )

    refined_logical_px = g2d.refine_logical_positions_only(
        specials=specials,
        selected_sids=best_selected,
        assignment_by_special=best_assignment,
        ll_tree_edges=best_ll_tree,
        boxes=boxes,
        W=W,
        H=H,
        resolution_m_per_px=resolution_m_per_px,
        seed=seed,
    )
    pos_skel_ref = dict(pos_skel)
    for sid in best_selected:
        pos_skel_ref[sid] = refined_logical_px[sid]

    dbg_path = os.path.join(out_dir, "graph_debug.png")
    g2d.save_graph_debug_png(
        out_path=dbg_path,
        map_bgr=img,
        boxes=boxes,
        special_nodes=specials,
        logical_sids_selected=best_selected,
        pos_skel=pos_skel_ref,
        assignment_sid_pairs=best_assignment,
        ll_tree_local=best_ll_tree,
    )

    role_bgr = {
        "vertiport": (0, 0, 255),
        "supplier": (0, 255, 0),
        "client": (0, 165, 255),
        "charging": (255, 80, 0),
    }
    img_special = img.copy()
    for _, x, y, w, h in boxes:
        cv2.rectangle(img_special, (int(x), int(y)), (int(x + w), int(y + h)), (70, 70, 70), 1)

    for sp in specials:
        cx, cy = int(sp["px"]), int(sp["py"])
        role = sp["role"]
        color = role_bgr.get(role, (0, 0, 255))
        cv2.circle(img_special, (cx, cy), 9, color, -1)

    special_nodes_path = os.path.join(out_dir, "special_nodes_colored.png")
    cv2.imwrite(special_nodes_path, img_special)
    print(f"Special nodes image saved: {special_nodes_path}")

    logger.info(f"Debug PNG saved: {dbg_path}")

    node_rows = []
    edge_rows = []

    num_logical_layers = logical_layer_count(num_vehicles)
    num_extra_logical_layers = num_logical_layers - 1
    logical_layer_z = [
        float(z_logical_final) + layer_idx * float(LOGICAL_LAYER_STEP_M)
        for layer_idx in range(num_logical_layers)
    ]
    num_base_logical_nodes = len(best_selected)

    logger.info(
        f"Logical layers: total={num_logical_layers} "
        f"(base + {num_extra_logical_layers} extra), "
        f"step={LOGICAL_LAYER_STEP_M:.2f}m, "
        f"z_min={logical_layer_z[0]:.2f}m, z_max={logical_layer_z[-1]:.2f}m"
    )

    # Logical-node IDs are globally sequential across all layers:
    # layer 0 -> LOGICAL_000 ... LOGICAL_{M-1};
    # layer 1 -> LOGICAL_M ... LOGICAL_{2M-1}; etc.
    # Upper layers replicate the same x,y coordinates and increase z by 5 m per layer.
    logical_nodes_world = []
    logical_layer_id_by_sid = {layer_idx: {} for layer_idx in range(num_logical_layers)}

    for i, sid in enumerate(best_selected):
        px, py = pos_skel_ref[sid]
        xw, yw = city.px_to_world(px, py, W, H, resolution_m_per_px)
        logical_nodes_world.append((sid, i, xw, yw))

        for layer_idx, z_layer in enumerate(logical_layer_z):
            nid = logical_node_id(layer_idx, i, num_base_logical_nodes)
            node_rows.append((nid, "logical", xw, yw, float(z_layer)))
            logical_layer_id_by_sid[layer_idx][sid] = nid

    sid_to_logical_id = logical_layer_id_by_sid[0]

    vertiports_world = []
    for sp in specials:
        bi = sp["bi"]
        xw, yw = city.px_to_world(sp["px"], sp["py"], W, H, resolution_m_per_px)
        roof_z = float(building_heights[bi])
        sp_z = roof_z + float(z_special)
        node_rows.append((sp["id"], sp["role"], xw, yw, sp_z))
        if sp["role"] == "vertiport":
            vertiports_world.append({"id": sp["id"], "x": xw, "y": yw, "z": sp_z})

    # Special nodes connect only to the base logical layer.
    for sp in specials:
        s_id = sp["id"]
        for sid_k in best_assignment[s_id]:
            lk = sid_to_logical_id[sid_k]
            edge_rows.append((s_id, lk))

    # Each logical layer reproduces the same logical-logical topology.
    for layer_idx in range(num_logical_layers):
        for (i, j) in best_ll_tree:
            edge_rows.append((logical_node_id(layer_idx, i, num_base_logical_nodes), logical_node_id(layer_idx, j, num_base_logical_nodes)))

    # Upper logical nodes connect only to their corresponding node in the layer below.
    # This creates vertical inter-layer channels with identical x,y and z separated by 5 m.
    for layer_idx in range(1, num_logical_layers):
        for i, _sid in enumerate(best_selected):
            edge_rows.append((logical_node_id(layer_idx, i, num_base_logical_nodes), logical_node_id(layer_idx - 1, i, num_base_logical_nodes)))

    vehicles = city.distribute_vehicles_over_vertiports(
        num_vehicles, vertiports_world, z_vehicle_above_vertiport=float(z_vehicle)
    )
    for v in vehicles:
        node_rows.append((v["id"], "vehicle", v["x"], v["y"], float(v["z"])))

    nodes_csv = os.path.join(out_dir, "graph_nodes.csv")
    edges_csv = os.path.join(out_dir, "graph_edges.csv")
    write_nodes_csv(nodes_csv, node_rows)
    write_edges_csv(edges_csv, edge_rows)
    logger.info(f"Wrote nodes: {nodes_csv} (N={len(node_rows)})")
    logger.info(f"Wrote edges: {edges_csv} (E={len(edge_rows)})")

    hm_raw = np.zeros((H, W), dtype=np.uint8)
    hm_fixed = city.pad_to_square_pow2p1(hm_raw)
    heightmap_path = os.path.join(out_dir, "heightmap.png")
    Image.fromarray(hm_fixed, mode="L").save(heightmap_path)

    texture_root_path = os.path.join(out_dir, "finalmap.png")
    Image.open(map_png).save(texture_root_path)

    city.write_ogre_material(out_dir=out_dir, texture_filename="finalmap.png")
    textures_dir = os.path.join(out_dir, "materials", "textures")
    os.makedirs(textures_dir, exist_ok=True)
    shutil.copyfile(texture_root_path, os.path.join(textures_dir, "finalmap.png"))

    city.write_stub_model_config(os.path.join(out_dir, "materials"), model_name="utm_materials_stub")

    node_pos = {}

    for i, sid in enumerate(best_selected):
        px, py = pos_skel_ref[sid]
        xw, yw = city.px_to_world(px, py, W, H, resolution_m_per_px)
        for layer_idx, z_layer in enumerate(logical_layer_z):
            nid = logical_node_id(layer_idx, i, num_base_logical_nodes)
            node_pos[nid] = (float(xw), float(yw), float(z_layer))

    for sp in specials:
        bi = sp["bi"]
        xw, yw = city.px_to_world(sp["px"], sp["py"], W, H, resolution_m_per_px)
        roof_z = float(building_heights[bi])
        sp_z = roof_z + float(z_special)
        node_pos[sp["id"]] = (float(xw), float(yw), float(sp_z))

    custom_models = []

    if spawn_markers:
        edge_id = 0
        for (a, b) in edge_rows:
            if a not in node_pos or b not in node_pos:
                continue

            x1, y1, z1 = node_pos[a]
            x2, y2, z2 = node_pos[b]

            custom_models.append(
                make_channel_sdf(
                    name=f"EDGE_{edge_id:05d}",
                    x1=x1,
                    y1=y1,
                    z1=z1,
                    x2=x2,
                    y2=y2,
                    z2=z2,
                    radius=CHANNEL_RADIUS,
                    alpha=CHANNEL_ALPHA,
                )
            )
            edge_id += 1

        for i, v in enumerate(vehicles):
            color = UAV_COLORS_RGBA[i % len(UAV_COLORS_RGBA)]

            custom_models.append(
                make_uav_sdf(
                    vehicle_id=v["id"],
                    x=v["x"],
                    y=v["y"],
                    z=float(v["z"]),
                    rgba=color,
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
        seed=seed,

        # Não cria esferas amarelas dos nós lógicos.
        logical_markers=None,

        # Não cria veículos padrão, porque vamos criar modelos coloridos
        # com os nomes corretos: VEHICLE_000, VEHICLE_001, etc.
        vehicle_markers=None,

        park_models=park_models,
        vehicle_model_uri="model://quadrotor",

        # Não cria canais padrão. Vamos injetar canais transparentes.
        edge_markers=None,
    )

    sdf = inject_models_before_world_close(sdf, custom_models)

    sdf_path = os.path.join(out_dir, "utm_world.sdf")
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf)

    print("\n[SUCCESS] World + constrained graph generated")
    print(f"  • Output dir: {os.path.abspath(out_dir)}")
    print(f"  • Buildings: {len(boxes)} (special={len(roles_by_index)})")
    print(f"  • Tallest building: {tallest:.2f} m")
    print(f"  • Logical base Z: {logical_layer_z[0]:.2f} m (tallest + {LOGICAL_ABOVE_TALLEST})")
    print(f"  • Logical layers: {num_logical_layers} total = 1 base + {num_extra_logical_layers} extra")
    print(f"  • Logical layer vertical step: {LOGICAL_LAYER_STEP_M:.2f} m")
    print(f"  • Logical top Z: {logical_layer_z[-1]:.2f} m")
    print(f"  • Logical node spheres in Gazebo: OFF")
    print(f"  • UAV model names preserved: VEHICLE_000, VEHICLE_001, ...")
    print(f"  • UAV scale: {UAV_SCALE}x")
    print(f"  • Channel alpha: {CHANNEL_ALPHA}")
    print(f"  • Nodes CSV: {nodes_csv}")
    print(f"  • Edges CSV: {edges_csv}")
    print(f"  • Debug PNG: {dbg_path}")
    print(f"  • World SDF: {sdf_path}")
    print("\n[RUN] (from out_dir)")
    print(f"  cd {os.path.abspath(out_dir)}")
    print("  export GAZEBO_RESOURCE_PATH=$PWD:$GAZEBO_RESOURCE_PATH")
    print("  gazebo --verbose utm_world.sdf")


def parse_args():
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("num_vehicles", type=int)
    ap.add_argument("num_vertiports", type=int)
    ap.add_argument("num_charging", type=int)
    ap.add_argument("num_suppliers", type=int)
    ap.add_argument("num_clients", type=int)

    ap.add_argument("--map", dest="map_png", default="./assets/finalmap.png")
    ap.add_argument("--out", dest="out_dir", default="gz_world_out")
    ap.add_argument("--res", dest="resolution", type=float, default=0.2)
    ap.add_argument("--seed", dest="seed", type=int, default=42)

    ap.add_argument("--z-special", dest="z_special", type=float, default=2.0)
    ap.add_argument("--z-vehicle", dest="z_vehicle", type=float, default=1.0)

    ap.add_argument("--max-candidates", dest="max_candidates", type=int, default=40)
    ap.add_argument("--max-deg-logical", dest="max_deg_logical", type=int, default=4)
    ap.add_argument("--restarts", dest="restarts", type=int, default=500)
    ap.add_argument("--no-markers", dest="no_markers", action="store_true")
    ap.add_argument("--no-overflight", dest="no_overflight", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.num_vehicles,
        args.num_vertiports,
        args.num_charging,
        args.num_suppliers,
        args.num_clients,
        map_png=args.map_png,
        out_dir=args.out_dir,
        resolution_m_per_px=args.resolution,
        seed=args.seed,
        z_special=args.z_special,
        z_vehicle=args.z_vehicle,
        max_candidates_per_special=args.max_candidates,
        max_deg_logical=args.max_deg_logical,
        spawn_markers=(not args.no_markers),
        restarts=args.restarts,
        no_overflight=args.no_overflight,
    )
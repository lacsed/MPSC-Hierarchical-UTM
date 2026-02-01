#!/usr/bin/env python3

import os
import math
import random
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
    cv2.imwrite(os.path.join(out_dir, "mask_building.png"), mask_building)
    cv2.imwrite(os.path.join(out_dir, "mask_roads.png"), mask_roads)

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

    building_heights = city.sample_building_heights(len(boxes), seed=seed, random_height=True, height_fixed=20.0)
    tallest = float(max(building_heights))
    z_logical_final = tallest + float(LOGICAL_ABOVE_TALLEST)
    allow_overflight = (not no_overflight) and (z_logical_final >= tallest + OVERFLIGHT_MARGIN_M)

    logger.info(
        f"Height stats: tallest={tallest:.2f}m | z_logical={z_logical_final:.2f}m | "
        f"z_special={z_special:.2f}m | z_vehicle={z_vehicle:.2f}m | overflight={'ON' if allow_overflight else 'OFF'}"
    )

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
        max_candidates_per_special=max_candidates_per_special,
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

    E_special = g2d.K_SPECIAL * Ns

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
    logger.info(f"Debug PNG saved: {dbg_path}")

    node_rows = []
    edge_rows = []

    logical_nodes_world = []
    for i, sid in enumerate(best_selected):
        px, py = pos_skel_ref[sid]
        xw, yw = city.px_to_world(px, py, W, H, resolution_m_per_px)
        nid = f"LOGICAL_{i:03d}"
        node_rows.append((nid, "logical", xw, yw, float(z_logical_final)))
        logical_nodes_world.append((sid, nid, xw, yw))

    sid_to_logical_id = {sid: nid for sid, nid, _, _ in logical_nodes_world}

    vertiports_world = []
    for sp in specials:
        bi = sp["bi"]
        xw, yw = city.px_to_world(sp["px"], sp["py"], W, H, resolution_m_per_px)
        roof_z = float(building_heights[bi])
        sp_z = roof_z + float(z_special)
        node_rows.append((sp["id"], sp["role"], xw, yw, sp_z))
        if sp["role"] == "vertiport":
            vertiports_world.append({"id": sp["id"], "x": xw, "y": yw, "z": sp_z})

    for sp in specials:
        s_id = sp["id"]
        for sid_k in best_assignment[s_id]:
            lk = sid_to_logical_id[sid_k]
            edge_rows.append((s_id, lk))

    for (i, j) in best_ll_tree:
        edge_rows.append((f"LOGICAL_{i:03d}", f"LOGICAL_{j:03d}"))

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

    logical_markers = []
    vehicle_markers = []
    if spawn_markers:
        for _sid, nid, xw, yw in logical_nodes_world:
            logical_markers.append((nid, xw, yw, float(z_logical_final)))
        for v in vehicles:
            vehicle_markers.append((v["id"], v["x"], v["y"], float(v["z"])))

    sdf = city.make_world_sdf(
        W_px=W,
        H_px=H,
        boxes=boxes,
        roles_by_index=roles_by_index,
        building_heights=building_heights,
        resolution_m_per_px=resolution_m_per_px,
        seed=seed,
        logical_markers=logical_markers if spawn_markers else None,
        vehicle_markers=vehicle_markers if spawn_markers else None,
    )
    sdf_path = os.path.join(out_dir, "utm_world.sdf")
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf)

    print("\n[SUCCESS] World + constrained graph generated")
    print(f"  • Output dir: {os.path.abspath(out_dir)}")
    print(f"  • Buildings: {len(boxes)} (special={len(roles_by_index)})")
    print(f"  • Tallest building: {tallest:.2f} m")
    print(f"  • Logical nodes Z: {z_logical_final:.2f} m (tallest + {LOGICAL_ABOVE_TALLEST})")
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

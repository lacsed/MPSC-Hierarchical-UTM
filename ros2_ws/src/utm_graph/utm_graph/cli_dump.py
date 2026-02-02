from __future__ import annotations

import argparse

from .loader import load_graph_data, validate_world_bounds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True, help="Path to graph_nodes.csv")
    p.add_argument("--edges", required=True, help="Path to graph_edges.csv")
    p.add_argument("--check-bounds", action="store_true", help="Heuristic bounds check (ground size ~102.6).")
    args = p.parse_args()

    gd = load_graph_data(args.nodes, args.edges, add_euclidean_weight=True)

    print("=== Graph summary ===")
    print(f"Navigable Nodes:   {len(gd.nodes)}")
    print(f"Spawns:      {len(gd.spawns)}")
    print(f"Edges:       {gd.graph.number_of_edges()} (directed, includes both directions)")


    if args.check_bounds:
        ok = validate_world_bounds(gd, world_size_x=102.6, world_size_y=102.6, warn_only=True)
        print(f"Bounds check: {'OK' if ok else 'WARNINGS'}")

    print("\n=== Vehicle spawns ===")
    for s in gd.spawns:
        print(f"{s.vehicle_id}: ({s.x:.3f}, {s.y:.3f}, {s.z:.3f}) yaw={s.yaw:.3f}")


if __name__ == "__main__":
    main()

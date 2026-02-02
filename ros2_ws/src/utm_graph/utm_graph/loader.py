from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple

import networkx as nx

from .node_types import NodeType, DES_NODE_TYPES, SPAWN_TYPES
from .models import NodeRecord, SpawnRecord, GraphData


def _read_nodes_csv(nodes_csv: str) -> List[NodeRecord]:
    if not os.path.exists(nodes_csv):
        raise FileNotFoundError(f"graph_nodes.csv not found: {nodes_csv}")

    nodes: List[NodeRecord] = []
    with open(nodes_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"id", "type", "x", "y", "z"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid nodes CSV header. Expected at least {expected}, got {reader.fieldnames}"
            )

        for row in reader:
            node_id = (row.get("id") or "").strip()
            node_type_raw = (row.get("type") or "").strip()
            if not node_id or not node_type_raw:
                continue

            nt = NodeType.parse(node_type_raw)

            try:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except Exception as e:
                raise ValueError(f"Invalid numeric coordinate for node '{node_id}': {e}")

            nodes.append(NodeRecord(node_id=node_id, node_type=nt, x=x, y=y, z=z))

    if not nodes:
        raise ValueError(f"No nodes parsed from {nodes_csv}")

    return nodes


def _read_edges_csv(edges_csv: str) -> List[Tuple[str, str]]:
    if not os.path.exists(edges_csv):
        raise FileNotFoundError(f"graph_edges.csv not found: {edges_csv}")

    edges: List[Tuple[str, str]] = []
    with open(edges_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"src", "dst"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid edges CSV header. Expected at least {expected}, got {reader.fieldnames}"
            )

        for row in reader:
            src = (row.get("src") or "").strip()
            dst = (row.get("dst") or "").strip()
            if not src or not dst:
                continue
            if src == dst:
                continue
            edges.append((src, dst))

    if not edges:
        raise ValueError(f"No edges parsed from {edges_csv}")

    return edges


def load_graph_data(
    nodes_csv: str,
    edges_csv: str,
    *,
    add_euclidean_weight: bool = True,
) -> GraphData:
    """
    Main loader:
    - Reads graph_nodes.csv and graph_edges.csv
    - Splits into DES nodes and vehicle spawns
    - Builds directed MultiDiGraph where each (src,dst) becomes two directed edges
    - Adds 'weight' as Euclidean distance (optional)
    """

    all_nodes = _read_nodes_csv(nodes_csv)
    all_edges = _read_edges_csv(edges_csv)

    des_nodes: Dict[str, NodeRecord] = {}
    spawns: List[SpawnRecord] = []

    # split nodes
    for n in all_nodes:
        if n.node_type in SPAWN_TYPES:
            spawns.append(SpawnRecord(vehicle_id=n.node_id, x=n.x, y=n.y, z=n.z, yaw=0.0))
        elif n.node_type in DES_NODE_TYPES:
            # keep last occurrence if duplicated (but generator should avoid duplicates)
            des_nodes[n.node_id] = n

    if not des_nodes:
        raise ValueError("No DES nodes found (logical/supplier/client/charging/vertiport).")

    positions: Dict[str, Tuple[float, float, float]] = {
        nid: (nr.x, nr.y, nr.z) for nid, nr in des_nodes.items()
    }

    G = nx.MultiDiGraph()

    # add nodes with attributes
    for nid, nr in des_nodes.items():
        G.add_node(
            nid,
            type=nr.node_type.value,
            x=nr.x,
            y=nr.y,
            z=nr.z,
        )

    # add edges as bidirectional
    skipped = 0
    for src, dst in all_edges:
        if src not in des_nodes or dst not in des_nodes:
            skipped += 1
            continue

        w = 1.0
        if add_euclidean_weight:
            x1, y1, _ = positions[src]
            x2, y2, _ = positions[dst]
            w = math.hypot(x2 - x1, y2 - y1)

        G.add_edge(src, dst, weight=w)
        G.add_edge(dst, src, weight=w)

    if G.number_of_edges() == 0:
        raise ValueError("No usable edges added. Check if edges reference correct node ids.")

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} edges referencing non-DES nodes (e.g., vehicles).")

    return GraphData(graph=G, nodes=des_nodes, spawns=spawns, positions=positions)


def validate_world_bounds(
    graph_data: GraphData,
    *,
    world_size_x: float = 102.6,
    world_size_y: float = 102.6,
    warn_only: bool = True,
) -> bool:
    """
    Heuristic bounds check:
    If your ground plane/heightmap size is ~102.6 x 102.6 and centered at (0,0),
    typical valid region is x,y in [-51.3, 51.3].

    This is only to catch coordinate-transform mistakes.
    """
    half_x = world_size_x / 2.0
    half_y = world_size_y / 2.0

    ok = True
    for nid, (x, y, _z) in graph_data.positions.items():
        if not (-half_x <= x <= half_x and -half_y <= y <= half_y):
            ok = False
            msg = (
                f"[WARN] Node '{nid}' out of expected bounds: "
                f"x={x:.3f}, y={y:.3f} (expected about [-{half_x:.1f},{half_x:.1f}] / [-{half_y:.1f},{half_y:.1f}])"
            )
            if warn_only:
                print(msg)
            else:
                raise ValueError(msg)

    return ok

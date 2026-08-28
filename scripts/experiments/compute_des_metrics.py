#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


# -----------------------------------------------------------------------------
# Path handling
# -----------------------------------------------------------------------------


def project_root() -> Path:
    env_root = os.environ.get("MPSC_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.home() / "MPSC-Hierarchical-UTM"


def prepare_python_path() -> None:
    root = project_root()
    candidates = [
        root / "ros2_ws" / "src" / "utm_fleet",
        root / "ros2_ws" / "src" / "utm_graph",
        root / "ros2_ws" / "install" / "utm_fleet" / "lib" / "python3.8" / "site-packages",
        root / "ros2_ws" / "install" / "utm_graph" / "lib" / "python3.8" / "site-packages",
    ]
    for path in candidates:
        if path.exists():
            sp = str(path)
            if sp not in sys.path:
                sys.path.insert(0, sp)


def default_run_dir() -> Path:
    run_dir = os.environ.get("UTM_RUN_DIR", "").strip()
    if run_dir:
        return Path(run_dir).expanduser().resolve()

    run_id = os.environ.get("UTM_RUN_ID", "").strip()
    if not run_id:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    return Path.home() / "utm_runs" / run_id


# -----------------------------------------------------------------------------
# UltraDES transition handling
# -----------------------------------------------------------------------------


def iter_transition_triples(A: Any) -> Iterator[Tuple[Any, Any, Any]]:
    """
    Return transitions as (source, event, target) triples.

    Important: the installed UltraDES version used in this project is not fully
    consistent with the public wiki. In this installation, dfa(...) receives an
    iterable of triples, while transitions(A) may be exposed either as triples
    or as a nested dictionary depending on the wrapper/version. This helper
    accepts both representations.
    """
    from ultrades.automata import transitions

    T = transitions(A)

    # Wiki/documentation-style representation: {q: {e: nq}}
    if hasattr(T, "items"):
        for q, out in T.items():
            if out is None:
                continue
            if hasattr(out, "items"):
                for e, nq in out.items():
                    yield q, e, nq
            else:
                for item in out:
                    if len(item) == 2:
                        e, nq = item
                        yield q, e, nq
                    elif len(item) == 3:
                        q2, e, nq = item
                        yield q2, e, nq
                    else:
                        raise TypeError(f"Unsupported transition item: {item!r}")
        return

    # Installed UltraDES-style representation: iterable of triples.
    for item in T:
        try:
            q, e, nq = item[0], item[1], item[2]
        except Exception as exc:
            raise TypeError(
                "Unsupported transitions(A) representation. Expected either "
                "{state: {event: state}} or an iterable of transition triples."
            ) from exc
        yield q, e, nq


def add_transition(
    transition_list: List[Tuple[Any, Any, Any]],
    q: Any,
    e: Any,
    nq: Any,
) -> None:
    """Add one transition using the installed UltraDES triple-list format."""
    transition_list.append((q, e, nq))


def make_dfa_from_triples(
    triples: Iterable[Tuple[Any, Any, Any]],
    initial: Any,
    name: str,
):
    """Build a DFA using the installed UltraDES transition-triple API."""
    from ultrades.automata import dfa

    return dfa(list(triples), initial, name)


# -----------------------------------------------------------------------------
# Generic counters and CSV writer
# -----------------------------------------------------------------------------


def count_automaton(A: Any) -> Tuple[int, int]:
    from ultrades.automata import states

    q_count = len(list(states(A)))
    tr_count = sum(1 for _ in iter_transition_triples(A))
    return q_count, tr_count


def count_collection(automata: Iterable[Any]) -> Tuple[int, int]:
    q_total = 0
    tr_total = 0
    for A in automata:
        q, tr = count_automaton(A)
        q_total += q
        tr_total += tr
    return q_total, tr_total


def write_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Shared event registry and automaton cloning
# -----------------------------------------------------------------------------


class EventRegistry:
    def __init__(self):
        self._events: Dict[str, Any] = {}
        self._ctrl: Dict[str, bool] = {}

    def get(self, name: str, controllable: bool = True):
        from ultrades.automata import event

        name = str(name)
        ctrl = bool(controllable)

        if name in self._events:
            if self._ctrl[name] != ctrl:
                raise ValueError(
                    f"Event controllability conflict for '{name}': "
                    f"existing={self._ctrl[name]}, new={ctrl}"
                )
            return self._events[name]

        ev = event(name, controllable=ctrl)
        self._events[name] = ev
        self._ctrl[name] = ctrl
        return ev

    def __contains__(self, name: str) -> bool:
        return str(name) in self._events

    def __len__(self) -> int:
        return len(self._events)


def clone_automaton_for_agent(A: Any, agent_id: int, registry: EventRegistry, name: str):
    from ultrades.automata import initial_state, is_controllable, is_marked, state, states

    aid = int(agent_id)
    state_map: Dict[Any, Any] = {}

    def mapped_state(q: Any) -> Any:
        if q not in state_map:
            q_name = f"A{aid}::{str(q)}"
            state_map[q] = state(q_name, marked=is_marked(q))
        return state_map[q]

    for q in states(A):
        mapped_state(q)

    T: List[Tuple[Any, Any, Any]] = []
    for q, e, nq in iter_transition_triples(A):
        ev_name = f"{str(e)}_{aid}"
        ev_obj = registry.get(ev_name, controllable=is_controllable(e))
        add_transition(T, mapped_state(q), ev_obj, mapped_state(nq))

    q0 = mapped_state(initial_state(A))

    return make_dfa_from_triples(T, q0, name)


def clone_automata_for_agent(
    automata: Iterable[Any],
    agent_id: int,
    registry: EventRegistry,
    prefix: str,
) -> List[Any]:
    cloned = []
    for i, A in enumerate(automata):
        cloned.append(
            clone_automaton_for_agent(
                A,
                agent_id=agent_id,
                registry=registry,
                name=f"{prefix}_a{agent_id}_{i}",
            )
        )
    return cloned


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------


def kind_static(node_id: str, G: Any) -> str:
    node_id = str(node_id)
    s = node_id.upper()
    t = ""

    try:
        nd = G.nodes[node_id]
        t = str(nd.get("type", nd.get("tipo", nd.get("kind", "")))).upper()
    except Exception:
        t = ""

    if "VERTIPORT" in s or "VERTIPORT" in t:
        return "VERTIPORT"
    if any(x in s or x in t for x in ("STATION", "ESTACAO", "CHARG", "CHARGING")):
        return "STATION"
    if any(x in s or x in t for x in ("SUPPLIER", "FORNECEDOR")):
        return "SUPPLIER"
    if any(x in s or x in t for x in ("CLIENT", "CLIENTE")):
        return "CLIENT"
    return "LOGICAL"


def first_vertiport(G: Any) -> str:
    for n in G.nodes():
        if kind_static(str(n), G) == "VERTIPORT":
            return str(n)
    return str(next(iter(G.nodes())))


def is_mutex_vertex(node_id: str, G: Any, mode: str) -> bool:
    mode = str(mode or "all_except_vertiport").strip().lower()
    k = kind_static(str(node_id), G)

    if k == "VERTIPORT":
        return False
    if mode == "none":
        return False
    if mode == "logical":
        return k == "LOGICAL"
    if mode == "all_except_vertiport":
        return True
    if mode == "all":
        return True
    return True


def is_blockable_vertex(node_id: str, G: Any) -> bool:
    return kind_static(str(node_id), G) != "VERTIPORT"


# -----------------------------------------------------------------------------
# Exact centralized UTM constraints with agent-indexed events
# -----------------------------------------------------------------------------


def build_centralized_utm_command_plant(G: Any, registry: EventRegistry):
    from ultrades.automata import state

    ready = state("CENTRAL_UTM_BLOCK_COMMAND_READY", marked=True)
    T: List[Tuple[Any, Any, Any]] = []

    for n in G.nodes():
        n = str(n)
        add_transition(T, ready, registry.get(f"block::{n}", controllable=True), ready)
        add_transition(T, ready, registry.get(f"unblock::{n}", controllable=True), ready)

    return make_dfa_from_triples(T, ready, "central_utm_block_command_plant")


def build_centralized_vertex_block_specs(
    G: Any,
    num_uavs: int,
    registry: EventRegistry,
) -> List[Any]:
    from ultrades.automata import accessible, state

    specs = []
    N = int(num_uavs)

    for v in G.nodes():
        v = str(v)
        if not is_blockable_vertex(v, G):
            continue

        unblocked = state(f"CENTRAL_UNBLOCKED::{v}", marked=True)
        blocked = state(f"CENTRAL_BLOCKED::{v}")
        T: List[Tuple[Any, Any, Any]] = []

        add_transition(T, unblocked, registry.get(f"block::{v}", controllable=True), blocked)
        add_transition(T, blocked, registry.get(f"unblock::{v}", controllable=True), unblocked)

        for aid in range(N):
            for u in set(str(x) for x in G.predecessors(v)):
                ev_name = f"edge_take::{u}::{v}_{aid}"
                add_transition(T, unblocked, registry.get(ev_name, controllable=True), unblocked)

        specs.append(accessible(make_dfa_from_triples(T, unblocked, f"central_utm_block_spec::{v}")))

    return specs


def build_centralized_global_block_spec(G: Any, registry: EventRegistry):
    from ultrades.automata import state

    free = state("CENTRAL_UTM_GLOBAL_UNBLOCKED", marked=True)
    blocked = state("CENTRAL_UTM_GLOBAL_BLOCKED")
    T: List[Tuple[Any, Any, Any]] = []

    for n in G.nodes():
        n = str(n)
        if not is_blockable_vertex(n, G):
            continue
        add_transition(T, free, registry.get(f"block::{n}", controllable=True), blocked)
        add_transition(T, blocked, registry.get(f"unblock::{n}", controllable=True), free)

    return make_dfa_from_triples(T, free, "central_utm_global_block_spec")


def build_centralized_vertex_mutex_specs(
    G: Any,
    num_uavs: int,
    registry: EventRegistry,
    mutex_vertices: str,
) -> List[Any]:
    from ultrades.automata import accessible, state

    specs = []
    N = int(num_uavs)

    for v in G.nodes():
        v = str(v)
        if not is_mutex_vertex(v, G, mutex_vertices):
            continue

        free = state(f"CENTRAL_VERTEX_FREE::{v}", marked=True)
        occ = {aid: state(f"CENTRAL_VERTEX_OCC::{v}::A{aid}") for aid in range(N)}
        T: List[Tuple[Any, Any, Any]] = []

        for aid in range(N):
            for u in set(str(x) for x in G.predecessors(v)):
                ev_name = f"edge_take::{u}::{v}_{aid}"
                add_transition(T, free, registry.get(ev_name, controllable=True), occ[aid])

            for w in set(str(x) for x in G.successors(v)):
                ev_name = f"edge_take::{v}::{w}_{aid}"
                add_transition(T, occ[aid], registry.get(ev_name, controllable=True), free)

        if T:
            specs.append(accessible(make_dfa_from_triples(T, free, f"central_utm_vertex_mutex::{v}")))

    return specs


def build_exact_centralized_automata(
    uav_model: Any,
    utm_model: Any,
    num_uavs: int,
    mutex_vertices: str,
) -> Tuple[List[Any], List[Any], EventRegistry]:
    registry = EventRegistry()
    plants: List[Any] = []
    specs: List[Any] = []

    for aid in range(int(num_uavs)):
        plants.extend(
            clone_automata_for_agent(
                uav_model.plants,
                agent_id=aid,
                registry=registry,
                prefix="central_uav_plant",
            )
        )
        specs.extend(
            clone_automata_for_agent(
                uav_model.specs,
                agent_id=aid,
                registry=registry,
                prefix="central_uav_spec",
            )
        )

    G = utm_model.G

    plants.append(build_centralized_utm_command_plant(G, registry))
    specs.extend(build_centralized_vertex_block_specs(G, int(num_uavs), registry))
    specs.append(build_centralized_global_block_spec(G, registry))
    specs.extend(
        build_centralized_vertex_mutex_specs(
            G,
            int(num_uavs),
            registry,
            mutex_vertices=mutex_vertices,
        )
    )

    return plants, specs, registry


# -----------------------------------------------------------------------------
# Metric computation
# -----------------------------------------------------------------------------


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf")
    return float(numerator) / float(denominator)


def safe_saving(reference: float, proposed: float) -> float:
    """
    Percentage reduction of proposed relative to reference.
    Positive value means proposed is smaller than reference.
    """
    if reference == 0:
        return 0.0
    return 100.0 * (float(reference) - float(proposed)) / float(reference)


def compute_metrics(
    nodes_csv: str,
    edges_csv: str,
    init_node: str,
    num_uavs: int,
    mutex_vertices: str,
) -> Dict[str, Any]:
    prepare_python_path()

    from ultrades.automata import monolithic_supervisor
    from utm_fleet.GenericUAVModel import GenericUAVModel
    from utm_fleet.utm_supervisor import GenericUTMModel

    t_all0 = time.perf_counter()

    uav_model = GenericUAVModel(nodes_csv, edges_csv, init_node)
    q_uav, tr_uav = count_automaton(uav_model.supervisor_mono)
    q_uav_plants, tr_uav_plants = count_collection(uav_model.plants)
    q_uav_specs, tr_uav_specs = count_collection(uav_model.specs)

    utm_model = GenericUTMModel(nodes_csv, edges_csv, init_node)
    q_utm, tr_utm = count_automaton(utm_model.supervisor_mono)
    q_utm_plants, tr_utm_plants = count_collection(utm_model.plants)
    q_utm_specs, tr_utm_specs = count_collection(utm_model.specs)

    N = int(num_uavs)

    # Hierarchical structural size: one generic UAV supervisor instantiated N times
    # plus one UTM supervisor. This is the structural DES size actually manipulated
    # by the hierarchical architecture, not the full synchronous product.
    hierarchical_states = N * q_uav + q_utm
    hierarchical_transitions = N * tr_uav + tr_utm

    t_build0 = time.perf_counter()
    centralized_plants, centralized_specs, registry = build_exact_centralized_automata(
        uav_model=uav_model,
        utm_model=utm_model,
        num_uavs=N,
        mutex_vertices=mutex_vertices,
    )
    t_build1 = time.perf_counter()

    q_cent_plants_sum, tr_cent_plants_sum = count_collection(centralized_plants)
    q_cent_specs_sum, tr_cent_specs_sum = count_collection(centralized_specs)

    t_synth0 = time.perf_counter()
    centralized_supervisor = monolithic_supervisor(centralized_plants, centralized_specs)
    t_synth1 = time.perf_counter()

    q_cent, tr_cent = count_automaton(centralized_supervisor)

    t_all1 = time.perf_counter()

    state_expansion_factor = safe_ratio(q_cent, hierarchical_states)
    transition_expansion_factor = safe_ratio(tr_cent, hierarchical_transitions)
    hierarchical_state_saving_pct = safe_saving(q_cent, hierarchical_states)
    hierarchical_transition_saving_pct = safe_saving(tr_cent, hierarchical_transitions)

    return {
        "t_wall": time.time(),
        "run_id": os.environ.get("UTM_RUN_ID", ""),
        "scenario_id": os.environ.get("UTM_SCENARIO", "default"),
        "baseline": os.environ.get("UTM_BASELINE", "proposed"),
        "graph_size": os.environ.get("UTM_GRAPH_SIZE", "current"),
        "density": os.environ.get("UTM_DENSITY", "medium"),
        "seed": os.environ.get("UTM_SEED", ""),
        "num_uavs": N,
        "num_nodes": len(list(uav_model.G.nodes())),
        "num_edges_bidirected": len(list(uav_model.G.edges())),
        "init_node": str(init_node),
        "mutex_vertices": str(mutex_vertices),
        "uav_plants": len(uav_model.plants),
        "uav_specs": len(uav_model.specs),
        "uav_atomic_plant_states_sum": q_uav_plants,
        "uav_atomic_plant_transitions_sum": tr_uav_plants,
        "uav_atomic_spec_states_sum": q_uav_specs,
        "uav_atomic_spec_transitions_sum": tr_uav_specs,
        "uav_supervisor_states": q_uav,
        "uav_supervisor_transitions": tr_uav,
        "utm_plants": len(utm_model.plants),
        "utm_specs": len(utm_model.specs),
        "utm_atomic_plant_states_sum": q_utm_plants,
        "utm_atomic_plant_transitions_sum": tr_utm_plants,
        "utm_atomic_spec_states_sum": q_utm_specs,
        "utm_atomic_spec_transitions_sum": tr_utm_specs,
        "utm_supervisor_states": q_utm,
        "utm_supervisor_transitions": tr_utm,
        "hierarchical_state_count": hierarchical_states,
        "hierarchical_transition_count": hierarchical_transitions,
        "hierarchical_total_construction_and_synthesis_time_s":  t_build0-t_all0,
        "centralized_exact_plants": len(centralized_plants),
        "centralized_exact_specs": len(centralized_specs),
        "centralized_exact_event_count": len(registry),
        "centralized_exact_atomic_plant_states_sum": q_cent_plants_sum,
        "centralized_exact_atomic_plant_transitions_sum": tr_cent_plants_sum,
        "centralized_exact_atomic_spec_states_sum": q_cent_specs_sum,
        "centralized_exact_atomic_spec_transitions_sum": tr_cent_specs_sum,
        "centralized_exact_supervisor_states": q_cent,
        "centralized_exact_supervisor_transitions": tr_cent,
        "centralized_exact_automata_build_time_s": t_build1 - t_build0,
        "centralized_exact_synthesis_time_s": t_synth1 - t_synth0,
        "total_metric_time_s": t_all1 - t_all0,
        "centralized_vs_hierarchical_state_factor": state_expansion_factor,
        "centralized_vs_hierarchical_transition_factor": transition_expansion_factor,
        "hierarchical_state_saving_pct_vs_centralized": hierarchical_state_saving_pct,
        "hierarchical_transition_saving_pct_vs_centralized": hierarchical_transition_saving_pct,
        "centralized_metric_type": "exact_monolithic_supervisor",
        "hierarchical_metric_type": "N_times_generic_UAV_supervisor_plus_one_UTM_supervisor",
        "centralized_model_definition": "N_agent_indexed_GenericUAVModel_copies_plus_agent_indexed_UTM_block_mutex_specs",
    }


def infer_init_node(nodes_csv: str, edges_csv: str) -> str:
    prepare_python_path()

    from utm_graph import load_graph_data
    from utm_fleet.utm_supervisor import GenericUTMModel

    gd = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
    G = GenericUTMModel._to_bidirectional_multidigraph(gd.graph)
    return first_vertiport(G)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DES/SCT structural metrics comparing the hierarchical "
            "MPSC/SCT architecture with an exact centralized monolithic supervisor."
        )
    )
    parser.add_argument("--nodes", required=True, help="Path to graph_nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to graph_edges.csv")
    parser.add_argument("--init-node", default="", help="Initial node, usually VERTIPORT_000")
    parser.add_argument("--num-uavs", type=int, default=0, help="Number of UAVs in this run")
    parser.add_argument(
        "--mutex-vertices",
        default="all_except_vertiport",
        choices=["all", "all_except_vertiport", "logical", "none"],
        help="Vertices protected by the centralized UTM mutex constraints.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path. Default: $UTM_RUN_DIR/des_metrics.csv",
    )
    args = parser.parse_args()

    prepare_python_path()

    init_node = str(args.init_node or "").strip()
    if not init_node:
        init_node = infer_init_node(args.nodes, args.edges)

    num_uavs = int(args.num_uavs or os.environ.get("UTM_UAVS", "0") or 0)
    if num_uavs <= 0:
        raise ValueError("num_uavs must be provided through --num-uavs or UTM_UAVS.")

    row = compute_metrics(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        init_node=init_node,
        num_uavs=num_uavs,
        mutex_vertices=args.mutex_vertices,
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = default_run_dir() / "des_metrics.csv"

    write_csv(out_path, row)

    print(f"[OK] DES metrics written to: {out_path}")
    print(
        "[SUMMARY] "
        f"N={row['num_uavs']} | "
        f"Q_uav={row['uav_supervisor_states']} | "
        f"delta_uav={row['uav_supervisor_transitions']} | "
        f"Q_utm={row['utm_supervisor_states']} | "
        f"delta_utm={row['utm_supervisor_transitions']} | "
        f"Q_hier={row['hierarchical_state_count']} | "
        f"delta_hier={row['hierarchical_transition_count']} | "
        f"Q_central_exact={row['centralized_exact_supervisor_states']} | "
        f"delta_central_exact={row['centralized_exact_supervisor_transitions']} | "
        f"state_factor={row['centralized_vs_hierarchical_state_factor']:.3f} | "
        f"transition_factor={row['centralized_vs_hierarchical_transition_factor']:.3f} | "
        f"central_synthesis_s={row['centralized_exact_synthesis_time_s']:.3f}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# test_cost_engine.py

import os
os.environ.setdefault("PYTHONNET_CLEANUP", "0")

import argparse
import csv
import sys
import tempfile
import traceback

from ultrades.automata import states, transitions, events, marked_states, monolithic_supervisor
from utm_graph import load_graph_data

from .GenericUAVModel import GenericUAVModel
from .help_cost import (
    build_atomic_cost_dict,
    build_supervisor_cost_index,
    supervisor_state_cost_from_atomic,
    build_supervisor_cost_dict,
    update_supervisor_cost_entry,
    update_supervisor_costs_for_atomic_changes,
    build_cost_engine,
    rebuild_all_costs,
    set_atomic_cost_and_update_supervisor,
)


def _csv_ids(nodes_csv):
    ids = []
    with open(nodes_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row and "id" in row and row["id"] is not None:
                ids.append(str(row["id"]).strip())
    return ids


def _graph_diagnostics(nodes_csv, edges_csv, init_node):
    try:
        gd = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
    except Exception as e:
        print("\n[DIAG] load_graph_data failed:")
        print(f"       {e}")
        return

    G = gd.graph
    init_node = str(init_node)

    print("\n[DIAG] Graph loaded OK.")
    print(f"       nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
    print(f"       directed={getattr(G, 'is_directed', lambda: False)()}")

    if init_node not in G.nodes:
        print(f"[DIAG] init_node NOT IN GRAPH: {init_node}")
    else:
        try:
            outdeg = G.out_degree(init_node) if hasattr(G, "out_degree") else len(list(G.neighbors(init_node)))
            indeg = G.in_degree(init_node) if hasattr(G, "in_degree") else len(list(G.neighbors(init_node)))
            print(f"[DIAG] init_node degrees: out={outdeg}  in={indeg}")
        except Exception:
            pass

    dead_ends = []
    for n in G.nodes:
        try:
            if hasattr(G, "successors"):
                if len(list(G.successors(n))) == 0:
                    dead_ends.append(str(n))
            else:
                if len(list(G.neighbors(n))) == 0:
                    dead_ends.append(str(n))
        except Exception:
            continue

    if dead_ends:
        print(f"[DIAG] nodes with NO successors/neighbors: {len(dead_ends)}")
        for x in dead_ends[:25]:
            print(f"       - {x}")
        if len(dead_ends) > 25:
            print(f"       ... +{len(dead_ends) - 25} more")


def _count_transitions(A):
    try:
        t = transitions(A)
        if not t:
            return 0
        total = 0
        for _q, mp in t.items():
            if mp:
                total += len(mp)
        return total
    except Exception:
        return -1


def _name_of(A):
    nm = getattr(A, "name", None)
    if nm:
        return str(nm)
    return str(A)


def _dump_automata_list(title, lst):
    print(f"\n[DIAG] {title}: {len(lst)}")
    for i, A in enumerate(lst):
        nm = _name_of(A)

        ns = -1
        ne = -1
        nt = -1
        nmks = -1

        try:
            ns = len(list(states(A)))
        except Exception:
            pass
        try:
            ne = len(list(events(A)))
        except Exception:
            pass
        try:
            nt = _count_transitions(A)
        except Exception:
            pass
        try:
            nmks = len(list(marked_states(A)))
        except Exception:
            pass

        flag = []
        if nt == 0:
            flag.append("ZERO_TRANS")
        if nmks == 0:
            flag.append("NO_MARKED")
        tag = (" [" + ",".join(flag) + "]") if flag else ""

        print(f"  ({i:03d}) {nm}  states={ns}  events={ne}  trans={nt}  marked={nmks}{tag}")


def _incremental_supervisor_debug(plants, specs):
    print("\n[DIAG] Trying incremental supervisor build to find first failing spec...")
    ok_specs = []
    for i, S in enumerate(specs):
        nm = _name_of(S)
        try:
            _ = monolithic_supervisor(plants, ok_specs + [S])
            ok_specs.append(S)
            print(f"  [OK ] add spec ({i:03d}) {nm}")
        except Exception as e:
            print(f"  [FAIL] adding spec ({i:03d}) {nm}")
            print(f"        {e}")
            return False, ("spec", i, nm)

    print("\n[DIAG] All specs OK incrementally; trying incremental plants...")
    ok_plants = []
    for i, P in enumerate(plants):
        nm = _name_of(P)
        try:
            _ = monolithic_supervisor(ok_plants + [P], specs)
            ok_plants.append(P)
            print(f"  [OK ] add plant ({i:03d}) {nm}")
        except Exception as e:
            print(f"  [FAIL] adding plant ({i:03d}) {nm}")
            print(f"        {e}")
            return False, ("plant", i, nm)

    return True, None


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def _make_sample_graph(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    nodes_csv = os.path.join(out_dir, "nodes_sample.csv")
    nodes_rows = [
        ("VERTIPORT_0", "VERTIPORT", 0.0, 0.0, 5.0),
        ("STATION_0", "STATION", 5.0, 0.0, 5.0),
        ("SUPPLIER_0", "SUPPLIER", 10.0, 0.0, 5.0),
        ("CLIENT_0", "CLIENT", 15.0, 0.0, 5.0),
    ]
    _write_csv(nodes_csv, ["id", "type", "x", "y", "z"], nodes_rows)

    edges_pairs = [
        ("VERTIPORT_0", "SUPPLIER_0"),
        ("SUPPLIER_0", "CLIENT_0"),
        ("CLIENT_0", "VERTIPORT_0"),
        ("VERTIPORT_0", "STATION_0"),
        ("STATION_0", "VERTIPORT_0"),
        ("SUPPLIER_0", "VERTIPORT_0"),
        ("CLIENT_0", "SUPPLIER_0"),
    ]

    edge_header_candidates = [
        ("u", "v"),
        ("src", "dst"),
        ("from", "to"),
        ("source", "target"),
    ]

    last_exc = None
    for hdr in edge_header_candidates:
        edges_csv = os.path.join(out_dir, f"edges_sample_{hdr[0]}_{hdr[1]}.csv")
        _write_csv(edges_csv, list(hdr), edges_pairs)
        try:
            _ = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
            return nodes_csv, edges_csv, "VERTIPORT_0"
        except Exception as e:
            last_exc = e

    raise RuntimeError(
        "Could not create an edges.csv that your loader accepts. "
        "Check utm_graph.load_graph_data for expected edge column names. "
        f"Last error: {last_exc}"
    )


def _assert_non_negative_costs(cost_dict, name, sample=20):
    keys = list(cost_dict.keys())
    for k in keys[: min(sample, len(keys))]:
        e, tf, d = cost_dict[k]
        if e < -1e-12 or tf < -1e-12 or d < -1e-12:
            raise AssertionError(f"[{name}] negative cost at {k}: {(e, tf, d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", default=None)
    ap.add_argument("--edges_csv", default=None)
    ap.add_argument("--init_node", default=None)
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--energy_per_meter", type=float, default=0.10)
    ap.add_argument("--base_time_cost", type=float, default=0.10)
    args = ap.parse_args()

    if args.make_sample:
        out_dir = args.out_dir or os.path.join(tempfile.gettempdir(), "utm_cost_test")
        nodes_csv, edges_csv, init_node = _make_sample_graph(out_dir)
        print(f"[INFO] Sample graph written to: {out_dir}")
        print(f"[INFO] nodes_csv: {nodes_csv}")
        print(f"[INFO] edges_csv: {edges_csv}")
        print(f"[INFO] init_node: {init_node}")
    else:
        if not args.nodes_csv or not args.edges_csv or not args.init_node:
            print("[ERROR] Provide --nodes_csv --edges_csv --init_node (or use --make-sample).")
            return 2
        nodes_csv, edges_csv, init_node = args.nodes_csv, args.edges_csv, args.init_node

    if not os.path.isfile(nodes_csv):
        print(f"[ERROR] nodes_csv not found: {nodes_csv}")
        return 2
    if not os.path.isfile(edges_csv):
        print(f"[ERROR] edges_csv not found: {edges_csv}")
        return 2

    ids = _csv_ids(nodes_csv)
    if str(init_node) not in ids:
        print(f"[WARN] init_node not present in nodes_csv ids: {init_node}")

    _graph_diagnostics(nodes_csv, edges_csv, init_node)

    print("\n[TEST] Building GenericUAVModel (with supervisor)...")
    try:
        model = GenericUAVModel(nodes_csv, edges_csv, init_node)
    except Exception as e:
        print("\n[FAIL] GenericUAVModel failed while computing supervisor:")
        print(str(e))
        print("\n--- traceback ---")
        traceback.print_exc()

        print("\n[DIAG] Rebuilding model while skipping supervisor to inspect automata...")
        orig = GenericUAVModel.compute_monolithic_supervisor

        def _skip_supervisor(self, force=False):
            return None

        GenericUAVModel.compute_monolithic_supervisor = _skip_supervisor
        try:
            model2 = GenericUAVModel(nodes_csv, edges_csv, init_node)
        except Exception as e2:
            print("\n[DIAG] Even skipping supervisor, model build failed:")
            print(str(e2))
            print("\n--- traceback ---")
            traceback.print_exc()
            GenericUAVModel.compute_monolithic_supervisor = orig
            return 1

        GenericUAVModel.compute_monolithic_supervisor = orig

        _dump_automata_list("PLANTS", getattr(model2, "plants", []))
        _dump_automata_list("SPECS", getattr(model2, "specs", []))

        plants = list(getattr(model2, "plants", []))
        specs = list(getattr(model2, "specs", []))

        print("\n[DIAG] Trying supervisor now (manual call) to reproduce the failure...")
        try:
            _ = monolithic_supervisor(plants, specs)
            print("[DIAG] Manual monolithic_supervisor succeeded (unexpected).")
        except Exception as e3:
            print("[DIAG] Manual monolithic_supervisor failed:")
            print(str(e3))

        ok, where = _incremental_supervisor_debug(plants, specs)
        if (not ok) and where:
            kind, idx, nm = where
            print(f"\n[DIAG] First failing {kind}: ({idx:03d}) {nm}")
            print("[DIAG] Very common cause: DFA with ZERO transitions in plants/specs.")
            print("[DIAG] Fix: ensure loc/task specs have entry transitions (edge_release::x::n exists).")
        return 1

    if getattr(model, "supervisor_mono", None) is None:
        print("[FAIL] Supervisor was not built (model.supervisor_mono is None).")
        return 1

    print("[OK] Model created and supervisor built.")
    print(f"     |G|={model.G.number_of_nodes()} nodes, |E|={model.G.number_of_edges()} edges")
    print(f"     supervisor_states={len(list(states(model.supervisor_mono)))}")

    print("\n[TEST] build_cost_engine(...)")
    params = {
        "IDLE_PENALTY": 0.5,
        "REPEAT_PENALTY": 0.3,
        "EARLY_CHARGE_PENALTY": 2.0,
        "NO_CHARGE_PENALTY": 15.0,
        "NO_WORK_PENALTY": 3.0,
        "NO_BASE_PENALTY": 2.0,
        "TASK_PROGRESS_BONUS": 5.0,
    }

    engine = build_cost_engine(
        model,
        speed_mps=args.speed,
        energy_per_meter=args.energy_per_meter,
        base_time_cost=args.base_time_cost,
        params=params,
    )

    atomic_cost = engine["atomic_cost"]
    sup_cost = engine["sup_cost"]
    sup_index = engine["sup_index"]
    rev_index = engine["rev_index"]

    print(f"[OK] atomic_cost: {len(atomic_cost)} states")
    print(f"[OK] sup_cost:    {len(sup_cost)} states")

    _assert_non_negative_costs(atomic_cost, "atomic_cost")
    _assert_non_negative_costs(sup_cost, "sup_cost")
    print("[OK] Sampled costs are non-negative.")

    print("\n[TEST] supervisor_state_cost_from_atomic (one supervisor state)")
    one_sup = next(iter(sup_cost.keys()))
    c1 = supervisor_state_cost_from_atomic(model, one_sup, atomic_cost, params=params, sup_index=sup_index)
    print(f"[OK] Computed cost for supervisor state: {one_sup}")
    print(f"     cost(E,Tf,D)={c1}")

    print("\n[TEST] update_supervisor_cost_entry (recompute one entry)")
    c2 = update_supervisor_cost_entry(model, sup_cost, one_sup, atomic_cost, params=params, sup_index=sup_index)
    print(f"[OK] Updated one supervisor state: cost={c2}")

    print("\n[TEST] set_atomic_cost_and_update_supervisor (incremental updates)")
    one_atomic = next(iter(atomic_cost.keys()))
    old_atomic = atomic_cost[one_atomic]
    new_atomic = (old_atomic[0] + 1.0, old_atomic[1] + 0.5, old_atomic[2] + 0.2)
    written, nupd = set_atomic_cost_and_update_supervisor(engine, model, one_atomic, new_atomic)
    print(f"[OK] Atomic updated: {one_atomic}")
    print(f"     old={old_atomic}")
    print(f"     new={written}")
    print(f"     supervisor states recomputed (incremental) = {nupd}")

    _assert_non_negative_costs(engine["atomic_cost"], "atomic_cost (after atomic update)")
    _assert_non_negative_costs(engine["sup_cost"], "sup_cost (after atomic update)")
    print("[OK] Costs remain non-negative after incremental update.")

    print("\n[TEST] update_supervisor_costs_for_atomic_changes (batch)")
    some_atomic = list(engine["atomic_cost"].keys())[:3]
    for a in some_atomic:
        e, tf, d = engine["atomic_cost"][a]
        engine["atomic_cost"][a] = (e, tf, d + 0.123)

    nupd2 = update_supervisor_costs_for_atomic_changes(
        model,
        engine["sup_cost"],
        some_atomic,
        engine["atomic_cost"],
        params=params,
        sup_index=sup_index,
        rev_index=rev_index,
    )
    print(f"[OK] Batch update: atomic_changed={some_atomic}")
    print(f"     supervisor states recomputed = {nupd2}")

    print("\n[TEST] rebuild_all_costs (full rebuild)")
    rebuild_all_costs(engine, model, speed_mps=args.speed * 1.5)
    print(f"[OK] Rebuilt: speed_mps={engine['speed_mps']}")
    _assert_non_negative_costs(engine["atomic_cost"], "atomic_cost (after rebuild)")
    _assert_non_negative_costs(engine["sup_cost"], "sup_cost (after rebuild)")
    print("[OK] Costs remain non-negative after full rebuild.")

    print("\n[TEST] build_supervisor_cost_index (direct)")
    si2, ri2 = build_supervisor_cost_index(model)
    print(f"[OK] sup_index size={len(si2)}, rev_index keys={len(ri2)}")

    print("\n[TEST] build_atomic_cost_dict + build_supervisor_cost_dict (direct)")
    ac2 = build_atomic_cost_dict(
        model,
        speed_mps=args.speed,
        energy_per_meter=args.energy_per_meter,
        base_time_cost=args.base_time_cost,
    )
    sc2 = build_supervisor_cost_dict(model, ac2, params=params, sup_index=si2)
    print(f"[OK] atomic_cost(direct)={len(ac2)}  sup_cost(direct)={len(sc2)}")

    print("\n✅ ALL TESTS PASSED.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("\n[FAIL] Exception during tests:")
        print(str(e))
        print("\n--- traceback ---")
        traceback.print_exc()
        sys.exit(1)

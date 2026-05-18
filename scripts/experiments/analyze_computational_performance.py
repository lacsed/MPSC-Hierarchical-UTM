#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple


def _read_csvs(paths: Iterable[Path]) -> List[dict]:
    rows = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = path.name
                rows.append(row)
    return rows


def _float(x, default=math.nan) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _int(x, default=0) -> int:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _quantile(values: List[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def _safe_std(values: List[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return stdev(vals) if len(vals) >= 2 else 0.0 if len(vals) == 1 else math.nan


def _fmt(x) -> str:
    if isinstance(x, float):
        if not math.isfinite(x):
            return ""
        return f"{x:.6g}"
    return str(x)


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fieldnames})


def _group_key(row: dict) -> Tuple[str, ...]:
    return (
        str(row.get("baseline", "")),
        str(row.get("scenario_id", "")),
        str(row.get("graph_size", "")),
        str(row.get("density", "")),
        str(row.get("seed", "")),
        str(row.get("num_uavs", "")),
        str(row.get("num_nodes", "")),
        str(row.get("num_edges", "")),
        str(row.get("planning_horizon", "")),
    )


def _group_key_no_seed(row: dict) -> Tuple[str, ...]:
    return (
        str(row.get("baseline", "")),
        str(row.get("scenario_id", "")),
        str(row.get("graph_size", "")),
        str(row.get("density", "")),
        str(row.get("num_uavs", "")),
        str(row.get("num_nodes", "")),
        str(row.get("num_edges", "")),
        str(row.get("planning_horizon", "")),
    )


def summarize_planner(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    out = []
    online = []
    fail_status = {"ERROR", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "NO_SELECTED_EVENT"}

    for key, items in sorted(groups.items()):
        baseline, scenario, graph_size, density, seed, n_uav, n_nodes, n_edges, H = key
        runtimes = [_float(r.get("runtime_ms")) for r in items]
        runtimes = [x for x in runtimes if math.isfinite(x)]
        status = [str(r.get("milp_status", "")) for r in items]
        selected = [str(r.get("selected_event", "")).strip() for r in items]
        failures = 0
        for st, ev in zip(status, selected):
            if st in fail_status or not ev:
                failures += 1

        calls = len(items)
        success = max(calls - failures, 0)
        out.append({
            "baseline": baseline,
            "scenario_id": scenario,
            "graph_size": graph_size,
            "density": density,
            "seed": seed,
            "num_uavs": n_uav,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "planning_horizon": H,
            "num_milp_calls": calls,
            "success_count": success,
            "failure_count": failures,
            "failure_rate_pct": 100.0 * failures / calls if calls else math.nan,
            "mean_runtime_ms": _safe_mean(runtimes),
            "std_runtime_ms": _safe_std(runtimes),
            "max_runtime_ms": max(runtimes) if runtimes else math.nan,
            "p95_runtime_ms": _quantile(runtimes, 0.95),
            "p99_runtime_ms": _quantile(runtimes, 0.99),
        })

        available = [_float(r.get("available_time_ms")) for r in items]
        ratios = [_float(r.get("online_ratio")) for r in items]
        available = [x for x in available if math.isfinite(x) and x > 0]
        ratios = [x for x in ratios if math.isfinite(x) and x >= 0]
        if available:
            p95_rt = _quantile(runtimes, 0.95)
            min_available = min(available)
            p95_ratio = _quantile(ratios, 0.95) if ratios else math.nan
            online.append({
                "baseline": baseline,
                "scenario_id": scenario,
                "graph_size": graph_size,
                "density": density,
                "seed": seed,
                "num_uavs": n_uav,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "planning_horizon": H,
                "p95_runtime_ms": p95_rt,
                "min_available_time_ms": min_available,
                "mean_available_time_ms": _safe_mean(available),
                "p95_online_ratio": p95_ratio,
                "online_feasible_p95": int(math.isfinite(p95_rt) and p95_rt < min_available),
            })

    return out, online


def summarize_utm(rows: List[dict]) -> List[dict]:
    request_rows = [r for r in rows if r.get("record_type") == "request_event"]
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in request_rows:
        key = (
            str(row.get("baseline", "")),
            str(row.get("scenario_id", "")),
            str(row.get("graph_size", "")),
            str(row.get("density", "")),
            str(row.get("seed", "")),
            str(row.get("num_uavs", "")),
            str(row.get("num_nodes", "")),
            str(row.get("num_edges", "")),
        )
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items()):
        baseline, scenario, graph_size, density, seed, n_uav, n_nodes, n_edges = key
        accepted = sum(1 for r in items if _int(r.get("accepted")) == 1)
        rejected = len(items) - accepted
        runtimes = [_float(r.get("request_runtime_ms")) for r in items]
        forb = [_float(r.get("forbidden_count")) for r in items]
        out.append({
            "baseline": baseline,
            "scenario_id": scenario,
            "graph_size": graph_size,
            "density": density,
            "seed": seed,
            "num_uavs": n_uav,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "utm_requests": len(items),
            "accepted": accepted,
            "rejected": rejected,
            "rejection_rate_pct": 100.0 * rejected / len(items) if items else math.nan,
            "mean_request_runtime_ms": _safe_mean(runtimes),
            "p95_request_runtime_ms": _quantile(runtimes, 0.95),
            "mean_forbidden_count": _safe_mean(forb),
            "max_forbidden_count": max([x for x in forb if math.isfinite(x)], default=math.nan),
        })
    return out


def summarize_events(rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("baseline", "")),
                str(row.get("scenario_id", "")),
                str(row.get("graph_size", "")),
                str(row.get("density", "")),
                str(row.get("seed", "")),
                str(row.get("num_uavs", "")),
                str(row.get("num_nodes", "")),
                str(row.get("num_edges", "")),
            )
        ].append(row)

    out = []
    for key, items in sorted(groups.items()):
        baseline, scenario, graph_size, density, seed, n_uav, n_nodes, n_edges = key
        tx = [r for r in items if r.get("direction") == "tx"]
        evs = [str(r.get("event_generic", "")) for r in tx]
        out.append({
            "baseline": baseline,
            "scenario_id": scenario,
            "graph_size": graph_size,
            "density": density,
            "seed": seed,
            "num_uavs": n_uav,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "published_events": len(tx),
            "edge_take_count": sum(e.startswith("edge_take::") for e in evs),
            "edge_release_count": sum(e.startswith("edge_release::") for e in evs),
            "work_start_count": sum(e.startswith("work_start::") for e in evs),
            "work_end_count": sum(e.startswith("work_end::") for e in evs),
            "charge_start_count": sum(e.startswith("charge_start::") for e in evs),
            "task_done_count": sum(e == "task_done" for e in evs),
        })
    return out


def summarize_centralized_proxy(planner_rows: List[dict], cycle_width_s: float) -> List[dict]:
    bins: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    local_groups: Dict[Tuple[str, ...], List[float]] = defaultdict(list)

    for row in planner_rows:
        rt = _float(row.get("runtime_ms"))
        tw = _float(row.get("t_wall"))
        if not (math.isfinite(rt) and math.isfinite(tw)):
            continue
        base_key = _group_key_no_seed(row)
        seed = str(row.get("seed", ""))
        tbin = int(math.floor(tw / max(cycle_width_s, 1e-9)))
        bins[base_key + (seed, str(tbin))].append(rt)
        local_groups[base_key].append(rt)

    cycle_groups: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for key, vals in bins.items():
        base_key = key[:-2]
        cycle_groups[base_key].append(sum(vals))

    out = []
    for key, cycle_vals in sorted(cycle_groups.items()):
        baseline, scenario, graph_size, density, n_uav, n_nodes, n_edges, H = key
        local_vals = local_groups.get(key, [])
        out.append({
            "baseline": baseline,
            "scenario_id": scenario,
            "graph_size": graph_size,
            "density": density,
            "num_uavs": n_uav,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "planning_horizon": H,
            "cycle_width_s": cycle_width_s,
            "mean_local_runtime_ms": _safe_mean(local_vals),
            "p95_local_runtime_ms": _quantile(local_vals, 0.95),
            "mean_centralized_proxy_cycle_ms": _safe_mean(cycle_vals),
            "max_centralized_proxy_cycle_ms": max(cycle_vals) if cycle_vals else math.nan,
            "p95_centralized_proxy_cycle_ms": _quantile(cycle_vals, 0.95),
        })
    return out


def write_markdown_report(path: Path, planner_summary, online_summary, utm_summary, event_summary, proxy_summary) -> None:
    def table(title, rows, columns, limit=20):
        lines = [f"\n## {title}\n"]
        if not rows:
            lines.append("No data.\n")
            return lines
        lines.append("| " + " | ".join(columns) + " |\n")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for r in rows[:limit]:
            lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |\n")
        return lines

    lines = ["# Computational Performance Report\n"]
    lines += table(
        "MILP runtime summary",
        planner_summary,
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "planning_horizon", "num_milp_calls", "mean_runtime_ms", "std_runtime_ms", "max_runtime_ms", "p95_runtime_ms", "failure_rate_pct"],
    )
    lines += table(
        "Online feasibility summary",
        online_summary,
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "planning_horizon", "p95_runtime_ms", "min_available_time_ms", "p95_online_ratio", "online_feasible_p95"],
    )
    lines += table(
        "UTM communication and arbitration summary",
        utm_summary,
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "utm_requests", "accepted", "rejected", "rejection_rate_pct", "mean_request_runtime_ms", "mean_forbidden_count"],
    )
    lines += table(
        "Event/task summary",
        event_summary,
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "published_events", "edge_take_count", "charge_start_count", "task_done_count"],
    )
    lines += table(
        "Centralized proxy summary",
        proxy_summary,
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "planning_horizon", "mean_local_runtime_ms", "p95_local_runtime_ms", "mean_centralized_proxy_cycle_ms", "p95_centralized_proxy_cycle_ms"],
    )
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate UTM/MPSC computational-performance metrics.")
    parser.add_argument("--run-dir", required=True, help="Directory containing planner_agent_*.csv and UTM metric CSV files.")
    parser.add_argument("--out-dir", default="", help="Output directory. Defaults to run-dir/analysis.")
    parser.add_argument("--cycle-width", type=float, default=1.0, help="Time-bin width in seconds for centralized proxy analysis.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    planner_rows = _read_csvs(sorted(run_dir.glob("planner_agent_*.csv")))
    utm_rows = _read_csvs(sorted(run_dir.glob("utm_supervisor.csv")))
    event_rows = _read_csvs(sorted(run_dir.glob("events_agent_*.csv")))

    planner_summary, online_summary = summarize_planner(planner_rows)
    utm_summary = summarize_utm(utm_rows)
    event_summary = summarize_events(event_rows)
    proxy_summary = summarize_centralized_proxy(planner_rows, args.cycle_width)

    _write_csv(
        out_dir / "computational_runtime_summary.csv",
        ["baseline", "scenario_id", "graph_size", "density", "seed", "num_uavs", "num_nodes", "num_edges", "planning_horizon", "num_milp_calls", "success_count", "failure_count", "failure_rate_pct", "mean_runtime_ms", "std_runtime_ms", "max_runtime_ms", "p95_runtime_ms", "p99_runtime_ms"],
        planner_summary,
    )
    _write_csv(
        out_dir / "online_feasibility_summary.csv",
        ["baseline", "scenario_id", "graph_size", "density", "seed", "num_uavs", "num_nodes", "num_edges", "planning_horizon", "p95_runtime_ms", "min_available_time_ms", "mean_available_time_ms", "p95_online_ratio", "online_feasible_p95"],
        online_summary,
    )
    _write_csv(
        out_dir / "utm_communication_summary.csv",
        ["baseline", "scenario_id", "graph_size", "density", "seed", "num_uavs", "num_nodes", "num_edges", "utm_requests", "accepted", "rejected", "rejection_rate_pct", "mean_request_runtime_ms", "p95_request_runtime_ms", "mean_forbidden_count", "max_forbidden_count"],
        utm_summary,
    )
    _write_csv(
        out_dir / "event_task_summary.csv",
        ["baseline", "scenario_id", "graph_size", "density", "seed", "num_uavs", "num_nodes", "num_edges", "published_events", "edge_take_count", "edge_release_count", "work_start_count", "work_end_count", "charge_start_count", "task_done_count"],
        event_summary,
    )
    _write_csv(
        out_dir / "centralized_proxy_summary.csv",
        ["baseline", "scenario_id", "graph_size", "density", "num_uavs", "num_nodes", "num_edges", "planning_horizon", "cycle_width_s", "mean_local_runtime_ms", "p95_local_runtime_ms", "mean_centralized_proxy_cycle_ms", "max_centralized_proxy_cycle_ms", "p95_centralized_proxy_cycle_ms"],
        proxy_summary,
    )

    write_markdown_report(
        out_dir / "computational_performance_report.md",
        planner_summary,
        online_summary,
        utm_summary,
        event_summary,
        proxy_summary,
    )

    print(f"[OK] Analysis written to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from utm_graph import load_graph_data
from .GenericUAVModel import GenericUAVModel
from .SupervisorAgent import UAVAgentNode


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------

def dist2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def nearest_node_id(positions, x, y):
    return min(
        positions.items(),
        key=lambda kv: dist2((kv[1][0], kv[1][1]), (x, y)),
    )[0]


# ----------------------------------------------------------------------
# process entrypoint
# ----------------------------------------------------------------------

def run_agent_process(
    agent_id,
    entity_name,
    init_pose,
    base_model,
    set_state_service,
    rate_hz,
    speed_mps,
    vspeed_mps,
    tol,
    clearance,
    alt_offset,
    planning_horizon,
    seed,
    work_time_s,
    charge_time_s,
    low_batt_threshold,
    battery_log_period_s,
):
    rclpy.init()

    node = None
    executor = None

    try:
        node = UAVAgentNode(
            agent_id=agent_id,
            entity_name=entity_name,
            init_pose=init_pose,
            base_model=base_model,
            set_state_service=set_state_service,
            rate_hz=rate_hz,
            speed_mps=speed_mps,
            vspeed_mps=vspeed_mps,
            tol=tol,
            clearance=clearance,
            alt_offset=alt_offset,
            planning_horizon=planning_horizon,
            seed=seed,
            work_time_s=work_time_s,
            charge_time_s=charge_time_s,
            low_batt_threshold=low_batt_threshold,
            battery_log_period_s=battery_log_period_s,
        )

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()

    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass

        if node is not None:
            try:
                node.close()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            rclpy.shutdown()
        except Exception:
            pass


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Launch one ROS 2 UAV agent process per spawn point."
    )
    parser.add_argument("--nodes", required=True, help="CSV with graph nodes")
    parser.add_argument("--edges", required=True, help="CSV with graph edges")
    parser.add_argument("--rate", type=float, default=20.0, help="Agent update rate [Hz]")
    parser.add_argument("--speed", type=float, default=3.0, help="Horizontal speed [m/s]")
    parser.add_argument("--vspeed", type=float, default=1.0, help="Vertical speed [m/s]")
    parser.add_argument("--tol", type=float, default=0.25, help="Waypoint tolerance [m]")
    parser.add_argument("--clearance", type=float, default=0.0, help="Clearance above graph altitude [m]")
    parser.add_argument("--alt-spread", type=float, default=1.0, help="Altitude offset spread across agents [m]")
    parser.add_argument("--set-service", default="/gazebo/set_entity_state", help="Gazebo SetEntityState service")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--planning-horizon", type=int, default=10, help="MILP planning horizon")
    parser.add_argument("--work-time", type=float, default=2.0, help="Pick/deliver time [s]")
    parser.add_argument("--charge-time", type=float, default=5.0, help="Charge time [s]")
    parser.add_argument("--low-batt-threshold", type=float, default=0.40, help="Low battery threshold [0,1]")
    parser.add_argument("--battery-log-period", type=float, default=0.0, help="Battery log period [s], 0 disables")
    parser.add_argument("--startup-stagger", type=float, default=0.25, help="Delay between process launches [s]")
    args = parser.parse_args()

    graph_data = load_graph_data(
        str(args.nodes),
        str(args.edges),
        add_euclidean_weight=True,
    )
    spawns = list(graph_data.spawns)

    if not spawns:
        raise RuntimeError("No spawn points found in graph_data.spawns")

    common_init_node = nearest_node_id(
        graph_data.positions,
        float(spawns[0].x),
        float(spawns[0].y),
    )

    for sp in spawns:
        sp_init = nearest_node_id(
            graph_data.positions,
            float(sp.x),
            float(sp.y),
        )
        if sp_init != common_init_node:
            raise RuntimeError(
                "Single GenericUAVModel requires all agents to share the same init node. "
                "Found '%s' and '%s'."
                % (common_init_node, sp_init)
            )

    base_model = GenericUAVModel(
        str(args.nodes),
        str(args.edges),
        common_init_node,
    )

    n_agents = len(spawns)
    alt_step = 0.0 if n_agents <= 1 else float(args.alt_spread) / float(n_agents - 1)

    print("[INFO] Launching UAV agents")
    print(f"[INFO] nodes: {args.nodes}")
    print(f"[INFO] edges: {args.edges}")
    print(f"[INFO] agents: {n_agents}")
    print(f"[INFO] common init node: {common_init_node}")
    print(f"[INFO] gazebo service: {args.set_service}")
    print("-" * 60)

    procs = []

    try:
        for i, sp in enumerate(spawns):
            agent_id = int(i)
            init_pose = (
                float(sp.x),
                float(sp.y),
                float(sp.z),
                0.0,
            )

            p = mp.Process(
                target=run_agent_process,
                args=(
                    agent_id,
                    sp.vehicle_id,
                    init_pose,
                    base_model,
                    args.set_service,
                    float(args.rate),
                    float(args.speed),
                    float(args.vspeed),
                    float(args.tol),
                    float(args.clearance),
                    float(i * alt_step),
                    int(args.planning_horizon),
                    int(args.seed),
                    float(args.work_time),
                    float(args.charge_time),
                    float(args.low_batt_threshold),
                    float(args.battery_log_period),
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

            print(
                "[INFO] started agent=%d entity='%s' pid=%s"
                % (agent_id, str(sp.vehicle_id), str(p.pid))
            )

            time.sleep(float(args.startup_stagger))

        for p in procs:
            p.join()

    except KeyboardInterrupt:
        print("\n[INFO] Stopping agent processes...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=2.0)


if __name__ == "__main__":
    main()
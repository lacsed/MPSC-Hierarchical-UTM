import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import networkx as nx
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

from utm_graph import load_graph_data, NodeType


def yaw_to_quaternion(yaw: float) -> Quaternion:
    # roll=pitch=0
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw * 0.5),
        w=math.cos(yaw * 0.5),
    )


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def nearest_node_id(positions: Dict[str, Tuple[float, float, float]], x: float, y: float) -> str:
    return min(positions.items(), key=lambda kv: dist2((kv[1][0], kv[1][1]), (x, y)))[0]


@dataclass
class VehicleAgent:
    vehicle_id: str
    x: float
    y: float
    z: float
    yaw: float

    path: List[str]
    wp_idx: int
    goal_id: Optional[str]

    def has_path(self) -> bool:
        return self.path and self.wp_idx < len(self.path)


class FleetController(Node):
    def __init__(
        self,
        nodes_csv: str,
        edges_csv: str,
        *,
        set_state_service: str = "/gazebo/set_entity_state",
        rate_hz: float = 10.0,
        speed_mps: float = 4.0,
        waypoint_tol_m: float = 0.7,
        cruise_clearance_m: float = 2.0,
        altitude_layer_step_m: float = 0.5,
        seed: int = 7,
    ) -> None:
        super().__init__("utm_fleet_controller")

        random.seed(seed)

        self.graph_data = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
        self.G = self.graph_data.graph

        # ---- build goal pool: only "special" nodes (not logical) by default
        self.goal_pool: List[str] = []
        for nid, rec in self.graph_data.nodes.items():
            if rec.node_type in (NodeType.SUPPLIER, NodeType.CLIENT, NodeType.CHARGING, NodeType.VERTIPORT):
                self.goal_pool.append(nid)

        if not self.goal_pool:
            # fallback: allow logical nodes too
            self.goal_pool = list(self.graph_data.nodes.keys())

        # ---- cruise altitude: use max logical z if available, else max z among nodes
        logical_z = [rec.z for rec in self.graph_data.nodes.values() if rec.node_type == NodeType.LOGICAL]
        if logical_z:
            base_cruise_z = max(logical_z) + cruise_clearance_m
        else:
            base_cruise_z = max(rec.z for rec in self.graph_data.nodes.values()) + cruise_clearance_m

        self.get_logger().info(f"Loaded graph: DES nodes={len(self.graph_data.nodes)} spawns={len(self.graph_data.spawns)}")
        self.get_logger().info(f"Edges (directed)={self.G.number_of_edges()}")
        self.get_logger().info(f"Goal pool size={len(self.goal_pool)}")
        self.get_logger().info(f"Cruise altitude base z={base_cruise_z:.3f} (layers step {altitude_layer_step_m:.2f})")

        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.speed = float(speed_mps)
        self.waypoint_tol = float(waypoint_tol_m)

        # ---- Gazebo service client
        self.cli_set = self.create_client(SetEntityState, set_state_service)
        if not self.cli_set.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(
                f"Service not available: {set_state_service}\n"
                f"Make sure you started Gazebo using: ros2 launch gazebo_ros gazebo.launch.py ..."
            )

        # ---- initialize agents at spawns (override z to cruise + layers)
        self.agents: List[VehicleAgent] = []
        for i, sp in enumerate(self.graph_data.spawns):
            z = base_cruise_z + i * altitude_layer_step_m
            agent = VehicleAgent(
                vehicle_id=sp.vehicle_id,
                x=sp.x,
                y=sp.y,
                z=z,
                yaw=0.0,
                path=[],
                wp_idx=0,
                goal_id=None,
            )
            self.agents.append(agent)

        # Place all vehicles at start pose immediately
        for a in self.agents:
            self._send_pose(a.vehicle_id, a.x, a.y, a.z, a.yaw)

        # Create initial missions
        for a in self.agents:
            self._assign_new_mission(a)

        self.timer = self.create_timer(self.dt, self._on_timer)

    # -------------------------
    # Mission + planning
    # -------------------------
    def _assign_new_mission(self, a: VehicleAgent) -> None:
        start = nearest_node_id(self.graph_data.positions, a.x, a.y)

        # choose goal different from start
        goal = start
        tries = 0
        while goal == start and tries < 50:
            goal = random.choice(self.goal_pool)
            tries += 1

        try:
            path = nx.shortest_path(self.G, start, goal, weight="weight")
        except Exception as e:
            self.get_logger().warn(f"[{a.vehicle_id}] planning failed start={start} goal={goal}: {e}")
            a.path = []
            a.wp_idx = 0
            a.goal_id = None
            return

        # path includes start; waypoint index should target path[1] first
        a.path = path
        a.wp_idx = 1 if len(path) > 1 else 0
        a.goal_id = goal

        self.get_logger().info(f"[{a.vehicle_id}] mission: {start} -> {goal} | hops={len(path)-1}")

    # -------------------------
    # Control loop
    # -------------------------
    def _on_timer(self) -> None:
        for a in self.agents:
            self._step_agent(a, self.dt)

    def _step_agent(self, a: VehicleAgent, dt: float) -> None:
        if not a.has_path():
            self._assign_new_mission(a)
            return

        target_id = a.path[a.wp_idx]
        tx, ty, _tz = self.graph_data.positions[target_id]

        dx = tx - a.x
        dy = ty - a.y
        d = math.hypot(dx, dy)

        if d <= self.waypoint_tol:
            a.wp_idx += 1
            if a.wp_idx >= len(a.path):
                # mission finished
                self.get_logger().info(f"[{a.vehicle_id}] reached goal {a.goal_id}")
                a.path = []
                a.wp_idx = 0
                a.goal_id = None
            return

        # move towards target with bounded step
        step = min(self.speed * dt, d)
        ux = dx / (d + 1e-9)
        uy = dy / (d + 1e-9)

        a.x += ux * step
        a.y += uy * step
        a.yaw = math.atan2(uy, ux)

        self._send_pose(a.vehicle_id, a.x, a.y, a.z, a.yaw)

    # -------------------------
    # Gazebo integration
    # -------------------------
    def _send_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> None:
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = yaw_to_quaternion(yaw)

        twist = Twist()  # leave as zero; we are doing kinematic updates

        state = EntityState()
        state.name = name
        state.pose = pose
        state.twist = twist
        state.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = state

        # fire-and-forget (fast enough for demo)
        self.cli_set.call_async(req)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True, help="Path to graph_nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to graph_edges.csv")
    parser.add_argument("--rate", type=float, default=10.0, help="Control rate (Hz)")
    parser.add_argument("--speed", type=float, default=4.0, help="Cruise speed (m/s)")
    parser.add_argument("--tol", type=float, default=0.7, help="Waypoint tolerance (m)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--set-service", default="/gazebo/set_entity_state", help="Gazebo set state service name")

    args = parser.parse_args()

    rclpy.init()
    node = FleetController(
        args.nodes,
        args.edges,
        set_state_service=args.set_service,
        rate_hz=args.rate,
        speed_mps=args.speed,
        waypoint_tol_m=args.tol,
        seed=args.seed,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

# fleet_node.py
import argparse
import random

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String

from utm_graph import load_graph_data
from .uav_hardware import UAVHardware, BatteryModel


class FleetController(Node):
    def __init__(
        self,
        nodes_csv,
        edges_csv,
        *,
        set_state_service="/gazebo/set_entity_state",
        rate_hz=20.0,
        speed_mps=3.0,
        vspeed_mps=1.0,
        tol=0.25,
        clearance=0.0,
        alt_spread=1.0,
        seed=7,
        naming="english",
    ):
        super().__init__("utm_fleet_controller")
        random.seed(int(seed))

        self.graph_data = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
        self.positions = self.graph_data.positions

        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / max(1e-6, self.rate_hz)

        self.cli_set = self.create_client(SetEntityState, set_state_service)
        if not self.cli_set.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"Service not available: {set_state_service}")

        self.sub_event = self.create_subscription(String, "/event", self._on_event, 50)

        spawns = list(self.graph_data.spawns)
        n = len(spawns)
        step = 0.0 if n <= 1 else float(alt_spread) / float(n - 1)

        self.uavs = []
        batt = BatteryModel()

        for i, sp in enumerate(spawns):
            agent_id = i
            uav = UAVHardware(
                self,
                entity_name=sp.vehicle_id,
                agent_id=agent_id,
                graph_positions=self.positions,
                set_state_client=self.cli_set,
                speed_mps=float(speed_mps),
                vspeed_mps=float(vspeed_mps),
                waypoint_tol_m=float(tol),
                clearance_m=float(clearance),
                alt_offset_m=float(i * step),
                battery_model=batt,
                init_pose=(float(sp.x), float(sp.y), float(sp.z), 0.0),
            )
            uav.send_pose()
            self.uavs.append(uav)

        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(f"Loaded nodes={len(self.graph_data.nodes)} spawns={len(self.graph_data.spawns)}")
        self.get_logger().info("Fleet simulator ready (event-driven, edge-only motion).")

    def _on_event(self, msg):
        ev = str(msg.data or "").strip()
        if not ev:
            return
        for u in self.uavs:
            u.on_event(ev)

    def _on_timer(self):
        for u in self.uavs:
            u.step(self.dt)
            u.send_pose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--vspeed", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument("--alt-spread", type=float, default=1.0)
    parser.add_argument("--set-service", default="/gazebo/set_entity_state")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--naming", default="english", choices=["english", "legacy"])
    args = parser.parse_args()

    rclpy.init()
    node = FleetController(
        args.nodes,
        args.edges,
        set_state_service=args.set_service,
        rate_hz=args.rate,
        speed_mps=args.speed,
        vspeed_mps=args.vspeed,
        tol=args.tol,
        clearance=args.clearance,
        alt_spread=args.alt_spread,
        seed=args.seed,
        naming=args.naming,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

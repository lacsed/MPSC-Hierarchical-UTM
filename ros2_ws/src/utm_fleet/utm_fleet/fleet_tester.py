import argparse
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from utm_graph import load_graph_data


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def nearest_node_id(positions, x, y):
    return min(positions.items(), key=lambda kv: dist2((kv[1][0], kv[1][1]), (x, y)))[0]


class FleetTester(Node):
    """
    Event generator for the fleet simulator.
    Publishes commands on /event and tracks completion via the same topic.
    Event format expected by UAVHardware:

      edge_take::<U>::<V>_<id>
      work_start::<NODE>_<id>
      charge_start::<NODE>_<id>

    Completion events produced by UAVHardware that we track:

      edge_release::<U>::<V>_<id>
      work_end::<NODE>_<id>
      charge_end::<NODE>_<id>
      battery_low_<id> / battery_empty_<id> / crashed_<id> (ignored for busy flag)
    """

    def __init__(self, nodes_csv, edges_csv, p_work=0.25, p_charge=0.15, rate_hz=2.0, seed=7):
        super().__init__("utm_fleet_tester")
        random.seed(int(seed))

        self.p_work = float(p_work)
        self.p_charge = float(p_charge)

        self.graph_data = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
        self.G = self.graph_data.graph
        self.pos = self.graph_data.positions

        self.pub = self.create_publisher(String, "/event", 50)
        self.sub = self.create_subscription(String, "/event", self._on_event, 50)

        self.agent_count = len(self.graph_data.spawns)
        self.agent_node = {}
        self.agent_busy = {}

        for i, sp in enumerate(self.graph_data.spawns):
            nid = nearest_node_id(self.pos, float(sp.x), float(sp.y))
            self.agent_node[i] = nid
            self.agent_busy[i] = False

        self.work_nodes = self._infer_work_nodes()
        self.charge_nodes = self._infer_charge_nodes()

        self.dt = 1.0 / max(1e-6, float(rate_hz))
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"Tester ready: agents={self.agent_count} "
            f"charge_nodes={len(self.charge_nodes)} work_nodes={len(self.work_nodes)} "
            f"rate={rate_hz}Hz"
        )

    def _infer_work_nodes(self):
        out = []
        for nid in self.graph_data.nodes.keys():
            u = str(nid).upper()
            if ("SUPPLIER" in u) or ("CLIENT" in u) or ("FORNECEDOR" in u) or ("CLIENTE" in u):
                out.append(nid)
        return out

    def _infer_charge_nodes(self):
        out = []
        for nid in self.graph_data.nodes.keys():
            u = str(nid).upper()
            if ("CHARG" in u) or ("ESTACAO" in u) or ("STATION" in u):
                out.append(nid)
        return out

    def _ev_take(self, u, v, agent_id):
        return f"edge_take::{u}::{v}_{agent_id}"

    def _ev_start_work(self, node_id, agent_id):
        return f"work_start::{node_id}_{agent_id}"

    def _ev_start_charge(self, node_id, agent_id):
        return f"charge_start::{node_id}_{agent_id}"

    def _publish(self, ev):
        self.pub.publish(String(data=str(ev)))
        self.get_logger().info(f"publish: {ev}")

    def _parse_suffix_id(self, ev):
        if "_" not in ev:
            return None, None
        base, suf = ev.rsplit("_", 1)
        if not suf.isdigit():
            return None, None
        return base, int(suf)

    def _on_event(self, msg):
        ev = str(msg.data or "").strip()
        if not ev:
            return

        base, agent_id = self._parse_suffix_id(ev)
        if agent_id is None:
            return
        if agent_id not in self.agent_node:
            return

        # Movement completion: edge_release::<U>::<V>_<id>
        if base.startswith("edge_release::"):
            rest = base.split("edge_release::", 1)[1]
            parts = rest.split("::")
            if len(parts) == 2:
                u = parts[0]
                v = parts[1]
                self.agent_node[agent_id] = v
                self.agent_busy[agent_id] = False
            return

        # Local action completion: work_end / charge_end
        if base.startswith("work_end::") or base.startswith("charge_end::"):
            self.agent_busy[agent_id] = False
            return

    def _tick(self):
        if self.agent_count <= 0:
            return

        for agent_id in range(self.agent_count):
            if self.agent_busy.get(agent_id, False):
                continue

            curr = self.agent_node.get(agent_id)
            if curr is None:
                continue

            # Local actions
            if self.work_nodes and (curr in self.work_nodes) and (random.random() < self.p_work):
                self._publish(self._ev_start_work(curr, agent_id))
                self.agent_busy[agent_id] = True
                continue

            if self.charge_nodes and (curr in self.charge_nodes) and (random.random() < self.p_charge):
                self._publish(self._ev_start_charge(curr, agent_id))
                self.agent_busy[agent_id] = True
                continue

            # Move along graph
            succ = list(self.G.successors(curr)) if hasattr(self.G, "successors") else []
            if not succ:
                succ = list(self.G.neighbors(curr))

            if not succ:
                continue

            nxt = random.choice(succ)
            self._publish(self._ev_take(curr, nxt, agent_id))
            self.agent_busy[agent_id] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--p-work", type=float, default=0.25)
    parser.add_argument("--p-charge", type=float, default=0.15)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rclpy.init()
    node = FleetTester(
        args.nodes,
        args.edges,
        p_work=args.p_work,
        p_charge=args.p_charge,
        rate_hz=args.rate,
        seed=args.seed,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

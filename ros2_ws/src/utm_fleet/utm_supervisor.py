#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple

import networkx as nx
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

from utm_graph import load_graph_data


_RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def dist2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def nearest_node_id(positions, x, y):
    return min(
        positions.items(),
        key=lambda kv: dist2((kv[1][0], kv[1][1]), (x, y)),
    )[0]


def to_bidirectional_multidigraph(G_in):
    H = nx.MultiDiGraph()

    for n, d in G_in.nodes(data=True):
        H.add_node(str(n), **(d or {}))

    def _add(u, v, k, data):
        u = str(u)
        v = str(v)
        if not H.has_edge(u, v, key=k):
            H.add_edge(u, v, key=k, **(data or {}))

    if isinstance(G_in, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, k, d in G_in.edges(keys=True, data=True):
            _add(u, v, k, d)
            _add(v, u, k, d)
    else:
        for u, v, d in G_in.edges(data=True):
            _add(u, v, 0, d)
            _add(v, u, 0, d)

    return H


@dataclass
class AgentShadowState:
    current_node: Optional[str] = None
    occupied_edge: Optional[Tuple[str, str]] = None
    has_task: bool = False


class UTMSupervisorNode(Node):
    def __init__(self, nodes_csv: str, edges_csv: str, use_task_queue: bool = True):
        super().__init__("utm_supervisor")

        self.nodes_csv = str(nodes_csv)
        self.edges_csv = str(edges_csv)
        self.use_task_queue = bool(use_task_queue)

        gd = load_graph_data(self.nodes_csv, self.edges_csv, add_euclidean_weight=True)
        self.G = to_bidirectional_multidigraph(gd.graph)
        self.positions = getattr(gd, "positions", {}) or {}
        self.spawns = list(getattr(gd, "spawns", []) or [])

        self._lock = threading.RLock()

        self.agent_states: Dict[int, AgentShadowState] = {}
        self.active_agents: Set[int] = set()

        self.task_counter = 0
        self.task_queue: Deque[str] = deque()

        self.all_take_events: Set[str] = set()
        self.incoming_take_events_by_node: Dict[str, Set[str]] = {}

        self._build_event_index()
        self._init_agent_states()

        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub_prohibited = self.create_publisher(String, "/prohibited_events", qos_latched)
        self.pub_shadow = self.create_publisher(String, "/utm_shadow_state", qos_latched)
        self.pub_task = self.create_publisher(String, "/task", 20)

        self.sub_event = self.create_subscription(String, "/event", self._on_event, 50)

        if self.use_task_queue:
            self.sub_task_todo = self.create_subscription(
                String,
                "/task_todo",
                self._on_task_todo,
                20,
            )
        else:
            self.sub_task_todo = None

        self._publish_shadow_state()
        self._publish_prohibited_events()

        self.get_logger().info(
            "utm_supervisor ready | agents=%d | queue=%s"
            % (len(self.agent_states), str(self.use_task_queue))
        )

    # ------------------------------------------------------------------
    # graph semantics
    # ------------------------------------------------------------------

    def _kind(self, node_id: str) -> str:
        n = str(node_id)
        s = n.upper()

        t = ""
        try:
            nd = self.G.nodes[n]
            t = str(nd.get("type", nd.get("tipo", nd.get("kind", "")))).upper()
        except Exception:
            t = ""

        if "VERTIPORT" in s or "VERTIPORT" in t:
            return "VERTIPORT"
        if ("STATION" in s) or ("ESTACAO" in s) or ("CHARG" in s) or ("STATION" in t) or ("ESTACAO" in t) or ("CHARG" in t):
            return "STATION"
        if ("SUPPLIER" in s) or ("FORNECEDOR" in s) or ("SUPPLIER" in t) or ("FORNECEDOR" in t):
            return "SUPPLIER"
        if ("CLIENT" in s) or ("CLIENTE" in s) or ("CLIENT" in t) or ("CLIENTE" in t):
            return "CLIENT"
        return "NORMAL"

    def _node_allows_shared_occupancy(self, node_id: str) -> bool:
        return self._kind(node_id) in {"VERTIPORT", "SUPPLIER", "CLIENT", "STATION"}

    @staticmethod
    def _edge_bundle(u: str, v: str) -> Tuple[str, str]:
        return (u, v) if u <= v else (v, u)

    def _build_event_index(self):
        incoming = {}

        for _u, v in self.G.edges():
            incoming.setdefault(str(v), set())

        for u, v, _k, _d in self.G.edges(keys=True, data=True):
            u = str(u)
            v = str(v)
            ev = f"edge_take::{u}::{v}"
            self.all_take_events.add(ev)
            incoming.setdefault(v, set()).add(ev)

        self.incoming_take_events_by_node = incoming

    def _init_agent_states(self):
        if self.spawns:
            for i, sp in enumerate(self.spawns):
                node_id = nearest_node_id(
                    self.positions,
                    float(sp.x),
                    float(sp.y),
                )
                self.agent_states[int(i)] = AgentShadowState(
                    current_node=str(node_id),
                    occupied_edge=None,
                    has_task=False,
                )
        else:
            # fallback minimal
            first = str(next(iter(self.G.nodes())))
            self.agent_states[0] = AgentShadowState(
                current_node=first,
                occupied_edge=None,
                has_task=False,
            )

    # ------------------------------------------------------------------
    # task queue
    # ------------------------------------------------------------------

    def _on_task_todo(self, msg: String):
        raw = str(msg.data or "").strip()
        if not raw:
            return

        if ":" in raw:
            task_full = raw
        else:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != 2:
                self.get_logger().warning("invalid /task_todo payload: '%s'" % raw)
                return

            with self._lock:
                self.task_counter += 1
                task_full = f"T{self.task_counter}:{parts[0]},{parts[1]}"

        with self._lock:
            self.task_queue.append(task_full)

        self.get_logger().info("queued task '%s'" % task_full)
        self._dispatch_tasks_if_possible()

    def _dispatch_tasks_if_possible(self):
        if not self.use_task_queue:
            return

        to_publish = []

        with self._lock:
            free_slots = max(0, len(self.agent_states) - len(self.active_agents))

            while free_slots > 0 and self.task_queue:
                task = self.task_queue.popleft()
                to_publish.append(task)
                free_slots -= 1

        for task in to_publish:
            self.pub_task.publish(String(data=task))
            self.get_logger().info("dispatched task '%s'" % task)

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_event_and_agent(ev_raw: str) -> Tuple[str, Optional[int]]:
        m = _RE_SUFFIX.match(str(ev_raw))
        if not m:
            return str(ev_raw), None
        return m.group(1), int(m.group(2))

    def _on_event(self, msg: String):
        raw = str(msg.data or "").strip()
        if not raw:
            return

        generic_ev, agent_id = self._parse_event_and_agent(raw)

        publish_tasks = False
        publish_state = False
        publish_prohibited = False

        with self._lock:
            if agent_id is not None and agent_id not in self.agent_states:
                self.agent_states[agent_id] = AgentShadowState()

            shadow = self.agent_states.get(agent_id) if agent_id is not None else None

            if generic_ev == "task_accept" and shadow is not None:
                shadow.has_task = True
                self.active_agents.add(agent_id)
                publish_state = True

            elif generic_ev == "task_done" and shadow is not None:
                shadow.has_task = False
                self.active_agents.discard(agent_id)
                publish_state = True
                publish_tasks = True

            elif generic_ev.startswith("edge_take::") and shadow is not None:
                parts = generic_ev.split("::")
                if len(parts) == 3:
                    _tag, u, v = parts
                    shadow.current_node = str(v)
                    shadow.occupied_edge = self._edge_bundle(str(u), str(v))
                    publish_state = True
                    publish_prohibited = True

            elif generic_ev.startswith("edge_release::") and shadow is not None:
                shadow.occupied_edge = None
                publish_state = True
                publish_prohibited = True

        if publish_state:
            self._publish_shadow_state()

        if publish_prohibited or generic_ev in ("task_accept", "task_done"):
            self._publish_prohibited_events()

        if publish_tasks:
            self._dispatch_tasks_if_possible()

    # ------------------------------------------------------------------
    # prohibited-event computation
    # ------------------------------------------------------------------

    def _compute_prohibited_take_events(self) -> Set[str]:
        with self._lock:
            occupied_edges = set()
            reserved_nodes = set()

            for shadow in self.agent_states.values():
                if shadow.occupied_edge is not None:
                    occupied_edges.add(shadow.occupied_edge)

                if shadow.current_node and not self._node_allows_shared_occupancy(shadow.current_node):
                    reserved_nodes.add(str(shadow.current_node))

        prohibited = set()

        for ev in self.all_take_events:
            parts = ev.split("::")
            if len(parts) != 3:
                continue

            _tag, u, v = parts
            bundle = self._edge_bundle(str(u), str(v))

            if bundle in occupied_edges:
                prohibited.add(ev)
                continue

            if str(v) in reserved_nodes:
                prohibited.add(ev)

        return prohibited

    def _publish_prohibited_events(self):
        prohibited = self._compute_prohibited_take_events()
        payload = ",".join(sorted(prohibited))
        self.pub_prohibited.publish(String(data=payload))
        self.get_logger().info("published %d prohibited edge_take events" % len(prohibited))

    def _publish_shadow_state(self):
        with self._lock:
            chunks = []
            for agent_id in sorted(self.agent_states.keys()):
                st = self.agent_states[agent_id]
                chunks.append(
                    "agent=%d node=%s edge=%s task=%s" % (
                        agent_id,
                        str(st.current_node),
                        str(st.occupied_edge),
                        str(st.has_task),
                    )
                )

        self.pub_shadow.publish(String(data=" | ".join(chunks)))


def main():
    parser = argparse.ArgumentParser(description="Global UTM supervisor for fleet conflict management")
    parser.add_argument("--nodes", required=True, help="CSV with graph nodes")
    parser.add_argument("--edges", required=True, help="CSV with graph edges")
    parser.add_argument(
        "--no-task-queue",
        action="store_true",
        help="Disable /task_todo -> /task queue dispatch",
    )
    args = parser.parse_args()

    rclpy.init()
    node = None

    try:
        node = UTMSupervisorNode(
            nodes_csv=str(args.nodes),
            edges_csv=str(args.edges),
            use_task_queue=not bool(args.no_task_queue),
        )
        rclpy.spin(node)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
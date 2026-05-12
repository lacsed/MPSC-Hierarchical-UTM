#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from utm_graph import load_graph_data
from utm_interfaces.srv import RequestEvent
from ultrades.automata import *


_RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def map_enabled_events(automaton: Any) -> Dict[Any, Set[Any]]:
    enabled_by_state: Dict[Any, Set[Any]] = {q: set() for q in states(automaton)}

    for q, e, _nq in transitions(automaton):
        if q in enabled_by_state:
            enabled_by_state[q].add(e)

    return enabled_by_state


def compute_forbidden_take_events(supervisor: Any, plants: List[Any]) -> Dict[Any, Set[Any]]:
    supervisor_enabled = map_enabled_events(supervisor)
    plant_enabled_maps = [map_enabled_events(A) for A in plants]
    plant_state_maps = [{str(q): q for q in states(A)} for A in plants]

    forbidden_by_state: Dict[Any, Set[Any]] = {}

    for q_sup in states(supervisor):
        q_sup_str = str(q_sup)
        components = q_sup_str.split("|")

        enabled_in_supervisor = supervisor_enabled.get(q_sup, set())
        feasible_in_plants: Set[Any] = set()

        for i in range(min(len(plants), len(components))):
            q_name = components[i]
            q_plant = plant_state_maps[i].get(q_name)

            if q_plant is None:
                continue

            feasible_in_plants.update(plant_enabled_maps[i].get(q_plant, set()))

        feasible_take = {
            e for e in feasible_in_plants
            if str(e).startswith("edge_take::")
        }

        enabled_take = {
            e for e in enabled_in_supervisor
            if str(e).startswith("edge_take::")
        }

        forbidden_by_state[q_sup] = feasible_take.difference(enabled_take)

    return forbidden_by_state


class GenericUTMModel:
    def __init__(self, nodes_csv, edges_csv, init_node):
        self.nodes_csv = str(nodes_csv)
        self.edges_csv = str(edges_csv)
        self.init_node = str(init_node)

        gd = load_graph_data(self.nodes_csv, self.edges_csv, add_euclidean_weight=True)
        G0 = gd.graph
        self.G = self._to_bidirectional_multidigraph(G0)

        self.pos = {}
        for nid, p in (getattr(gd, "positions", None) or {}).items():
            nid = str(nid)
            if p is None:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2]) if len(p) >= 3 else 0.0
                self.pos[nid] = (x, y, z)
            except Exception:
                continue

        self.events = {}
        self.edge_bundle = {}
        self.node_states = {}
        self.automata = {}
        self.plants = []
        self.specs = []

        self._build_alphabet()
        self._build_map_plant()
        self._build_block_command_plant()
        self._build_vertex_block_specs()
        self._build_global_block_spec()
        self._build_vertex_mutex_specs()

        self.supervisor_mono = None
        self.supervisor_mono = self.compute_monolithic_supervisor()
        self.forbidden_events_by_state = compute_forbidden_take_events(
            self.supervisor_mono,
            self.plants,
        )

    def __getstate__(self):
        return {
            "nodes_csv": self.nodes_csv,
            "edges_csv": self.edges_csv,
            "init_node": self.init_node,
        }

    def __setstate__(self, state):
        self.__init__(
            nodes_csv=state["nodes_csv"],
            edges_csv=state["edges_csv"],
            init_node=state["init_node"],
        )

    def ev(self, name):
        return self.events[name]

    @staticmethod
    def _to_bidirectional_multidigraph(G_in):
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

    def _kind(self, node_id):
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

    def _build_alphabet(self):
        E = self.events

        for u, v, k, _d in self.G.edges(keys=True, data=True):
            u = str(u)
            v = str(v)

            take_uv = f"edge_take::{u}::{v}"
            take_vu = f"edge_take::{v}::{u}"

            for nm in (take_uv, take_vu):
                if nm not in E:
                    E[nm] = event(nm, controllable=True)

            a, b = (u, v) if u <= v else (v, u)
            key = ((a, b), k)

            if key not in self.edge_bundle:
                take_ab = E[f"edge_take::{a}::{b}"]
                take_ba = E[f"edge_take::{b}::{a}"]
                self.edge_bundle[key] = (take_ab, take_ba)

        for n in self.G.nodes():
            n = str(n)

            block = f"block::{n}"
            unblock = f"unblock::{n}"

            if block not in E:
                E[block] = event(block, controllable=True)

            if unblock not in E:
                E[unblock] = event(unblock, controllable=True)

    def _build_map_plant(self):
        initial = None

        for n in self.G.nodes():
            n = str(n)
            st = state(n, marked=(n == self.init_node))
            self.node_states[n] = st

            if n == self.init_node:
                initial = st

        if initial is None:
            first = str(next(iter(self.G.nodes())))
            initial = self.node_states[first]

        trs = []

        for (ab, _k), (take_ab, take_ba) in self.edge_bundle.items():
            a, b = ab
            sa = self.node_states[a]
            sb = self.node_states[b]

            trs.append((sa, take_ab, sb))
            trs.append((sb, take_ba, sa))

        A = dfa(trs, initial, "utm_map")
        self.automata["utm_map"] = A
        self.plants.append(A)

    def _build_block_command_plant(self):
        Ready = state("BLOCK_COMMAND_READY", marked=True)
        trs = []

        for n in self.G.nodes():
            n = str(n)

            block = self.ev(f"block::{n}")
            unblock = self.ev(f"unblock::{n}")

            trs.append((Ready, block, Ready))
            trs.append((Ready, unblock, Ready))

        A = accessible(dfa(trs, Ready, "utm_block_command"))
        self.automata["utm_block_command"] = A
        self.plants.append(A)

    def _build_vertex_block_specs(self):
        for v in self.G.nodes():
            v = str(v)

            Free = state(f"VERTEX_UNBLOCKED::{v}", marked=True)
            Blocked = state(f"VERTEX_BLOCKED::{v}")

            block = self.ev(f"block::{v}")
            unblock = self.ev(f"unblock::{v}")

            trs = [
                (Free, block, Blocked),
                (Blocked, unblock, Free),
            ]

            for u in self.G.predecessors(v):
                u = str(u)
                evn = f"edge_take::{u}::{v}"

                if evn in self.events:
                    trs.append((Free, self.ev(evn), Free))

            A = accessible(dfa(trs, Free, f"utm_vertex_block::{v}"))
            self.automata[f"utm_vertex_block::{v}"] = A
            self.specs.append(A)

    def _build_global_block_spec(self):
        Free = state("GLOBAL_UNBLOCKED", marked=True)
        Blocked = state("GLOBAL_BLOCKED")

        trs = []

        for n in self.G.nodes():
            n = str(n)

            block = self.ev(f"block::{n}")
            unblock = self.ev(f"unblock::{n}")

            trs.append((Free, block, Blocked))
            trs.append((Blocked, unblock, Free))

        A = dfa(trs, Free, "utm_global_block")
        self.automata["utm_global_block"] = A
        self.specs.append(A)

    def _build_vertex_mutex_specs(self):
        special_nodes = {
            "VERTIPORT",
            "FORNECEDOR",
            "CLIENTE",
            "ESTACAO",
            "SUPPLIER",
            "CLIENT",
            "STATION",
            "CHARG",
            "CHARGING",
        }

        for v, _data in self.G.nodes(data=True):
            v = str(v)

            if any(special in v.upper() for special in special_nodes):
                continue

            Free = state(f"VERTEX_FREE::{v}", marked=True)
            Occupied = state(f"VERTEX_OCC::{v}")

            trs = []

            for u in set(self.G.predecessors(v)):
                u = str(u)
                evn = f"edge_take::{u}::{v}"

                if evn in self.events:
                    trs.append((Free, self.ev(evn), Occupied))

            for w in set(self.G.successors(v)):
                w = str(w)
                evn = f"edge_take::{v}::{w}"

                if evn in self.events:
                    trs.append((Occupied, self.ev(evn), Free))

            if not trs:
                continue

            A = accessible(dfa(trs, Free, f"utm_vertex_mutex::{v}"))
            self.automata[f"utm_vertex_mutex::{v}"] = A
            self.specs.append(A)

    def compute_monolithic_supervisor(self, force=False):
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plants, self.specs)
        return self.supervisor_mono


class UTMSupervisorNode(Node):
    def __init__(
        self,
        nodes_csv: str,
        edges_csv: str,
        init_node: str,
        mutex_vertices: str = "all_except_vertiport",
        grant_timeout_s: float = 3.0,
    ):
        super().__init__("utm_supervisor")

        gd = load_graph_data(nodes_csv, edges_csv, add_euclidean_weight=True)
        G0 = GenericUTMModel._to_bidirectional_multidigraph(gd.graph)

        if init_node is None or not str(init_node).strip():
            init_node = next(
                (str(n) for n in G0.nodes() if self._kind_static(str(n), G0) == "VERTIPORT"),
                str(next(iter(G0.nodes()))),
            )

        self.model = GenericUTMModel(nodes_csv, edges_csv, str(init_node))
        self.supervisor = self.model.supervisor_mono
        self.forbidden_events_by_state = self.model.forbidden_events_by_state

        self.mutex_vertices = str(mutex_vertices)
        self.grant_timeout_s = float(grant_timeout_s)

        self.lock = threading.RLock()

        self.agent_states: Dict[str, Any] = {}
        self.agent_current_node: Dict[str, str] = {}
        self.pending_grant: Dict[str, Tuple[str, str, str, float]] = {}

        self.edge_owner: Dict[Tuple[str, str], str] = {}
        self.vertex_owner: Dict[str, str] = {}
        self.blocked_vertices: Set[str] = set()

        self.vertiports: Set[str] = {
            str(n)
            for n in self.model.G.nodes()
            if self.model._kind(str(n)) == "VERTIPORT"
        }

        self.task_counter = 0
        self.task_queue = deque()

        qos = rclpy.qos.QoSProfile(depth=100)

        self.srv_request = self.create_service(
            RequestEvent,
            "/utm/request_event",
            self._on_request_event,
        )

        self.sub_event = self.create_subscription(
            String,
            "/event",
            self._on_event,
            qos,
        )

        self.sub_task_todo = self.create_subscription(
            String,
            "/task_todo",
            self._on_task_todo,
            qos,
        )

        self.sub_geofence = self.create_subscription(
            String,
            "/utm/geofence_cmd",
            self._on_geofence_cmd,
            qos,
        )

        self.pub_prohibited = self.create_publisher(
            String,
            "/prohibited_events",
            qos,
        )

        self.pub_task = self.create_publisher(
            String,
            "/task",
            qos,
        )

        self.pub_occupancy = self.create_publisher(
            String,
            "/utm/occupancy",
            qos,
        )

        self.transition_map = {}
        for q, e, nq in transitions(self.supervisor):
            self.transition_map[(q, e)] = nq

        self.create_timer(0.5, self._broadcast_status)

        self.get_logger().info(
            f"UTM supervisor ready | plants={len(self.model.plants)} | "
            f"specs={len(self.model.specs)} | states={len(list(states(self.supervisor)))}"
        )

    @staticmethod
    def _kind_static(node_id, G):
        n = str(node_id)
        s = n.upper()
        t = ""

        try:
            nd = G.nodes[n]
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

    def _parse_event(self, raw):
        raw = str(raw or "").strip()

        if not raw:
            return "", None

        if raw in self.model.events:
            return raw, None

        m = _RE_SUFFIX.match(raw)

        if m:
            candidate = m.group(1)
            agent_id = m.group(2)

            if candidate in self.model.events:
                return candidate, agent_id

            if candidate.startswith("edge_release::"):
                return candidate, agent_id

            return candidate, agent_id

        return raw, None

    def _edge_from_take(self, ev_name):
        if not ev_name.startswith("edge_take::"):
            return None

        parts = ev_name.split("::")

        if len(parts) != 3:
            return None

        return parts[1], parts[2]

    def _edge_from_release(self, ev_name):
        if not ev_name.startswith("edge_release::"):
            return None

        parts = ev_name.split("::")

        if len(parts) != 3:
            return None

        return parts[1], parts[2]

    def _edge_key(self, u, v):
        u = str(u)
        v = str(v)
        return (u, v) if u <= v else (v, u)

    def _is_vertiport(self, n):
        return str(n) in self.vertiports

    def _is_mutex_vertex(self, n):
        n = str(n)

        if self._is_vertiport(n):
            return False

        if self.mutex_vertices == "none":
            return False

        if self.mutex_vertices == "logical":
            return self.model._kind(n) == "NORMAL"

        return True

    def _sanitize(self):
        for v in list(self.vertiports):
            self.blocked_vertices.discard(v)
            self.vertex_owner.pop(v, None)

    def _ensure_agent(self, agent_id, current_node_hint=None):
        agent_id = str(agent_id)

        if agent_id not in self.agent_current_node:
            if current_node_hint is not None:
                self.agent_current_node[agent_id] = str(current_node_hint)
            else:
                self.agent_current_node[agent_id] = self.model.init_node

        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = initial_state(self.supervisor)

    def _advance_supervisor(self, agent_id, ev_name):
        if ev_name not in self.model.events:
            return True

        self._ensure_agent(agent_id)

        q = self.agent_states[agent_id]
        e = self.model.ev(ev_name)
        nq = self.transition_map.get((q, e))

        if nq is None:
            return False

        self.agent_states[agent_id] = nq
        return True

    def _get_supervisor_forbidden(self):
        out = set()

        for q in self.agent_states.values():
            for e in self.forbidden_events_by_state.get(q, set()):
                name = str(e)

                if name.startswith("edge_take::"):
                    out.add(name)

        return out

    def _get_runtime_forbidden(self):
        self._sanitize()

        out = set()

        for name in self.model.events:
            if not name.startswith("edge_take::"):
                continue

            edge = self._edge_from_take(name)

            if edge is None:
                continue

            u, v = edge
            key = self._edge_key(u, v)

            if key in self.edge_owner:
                out.add(name)
                continue

            if (not self._is_vertiport(v)) and v in self.blocked_vertices:
                out.add(name)
                continue

            if (not self._is_vertiport(v)) and self._is_mutex_vertex(v):
                if v in self.vertex_owner:
                    out.add(name)
                    continue

        return out

    def _get_forbidden(self):
        out = set()
        out.update(self._get_supervisor_forbidden())
        out.update(self._get_runtime_forbidden())
        return out

    def _publish_prohibited(self):
        disabled = sorted(self._get_forbidden())
        self.pub_prohibited.publish(String(data=",".join(disabled)))

    def _publish_occupancy(self):
        self._sanitize()

        data = {
            "blocked_vertices": sorted(self.blocked_vertices),
            "edge_owner": {
                f"{k[0]}--{k[1]}": owner
                for k, owner in sorted(self.edge_owner.items())
            },
            "vertex_owner": dict(sorted(self.vertex_owner.items())),
            "agent_current_node": dict(sorted(self.agent_current_node.items())),
            "pending_grant": {
                aid: {
                    "u": item[0],
                    "v": item[1],
                    "event": item[2],
                    "age_s": round(time.time() - item[3], 2),
                }
                for aid, item in sorted(self.pending_grant.items())
            },
            "vertiports": sorted(self.vertiports),
        }

        self.pub_occupancy.publish(String(data=json.dumps(data, sort_keys=True)))

    def _cleanup_stale_grants(self):
        now = time.time()
        expired = []

        for aid, item in list(self.pending_grant.items()):
            if now - item[3] >= self.grant_timeout_s:
                expired.append(aid)

        for aid in expired:
            item = self.pending_grant.pop(aid, None)

            if item is None:
                continue

            u, v, _ev, _ts = item
            key = self._edge_key(u, v)

            if self.edge_owner.get(key) == aid:
                self.edge_owner.pop(key, None)

            if self.vertex_owner.get(v) == aid:
                self.vertex_owner.pop(v, None)

            if self._is_mutex_vertex(u):
                self.vertex_owner[u] = aid

            self.get_logger().warn(f"grant timeout cancelled: agent={aid}, event={_ev}")

    def _broadcast_status(self):
        with self.lock:
            self._cleanup_stale_grants()
            self._publish_prohibited()
            self._publish_occupancy()

    def _on_request_event(self, req, resp):
        raw = str(req.event or "").strip()
        ev_name, parsed_agent = self._parse_event(raw)
        agent_id = str(req.agent_id or parsed_agent or "").strip()

        with self.lock:
            self._cleanup_stale_grants()

            if not agent_id:
                resp.accepted = False
                resp.reason = "missing agent_id"
                resp.prohibited_events = sorted(self._get_forbidden())
                return resp

            if bool(req.cancel):
                ok, reason = self._cancel_grant(agent_id, ev_name)
                resp.accepted = ok
                resp.reason = reason
                resp.prohibited_events = sorted(self._get_forbidden())
                self._publish_prohibited()
                self._publish_occupancy()
                return resp

            if not ev_name.startswith("edge_take::"):
                resp.accepted = True
                resp.reason = "non-edge-take event accepted"
                resp.prohibited_events = sorted(self._get_forbidden())
                return resp

            ok, reason = self._grant_edge_take(agent_id, ev_name)

            resp.accepted = ok
            resp.reason = reason
            resp.prohibited_events = sorted(self._get_forbidden())

            self._publish_prohibited()
            self._publish_occupancy()

            if ok:
                self.get_logger().info(f"ACCEPT {ev_name}_{agent_id}: {reason}")
            else:
                self.get_logger().warn(f"REJECT {ev_name}_{agent_id}: {reason}")

            return resp

    def _grant_edge_take(self, agent_id, ev_name):
        edge = self._edge_from_take(ev_name)

        if edge is None:
            return False, f"malformed edge_take event: {ev_name}"

        if ev_name not in self.model.events:
            return False, f"unknown UTM event: {ev_name}"

        if ev_name in self._get_supervisor_forbidden():
            return False, f"disabled by UTM supervisor: {ev_name}"

        u, v = edge

        self._ensure_agent(agent_id, current_node_hint=u)

        current = self.agent_current_node.get(agent_id)

        if current is not None and str(current) != str(u):
            return False, f"source mismatch: agent at {current}, requested {u}"

        if agent_id in self.pending_grant:
            old_u, old_v, old_ev, _ts = self.pending_grant[agent_id]

            if old_ev == ev_name:
                return True, "grant already pending"

            return False, f"agent has another pending grant: {old_ev}"

        key = self._edge_key(u, v)

        if key in self.edge_owner:
            return False, f"edge occupied by {self.edge_owner[key]}"

        if (not self._is_vertiport(v)) and v in self.blocked_vertices:
            return False, f"destination blocked: {v}"

        if (not self._is_vertiport(v)) and self._is_mutex_vertex(v):
            owner = self.vertex_owner.get(v)

            if owner is not None:
                return False, f"destination occupied by {owner}"

        self.edge_owner[key] = agent_id

        if (not self._is_vertiport(v)) and self._is_mutex_vertex(v):
            self.vertex_owner[v] = agent_id

        if self._is_mutex_vertex(u) and self.vertex_owner.get(u) == agent_id:
            self.vertex_owner.pop(u, None)

        self.pending_grant[agent_id] = (u, v, ev_name, time.time())

        return True, "grant issued"

    def _cancel_grant(self, agent_id, ev_name):
        item = self.pending_grant.get(agent_id)

        if item is None:
            return True, "no pending grant"

        u, v, old_ev, _ts = item

        if ev_name and ev_name.startswith("edge_take::") and ev_name != old_ev:
            return False, f"pending grant is {old_ev}, not {ev_name}"

        self.pending_grant.pop(agent_id, None)

        key = self._edge_key(u, v)

        if self.edge_owner.get(key) == agent_id:
            self.edge_owner.pop(key, None)

        if self.vertex_owner.get(v) == agent_id:
            self.vertex_owner.pop(v, None)

        if self._is_mutex_vertex(u):
            self.vertex_owner[u] = agent_id

        return True, "grant cancelled"

    def _on_event(self, msg):
        raw = str(msg.data or "").strip()

        if not raw:
            return

        ev_name, agent_id = self._parse_event(raw)

        with self.lock:
            self._cleanup_stale_grants()

            if ev_name.startswith("block::") or ev_name.startswith("unblock::"):
                self._apply_geofence(ev_name)
                self._publish_prohibited()
                self._publish_occupancy()
                return

            if ev_name.startswith("edge_take::"):
                if agent_id is not None:
                    self._confirm_edge_take(agent_id, ev_name)
                    self._publish_prohibited()
                    self._publish_occupancy()
                return

            if ev_name.startswith("edge_release::"):
                if agent_id is not None:
                    self._release_edge(agent_id, ev_name)
                    self._publish_prohibited()
                    self._publish_occupancy()
                return

    def _confirm_edge_take(self, agent_id, ev_name):
        edge = self._edge_from_take(ev_name)

        if edge is None:
            return

        u, v = edge
        key = self._edge_key(u, v)

        item = self.pending_grant.get(agent_id)

        if item is not None:
            self.pending_grant.pop(agent_id, None)
        else:
            owner = self.edge_owner.get(key)

            if owner is not None and owner != agent_id:
                self.get_logger().error(f"edge_take without grant and occupied: {ev_name}_{agent_id}")
                return

            self.edge_owner[key] = agent_id

        self._ensure_agent(agent_id, current_node_hint=u)
        self.agent_current_node[agent_id] = u

        self._advance_supervisor(agent_id, ev_name)

        self.get_logger().info(f"CONFIRMED {ev_name}_{agent_id}")

    def _release_edge(self, agent_id, ev_name):
        edge = self._edge_from_release(ev_name)

        if edge is None:
            return

        u, v = edge
        key = self._edge_key(u, v)

        if self.edge_owner.get(key) == agent_id:
            self.edge_owner.pop(key, None)

        self.pending_grant.pop(agent_id, None)

        if (not self._is_vertiport(v)) and self._is_mutex_vertex(v):
            self.vertex_owner[v] = agent_id

        self.agent_current_node[agent_id] = v

        self.get_logger().info(f"RELEASE {ev_name}_{agent_id}")

    def _on_geofence_cmd(self, msg):
        ev_name = str(msg.data or "").strip()

        if not ev_name:
            return

        with self.lock:
            self._apply_geofence(ev_name)
            self._publish_prohibited()
            self._publish_occupancy()

    def _apply_geofence(self, ev_name):
        if ev_name.startswith("block::"):
            n = ev_name.split("block::", 1)[1]

            if n not in self.model.G.nodes:
                return

            if self._is_vertiport(n):
                self.blocked_vertices.discard(n)
                self.vertex_owner.pop(n, None)
                return

            self.blocked_vertices.add(n)

            for aid in list(self.agent_states.keys()):
                self._advance_supervisor(aid, ev_name)

            return

        if ev_name.startswith("unblock::"):
            n = ev_name.split("unblock::", 1)[1]
            self.blocked_vertices.discard(n)

            if self._is_vertiport(n):
                self.vertex_owner.pop(n, None)

            for aid in list(self.agent_states.keys()):
                self._advance_supervisor(aid, ev_name)

    def _on_task_todo(self, msg):
        raw = str(msg.data or "").strip()

        if not raw or "," not in raw:
            return

        with self.lock:
            self.task_counter += 1
            task_id = f"T{self.task_counter}"
            payload = f"{task_id}:{raw}"

        self.pub_task.publish(String(data=payload))
        self.get_logger().info(f"DISPATCH TASK {payload}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="ROS 2 UTM Supervisor with atomic edge authorization"
    )

    parser.add_argument(
        "--nodes",
        required=True,
        help="Path to graph_nodes.csv",
    )

    parser.add_argument(
        "--edges",
        required=True,
        help="Path to graph_edges.csv",
    )

    parser.add_argument(
        "--init-node",
        default="",
        help="Initial DES node, usually the common vertiport",
    )

    parser.add_argument(
        "--mutex-vertices",
        default="all_except_vertiport",
        choices=["all", "all_except_vertiport", "logical", "none"],
        help="Which vertices are protected by mutex. Vertiports are always excluded.",
    )

    parser.add_argument(
        "--grant-timeout",
        type=float,
        default=3.0,
        help="Seconds to keep an accepted grant if the corresponding /event edge_take is not observed.",
    )

    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = UTMSupervisorNode(
        nodes_csv=parsed.nodes,
        edges_csv=parsed.edges,
        init_node=parsed.init_node,
        mutex_vertices=parsed.mutex_vertices,
        grant_timeout_s=float(parsed.grant_timeout),
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
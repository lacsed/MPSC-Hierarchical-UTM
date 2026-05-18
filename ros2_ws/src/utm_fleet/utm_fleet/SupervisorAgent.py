#!/usr/bin/env python3

from __future__ import annotations

import os

import re
import threading
import time
import networkx as nx
from typing import List, Optional, Set, Tuple
from .sim_metrics import CSVMetricLogger

from ultrades.automata import (
    dfa,
    event,
    initial_state,
    is_controllable,
    is_marked,
    transitions,
)

from .help_cost import build_cost_engine, rebuild_all_costs
from .milp_optimizer import otimizador

#!/usr/bin/env python3



import argparse
import multiprocessing as mp
import random
import re
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String
from utm_interfaces.srv import RequestEvent
from .uav_hardware import *
from .dispatch_hw import *

class SupervisorAgent:
    _RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")

    def __init__(
        self,
        model,
        agent_id: int,
        supervisor_mono=None,
        planning_horizon: int = 10,
        optimize_fn=None,
        speed_mps: float = 2.0,
        energy_per_meter: float = 0.10,
        base_time_cost: float = 0.10,
        cost_params=None,
    ):
        self.model = model
        self.id = int(agent_id)
        self.optimize_fn = optimize_fn if optimize_fn is not None else otimizador
        self.planning_horizon = int(planning_horizon)

        self.baseline = os.environ.get("UTM_BASELINE", "proposed").strip().lower()
        self.run_id = os.environ.get("UTM_RUN_ID", "").strip()
        self.scenario_id = os.environ.get("UTM_SCENARIO", "default").strip()
        self.graph_size = os.environ.get("UTM_GRAPH_SIZE", "current").strip()
        self.density = os.environ.get("UTM_DENSITY", "medium").strip()
        self.seed = os.environ.get("UTM_SEED", "").strip()
        self.num_uavs = int(os.environ.get("UTM_UAVS", "0") or 0)
        self.num_nodes = int(len(self.model.G.nodes()))
        self.num_edges = int(len(self.model.G.edges()))
        self.num_tasks_config = int(os.environ.get("UTM_NUM_TASKS", "0") or 0)
        self.speed_mps = float(speed_mps) if float(speed_mps) > 0.0 else 1.0
        self.work_time_s = float(os.environ.get("UTM_WORK_TIME_S", "2.0") or 2.0)
        self.charge_time_s = float(os.environ.get("UTM_CHARGE_TIME_S", "5.0") or 5.0)
        self.control_rate_hz = float(os.environ.get("UTM_CONTROL_RATE_HZ", "0.0") or 0.0)

        self._accepted_task_count = 0
        self._last_plan_status = "NOT_RUN"
        self._last_plan_status_code = ""
        self._last_plan_interest_count = 0
        self._last_plan_forbidden_count = 0

        self._validate_model()

        self._task_raw: Optional[str] = None
        self._task_lock = threading.RLock()
        self._claimed_tasks: Set[str] = set()
        self.terminated = [False, False, False]  # supplier_done, client_done, returned_to_vertiport

        self._execution_buffer: List[str] = []
        self._buffer_lock = threading.RLock()

        self._prohibited_generic: Set[str] = set()
        self._prohibited_lock = threading.RLock()

        self._planner_thread: Optional[threading.Thread] = None
        self._planner_lock = threading.RLock()
        self._is_calculating = False
        self._last_plan_request = 0.0
        self._min_plan_interval = 0.05

        self._pending_command: Optional[str] = None
        self._predicted_completion_event: Optional[str] = None
        self._pending_lock = threading.RLock()

        self.last_state_entry_time = time.time()

        self._forbid_unassigned_special_nodes = (
            os.environ.get("UTM_FORBID_UNASSIGNED_SPECIALS", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        self._idle_same_position_penalty_per_s = float(
            os.environ.get("UTM_IDLE_POSITION_PENALTY_PER_S", "4.0") or 4.0
        )
        self._idle_same_position_penalty_cap = float(
            os.environ.get("UTM_IDLE_POSITION_PENALTY_CAP", "40.0") or 40.0
        )
        self._same_position_future_penalty = float(
            os.environ.get("UTM_SAME_POSITION_FUTURE_PENALTY", "10.0") or 10.0
        )

        # Anti-gridlock parameters.  These parameters are intentionally
        # local to the UAV layer: SCT/UTM still determines admissibility,
        # while the MPSC layer is biased against remaining stopped at a
        # congested vertex when an admissible outgoing edge exists.
        self._stuck_escape_after_s = float(
            os.environ.get("UTM_STUCK_ESCAPE_AFTER_S", "0.75") or 0.75
        )
        self._rejected_edge_ttl_s = float(
            os.environ.get("UTM_REJECTED_EDGE_TTL_S", "2.5") or 2.5
        )
        self._altitude_escape_bonus = float(
            os.environ.get("UTM_ALTITUDE_ESCAPE_BONUS", "80.0") or 80.0
        )
        self._escape_edge_distance_weight = float(
            os.environ.get("UTM_ESCAPE_EDGE_DISTANCE_WEIGHT", "1.0") or 1.0
        )
        self._escape_target_hop_weight = float(
            os.environ.get("UTM_ESCAPE_TARGET_HOP_WEIGHT", "1000.0") or 1000.0
        )
        self._recently_rejected_generic = {}
        self._recently_rejected_lock = threading.RLock()

        self._planner_logger = CSVMetricLogger(
            f"planner_agent_{self.id}.csv",
            [
                "t_wall",
                "run_id",
                "scenario_id",
                "baseline",
                "graph_size",
                "density",
                "seed",
                "num_uavs",
                "num_nodes",
                "num_edges",
                "num_tasks",
                "planning_horizon",
                "agent_id",
                "reason",
                "task",
                "task_phase",
                "state",
                "terminated_supplier",
                "terminated_client",
                "terminated_base",
                "runtime_ms",
                "available_time_ms",
                "online_ratio",
                "selected_event",
                "selected_event_generic",
                "milp_status",
                "milp_status_code",
                "buffered_event",
                "forbidden_count",
                "interest_count",
                "soc",
            ],
        )

        self.cost_engine = build_cost_engine(
            model,
            speed_mps=speed_mps,
            energy_per_meter=energy_per_meter,
            base_time_cost=base_time_cost,
            params=cost_params,
        )
        self.base_sup_cost = dict(self.cost_engine["sup_cost"])
        self.dynamic_cost_dict = dict(self.base_sup_cost)

        if supervisor_mono is None:
            supervisor_mono = model.supervisor_mono or model.compute_monolithic_supervisor()
        self._sup_gen = supervisor_mono

        self.event_map = {}
        self.rev_event_map = {}
        self._event_objects = {}

        renamed_trs = []
        for q, e, q2 in list(transitions(self._sup_gen)):
            ev_gen = str(e)
            ev_id = f"{ev_gen}_{self.id}"

            if ev_id not in self._event_objects:
                ev_obj = event(ev_id, controllable=is_controllable(e))
                self._event_objects[ev_id] = ev_obj
                self.event_map[ev_gen] = ev_id
                self.rev_event_map[ev_id] = ev_gen

            renamed_trs.append((q, self._event_objects[ev_id], q2))

        self.supervisor = dfa(
            renamed_trs,
            initial_state(self._sup_gen),
            f"sup_id_{self.id}",
        )
        self._trs_id = list(transitions(self.supervisor))
        self._state = initial_state(self.supervisor)

    # ------------------------------------------------------------------
    # model validation
    # ------------------------------------------------------------------

    def _validate_model(self) -> None:
        vertiports = [str(n) for n in self.model.G.nodes() if self.model._kind(n) == "VERTIPORT"]
        if not vertiports:
            raise ValueError(
                "GenericUAVModel must contain at least one VERTIPORT. "
                "This SupervisorAgent does not support base fallback."
            )

    # ------------------------------------------------------------------
    # basic state API
    # ------------------------------------------------------------------

    def state(self):
        return self._state

    def state_str(self) -> str:
        return str(self._state)

    def is_marked_state(self) -> bool:
        return bool(is_marked(self._state))

    def to_generic(self, ev_with_id: str) -> str:
        return self.rev_event_map.get(str(ev_with_id), str(ev_with_id))

    def to_id(self, ev_generic: str) -> Optional[str]:
        return self.event_map.get(str(ev_generic))

    def current_task(self) -> Optional[str]:
        with self._task_lock:
            return self._task_raw

    def has_active_task(self) -> bool:
        return self.current_task() is not None

    def _should_process(self, ev: str) -> bool:
        m = self._RE_SUFFIX.match(str(ev))
        if not m:
            return False
        return int(m.group(2)) == self.id

    def enabled_events(self) -> List[str]:
        return self._enabled_events_from_state(self._state)

    def _enabled_events_from_state(self, state_obj) -> List[str]:
        s = str(state_obj)
        feasible = set()

        for q, e, _d in self._trs_id:
            if str(q) == s:
                feasible.add(str(e))

        with self._prohibited_lock:
            prohibited_generic = set(self._prohibited_generic)

        prohibited_generic.update(
            self._local_task_forbidden_events(
                state_obj=state_obj,
                terminated_flags=list(self.terminated),
            )
        )
        prohibited_generic.update(self._temporary_forbidden_events())

        out = []
        for ev_id in feasible:
            ev_gen = self.rev_event_map.get(ev_id)
            if ev_gen is None or ev_gen not in prohibited_generic:
                out.append(ev_id)

        return sorted(out)

    # ------------------------------------------------------------------
    # local task restrictions
    # ------------------------------------------------------------------

    def _is_restricted_task_special_node(self, node: str) -> bool:
        if node is None:
            return False

        kind = self.model._kind(str(node))

        # VERTIPORT and CHARGING/STATION nodes are always admissible local
        # destinations. They are safety-managed by the UTM/resource layer, not
        # by task-assignment filtering.
        if kind in ("VERTIPORT", "STATION"):
            return False

        return kind in ("SUPPLIER", "CLIENT")

    def _edge_take_destination(self, ev_generic: str) -> Optional[str]:
        ev_generic = str(ev_generic or "")
        if not ev_generic.startswith("edge_take::"):
            return None

        parts = ev_generic.split("::")
        if len(parts) != 3:
            return None

        return parts[2]

    def _allowed_supplier_client_nodes_for_task(self, terminated_flags=None) -> Set[str]:
        allowed: Set[str] = set()

        raw_task = self.current_task()
        if raw_task is None:
            return allowed

        parsed = self._parse_task(raw_task)
        if parsed is None:
            return allowed

        _task_id, supplier, client = parsed

        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        supplier_done = bool(terminated_flags[0])
        client_done = bool(terminated_flags[1])

        if not supplier_done:
            allowed.add(str(supplier))
        elif supplier_done and not client_done:
            allowed.add(str(client))

        return allowed

    def _local_task_forbidden_events(
        self,
        state_obj=None,
        terminated_flags=None,
    ) -> Set[str]:
        if not self._forbid_unassigned_special_nodes:
            return set()

        raw_task = self.current_task()
        if raw_task is None:
            return set()

        parsed = self._parse_task(raw_task)
        if parsed is None:
            return set()

        _task_id, supplier, client = parsed

        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        supplier_done = bool(terminated_flags[0])
        client_done = bool(terminated_flags[1])
        allowed_nodes = self._allowed_supplier_client_nodes_for_task(terminated_flags)

        forbidden: Set[str] = set()

        for ev_gen in self.event_map.keys():
            ev_gen = str(ev_gen)

            # Movement restriction: only SUPPLIER/CLIENT nodes are filtered by
            # task assignment. VERTIPORT and CHARGING/STATION remain reachable.
            if ev_gen.startswith("edge_take::"):
                dst = self._edge_take_destination(ev_gen)
                if dst is not None and self._is_restricted_task_special_node(dst):
                    if str(dst) not in allowed_nodes:
                        forbidden.add(ev_gen)
                continue

            # Work restriction: the UAV may start work only at the assigned
            # supplier/client and only in the correct task phase.
            if ev_gen.startswith("work_start::"):
                parts = ev_gen.split("::")
                if len(parts) != 3:
                    forbidden.add(ev_gen)
                    continue

                node = parts[1]
                role = parts[2]

                if role == "SUPPLIER":
                    if supplier_done or str(node) != str(supplier):
                        forbidden.add(ev_gen)

                elif role == "CLIENT":
                    if (not supplier_done) or client_done or str(node) != str(client):
                        forbidden.add(ev_gen)

                else:
                    forbidden.add(ev_gen)

        return forbidden

    def _node_from_state_string(self, state_string: str) -> Optional[str]:
        node_set = set(str(n) for n in self.model.G.nodes())
        for part in [x.strip() for x in str(state_string).split("|") if x.strip()]:
            if part in node_set:
                return part
        return None

    def _state_string_is_moving(self, state_string: str) -> bool:
        parts = [x.strip() for x in str(state_string).split("|") if x.strip()]
        return ("MOVING" in parts) or ("Movendo" in parts)

    def _state_string_is_busy_service(self, state_string: str) -> bool:
        parts = [x.strip() for x in str(state_string).split("|") if x.strip()]
        return any(
            p.startswith((
                "MODE_WORK_",
                "MODE_CHARGE",
                "trabalhando_",
                "carregando_",
            ))
            or p in ("MUST_EXIT_AFTER_CHARGE", "carregou_precisa_sair")
            for p in parts
        )

    def _state_string_is_idle_at_current_position(
        self,
        state_string: str,
        current_node: Optional[str],
    ) -> bool:
        if current_node is None:
            return False

        node = self._node_from_state_string(state_string)
        if node is None:
            return False

        if str(node) != str(current_node):
            return False

        if self._state_string_is_moving(state_string):
            return False

        if self._state_string_is_busy_service(state_string):
            return False

        return True

    # ------------------------------------------------------------------
    # task management
    # ------------------------------------------------------------------

    def register_claim(self, raw_task: str) -> None:
        raw = str(raw_task or "").strip()
        if raw:
            self._claimed_tasks.add(raw)

    def set_prohibited_events(self, events_generic) -> None:
        cleaned = set()
        for x in events_generic:
            sx = str(x).strip()
            if sx:
                cleaned.add(sx)
        with self._prohibited_lock:
            self._prohibited_generic = cleaned

    def get_prohibited_events(self) -> Set[str]:
        with self._prohibited_lock:
            out = set(self._prohibited_generic)
        out.update(self._temporary_forbidden_events())
        return out

    def register_temporarily_rejected(self, ev_id_or_generic: str, ttl_s: Optional[float] = None) -> None:
        ev_gen = self.to_generic(str(ev_id_or_generic or "").strip())
        if not ev_gen.startswith("edge_take::"):
            return

        ttl = self._rejected_edge_ttl_s if ttl_s is None else float(ttl_s)
        with self._recently_rejected_lock:
            self._recently_rejected_generic[ev_gen] = time.time() + max(0.1, ttl)

    def _temporary_forbidden_events(self) -> Set[str]:
        now = time.time()
        out = set()
        with self._recently_rejected_lock:
            expired = [ev for ev, until in self._recently_rejected_generic.items() if until <= now]
            for ev in expired:
                self._recently_rejected_generic.pop(ev, None)
            out.update(self._recently_rejected_generic.keys())
        return out

    def task_progress(self) -> dict:
        return {
            "supplier_done": bool(self.terminated[0]),
            "client_done": bool(self.terminated[1]),
            "returned_to_vertiport": bool(self.terminated[2]),
        }

    def try_accept_task(self, raw_task: str) -> Tuple[bool, Optional[str]]:
        raw = str(raw_task or "").strip()

        reject_ev = self.to_id("task_reject") or f"task_reject_{self.id}"
        accept_ev = self.to_id("task_accept") or f"task_accept_{self.id}"

        parsed = self._parse_task(raw)
        if parsed is None:
            return False, reject_ev

        _task_id, supplier, client = parsed
        if not self._validate_task_nodes(supplier, client):
            return False, reject_ev

        if raw in self._claimed_tasks:
            return False, reject_ev

        with self._task_lock:
            if self._task_raw is not None:
                return False, reject_ev

            self._task_raw = raw
            self.terminated = [False, False, False]
            self._accepted_task_count += 1

        self._claimed_tasks.add(raw)
        self._clear_buffer()
        self._clear_pending_command()

        with self._planner_lock:
            self._last_plan_request = 0.0

        return True, accept_ev

    def clear_task(self) -> Optional[str]:
        done_ev = self.to_id("task_done") or f"task_done_{self.id}"

        with self._task_lock:
            if self._task_raw is None:
                return None
            self._task_raw = None
            self.terminated = [False, False, False]

        self._clear_buffer()
        self._clear_pending_command()

        with self._planner_lock:
            self._last_plan_request = 0.0

        return done_ev

    def _parse_task(self, raw_task: str) -> Optional[Tuple[str, str, str]]:
        raw = str(raw_task or "").strip()
        if ":" not in raw:
            return None

        try:
            task_id, nodes_raw = raw.split(":", 1)
            parts = [p.strip() for p in nodes_raw.split(",") if p.strip()]
            if len(parts) != 2:
                return None

            supplier, client = parts
            if not task_id.strip():
                return None

            return task_id.strip(), supplier, client
        except Exception:
            return None

    def _validate_task_nodes(self, supplier: str, client: str) -> bool:
        if supplier not in self.model.G.nodes():
            return False
        if client not in self.model.G.nodes():
            return False
        if self.model._kind(supplier) != "SUPPLIER":
            return False
        if self.model._kind(client) != "CLIENT":
            return False
        return True

    # ------------------------------------------------------------------
    # pending action tracking
    # ------------------------------------------------------------------

    def has_pending_command(self) -> bool:
        with self._pending_lock:
            return self._pending_command is not None

    def pending_published_event(self) -> Optional[str]:
        with self._pending_lock:
            return self._pending_command

    def pending_completion_event(self) -> Optional[str]:
        with self._pending_lock:
            return self._predicted_completion_event

    def is_calculating(self) -> bool:
        with self._planner_lock:
            return bool(self._is_calculating)

    def _completion_event_for(self, controllable_ev_id: str) -> Optional[str]:
        ev_gen = self.to_generic(controllable_ev_id)

        if ev_gen.startswith("edge_take::"):
            rest = ev_gen.split("edge_take::", 1)[1]
            return f"edge_release::{rest}_{self.id}"

        if ev_gen.startswith("work_start::"):
            rest = ev_gen.split("work_start::", 1)[1]
            return f"work_end::{rest}_{self.id}"

        if ev_gen.startswith("charge_start::"):
            rest = ev_gen.split("charge_start::", 1)[1]
            return f"charge_end::{rest}_{self.id}"

        return None

    def mark_event_published(self, ev_id: str) -> None:
        ev_id = str(ev_id)
        with self._pending_lock:
            self._pending_command = ev_id
            self._predicted_completion_event = self._completion_event_for(ev_id)

    def clear_pending_published_event(self, ev_id: Optional[str] = None) -> None:
        with self._pending_lock:
            if ev_id is None:
                self._pending_command = None
                self._predicted_completion_event = None
                return

            sev = str(ev_id)
            if self._pending_command == sev or self._predicted_completion_event == sev:
                self._pending_command = None
                self._predicted_completion_event = None

    def _clear_pending_command(self) -> None:
        self.clear_pending_published_event()

    # ------------------------------------------------------------------
    # execution buffer
    # ------------------------------------------------------------------

    def _clear_buffer(self) -> None:
        with self._buffer_lock:
            self._execution_buffer = []

    def _replace_buffer(self, ev_id: Optional[str]) -> None:
        if ev_id is None:
            return
        with self._buffer_lock:
            self._execution_buffer = [str(ev_id)]

    def buffered_event(self) -> Optional[str]:
        with self._buffer_lock:
            if self._execution_buffer:
                return self._execution_buffer[0]
            return None

    def pop_next_dispatchable_event(self) -> Optional[str]:
        if self.has_pending_command():
            return None

        with self._buffer_lock:
            if not self._execution_buffer:
                return None
            ev_id = self._execution_buffer[0]

        enabled = set(self.enabled_events())
        if ev_id not in enabled:
            # Critical anti-stall fix:
            # a plan computed while the UAV was moving may become stale after
            # the completion event is processed.  If the stale head of the
            # buffer is not removed, _try_dispatch() keeps seeing a non-empty
            # buffer and never requests a fresh plan, leaving the UAV idle
            # forever at a vertex.
            with self._buffer_lock:
                if self._execution_buffer and self._execution_buffer[0] == ev_id:
                    self._execution_buffer.pop(0)

            self._last_plan_status = "STALE_BUFFER_CLEARED"
            self._last_plan_status_code = "-5"

            with self._planner_lock:
                self._last_plan_request = 0.0

            return None

        with self._buffer_lock:
            if not self._execution_buffer:
                return None
            if self._execution_buffer[0] != ev_id:
                return None
            self._execution_buffer.pop(0)

        self.mark_event_published(ev_id)
        return ev_id

    def dispatch_failed(self, ev_id: Optional[str] = None) -> None:
        self.clear_pending_published_event(ev_id)
        self._clear_buffer()

    # ------------------------------------------------------------------
    # computational-performance helpers
    # ------------------------------------------------------------------

    def _status_label(self, status_code) -> str:
        try:
            code = int(status_code)
        except Exception:
            return str(status_code or "UNKNOWN")

        labels = {
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            9: "TIME_LIMIT",
            11: "INTERRUPTED",
            13: "SUBOPTIMAL",
            -1: "ERROR",
            -2: "BASELINE_POLICY",
            -3: "NO_SELECTED_EVENT",
            -4: "ANTI_GRIDLOCK_ESCAPE",
            -5: "STALE_BUFFER_CLEARED",
        }
        return labels.get(code, f"STATUS_{code}")

    def _task_phase(self) -> str:
        if self.current_task() is None:
            return "idle"
        if not self.terminated[0]:
            return "supplier"
        if not self.terminated[1]:
            return "client"
        if not self.terminated[2]:
            return "return_to_base"
        return "done"

    def _node_distance_m(self, u: str, v: str) -> float:
        u = str(u)
        v = str(v)

        try:
            if hasattr(self.model, "pos") and u in self.model.pos and v in self.model.pos:
                pu = self.model.pos[u]
                pv = self.model.pos[v]
                dx = float(pu[0]) - float(pv[0])
                dy = float(pu[1]) - float(pv[1])
                dz = 0.0
                if len(pu) >= 3 and len(pv) >= 3:
                    dz = float(pu[2]) - float(pv[2])
                return max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)
        except Exception:
            pass

        try:
            edge_data = self.model.G.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                if "weight" in edge_data:
                    return max(float(edge_data["weight"]), 1e-6)
                if "distance" in edge_data:
                    return max(float(edge_data["distance"]), 1e-6)
                for _k, data in edge_data.items():
                    if isinstance(data, dict):
                        if "weight" in data:
                            return max(float(data["weight"]), 1e-6)
                        if "distance" in data:
                            return max(float(data["distance"]), 1e-6)
        except Exception:
            pass

        return 1000.0

    def _edge_time_ms_from_event(self, ev_id_or_generic: Optional[str]) -> float:
        ev = str(ev_id_or_generic or "")
        if not ev:
            return 0.0

        ev_gen = self.to_generic(ev)
        if not ev_gen.startswith("edge_take::"):
            return 0.0

        parts = ev_gen.split("::")
        if len(parts) != 3:
            return 0.0

        u, v = parts[1], parts[2]
        distance_m = self._node_distance_m(u, v)
        return 1000.0 * distance_m / max(float(self.speed_mps), 1e-6)

    def _estimate_available_time_ms(self, reason: str, selected_event: Optional[str]) -> float:
        reason = str(reason or "")

        pending = self.pending_published_event()
        if pending:
            pending_generic = self.to_generic(pending)
        else:
            pending_generic = ""

        if "edge" in reason:
            t_edge = self._edge_time_ms_from_event(pending or selected_event)
            if t_edge > 0.0:
                return t_edge

        if "service" in reason:
            return 1000.0 * max(float(self.work_time_s), 0.0)

        if "charge" in reason:
            return 1000.0 * max(float(self.charge_time_s), 0.0)

        if pending_generic.startswith("edge_take::"):
            return self._edge_time_ms_from_event(pending_generic)

        if pending_generic.startswith("work_start::"):
            return 1000.0 * max(float(self.work_time_s), 0.0)

        if pending_generic.startswith("charge_start::"):
            return 1000.0 * max(float(self.charge_time_s), 0.0)

        return 0.0


    # ------------------------------------------------------------------
    # planner control
    # ------------------------------------------------------------------

    def request_plan(self, force: bool = False, reason: str = "") -> bool:
        if self.current_task() is None:
            return False

        now = time.time()

        with self._planner_lock:
            if self._is_calculating:
                return False

            if not force and (now - self._last_plan_request) < self._min_plan_interval:
                return False

            if not force and self.buffered_event() is not None:
                return False

            self._is_calculating = True
            self._last_plan_request = now

        self._planner_thread = threading.Thread(
            target=self._planner_worker,
            args=(str(reason),),
            daemon=True,
        )
        self._planner_thread.start()
        return True

    def _planner_worker(self, reason: str = "") -> None:
        t0 = time.perf_counter()
        selected = None

        try:
            selected = self._compute_next_event()
            if selected is None:
                return
            self._replace_buffer(selected)
        finally:
            runtime_ms = 1000.0 * (time.perf_counter() - t0)

            try:
                self._log_planner_step(
                    reason=reason,
                    runtime_ms=runtime_ms,
                    selected_event=selected,
                )
            except Exception:
                pass

            with self._planner_lock:
                self._is_calculating = False

    def _log_planner_step(
        self,
        reason: str,
        runtime_ms: float,
        selected_event: Optional[str],
    ) -> None:
        with self._prohibited_lock:
            forbidden_count = len(self._prohibited_generic)

        available_time_ms = self._estimate_available_time_ms(
            reason=reason,
            selected_event=selected_event,
        )

        if available_time_ms > 0.0:
            online_ratio = float(runtime_ms) / float(available_time_ms)
        else:
            online_ratio = ""

        selected_generic = self.to_generic(str(selected_event or "")) if selected_event else ""

        self._planner_logger.write(
            run_id=str(self.run_id),
            scenario_id=str(self.scenario_id),
            baseline=str(self.baseline),
            graph_size=str(self.graph_size),
            density=str(self.density),
            seed=str(self.seed),
            num_uavs=int(self.num_uavs),
            num_nodes=int(self.num_nodes),
            num_edges=int(self.num_edges),
            num_tasks=int(self.num_tasks_config or self._accepted_task_count),
            planning_horizon=int(self.planning_horizon),
            agent_id=self.id,
            reason=str(reason),
            task=str(self.current_task() or ""),
            task_phase=str(self._task_phase()),
            state=str(self._state),
            terminated_supplier=int(bool(self.terminated[0])),
            terminated_client=int(bool(self.terminated[1])),
            terminated_base=int(bool(self.terminated[2])),
            runtime_ms=float(runtime_ms),
            available_time_ms=available_time_ms if available_time_ms > 0.0 else "",
            online_ratio=online_ratio,
            selected_event=str(selected_event or ""),
            selected_event_generic=str(selected_generic),
            milp_status=str(self._last_plan_status),
            milp_status_code=str(self._last_plan_status_code),
            buffered_event=str(self.buffered_event() or ""),
            forbidden_count=int(forbidden_count),
            interest_count=int(self._last_plan_interest_count),
            soc="",
        )

    # ------------------------------------------------------------------
    # event progression
    # ------------------------------------------------------------------

    def step(self, ev: str) -> bool:
        ev = str(ev or "").strip()
        if not ev:
            return False

        if not self._should_process(ev):
            return False

        ev_obj = self._event_objects.get(ev)
        if ev_obj is None:
            return False

        current = str(self._state)
        transitioned = False

        for q, e, d in self._trs_id:
            if str(q) == current and e == ev_obj:
                self._state = d
                self.last_state_entry_time = time.time()
                transitioned = True
                break

        if not transitioned:
            return False

        ev_gen = self.to_generic(ev)

        with self._pending_lock:
            predicted_completion = self._predicted_completion_event

        if predicted_completion is not None and ev == predicted_completion:
            self._clear_pending_command()

        self._update_task_progress(ev)
        self._update_dynamic_cost()

        if self.current_task() is not None and all(self.terminated):
            self.clear_task()
            return True

        if self.current_task() is None:
            return True

        # Receding-horizon policy:
        # compute the next decision while the UAV is physically moving or
        # executing a service, not after it has already arrived at a vertex.
        if ev_gen.startswith("edge_take::"):
            self.request_plan(force=True, reason="in_edge_replanning")
            return True

        if ev_gen.startswith("work_start::"):
            self.request_plan(force=True, reason="during_service_replanning")
            return True

        if ev_gen.startswith("charge_start::"):
            self.request_plan(force=True, reason="during_charge_replanning")
            return True

        if ev_gen == "battery_low":
            self._clear_buffer()
            self.request_plan(force=True, reason="battery_low_replanning")
            return True

        if (
            ev_gen.startswith("edge_release::")
            or ev_gen.startswith("work_end::")
            or ev_gen.startswith("charge_end::")
        ):
            return True

        if is_controllable(ev_obj):
            self.request_plan(force=True, reason="control_event_replanning")
            return True

        return True

    # ------------------------------------------------------------------
    # baseline policies
    # ------------------------------------------------------------------

    def _target_node_for_planning(self, state_obj=None, terminated_flags=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        parsed = self._parse_task(self.current_task() or "")
        if parsed is None:
            return None

        _task_id, supplier, client = parsed
        current = self._current_node_from_state(state_obj)

        if self._state_has_low_battery_in(state_obj):
            if current is not None and self.model._kind(current) == "STATION":
                return current

            stations = self._station_nodes()
            if current is not None and stations:
                best_station = None
                best_dist = 10**9
                for st in stations:
                    try:
                        d = nx.shortest_path_length(self.model.G, current, st)
                    except Exception:
                        continue
                    if d < best_dist:
                        best_dist = d
                        best_station = st
                if best_station is not None:
                    return best_station

        if not terminated_flags[0]:
            return supplier

        if not terminated_flags[1]:
            return client

        bases = self._vertiport_nodes()
        if bases:
            return bases[0]

        return None

    def _immediate_local_event(self, state_obj=None, terminated_flags=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        current = self._current_node_from_state(state_obj)
        if current is None:
            return None

        parsed = self._parse_task(self.current_task() or "")
        if parsed is None:
            return None

        _task_id, supplier, client = parsed
        enabled = set(self._enabled_events_from_state(state_obj))

        if self._state_has_low_battery_in(state_obj) and self.model._kind(current) == "STATION":
            ev = self.to_id(f"charge_start::{current}")
            if ev in enabled:
                return ev

        if not terminated_flags[0] and current == supplier:
            ev = self.to_id(f"work_start::{supplier}::SUPPLIER")
            if ev in enabled:
                return ev

        if terminated_flags[0] and not terminated_flags[1] and current == client:
            ev = self.to_id(f"work_start::{client}::CLIENT")
            if ev in enabled:
                return ev

        return None

    def _first_enabled_control_event(self, state_obj=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state

        enabled = sorted(self._enabled_events_from_state(state_obj))
        for ev_id in enabled:
            ev_gen = self.to_generic(ev_id)
            if ev_gen.startswith(("work_start::", "charge_start::", "edge_take::")):
                return ev_id
        return None

    def _greedy_distance_event(self, state_obj=None, terminated_flags=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        local = self._immediate_local_event(state_obj=state_obj, terminated_flags=terminated_flags)
        if local is not None:
            return local

        current = self._current_node_from_state(state_obj)
        if current is None:
            return None

        target = self._target_node_for_planning(state_obj=state_obj, terminated_flags=terminated_flags)
        if target is None:
            return self._first_enabled_control_event(state_obj)

        enabled_id = set(self._enabled_events_from_state(state_obj))
        best = None
        best_len = 10**9

        for ev_id in sorted(enabled_id):
            ev_gen = self.to_generic(ev_id)
            if not ev_gen.startswith("edge_take::"):
                continue

            parts = ev_gen.split("::")
            if len(parts) != 3:
                continue

            u, v = parts[1], parts[2]
            if u != current:
                continue

            try:
                d = nx.shortest_path_length(self.model.G, v, target)
            except Exception:
                continue

            if d < best_len:
                best_len = d
                best = ev_id

        if best is not None:
            return best
        return self._first_enabled_control_event(state_obj)

    def _sct_only_event(self, state_obj=None, terminated_flags=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        local = self._immediate_local_event(state_obj=state_obj, terminated_flags=terminated_flags)
        if local is not None:
            return local

        current = self._current_node_from_state(state_obj)
        target = self._target_node_for_planning(state_obj=state_obj, terminated_flags=terminated_flags)
        enabled = sorted(self._enabled_events_from_state(state_obj))

        if current is not None and target is not None:
            try:
                current_dist = nx.shortest_path_length(self.model.G, current, target)
            except Exception:
                current_dist = None

            for ev_id in enabled:
                ev_gen = self.to_generic(ev_id)
                if not ev_gen.startswith("edge_take::"):
                    continue

                parts = ev_gen.split("::")
                if len(parts) != 3:
                    continue

                u, v = parts[1], parts[2]
                if u != current:
                    continue

                if current_dist is None:
                    return ev_id

                try:
                    next_dist = nx.shortest_path_length(self.model.G, v, target)
                except Exception:
                    continue

                if next_dist < current_dist:
                    return ev_id

        return self._first_enabled_control_event(state_obj)

    # ------------------------------------------------------------------
    # return-to-base policy
    # ------------------------------------------------------------------

    def _return_phase_active(self, terminated_flags=None) -> bool:
        if terminated_flags is None:
            terminated_flags = list(self.terminated)
        return bool(terminated_flags[0] and terminated_flags[1] and not terminated_flags[2])

    def _next_return_event(self, state_obj=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state

        current = self._current_node_from_state(state_obj)
        if current is None:
            return None

        if self.model._kind(current) == "VERTIPORT":
            return None

        bases = self._vertiport_nodes()
        if not bases:
            return None

        enabled_id = set(self._enabled_events_from_state(state_obj))

        with self._prohibited_lock:
            prohibited_generic = set(self._prohibited_generic)

        search_graph = nx.DiGraph()
        search_graph.add_nodes_from(str(n) for n in self.model.G.nodes())

        for u, v in self.model.G.edges():
            u = str(u)
            v = str(v)
            ev = f"edge_take::{u}::{v}"

            if ev not in self.model.events:
                continue

            if ev in prohibited_generic:
                continue

            search_graph.add_edge(u, v)

        candidates = []

        for base in bases:
            try:
                path = nx.shortest_path(search_graph, current, base)
            except Exception:
                continue

            if len(path) < 2:
                continue

            nxt = str(path[1])
            ev_gen = f"edge_take::{current}::{nxt}"
            ev_id = self.to_id(ev_gen)

            if ev_id is None:
                continue

            if ev_id not in enabled_id:
                continue

            candidates.append((len(path), base, nxt, ev_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return candidates[0][3]

    def _edge_take_parts(self, ev_generic: str):
        ev_generic = str(ev_generic or "")
        if not ev_generic.startswith("edge_take::"):
            return None
        parts = ev_generic.split("::")
        if len(parts) != 3:
            return None
        return str(parts[1]), str(parts[2])

    def _node_z(self, node_id: str) -> float:
        try:
            p = self.model.pos.get(str(node_id))
            if p is not None and len(p) >= 3:
                return float(p[2])
        except Exception:
            pass
        return 0.0

    def _candidate_edge_distance(self, u: str, v: str) -> float:
        return float(self._node_distance_m(str(u), str(v)))

    def _build_unblocked_search_graph(self, prohibited_generic: Set[str]) -> nx.DiGraph:
        Gs = nx.DiGraph()
        Gs.add_nodes_from(str(n) for n in self.model.G.nodes())

        for u, v in self.model.G.edges():
            u = str(u)
            v = str(v)
            ev = f"edge_take::{u}::{v}"
            if ev not in self.model.events:
                continue
            if ev in prohibited_generic:
                continue
            Gs.add_edge(u, v)

        return Gs

    def _anti_gridlock_escape_event(self, state_obj=None, terminated_flags=None, reason: str = "") -> Optional[str]:
        """
        Select a safe outgoing movement when the optimizer returns no useful
        command or repeatedly selects a congested edge.  This is not a bypass
        of SCT/UTM: candidates are taken only from currently enabled local
        controllable events and are filtered by the current prohibited set.

        The score prefers: (i) progress toward the current mission target,
        (ii) short outgoing edges, and (iii) upward/higher-altitude alternatives
        when they are admissible.  The third term is what prevents a UAV from
        remaining stopped at a congested vertex while a higher-layer edge is
        free.
        """
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        current = self._current_node_from_state(state_obj)
        if current is None:
            return None

        if self._state_string_is_busy_service(str(state_obj)):
            return None

        elapsed = time.time() - self.last_state_entry_time

        enabled_id = set(self._enabled_events_from_state(state_obj))
        if not enabled_id:
            return None

        with self._prohibited_lock:
            prohibited_generic = set(self._prohibited_generic)
        prohibited_generic.update(
            self._local_task_forbidden_events(
                state_obj=state_obj,
                terminated_flags=terminated_flags,
            )
        )
        prohibited_generic.update(self._temporary_forbidden_events())

        target = self._target_node_for_planning(
            state_obj=state_obj,
            terminated_flags=terminated_flags,
        )

        search_graph = self._build_unblocked_search_graph(prohibited_generic)
        z_current = self._node_z(current)

        candidates = []
        for ev_id in sorted(enabled_id):
            ev_gen = self.to_generic(ev_id)
            parts = self._edge_take_parts(ev_gen)
            if parts is None:
                continue

            u, v = parts
            if str(u) != str(current):
                continue
            if ev_gen in prohibited_generic:
                continue

            # Avoid leaving the task domain through an unassigned supplier/client.
            if self._is_restricted_task_special_node(v):
                allowed = self._allowed_supplier_client_nodes_for_task(terminated_flags)
                if str(v) not in allowed:
                    continue

            if target is not None:
                try:
                    hops = nx.shortest_path_length(search_graph, str(v), str(target))
                except Exception:
                    hops = 10**6
            else:
                hops = 0

            edge_d = self._candidate_edge_distance(u, v)
            z_gain = self._node_z(v) - z_current

            # Strongly discourage staying logically at the same vertex/layer-equivalent
            # representation if an actual outgoing edge exists.  The altitude bonus
            # makes free upper-layer transitions attractive in congestion.
            score = (
                float(self._escape_target_hop_weight) * float(hops)
                + float(self._escape_edge_distance_weight) * float(edge_d)
                - float(self._altitude_escape_bonus) * max(0.0, float(z_gain))
            )

            candidates.append((score, -z_gain, edge_d, str(v), ev_id))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        selected = candidates[0][4]

        # Status labels are logged in planner_agent_*.csv and make gridlock
        # interventions visible in the experiment results.
        if elapsed >= self._stuck_escape_after_s or str(reason):
            self._last_plan_status = "ANTI_GRIDLOCK_ESCAPE"
            self._last_plan_status_code = "-4"

        return selected

    # ------------------------------------------------------------------
    # planning core
    # ------------------------------------------------------------------


    def _compute_next_event(self) -> Optional[str]:
        self._last_plan_status = "NOT_RUN"
        self._last_plan_status_code = ""
        self._last_plan_interest_count = 0
        self._last_plan_forbidden_count = 0

        if self.current_task() is None:
            self._last_plan_status = "NO_TASK"
            return None

        if all(self.terminated):
            self._last_plan_status = "TASK_ALREADY_DONE"
            return None

        self._update_dynamic_cost()

        plan_state, plan_flags = self._planning_snapshot()

        if self.baseline in ("greedy_distance", "no_mpsc"):
            self._last_plan_status = "BASELINE_GREEDY"
            self._last_plan_status_code = "-2"
            return self._greedy_distance_event(plan_state, plan_flags)

        if self.baseline == "sct_only":
            self._last_plan_status = "BASELINE_SCT_ONLY"
            self._last_plan_status_code = "-2"
            return self._sct_only_event(plan_state, plan_flags)

        if self._return_phase_active(plan_flags):
            current = self._current_node_from_state(plan_state)
            if current is not None and self.model._kind(current) == "VERTIPORT":
                self.terminated[2] = True
                self._last_plan_status = "RETURN_ALREADY_AT_BASE"
                return None

            self._last_plan_status = "RETURN_POLICY"
            self._last_plan_status_code = "-2"
            ret_ev = self._next_return_event(plan_state)
            if ret_ev is not None:
                return ret_ev
            return self._anti_gridlock_escape_event(
                state_obj=plan_state,
                terminated_flags=plan_flags,
                reason="return_policy_escape",
            )

        selected = self._plan_with_optimizer(
            state_obj=plan_state,
            terminated_flags=plan_flags,
        )

        if selected is None:
            selected = self._anti_gridlock_escape_event(
                state_obj=plan_state,
                terminated_flags=plan_flags,
                reason="optimizer_no_selected",
            )

        if selected is None and self._last_plan_status in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL"):
            self._last_plan_status = "NO_SELECTED_EVENT"
            self._last_plan_status_code = "-3"

        return selected

    def _plan_with_optimizer(self, state_obj=None, terminated_flags=None) -> Optional[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        interest_generic = self._build_interest_events(
            state_obj=state_obj,
            terminated_flags=terminated_flags,
        )
        interest_id = [self.to_id(x) for x in interest_generic]
        interest_id = [x for x in interest_id if x is not None]

        with self._prohibited_lock:
            prohibited_generic = set(self._prohibited_generic)

        prohibited_generic.update(
            self._local_task_forbidden_events(
                state_obj=state_obj,
                terminated_flags=terminated_flags,
            )
        )

        prohibited_id = [self.to_id(x) for x in prohibited_generic]
        prohibited_id = [x for x in prohibited_id if x is not None]

        self._last_plan_interest_count = int(len(interest_id))
        self._last_plan_forbidden_count = int(len(prohibited_id))

        try:
            seq, _status = self.optimize_fn(
                self.supervisor,
                state_obj,
                self.planning_horizon,
                self.dynamic_cost_dict,
                interest_id,
                prohibited_id,
            )
            self._last_plan_status_code = str(_status)
            self._last_plan_status = self._status_label(_status)
        except Exception:
            self._last_plan_status_code = "-1"
            self._last_plan_status = "ERROR"
            return None

        enabled_predicted = set(self._enabled_events_from_state(state_obj))

        for ev_id in seq:
            if str(ev_id).startswith(("inspec_start::", "inspec_end::")):
                continue

            ev_obj = self._event_objects.get(ev_id)
            if ev_obj is None:
                continue
            if not is_controllable(ev_obj):
                continue
            if ev_id not in enabled_predicted:
                continue
            return ev_id

        return self._anti_gridlock_escape_event(
            state_obj=state_obj,
            terminated_flags=terminated_flags,
            reason="milp_sequence_without_dispatchable_event",
        )

    def _planning_snapshot(self):
        state_for_plan = self._state
        flags_for_plan = list(self.terminated)

        with self._pending_lock:
            pending_cmd = self._pending_command
            completion_ev = self._predicted_completion_event

        if pending_cmd is None or completion_ev is None:
            return state_for_plan, flags_for_plan

        predicted_state = self._transition_from(state_for_plan, completion_ev)
        if predicted_state is None:
            return state_for_plan, flags_for_plan

        flags_for_plan = self._apply_completion_effects(
            flags=flags_for_plan,
            completion_ev_id=completion_ev,
            predicted_state=predicted_state,
        )
        return predicted_state, flags_for_plan

    def _transition_from(self, state_obj, ev_id: str):
        ev_obj = self._event_objects.get(str(ev_id))
        if ev_obj is None:
            return None

        s = str(state_obj)
        for q, e, d in self._trs_id:
            if str(q) == s and e == ev_obj:
                return d
        return None

    def _apply_completion_effects(self, flags, completion_ev_id: str, predicted_state):
        parsed = self._parse_task(self.current_task() or "")
        if parsed is None:
            return flags

        _task_id, supplier, client = parsed
        ev_gen = self.to_generic(completion_ev_id)

        if ev_gen == f"work_end::{supplier}::SUPPLIER":
            flags[0] = True
        elif ev_gen == f"work_end::{client}::CLIENT":
            flags[1] = True

        if flags[0] and flags[1] and not flags[2]:
            if ev_gen.startswith("edge_release::"):
                rest = ev_gen.split("edge_release::", 1)[1]
                parts = rest.split("::")
                if len(parts) == 2:
                    dst = parts[1]
                    if self.model._kind(dst) == "VERTIPORT":
                        flags[2] = True
                        return flags

            node_now = self._current_node_from_state(predicted_state)
            if node_now is not None and self.model._kind(node_now) == "VERTIPORT":
                flags[2] = True

        return flags

    def _build_interest_events(self, state_obj=None, terminated_flags=None) -> List[str]:
        if state_obj is None:
            state_obj = self._state
        if terminated_flags is None:
            terminated_flags = list(self.terminated)

        parsed = self._parse_task(self.current_task() or "")
        if parsed is None:
            return []

        _task_id, supplier, client = parsed
        out = []
        current = self._current_node_from_state(state_obj)

        if terminated_flags[0] and terminated_flags[1] and not terminated_flags[2]:
            seen = set()
            for base in self._vertiport_nodes():
                preds = set(str(x) for x in self.model.G.predecessors(base))
                for u in preds:
                    ev = f"edge_take::{u}::{base}"
                    if ev in self.model.events and ev not in seen:
                        seen.add(ev)
                        out.append(ev)
            return out

        if self._state_has_low_battery_in(state_obj):
            if current is not None and self.model._kind(current) == "STATION":
                out.append(f"charge_start::{current}")
                return out

            seen = set()
            for st in self._station_nodes():
                preds = set(str(x) for x in self.model.G.predecessors(st))
                for u in preds:
                    ev = f"edge_take::{u}::{st}"
                    if ev in self.model.events and ev not in seen:
                        seen.add(ev)
                        out.append(ev)

            if out:
                return out

        if not terminated_flags[0]:
            out.append(f"work_start::{supplier}::SUPPLIER")
            return out

        if not terminated_flags[1]:
            out.append(f"work_start::{client}::CLIENT")
            return out

        if not terminated_flags[2]:
            seen = set()
            for base in self._vertiport_nodes():
                preds = set(str(x) for x in self.model.G.predecessors(base))
                for u in preds:
                    ev = f"edge_take::{u}::{base}"
                    if ev in self.model.events and ev not in seen:
                        seen.add(ev)
                        out.append(ev)

        return out

    # ------------------------------------------------------------------
    # progress tracking
    # ------------------------------------------------------------------

    def _update_task_progress(self, ev_with_id: str) -> None:
        if self.current_task() is None:
            return

        parsed = self._parse_task(self.current_task())
        if parsed is None:
            return

        _task_id, supplier, client = parsed
        ev_gen = self.to_generic(ev_with_id)

        if ev_gen == f"work_end::{supplier}::SUPPLIER":
            self.terminated[0] = True
        elif ev_gen == f"work_end::{client}::CLIENT":
            self.terminated[1] = True

        self._detect_base_return(ev_gen)

    def _detect_base_return(self, ev_generic: str) -> None:
        if not (self.terminated[0] and self.terminated[1]):
            return
        if self.terminated[2]:
            return

        if ev_generic.startswith("edge_release::"):
            rest = ev_generic.split("edge_release::", 1)[1]
            parts = rest.split("::")
            if len(parts) == 2:
                dst = parts[1]
                if self.model._kind(dst) == "VERTIPORT":
                    self.terminated[2] = True
                    return

        node_now = self._current_node()
        if node_now is not None and self.model._kind(node_now) == "VERTIPORT":
            self.terminated[2] = True

    # ------------------------------------------------------------------
    # costs
    # ------------------------------------------------------------------

    def _update_dynamic_cost(self) -> None:
        self.dynamic_cost_dict = dict(self.base_sup_cost)

        qs = self.state_str()
        current_node = self._current_node()
        time_spent = time.time() - self.last_state_entry_time

        # Remaining idle at a vertex is operationally undesirable in dense
        # UTM traffic.  The penalty is intentionally aggressive because safety
        # is already enforced by SCT/UTM; the optimizer should use any
        # admissible outgoing edge instead of creating a queue at a vertex.
        if current_node is not None:
            idle_penalty = min(
                float(self._idle_same_position_penalty_cap),
                float(self._same_position_future_penalty)
                + float(self._idle_same_position_penalty_per_s) * max(0.0, time_spent),
            )

            escape_pressure = 0.0
            if time_spent >= self._stuck_escape_after_s:
                escape_pressure = min(
                    2.0 * float(self._idle_same_position_penalty_cap),
                    10.0 + 8.0 * (time_spent - self._stuck_escape_after_s),
                )

            for state_str in list(self.dynamic_cost_dict.keys()):
                if not self._state_string_is_idle_at_current_position(state_str, current_node):
                    continue

                E, Tf, D = self.dynamic_cost_dict[state_str]
                self.dynamic_cost_dict[state_str] = (
                    E,
                    Tf + 0.05 * max(0.0, time_spent),
                    D + idle_penalty + escape_pressure,
                )

        # Keep a persistence penalty on the exact current supervisor state.
        if qs in self.dynamic_cost_dict:
            E, Tf, D = self.dynamic_cost_dict[qs]
            if time_spent > 0.5:
                self.dynamic_cost_dict[qs] = (
                    E,
                    Tf + min(0.10 * time_spent, 2.00),
                    D + min(2.00 * time_spent, 30.00),
                )

        if "BAT_LOW" not in qs:
            return

        sup_index = self.cost_engine["sup_index"]
        for state_str, info in sup_index.items():
            _parts, _rep, at_kind, _idle, _low, is_chg, _work, _wf = info
            if at_kind == "STATION" or is_chg:
                E, Tf, D = self.dynamic_cost_dict[state_str]
                factor = 0.50 if is_chg else 0.75
                self.dynamic_cost_dict[state_str] = (E, Tf, max(0.0, D * factor))

    def rebuild_costs(
        self,
        speed_mps=None,
        energy_per_meter=None,
        base_time_cost=None,
        params=None,
    ) -> None:
        rebuild_all_costs(
            self.cost_engine,
            self.model,
            speed_mps=speed_mps,
            energy_per_meter=energy_per_meter,
            base_time_cost=base_time_cost,
            params=params,
        )
        self.base_sup_cost = dict(self.cost_engine["sup_cost"])
        self.dynamic_cost_dict = dict(self.base_sup_cost)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _current_node(self) -> Optional[str]:
        return self._current_node_from_state(self._state)

    def _current_node_from_state(self, state_obj) -> Optional[str]:
        node_set = set(str(n) for n in self.model.G.nodes())
        for part in [x.strip() for x in str(state_obj).split("|") if x.strip()]:
            if part in node_set:
                return part
        return None

    def _state_has_low_battery_in(self, state_obj) -> bool:
        return "BAT_LOW" in str(state_obj)

    def _station_nodes(self) -> List[str]:
        return [str(n) for n in self.model.G.nodes() if self.model._kind(n) == "STATION"]

    def _vertiport_nodes(self) -> List[str]:
        return [str(n) for n in self.model.G.nodes() if self.model._kind(n) == "VERTIPORT"]
    


# ----------------------------------------------------------------------
# UAVAgentNode
# ----------------------------------------------------------------------

class UAVAgentNode(Node):
    def __init__(
        self,
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
        super().__init__(f"utm_uav_agent_{int(agent_id)}")

        random.seed(int(seed) + int(agent_id) * 1000)

        self.agent_id = int(agent_id)
        self.model = base_model
        self.positions = self.model.pos

        self.cli_set = self.create_client(SetEntityState, set_state_service)
        if not self.cli_set.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"agent={self.agent_id} service not available: {set_state_service}"
            )

        self._utm_client_group = ReentrantCallbackGroup()
        self.cli_utm_req = self.create_client(
            RequestEvent,
            "/utm/request_event",
            callback_group=self._utm_client_group,
        )
        if not self.cli_utm_req.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                f"agent={self.agent_id} service not available: /utm/request_event"
            )

        self.pub_event = self.create_publisher(String, "/event", 50)
        self.pub_task_claim = self.create_publisher(String, "/task_claims", 20)

        self.sub_event = self.create_subscription(
            String,
            "/event",
            self._on_event,
            50,
        )
        self.sub_task = self.create_subscription(
            String,
            "/task",
            self._on_task,
            20,
        )
        self.sub_task_claim = self.create_subscription(
            String,
            "/task_claims",
            self._on_task_claim,
            20,
        )
        self.sub_prohibited = self.create_subscription(
            String,
            "/prohibited_events",
            self._on_prohibited_events,
            20,
        )

        self.uav = UAVHardware(
            self,
            entity_name=entity_name,
            agent_id=self.agent_id,
            graph_positions=self.positions,
            set_state_client=self.cli_set,
            speed_mps=float(speed_mps),
            vspeed_mps=float(vspeed_mps),
            waypoint_tol_m=float(tol),
            clearance_m=float(clearance),
            alt_offset_m=float(alt_offset),
            pickup_time_s=float(work_time_s),
            delivery_time_s=float(work_time_s),
            charge_time_s=float(charge_time_s),
            battery_model=BatteryModel(),
            low_batt_threshold=float(low_batt_threshold),
            init_pose=init_pose,
            battery_log_period_s=float(battery_log_period_s),
            local_event_callback=None,
        )
        self.uav.vertiport_nodes = set(
            str(n) for n in self.model.G.nodes()
            if self.model._kind(n) == "VERTIPORT"
        )
        self.uav.send_pose()

        self.agent = SupervisorAgent(
            self.model,
            agent_id=self.agent_id,
            planning_horizon=int(planning_horizon),
            speed_mps=float(speed_mps),
            energy_per_meter=0.10,
            base_time_cost=0.10,
        )

        self._event_logger = CSVMetricLogger(
            f"events_agent_{self.agent_id}.csv",
            [
                "t_wall",
                "run_id",
                "scenario_id",
                "baseline",
                "graph_size",
                "density",
                "seed",
                "num_uavs",
                "num_nodes",
                "num_edges",
                "agent_id",
                "direction",
                "event",
                "event_generic",
                "task",
                "task_phase",
                "uav_mode",
                "soc",
            ],
        )

        self._utm_request_logger = CSVMetricLogger(
            f"utm_requests_agent_{self.agent_id}.csv",
            [
                "t_wall",
                "run_id",
                "scenario_id",
                "baseline",
                "graph_size",
                "density",
                "seed",
                "num_uavs",
                "num_nodes",
                "num_edges",
                "agent_id",
                "event",
                "event_generic",
                "accepted",
                "reason",
                "request_runtime_ms",
                "forbidden_count",
            ],
        )

        self._dispatch_lock = threading.RLock()
        self._claim_worker_lock = threading.Lock()
        self._claim_worker_active = False

        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / max(1e-6, self.rate_hz)
        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(
            "agent=%d entity='%s' init_node='%s' ready"
            % (
                self.agent_id,
                str(entity_name),
                str(self.agent._current_node()),
            )
        )

    def _battery_soc_for_log(self):
        for name in ("soc", "battery_soc", "state_of_charge"):
            try:
                value = getattr(self.uav, name)
                if callable(value):
                    value = value()
                return value
            except Exception:
                pass
        try:
            return self.uav.battery.soc
        except Exception:
            return ""

    def _write_event_log(self, direction: str, event_name: str) -> None:
        try:
            self._event_logger.write(
                run_id=str(self.agent.run_id),
                scenario_id=str(self.agent.scenario_id),
                baseline=str(self.agent.baseline),
                graph_size=str(self.agent.graph_size),
                density=str(self.agent.density),
                seed=str(self.agent.seed),
                num_uavs=int(self.agent.num_uavs),
                num_nodes=int(self.agent.num_nodes),
                num_edges=int(self.agent.num_edges),
                agent_id=int(self.agent_id),
                direction=str(direction),
                event=str(event_name),
                event_generic=str(self.agent.to_generic(str(event_name))),
                task=str(self.agent.current_task() or ""),
                task_phase=str(self.agent._task_phase()),
                uav_mode=str(getattr(self.uav, "mode", "")),
                soc=self._battery_soc_for_log(),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_event(self, msg):
        ev = str(msg.data or "").strip()
        if not ev:
            return

        self._write_event_log("rx", ev)

        _base, eid = split_suffix_id(ev)
        if eid is None or eid != self.agent_id:
            return

        had_task_before = self.agent.current_task() is not None
        transitioned = self.agent.step(ev)

        if not transitioned:
            return

        if had_task_before and self.agent.current_task() is None:
            done_ev = self.agent.to_id("task_done") or f"task_done_{self.agent_id}"
            self._publish_event(done_ev)
            return

        if self.agent.current_task() is None:
            return

        self._try_dispatch()

    def _on_task(self, msg):
        raw = str(msg.data or "").strip()
        if not raw:
            return

        if parse_task(raw) is None:
            return

        with self._claim_worker_lock:
            if self._claim_worker_active:
                return
            self._claim_worker_active = True

        th = threading.Thread(
            target=self._claim_task_worker,
            args=(raw,),
            daemon=True,
        )
        th.start()

    def _claim_task_worker(self, raw):
        try:
            if self.agent.current_task() is not None:
                return

            if raw in self.agent._claimed_tasks:
                return

            if self.agent.baseline == "random_allocation":
                delay = random.uniform(0.0, 0.50)
            else:
                delay = 0.10 * self.agent_id + random.uniform(0.0, 0.05)
            time.sleep(delay)

            if self.agent.current_task() is not None:
                return

            if raw in self.agent._claimed_tasks:
                return

            accepted, ack = self.agent.try_accept_task(raw)
            if not accepted:
                if ack:
                    self._publish_event(ack)
                return

            self.pub_task_claim.publish(String(data=raw))
            self.get_logger().info(
                "agent=%d accepted task '%s'" % (self.agent_id, raw)
            )

            if ack:
                self._publish_event(ack)

            self.agent.request_plan(force=True, reason="initial_task_plan")
            self._try_dispatch()

        finally:
            with self._claim_worker_lock:
                self._claim_worker_active = False

    def _on_task_claim(self, msg):
        raw = str(msg.data or "").strip()
        if not raw:
            return
        self.agent.register_claim(raw)

    def _on_prohibited_events(self, msg):
        raw = str(msg.data or "").strip()
        items = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        self.agent.set_prohibited_events(items)

        # If a previously planned command became prohibited before dispatch,
        # discard it immediately and force a new plan.  This avoids a vehicle
        # sitting at a vertex with an obsolete buffered command while upper
        # altitude/layer alternatives are available.
        try:
            buffered = self.agent.buffered_event()
            if buffered is not None:
                buffered_generic = self.agent.to_generic(buffered)
                if buffered_generic in set(items):
                    self.agent.dispatch_failed(buffered)
                    self.agent.request_plan(force=True, reason="prohibited_update_replanning")
        except Exception:
            pass

    def _on_timer(self):
        self.uav.step(self.dt)
        self.uav.send_pose()

        active_task = self.agent.current_task() is not None
        uav_mode = str(getattr(self.uav, "mode", ""))

        if (
            active_task
            and uav_mode == "MOVING"
            and self.agent.buffered_event() is None
            and not self.agent.is_calculating()
        ):
            self.agent.request_plan(force=False, reason="timer_in_edge_replanning")

        # If the UAV is idle at a vertex with an active task and no command,
        # do not wait indefinitely for a MILP solution.  Generate a safe
        # locally enabled escape edge; UTM authorization is still required
        # below in _try_dispatch().
        if (
            active_task
            and uav_mode == "IDLE"
            and not self.agent.has_pending_command()
            and self.agent.buffered_event() is None
            and not self.agent.is_calculating()
        ):
            idle_elapsed = time.time() - float(getattr(self.agent, "last_state_entry_time", time.time()))
            last_attempt = float(getattr(self, "_last_idle_escape_attempt_ts", 0.0))
            if idle_elapsed >= float(getattr(self.agent, "_stuck_escape_after_s", 0.75)) and (time.time() - last_attempt) >= 0.25:
                self._last_idle_escape_attempt_ts = time.time()
                try:
                    escape_ev = self.agent._anti_gridlock_escape_event(
                        state_obj=self.agent.state(),
                        terminated_flags=list(self.agent.terminated),
                        reason="timer_idle_escape",
                    )
                    if escape_ev is not None:
                        self.agent._replace_buffer(escape_ev)
                    else:
                        self.agent.request_plan(force=True, reason="timer_idle_replanning")
                except Exception:
                    self.agent.request_plan(force=True, reason="timer_idle_replanning")

        self._try_dispatch()

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------


    def _needs_utm_authorization(self, ev: str) -> bool:
        if os.environ.get("UTM_BASELINE", "proposed").strip().lower() == "no_utm":
            return False

        ev_gen = self.agent.to_generic(str(ev))
        return ev_gen.startswith("edge_take::")


    def _request_utm_authorization(self, ev: str) -> bool:
        if not self._needs_utm_authorization(ev):
            return True

        t0_request = time.perf_counter()

        def _log_request(accepted, reason, forbidden_count=0):
            try:
                self._utm_request_logger.write(
                    run_id=str(self.agent.run_id),
                    scenario_id=str(self.agent.scenario_id),
                    baseline=str(self.agent.baseline),
                    graph_size=str(self.agent.graph_size),
                    density=str(self.agent.density),
                    seed=str(self.agent.seed),
                    num_uavs=int(self.agent.num_uavs),
                    num_nodes=int(self.agent.num_nodes),
                    num_edges=int(self.agent.num_edges),
                    agent_id=int(self.agent_id),
                    event=str(ev),
                    event_generic=str(self.agent.to_generic(str(ev))),
                    accepted=int(bool(accepted)),
                    reason=str(reason),
                    request_runtime_ms=1000.0 * (time.perf_counter() - t0_request),
                    forbidden_count=int(forbidden_count),
                )
            except Exception:
                pass

        if not self.cli_utm_req.service_is_ready():
            if not self.cli_utm_req.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    "UTM authorization service unavailable for '%s'" % str(ev)
                )
                _log_request(False, "service_unavailable", 0)
                return False

        req = RequestEvent.Request()
        req.agent_id = str(self.agent_id)
        req.event = str(ev)
        req.cancel = False

        fut = self.cli_utm_req.call_async(req)
        deadline = time.time() + 5.0

        while rclpy.ok() and time.time() < deadline:
            if fut.done():
                break
            time.sleep(0.002)

        if not fut.done():
            self.get_logger().warning(
                "UTM authorization timeout for '%s'" % str(ev)
            )
            try:
                self._cancel_utm_authorization(ev)
            except Exception:
                pass
            _log_request(False, "timeout", 0)
            return False

        resp = fut.result()
        if resp is None:
            self.get_logger().warning(
                "UTM authorization returned no response for '%s'" % str(ev)
            )
            _log_request(False, "no_response", 0)
            return False

        try:
            self.agent.set_prohibited_events(list(resp.prohibited_events))
        except Exception:
            pass

        forbidden_count = len(list(resp.prohibited_events))

        if not bool(resp.accepted):
            self.get_logger().info(
                "UTM rejected '%s': %s" % (str(ev), str(resp.reason))
            )
            _log_request(False, str(resp.reason), forbidden_count)
            return False

        _log_request(True, str(resp.reason), forbidden_count)
        return True

    def _cancel_utm_authorization(self, ev: str) -> None:
        if not self._needs_utm_authorization(ev):
            return

        if not self.cli_utm_req.service_is_ready():
            return

        req = RequestEvent.Request()
        req.agent_id = str(self.agent_id)
        req.event = str(ev)
        req.cancel = True

        fut = self.cli_utm_req.call_async(req)
        deadline = time.time() + 1.0

        while rclpy.ok() and time.time() < deadline:
            if fut.done():
                break
            time.sleep(0.002)

    def _publish_event(self, ev):
        ev = str(ev or "").strip()
        if not ev:
            return

        self.pub_event.publish(String(data=ev))
        self._write_event_log("tx", ev)
        self.get_logger().info("publish: %s" % ev)

    def _try_dispatch(self):
        if self.agent.current_task() is None:
            return

        with self._dispatch_lock:
            if self.agent.current_task() is None:
                return

            if self.agent.has_pending_command():
                return

            if self.agent.buffered_event() is None and not self.agent.is_calculating():
                self.agent.request_plan(force=False, reason="dispatch_idle_replanning")

            ev = self.agent.pop_next_dispatchable_event()
            if ev is None:
                # If pop_next_dispatchable_event() cleared a stale buffered
                # command, request a new plan immediately.  Without this branch,
                # the node may remain IDLE until the next timer cycle, and with
                # repeated stale buffers this becomes apparent deadlock.
                if (
                    self.agent.current_task() is not None
                    and self.agent.buffered_event() is None
                    and not self.agent.is_calculating()
                ):
                    self.agent.request_plan(force=True, reason="dispatch_no_dispatchable_replanning")
                return

            auth_ok = self._request_utm_authorization(ev)
            if not auth_ok:
                self.agent.register_temporarily_rejected(ev)
                self.agent.dispatch_failed(ev)

                # Try an immediate anti-gridlock escape before waiting for the
                # next timer tick.  The selected edge is still authorized by UTM
                # below; this only changes which admissible candidate is tried.
                escape_ev = self.agent._anti_gridlock_escape_event(
                    state_obj=self.agent.state(),
                    terminated_flags=list(self.agent.terminated),
                    reason="utm_rejected_immediate_escape",
                )
                if escape_ev is not None and escape_ev != ev:
                    self.agent._replace_buffer(escape_ev)
                else:
                    self.agent.request_plan(force=True, reason="utm_rejected_replanning")
                return

            ok = dispatch_control_event_to_hardware(self.uav, ev)
            if not ok:
                self._cancel_utm_authorization(ev)
                self.get_logger().warning(
                    "agent=%d hardware rejected '%s'" % (self.agent_id, ev)
                )
                self.agent.dispatch_failed(ev)
                self.agent.request_plan(force=True, reason="hardware_rejected_replanning")
                return

            self._publish_event(ev)

    # ------------------------------------------------------------------
    # shutdown helper
    # ------------------------------------------------------------------

    def close(self):
        try:
            if self.timer is not None:
                self.timer.cancel()
        except Exception:
            pass
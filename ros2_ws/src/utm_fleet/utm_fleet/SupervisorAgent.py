#!/usr/bin/env python3

from __future__ import annotations

import re
import threading
import time
from typing import List, Optional, Set, Tuple

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
from rclpy.node import Node

from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String
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
            prohibited = set(self._prohibited_generic)

        out = []
        for ev_id in feasible:
            ev_gen = self.rev_event_map.get(ev_id)
            if ev_gen is None or ev_gen not in prohibited:
                out.append(ev_id)

        return sorted(out)

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
            return set(self._prohibited_generic)

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
    # planner control
    # ------------------------------------------------------------------

    def request_plan(self, force: bool = False) -> bool:
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
            daemon=True,
        )
        self._planner_thread.start()
        return True

    def _planner_worker(self) -> None:
        try:
            ev_id = self._compute_next_event()
            if ev_id is None:
                return
            self._replace_buffer(ev_id)
        finally:
            with self._planner_lock:
                self._is_calculating = False

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

        if is_controllable(ev_obj):
            self.request_plan(force=True)
            return True

        if ev_gen == "battery_low":
            self._clear_buffer()
            self.request_plan(force=True)
            return True

        if (
            ev_gen.startswith("edge_release::")
            or ev_gen.startswith("work_end::")
            or ev_gen.startswith("charge_end::")
        ):
            if self.buffered_event() is None and not self.is_calculating():
                self.request_plan(force=True)

        return True

    # ------------------------------------------------------------------
    # planning core
    # ------------------------------------------------------------------

    def _compute_next_event(self) -> Optional[str]:
        if self.current_task() is None:
            return None

        if all(self.terminated):
            return None

        self._update_dynamic_cost()

        plan_state, plan_flags = self._planning_snapshot()
        return self._plan_with_optimizer(
            state_obj=plan_state,
            terminated_flags=plan_flags,
        )

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
            prohibited_id = [self.to_id(x) for x in self._prohibited_generic]
        prohibited_id = [x for x in prohibited_id if x is not None]

        try:
            seq, _status = self.optimize_fn(
                self.supervisor,
                state_obj,
                self.planning_horizon,
                self.dynamic_cost_dict,
                interest_id,
                prohibited_id,
            )
        except Exception:
            return None

        enabled_predicted = set(self._enabled_events_from_state(state_obj))

        for ev_id in seq:
            ev_obj = self._event_objects.get(ev_id)
            if ev_obj is None:
                continue
            if not is_controllable(ev_obj):
                continue
            if ev_id not in enabled_predicted:
                continue
            return ev_id

        return None

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

        if qs in self.dynamic_cost_dict:
            E, Tf, D = self.dynamic_cost_dict[qs]
            time_spent = time.time() - self.last_state_entry_time
            if time_spent > 2.0:
                self.dynamic_cost_dict[qs] = (
                    E,
                    Tf + min(0.02 * time_spent, 0.30),
                    D + min(0.10 * time_spent, 0.50),
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
        self.uav.send_pose()

        self.agent = SupervisorAgent(
            self.model,
            agent_id=self.agent_id,
            planning_horizon=int(planning_horizon),
            speed_mps=float(speed_mps),
            energy_per_meter=0.10,
            base_time_cost=0.10,
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

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_event(self, msg):
        ev = str(msg.data or "").strip()
        if not ev:
            return

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

            self.agent.request_plan(force=True)
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

    def _on_timer(self):
        self.uav.step(self.dt)
        self.uav.send_pose()
        self._try_dispatch()

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def _publish_event(self, ev):
        ev = str(ev or "").strip()
        if not ev:
            return

        self.pub_event.publish(String(data=ev))
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
                self.agent.request_plan(force=False)

            ev = self.agent.pop_next_dispatchable_event()
            if ev is None:
                return

            ok = dispatch_control_event_to_hardware(self.uav, ev)
            if not ok:
                self.get_logger().warning(
                    "agent=%d hardware rejected '%s'" % (self.agent_id, ev)
                )
                self.agent.dispatch_failed(ev)
                self.agent.request_plan(force=True)
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



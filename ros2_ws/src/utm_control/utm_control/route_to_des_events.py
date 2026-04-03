#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def build_uav_event_sequence(uav_id, solver_solution: dict, des_model) -> dict:
    """
    Convert the route produced by ProblemSolver.solve() into a DES event sequence
    compatible with GenericUAVModel.

    Parameters
    ----------
    uav_id : int | str
        UAV index used in the MILP solution.
    solver_solution : dict
        Dictionary returned by ProblemSolver.solve().
    des_model : GenericUAVModel
        DES model instance for the same UAV / map.

    Returns
    -------
    dict
        {
            "uav_id": int,
            "route_arcs": [(stage, from_label, to_label), ...],
            "event_names": [str, ...],
            "events": [event_obj, ...],
        }

    Notes
    -----
    Expected solver_solution structure:
        - solver_solution["x"] with shape (num_nodes, num_nodes, H_max+1, num_vehicles)
        - solver_solution["required_task_labels"]
        - solver_solution["precedence_relations"]

    Label reconstruction priority:
        1. solver_solution["index_to_node_label"]
        2. solver_solution["node_labels"]
        3. sorted(des_model.G.nodes())

    Mission interpretation:
        - if precedence_relations is empty: inspection mission
        - otherwise: pickup-and-delivery mission
    """
    # ------------------------------------------------------------
    # Normalize UAV id
    # ------------------------------------------------------------
    if isinstance(uav_id, str):
        digits = "".join(ch for ch in uav_id if ch.isdigit())
        if digits == "":
            raise ValueError(f"Could not parse UAV id from '{uav_id}'.")
        k = int(digits)
    else:
        k = int(uav_id)

    if "x" not in solver_solution:
        raise ValueError("solver_solution must contain key 'x'.")

    x = np.asarray(solver_solution["x"])
    if x.ndim != 4:
        raise ValueError(
            "solver_solution['x'] must be a 4D tensor with shape "
            "(num_nodes, num_nodes, H_max+1, num_vehicles)."
        )

    n1, n2, h_size, num_vehicles = x.shape
    if n1 != n2:
        raise ValueError("solver_solution['x'] must have square first two dimensions.")
    if k < 0 or k >= num_vehicles:
        raise ValueError(
            f"uav_id={k} is outside the valid range [0, {num_vehicles - 1}]."
        )

    # ------------------------------------------------------------
    # Recover node labels
    # ------------------------------------------------------------
    if "index_to_node_label" in solver_solution:
        idx_to_label_raw = solver_solution["index_to_node_label"]
        if isinstance(idx_to_label_raw, dict):
            idx_to_label = {int(i): str(lbl) for i, lbl in idx_to_label_raw.items()}
        else:
            idx_to_label = {i: str(lbl) for i, lbl in enumerate(idx_to_label_raw)}
    elif "node_labels" in solver_solution:
        idx_to_label = {i: str(lbl) for i, lbl in enumerate(solver_solution["node_labels"])}
    else:
        # Fallback consistent with the simple Graph/ProblemParameters convention:
        # node indices follow sorted node labels.
        sorted_labels = sorted(str(n) for n in des_model.G.nodes())
        if len(sorted_labels) != n1:
            raise ValueError(
                "Could not reconstruct node labels: solution size and DES graph size differ. "
                "Provide 'index_to_node_label' or 'node_labels' inside solver_solution."
            )
        idx_to_label = {i: lbl for i, lbl in enumerate(sorted_labels)}

    # ------------------------------------------------------------
    # Extract active stage-indexed route
    # ------------------------------------------------------------
    route_arcs = []
    started = False

    for h in range(1, h_size):
        active = np.argwhere(x[:, :, h, k] > 0.5)

        if active.shape[0] == 0:
            if started:
                break
            continue

        if active.shape[0] > 1:
            raise ValueError(
                f"UAV {k} has more than one active arc at stage h={h}. "
                "The helper expects a single nominal arc per stage."
            )

        i, j = map(int, active[0])
        if i not in idx_to_label or j not in idx_to_label:
            raise ValueError(f"Missing label for node indices ({i}, {j}).")

        route_arcs.append((h, idx_to_label[i], idx_to_label[j]))
        started = True

    if not route_arcs:
        return {
            "uav_id": k,
            "route_arcs": [],
            "event_names": [],
            "events": [],
        }

    required_task_labels = {
        str(lbl) for lbl in solver_solution.get("required_task_labels", [])
    }
    precedence_relations = [
        (str(a), str(b)) for a, b in solver_solution.get("precedence_relations", [])
    ]
    is_pickup_delivery = len(precedence_relations) > 0

    event_names = []

    def add_event_name(name: str):
        if name not in des_model.events:
            raise ValueError(
                f"DES event '{name}' does not exist in GenericUAVModel.events."
            )
        event_names.append(name)

    def add_service_events_at_node(node_label: str):
        kind = des_model._kind(node_label)

        # Charging station
        if kind == "STATION":
            cs = f"charge_start::{node_label}"
            ce = f"charge_end::{node_label}"
            if cs in des_model.events and ce in des_model.events:
                add_event_name(cs)
                add_event_name(ce)
            return

        # Only required task nodes generate mission service events
        if node_label not in required_task_labels:
            return

        # Inspection mission
        if not is_pickup_delivery:
            insps = f"inspec_start::{node_label}::{kind}"
            inspe = f"inspec_end::{node_label}::{kind}"

            if insps in des_model.events and inspe in des_model.events:
                add_event_name(insps)
                add_event_name(inspe)
                return

            # fallback if inspection events are not present
            if kind == "SUPPLIER":
                ws = f"work_start::{node_label}::SUPPLIER"
                we = f"work_end::{node_label}::SUPPLIER"
                if ws in des_model.events and we in des_model.events:
                    add_event_name(ws)
                    add_event_name(we)
                    return

            if kind == "CLIENT":
                ws = f"work_start::{node_label}::CLIENT"
                we = f"work_end::{node_label}::CLIENT"
                if ws in des_model.events and we in des_model.events:
                    add_event_name(ws)
                    add_event_name(we)
                    return

            raise ValueError(
                f"No valid inspection/service event pair found for required node '{node_label}'."
            )

        # Pickup-and-delivery mission
        if kind == "SUPPLIER":
            ws = f"work_start::{node_label}::SUPPLIER"
            we = f"work_end::{node_label}::SUPPLIER"
            add_event_name(ws)
            add_event_name(we)
            return

        if kind == "CLIENT":
            ws = f"work_start::{node_label}::CLIENT"
            we = f"work_end::{node_label}::CLIENT"
            add_event_name(ws)
            add_event_name(we)
            return

        # If a required task is neither supplier nor client, try inspection pair
        insps = f"inspec_start::{node_label}::{kind}"
        inspe = f"inspec_end::{node_label}::{kind}"
        if insps in des_model.events and inspe in des_model.events:
            add_event_name(insps)
            add_event_name(inspe)
            return

        raise ValueError(
            f"No valid pickup/delivery service event pair found for required node '{node_label}'."
        )

    # ------------------------------------------------------------
    # Build event sequence
    # ------------------------------------------------------------
    for _h, from_label, to_label in route_arcs:
        take_ev = f"edge_take::{from_label}::{to_label}"
        rel_ev = f"edge_release::{from_label}::{to_label}"

        add_event_name(take_ev)
        add_event_name(rel_ev)

        # After arriving at the destination node, execute its local service if needed
        add_service_events_at_node(to_label)

    return {
        "uav_id": k,
        "route_arcs": route_arcs,
        "event_names": event_names,
        "events": [des_model.ev(name) for name in event_names],
    }
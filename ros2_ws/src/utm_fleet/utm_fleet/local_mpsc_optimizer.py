#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from gurobipy import GRB, Env, Model, quicksum
from ultrades.automata import dfa, event, states, transitions

from .extract_automaton_matrices import extract_automaton_matrices


_LOCAL_GUROBI_ENV = None


def _get_local_env():
    global _LOCAL_GUROBI_ENV
    if _LOCAL_GUROBI_ENV is None:
        _LOCAL_GUROBI_ENV = Env(empty=True)
        _LOCAL_GUROBI_ENV.setParam("OutputFlag", 0)
        _LOCAL_GUROBI_ENV.start()
    return _LOCAL_GUROBI_ENV


def _extract_subautomaton(supervisor, initial_state, horizon):
    """
    Breadth-first extraction up to depth = horizon.
    Dead-end states receive an uncontrollable self-loop 'epslon'
    to keep the local model well defined.
    """
    trans_list = list(transitions(supervisor))
    by_origin = defaultdict(list)
    for q, ev, nq in trans_list:
        by_origin[q].append((q, ev, nq))

    depth = {initial_state: 0}
    queue = deque([initial_state])
    kept_transitions = []

    while queue:
        q = queue.popleft()
        d = depth[q]
        if d >= horizon:
            continue

        for tr in by_origin.get(q, []):
            _, _, nq = tr
            kept_transitions.append(tr)
            if nq not in depth:
                depth[nq] = d + 1
                queue.append(nq)

    eps = event("epslon", controllable=False)

    origins = {q for q, _, _ in kept_transitions}
    reachable_states = set(depth.keys())
    dead_ends = reachable_states - origins

    for q in dead_ends:
        kept_transitions.append((q, eps, q))

    return dfa(kept_transitions, initial_state, f"local_sub_{str(initial_state)}")


def solve_local_mpsc(
    supervisor,
    current_state,
    horizon,
    nominal_event_sequence,
    nominal_state_sequence,
    forbidden_events=None,
    time_limit=5.0,
):
    """
    Solve the local MPSC repair MILP.

    Parameters
    ----------
    supervisor :
        Monolithic DES supervisor.
    current_state :
        Current supervisor state q^curr.
    horizon : int
        Prediction horizon H.
    nominal_event_sequence : list[str]
        Remaining nominal event string [sigma_bar_0, ..., sigma_bar_{H-1}].
    nominal_state_sequence : list
        States associated with the remaining nominal route.
        This defines Q^nom.
    forbidden_events : list[str] | None
        Events currently forbidden by the UTM layer.
    time_limit : float
        Gurobi time limit in seconds.

    Returns
    -------
    dict
        {
            "status": int,
            "objective_value": float | None,
            "event_sequence": list[str],
            "state_sequence": list[str],
            "rho": list[int],
            "event_names": list[str],
            "state_names": list[str],
        }
    """
    forbidden_events = set(forbidden_events or [])
    H = int(horizon)

    if H <= 0:
        raise ValueError("horizon must be > 0.")

    if len(nominal_event_sequence) < H:
        raise ValueError(
            "nominal_event_sequence must contain at least H events."
        )

    # ------------------------------------------------------------
    # Local sub-automaton and matrices
    # ------------------------------------------------------------
    sub = _extract_subautomaton(supervisor, current_state, H)

    A_csr, B_csr, C_csr, W, D_np, event_dict, state_index = extract_automaton_matrices(sub, 3)

    state_list = list(states(sub))
    state_names = [str(q) for q in state_list]

    n = A_csr.shape[0]
    m = C_csr.shape[1]

    event_names = list(event_dict.keys())
    event_to_idx = {name: i for i, name in enumerate(event_names)}

    # ------------------------------------------------------------
    # Nominal data
    # ------------------------------------------------------------
    nominal_event_sequence = [str(e) for e in nominal_event_sequence[:H]]
    nominal_state_set = {str(q) for q in nominal_state_sequence}

    forb_idx = [event_to_idx[e] for e in forbidden_events if e in event_to_idx]

    # d[l,e] = 0 if event matches nominal event at step l, 1 otherwise
    d = np.ones((H, m), dtype=np.int32)
    for l in range(H):
        e_nom = nominal_event_sequence[l]
        if e_nom in event_to_idx:
            d[l, event_to_idx[e_nom]] = 0

    # rho state mask: 1 if state belongs to Q^nom, 0 otherwise
    nominal_mask = np.zeros(n, dtype=np.int32)
    for q, idx in state_index.items():
        if str(q) in nominal_state_set:
            nominal_mask[idx] = 1

    # ------------------------------------------------------------
    # Reachability pruning by level
    # ------------------------------------------------------------
    A_bool = (A_csr != 0).astype(np.int8)

    reachable = []
    current = np.array([0], dtype=np.int32)  # local initial state is index 0
    reachable.append(current)

    for _ in range(H):
        if current.size == 0:
            nxt = np.array([], dtype=np.int32)
        else:
            nxt = np.unique(A_bool[current, :].indices).astype(np.int32)
        reachable.append(nxt)
        current = nxt

    pos = [{int(s): i for i, s in enumerate(reachable[l])} for l in range(H + 1)]

    # ------------------------------------------------------------
    # MILP
    # ------------------------------------------------------------
    env = _get_local_env()
    model = Model("local_mpsc_repair", env=env)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", float(time_limit))

    # x_l(j)
    x = [
        model.addMVar(len(reachable[l]), vtype=GRB.BINARY, name=f"x_{l}")
        for l in range(H + 1)
    ]

    # u_l(e)
    u = model.addMVar((H, m), vtype=GRB.BINARY, name="u")

    # rho_l
    rho = model.addMVar(H + 1, vtype=GRB.BINARY, name="rho")

    # initial state x_0 = q_curr
    if 0 not in pos[0]:
        raise RuntimeError("Current state was not mapped to local state index 0.")
    model.addConstr(x[0][pos[0][0]] == 1, name="init_state")

    # one-hot x
    for l in range(H + 1):
        model.addConstr(x[l].sum() == 1, name=f"onehot_x_{l}")

    # one-hot u
    for l in range(H):
        model.addConstr(u[l, :].sum() == 1, name=f"onehot_u_{l}")

    # enabled-event feasibility
    C_dense = np.asarray(C_csr.todense())
    for l in range(H):
        rl = reachable[l]
        if len(rl) == 0:
            continue
        C_sub = C_dense[rl, :]
        model.addConstr(
            u[l, :] <= x[l] @ C_sub,
            name=f"event_feas_{l}",
        )

    # forbidden events
    for l in range(H):
        for e_idx in forb_idx:
            model.addConstr(u[l, e_idx] == 0, name=f"forbidden_l{l}_e{e_idx}")

    # deterministic dynamics x_{l+1} = (A x_l) o (B u_l)
    A_T = A_csr.transpose().tocsr()
    B_dense = np.asarray(B_csr.todense())

    for l in range(H):
        rl = reachable[l]
        rlp1 = reachable[l + 1]

        incoming_sources = defaultdict(list)

        for idx_next, s_next in enumerate(rlp1):
            prev_states = A_T[s_next, :].indices

            for idx_curr, s_curr in enumerate(rl):
                if s_curr not in prev_states:
                    continue

                valid_events = np.where(
                    (C_dense[s_curr, :] > 0) & (B_dense[:, s_next] > 0)
                )[0]

                for e_idx in valid_events:
                    incoming_sources[idx_next].append((idx_curr, e_idx))

        for idx_next in range(len(rlp1)):
            sources = incoming_sources[idx_next]
            if sources:
                model.addConstr(
                    x[l + 1][idx_next] ==
                    quicksum(x[l][i] * u[l, e] for i, e in sources),
                    name=f"dyn_l{l}_s{idx_next}",
                )
            else:
                model.addConstr(
                    x[l + 1][idx_next] == 0,
                    name=f"dyn_zero_l{l}_s{idx_next}",
                )

    # rho_l = sum_{j in Q_nom} x_l,j
    for l in range(H + 1):
        rl = reachable[l]
        if len(rl) == 0:
            model.addConstr(rho[l] == 0, name=f"rho_zero_{l}")
            continue

        mask = np.array([nominal_mask[s] for s in rl], dtype=np.int32)
        model.addConstr(
            rho[l] == x[l] @ mask,
            name=f"rho_def_{l}",
        )

    # objective: minimize cumulative deviation from nominal event sequence
    model.setObjective(
        quicksum(float(d[l, e]) * u[l, e] for l in range(H) for e in range(m)),
        GRB.MINIMIZE,
    )

    model.optimize()

    status = int(model.status)

    if model.SolCount == 0:
        if status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("local_mpsc_repair_iis.ilp")
        model.dispose()
        return {
            "status": status,
            "objective_value": None,
            "event_sequence": [],
            "state_sequence": [],
            "rho": [],
            "event_names": event_names,
            "state_names": state_names,
        }

    # ------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------
    u_sol = np.asarray(u.X)
    rho_sol = np.asarray(rho.X).round().astype(int).tolist()

    event_sequence = []
    for l in range(H):
        e_idx = int(np.argmax(u_sol[l, :]))
        event_sequence.append(event_names[e_idx])

    state_sequence = []
    for l in range(H + 1):
        x_l = np.asarray(x[l].X).reshape(-1)
        idx_local = int(np.argmax(x_l))
        state_global = reachable[l][idx_local]
        state_sequence.append(str(state_list[state_global]))

    obj = float(model.ObjVal)

    model.dispose()

    return {
        "status": status,
        "objective_value": obj,
        "event_sequence": event_sequence,
        "state_sequence": state_sequence,
        "rho": rho_sol,
        "event_names": event_names,
        "state_names": state_names,
    }
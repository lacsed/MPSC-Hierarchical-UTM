import math
from ultrades.automata import *


# ----------------------------- core helpers -----------------------------

def _clamp0(x):
    return 0.0 if x < 0.0 else float(x)


def _norm_params(params, defaults):
    p = {} if params is None else dict(params)
    for k, v in defaults.items():
        if k not in p:
            p[k] = v
    for k in p:
        try:
            p[k] = float(p[k])
        except Exception:
            p[k] = float(defaults.get(k, 0.0))
        if p[k] < 0.0:
            p[k] = 0.0
    return p


def _xyz(model, n):
    p = model.pos.get(str(n))
    if not p:
        return None
    x = float(p[0])
    y = float(p[1])
    z = float(p[2]) if len(p) >= 3 else 0.0
    return (x, y, z)


def _dist3_cached(model, a, b):
    if not hasattr(model, "_dist_cache_3d"):
        model._dist_cache_3d = {}

    ka = str(a)
    kb = str(b)
    k = (ka, kb) if ka <= kb else (kb, ka)
    d = model._dist_cache_3d.get(k)
    if d is not None:
        return d

    pa = _xyz(model, a)
    pb = _xyz(model, b)
    if pa is None or pb is None:
        d = 1.0
    else:
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        dz = pb[2] - pa[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

    model._dist_cache_3d[k] = d
    return d


# ----------------------------- atomic cost -----------------------------

def build_atomic_cost_dict(model, speed_mps=2.0, energy_per_meter=0.10, base_time_cost=0.10):
    """
    atomic_cost[state_str] = (E, Tf, D)

    - Base: all atomic states get D = base_time_cost
    - EDGE_OCC::<a>::<b> and EDGE_OCC::<b>::<a>: E,Tf based on 3D distance and speed/energy_per_meter
    - Uses model.edge_bundle (the model DOES NOT have edge_pairs)
    All outputs clamped to >= 0.
    """
    speed_mps = float(speed_mps)
    if speed_mps <= 1e-9:
        speed_mps = 2.0
    energy_per_meter = abs(float(energy_per_meter))
    base_time_cost = _clamp0(float(base_time_cost))

    atomic_cost = {}

    # base time cost on every atomic state
    btc = base_time_cost
    for _, A in model.automata.items():
        for q in states(A):
            atomic_cost[str(q)] = (0.0, 0.0, btc)

    # physical cost on edge occupancy (undirected pair distance, applied to BOTH directions if state exists)
    undirected_pairs = set()
    for (ab, _k) in model.edge_bundle.keys():
        undirected_pairs.add(ab)  # ab already (a<=b) by construction in your model

    for (a, b) in undirected_pairs:
        d = _dist3_cached(model, a, b)
        tf = d / speed_mps
        e = d * energy_per_meter

        occ_ab = f"EDGE_OCC::{a}::{b}"
        if occ_ab in atomic_cost:
            atomic_cost[occ_ab] = (_clamp0(e), _clamp0(tf), btc)

        occ_ba = f"EDGE_OCC::{b}::{a}"
        if occ_ba in atomic_cost:
            atomic_cost[occ_ba] = (_clamp0(e), _clamp0(tf), btc)

    return atomic_cost


def set_atomic_cost_entry(atomic_cost, atomic_state_str, new_cost_vec):
    """
    Update ONE atomic state's cost (E,Tf,D) (clamped >=0). Returns the written vec.
    """
    if (not isinstance(new_cost_vec, tuple)) or len(new_cost_vec) != 3:
        raise ValueError("new_cost_vec must be a tuple(E,Tf,D).")
    e = _clamp0(float(new_cost_vec[0]))
    tf = _clamp0(float(new_cost_vec[1]))
    d = _clamp0(float(new_cost_vec[2]))
    atomic_cost[str(atomic_state_str)] = (e, tf, d)
    return (e, tf, d)


# ----------------------------- supervisor indexing (fast) -----------------------------

def build_supervisor_cost_index(model):
    """
    Pre-parse each supervisor state once.

    Returns:
      sup_index[qs] = (parts_tuple, repeat_count, at_kind, is_idle, is_low, is_chg, is_work, wf_count)
      rev_index[atomic_part] = [qs, qs, ...]
    """
    sup = getattr(model, "supervisor_mono", None)
    if sup is None:
        raise RuntimeError("model.supervisor_mono is None.")

    sup_index = {}
    rev_index = {}

    # fast membership check for "where am I?" using map-state == node_id
    node_set = set(str(n) for n in model.G.nodes())

    for q in states(sup):
        qs = str(q)
        parts = tuple(p for p in qs.split("|") if p)

        uniq = set(parts)
        repeat_count = len(parts) - len(uniq)

        is_idle = False
        is_low = False
        is_chg = False
        is_work = False
        wf_count = 0

        at_kind = ""
        at_node = None

        for p in parts:
            if (not is_idle) and p == "IDLE":
                is_idle = True
            elif (not is_low) and p == "BAT_LOW":
                is_low = True
            elif (not is_chg) and p.startswith("MODE_CHARGE::"):
                is_chg = True
            elif (not is_work) and (p.startswith("MODE_WORK_SUPPLIER::") or p.startswith("MODE_WORK_CLIENT::")):
                is_work = True

            if p == "WF_BASE" or p == "WF_PICK" or p == "WF_PLACE":
                wf_count += 1

            # old naming (if ever present)
            if (at_node is None) and p.startswith("AT::"):
                at_node = p[4:]

            # current model: map state is literally the node id
            if (at_node is None) and (p in node_set):
                at_node = p

        if at_node is not None:
            at_kind = model._kind(at_node)

        sup_index[qs] = (parts, repeat_count, at_kind, is_idle, is_low, is_chg, is_work, wf_count)

        for ap in uniq:
            lst = rev_index.get(ap)
            if lst is None:
                rev_index[ap] = [qs]
            else:
                lst.append(qs)

    return sup_index, rev_index


# ----------------------------- supervisor cost (from atomic) -----------------------------

def supervisor_state_cost_from_atomic(model, sup_state_str, atomic_cost, params=None, sup_index=None):
    """
    Compute cost of ONE supervisor state (E,Tf,D), non-negative.
    """
    defaults = {
        "REPEAT_PENALTY": 0.30,
        "EARLY_CHARGE_PENALTY": 2.00,
        "NO_CHARGE_PENALTY": 15.0,   # low battery and NOT charging
        "NO_WORK_PENALTY": 3.00,     # idle at supplier/client without working
        "NO_BASE_PENALTY": 2.00,     # idle outside base
        "TASK_PROGRESS_BONUS": 5.0,  # reduces D (clamped)
        "IDLE_PENALTY": 0.50,        # idle generic penalty
    }
    p = _norm_params(params, defaults)

    key = str(sup_state_str)

    if sup_index is not None and key in sup_index:
        parts, repeat_count, at_kind, is_idle, is_low, is_chg, is_work, wf_count = sup_index[key]
    else:
        parts = tuple(x for x in key.split("|") if x)
        uniq = set(parts)
        repeat_count = len(parts) - len(uniq)
        is_idle = ("IDLE" in uniq)
        is_low = ("BAT_LOW" in uniq)
        is_chg = any(x.startswith("MODE_CHARGE::") for x in uniq)
        is_work = any(x.startswith("MODE_WORK_SUPPLIER::") or x.startswith("MODE_WORK_CLIENT::") for x in uniq)
        wf_count = sum(1 for x in uniq if x in ("WF_BASE", "WF_PICK", "WF_PLACE"))

        at_kind = ""
        node_set = set(str(n) for n in model.G.nodes())
        for x in uniq:
            if x.startswith("AT::"):
                at_kind = model._kind(x[4:])
                break
            if x in node_set:
                at_kind = model._kind(x)
                break

    ac_get = atomic_cost.get

    E = 0.0
    Tf = 0.0
    D = 0.0

    for part in parts:
        c = ac_get(part)
        if c is not None:
            E += float(c[0])
            Tf += float(c[1])
            D += float(c[2])

    if repeat_count > 0:
        D += p["REPEAT_PENALTY"] * float(repeat_count)

    if is_idle:
        D += p["IDLE_PENALTY"]

    if is_chg and (not is_low):
        D += p["EARLY_CHARGE_PENALTY"]

    if is_low and (not is_chg):
        D += p["NO_CHARGE_PENALTY"]

    if is_idle and (not is_work) and (at_kind == "SUPPLIER" or at_kind == "CLIENT"):
        D += p["NO_WORK_PENALTY"]

    if is_idle and (not is_work) and (at_kind != ""):
        if (at_kind != "STATION") and (at_kind != "VERTIPORT"):
            D += p["NO_BASE_PENALTY"]

    if wf_count >= 2:
        D = max(0.0, D - (p["TASK_PROGRESS_BONUS"] / float(wf_count)))

    return (_clamp0(E), _clamp0(Tf), _clamp0(D))


def build_supervisor_cost_dict(model, atomic_cost, params=None, sup_index=None):
    """
    Build ALL supervisor costs (fast if sup_index provided).
    Returns sup_cost_dict.
    """
    sup = getattr(model, "supervisor_mono", None)
    if sup is None:
        raise RuntimeError("Supervisor not computed in model.")

    if sup_index is None:
        sup_index, _ = build_supervisor_cost_index(model)

    out = {}
    f = supervisor_state_cost_from_atomic
    for q in states(sup):
        qs = str(q)
        out[qs] = f(model, qs, atomic_cost, params=params, sup_index=sup_index)
    return out


# ----------------------------- updates (fast incremental) -----------------------------

def update_supervisor_cost_entry(model, sup_cost_dict, sup_state_str, atomic_cost, params=None, sup_index=None):
    """
    Recompute + overwrite ONE supervisor state's cost, based on atomic_cost.
    """
    key = str(sup_state_str)
    if key not in sup_cost_dict:
        raise KeyError(f"Supervisor state not found in sup_cost_dict: {key}")
    new_cost = supervisor_state_cost_from_atomic(model, key, atomic_cost, params=params, sup_index=sup_index)
    sup_cost_dict[key] = new_cost
    return new_cost


def update_supervisor_costs_for_atomic_changes(model, sup_cost_dict, changed_atomic_states,
                                              atomic_cost, params=None, sup_index=None, rev_index=None):
    """
    Incremental update:
    - changed_atomic_states: iterable of atomic state strings that changed in atomic_cost
    - uses rev_index to update ONLY affected supervisor states.
    Returns number of updated supervisor states.
    """
    if rev_index is None:
        sup_index, rev_index = build_supervisor_cost_index(model)
    elif sup_index is None:
        raise ValueError("If rev_index is provided, sup_index must also be provided for fast update.")

    affected = set()
    for a in changed_atomic_states:
        lst = rev_index.get(str(a))
        if lst:
            affected.update(lst)

    f = supervisor_state_cost_from_atomic
    for qs in affected:
        sup_cost_dict[qs] = f(model, qs, atomic_cost, params=params, sup_index=sup_index)

    return len(affected)


# ----------------------------- manager/orchestrator (one call) -----------------------------

def build_cost_engine(model, speed_mps=2.0, energy_per_meter=0.10, base_time_cost=0.10, params=None):
    """
    One-shot builder:
      - atomic_cost
      - sup_index, rev_index
      - sup_cost
    Returns an engine dict (pure data) for fast updates.
    """
    atomic_cost = build_atomic_cost_dict(
        model,
        speed_mps=speed_mps,
        energy_per_meter=energy_per_meter,
        base_time_cost=base_time_cost,
    )
    sup_index, rev_index = build_supervisor_cost_index(model)
    sup_cost = build_supervisor_cost_dict(model, atomic_cost, params=params, sup_index=sup_index)

    return {
        "atomic_cost": atomic_cost,
        "sup_cost": sup_cost,
        "sup_index": sup_index,
        "rev_index": rev_index,
        "params": {} if params is None else dict(params),
        "speed_mps": float(speed_mps),
        "energy_per_meter": abs(float(energy_per_meter)),
        "base_time_cost": _clamp0(float(base_time_cost)),
    }


def rebuild_all_costs(engine, model, speed_mps=None, energy_per_meter=None, base_time_cost=None, params=None):
    """
    Full rebuild:
      - rebuild atomic_cost (new physical params)
      - rebuild all supervisor costs (same index)
    """
    if speed_mps is None:
        speed_mps = engine.get("speed_mps", 2.0)
    if energy_per_meter is None:
        energy_per_meter = engine.get("energy_per_meter", 0.10)
    if base_time_cost is None:
        base_time_cost = engine.get("base_time_cost", 0.10)
    if params is None:
        params = engine.get("params", None)

    atomic_cost = build_atomic_cost_dict(
        model,
        speed_mps=speed_mps,
        energy_per_meter=energy_per_meter,
        base_time_cost=base_time_cost,
    )

    sup_index = engine.get("sup_index")
    rev_index = engine.get("rev_index")
    if (sup_index is None) or (rev_index is None):
        sup_index, rev_index = build_supervisor_cost_index(model)

    sup_cost = build_supervisor_cost_dict(model, atomic_cost, params=params, sup_index=sup_index)

    engine["atomic_cost"] = atomic_cost
    engine["sup_cost"] = sup_cost
    engine["sup_index"] = sup_index
    engine["rev_index"] = rev_index
    engine["params"] = {} if params is None else dict(params)
    engine["speed_mps"] = float(speed_mps)
    engine["energy_per_meter"] = abs(float(energy_per_meter))
    engine["base_time_cost"] = _clamp0(float(base_time_cost))

    return engine


def set_atomic_cost_and_update_supervisor(engine, model, atomic_state_str, new_cost_vec):
    """
    Fast path:
      - update ONE atomic state cost
      - update ONLY supervisor states that depend on it
    Returns (atomic_written_vec, updated_supervisor_count).
    """
    atomic_cost = engine["atomic_cost"]
    sup_cost = engine["sup_cost"]
    sup_index = engine["sup_index"]
    rev_index = engine["rev_index"]
    params = engine.get("params", None)

    written = set_atomic_cost_entry(atomic_cost, atomic_state_str, new_cost_vec)

    nup = update_supervisor_costs_for_atomic_changes(
        model,
        sup_cost,
        [str(atomic_state_str)],
        atomic_cost,
        params=params,
        sup_index=sup_index,
        rev_index=rev_index,
    )
    return written, nup

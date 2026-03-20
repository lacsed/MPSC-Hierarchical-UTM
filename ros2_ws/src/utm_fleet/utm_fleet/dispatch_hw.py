import re
_RE_SUFFIX = re.compile(r"^(.*)_(\d+)$")


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------

def dist2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def nearest_node_id(positions, x, y):
    return min(
        positions.items(),
        key=lambda kv: dist2((kv[1][0], kv[1][1]), (x, y)),
    )[0]


def parse_task(raw):
    raw = str(raw or "").strip()
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


def split_suffix_id(ev):
    ev = str(ev or "").strip()
    m = _RE_SUFFIX.match(ev)
    if not m:
        return ev, None
    return m.group(1), int(m.group(2))


def dispatch_control_event_to_hardware(hw, ev):
    ev = str(ev or "").strip()
    if not ev:
        return False

    base, eid = split_suffix_id(ev)
    if eid is None or eid != int(hw.agent_id):
        return False

    if base.startswith("edge_take::"):
        rest = base.split("edge_take::", 1)[1]
        parts = rest.split("::")
        if len(parts) != 2:
            return False
        u, v = parts
        return hw.start_move(u, v)

    if base.startswith("work_start::"):
        rest = base.split("work_start::", 1)[1]
        parts = rest.split("::")
        if len(parts) != 2:
            return False

        node_id, kind = parts
        kind = kind.upper().strip()

        if kind == "SUPPLIER":
            return hw.start_pick(node_id)
        if kind == "CLIENT":
            return hw.start_deliver(node_id)
        return False

    if base.startswith("charge_start::"):
        node_id = base.split("charge_start::", 1)[1]
        return hw.start_charge(node_id)

    return False


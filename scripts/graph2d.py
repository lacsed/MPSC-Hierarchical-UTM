#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# graph2d.py

import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from city_gen import point_in_any_box, seg_hits_any_box


# -----------------------------
# Tunables (may be overwritten by importer)
# -----------------------------

K_SPECIAL = 1

LOGICAL_MAX_DEG = 4
LOGICAL_LL_TARGET = 2
LOGICAL_SL_CAP = LOGICAL_MAX_DEG - LOGICAL_LL_TARGET  # 2 by default

SPECIAL_MIN_DEG = 1
SPECIAL_MAX_DEG = 2

TRI_MIN_AREA_PX2 = 1200.0

MIN_PT_CLEAR_PX = 12.0
MIN_EDGE_CLEAR_PX = 12.0
MIN_LOGICAL_TO_SPECIAL_PX = 10.0

POINT_EPS = 1e-9
TOUCH_EPS = 1e-9

KNN_LOCAL = 32
MAX_OUT_EDGES = 16

# LL path search
LL_GRID_CELL = 96.0
LL_R_STEP = 120.0
LL_R_STEPS = 12
LL_TRIES = 60


# -----------------------------
# Globals / cache
# -----------------------------

_GLOBAL_POS_SKEL = None
_GLOBAL_CENTER_PX = None

_GLOBAL_BOXES = None
_GLOBAL_ALLOW_OVERFLIGHT = False

_CACHE_KEY = None
_TRIANGLES = None
_FORBID_TRIS = None
_LOGICAL_SIDS = None
_LOGICAL_POS = None
_LOGICAL_TRI = None
_TRI_KEY_TO_LOGICAL = None
_SPECIAL_TO_LOGICALS = None

_LAST_STATE = None


def set_logical_globals(pos_skel, center_px):
    global _GLOBAL_POS_SKEL, _GLOBAL_CENTER_PX
    _GLOBAL_POS_SKEL = pos_skel
    _GLOBAL_CENTER_PX = (float(center_px[0]), float(center_px[1]))


def _ensure_logical_globals():
    if _GLOBAL_POS_SKEL is None or _GLOBAL_CENTER_PX is None:
        raise RuntimeError("Logical globals not initialized (call set_logical_globals).")


def _px(sid):
    _ensure_logical_globals()
    return _GLOBAL_POS_SKEL[sid]


# -----------------------------
# Geometry
# -----------------------------

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _sub(a, b):
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _add(a, b):
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]))


def _mul(a, s):
    return (float(a[0]) * float(s), float(a[1]) * float(s))


def _dot(a, b):
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _orient(a, b, c):
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


def _points_equal(p, q, eps=POINT_EPS):
    return abs(float(p[0]) - float(q[0])) <= eps and abs(float(p[1]) - float(q[1])) <= eps


def _on_segment(a, p, b):
    return (
        min(a[0], b[0]) - TOUCH_EPS <= p[0] <= max(a[0], b[0]) + TOUCH_EPS
        and min(a[1], b[1]) - TOUCH_EPS <= p[1] <= max(a[1], b[1]) + TOUCH_EPS
    )


def _segments_intersect_or_touch(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    if abs(o1) <= TOUCH_EPS and _on_segment(a, c, b):
        return True
    if abs(o2) <= TOUCH_EPS and _on_segment(a, d, b):
        return True
    if abs(o3) <= TOUCH_EPS and _on_segment(c, a, d):
        return True
    if abs(o4) <= TOUCH_EPS and _on_segment(c, b, d):
        return True

    return False


def _closest_point_on_segment(p, a, b):
    ab = _sub(b, a)
    ap = _sub(p, a)
    ab2 = _dot(ab, ab)
    if ab2 <= 1e-12:
        return a, 0.0
    t = _dot(ap, ab) / ab2
    t = _clamp(t, 0.0, 1.0)
    c = _add(a, _mul(ab, t))
    return c, t


def _closest_points_segments(a, b, c, d):
    d1 = _sub(b, a)
    d2 = _sub(d, c)
    r = _sub(a, c)
    a1 = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    eps = 1e-12

    if a1 <= eps and e <= eps:
        return a, c, _dist(a, c)

    if a1 <= eps:
        t = _clamp(f / e, 0.0, 1.0) if e > eps else 0.0
        p1 = a
        p2 = _add(c, _mul(d2, t))
        return p1, p2, _dist(p1, p2)

    c1 = _dot(d1, r)
    if e <= eps:
        s = _clamp(-c1 / a1, 0.0, 1.0)
        p1 = _add(a, _mul(d1, s))
        p2 = c
        return p1, p2, _dist(p1, p2)

    b1 = _dot(d1, d2)
    denom = a1 * e - b1 * b1
    if abs(denom) > eps:
        s = _clamp((b1 * f - c1 * e) / denom, 0.0, 1.0)
    else:
        s = 0.0

    t = (b1 * s + f) / e
    if t < 0.0:
        t = 0.0
        s = _clamp(-c1 / a1, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = _clamp((b1 - c1) / a1, 0.0, 1.0)

    p1 = _add(a, _mul(d1, s))
    p2 = _add(c, _mul(d2, t))
    return p1, p2, _dist(p1, p2)


def _triangle_area2(a, b, c):
    return abs(_orient(a, b, c))


def _point_in_tri_strict(p, a, b, c, eps=1e-9):
    o1 = _orient(a, b, p)
    o2 = _orient(b, c, p)
    o3 = _orient(c, a, p)
    has_pos = (o1 > eps) or (o2 > eps) or (o3 > eps)
    has_neg = (o1 < -eps) or (o2 < -eps) or (o3 < -eps)
    if has_pos and has_neg:
        return False
    if abs(o1) <= eps or abs(o2) <= eps or abs(o3) <= eps:
        return False
    return True


def _triangles_overlap_strict(t1, t2):
    (A, B, C) = t1
    (D, E, F) = t2
    e1 = [(A, B), (B, C), (C, A)]
    e2 = [(D, E), (E, F), (F, D)]

    for (p, q) in e1:
        for (r, s) in e2:
            if not _segments_intersect_or_touch(p, q, r, s):
                continue
            shared = _points_equal(p, r) or _points_equal(p, s) or _points_equal(q, r) or _points_equal(q, s)
            if not shared:
                return True

    c1 = ((A[0] + B[0] + C[0]) / 3.0, (A[1] + B[1] + C[1]) / 3.0)
    c2 = ((D[0] + E[0] + F[0]) / 3.0, (D[1] + E[1] + F[1]) / 3.0)
    if _point_in_tri_strict(c1, D, E, F):
        return True
    if _point_in_tri_strict(c2, A, B, C):
        return True
    return False


def _tri_bbox(tri):
    (A, B, C) = tri
    x0 = min(A[0], B[0], C[0])
    y0 = min(A[1], B[1], C[1])
    x1 = max(A[0], B[0], C[0])
    y1 = max(A[1], B[1], C[1])
    return (x0, y0, x1, y1)


def _seg_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def _segment_crosses_triangle_interior(a, b, tri):
    (A, B, C) = tri
    shared = (
        _points_equal(a, A) or _points_equal(a, B) or _points_equal(a, C)
        or _points_equal(b, A) or _points_equal(b, B) or _points_equal(b, C)
    )

    mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
    if _point_in_tri_strict(mid, A, B, C):
        return True

    if _segments_intersect_or_touch(a, b, A, B) or _segments_intersect_or_touch(a, b, B, C) or _segments_intersect_or_touch(a, b, C, A):
        if not shared:
            return True
    return False


def _circumradius(tri):
    (A, B, C) = tri
    a = _dist(B, C)
    b = _dist(A, C)
    c = _dist(A, B)
    area2 = _triangle_area2(A, B, C)
    if area2 <= 1e-12:
        return 0.0
    area = 0.5 * area2
    R = (a * b * c) / (4.0 * area)
    return float(R)


# -----------------------------
# Spatial indices
# -----------------------------

class _GridIndex:
    def __init__(self, cell_size):
        self.cell = float(max(1.0, cell_size))
        self.grid = defaultdict(list)

    def _cell_xy(self, x, y):
        return (int(math.floor(float(x) / self.cell)), int(math.floor(float(y) / self.cell)))

    def insert(self, key, p):
        self.grid[self._cell_xy(p[0], p[1])].append((key, p))

    def query_bbox(self, x0, y0, x1, y1):
        cx0, cy0 = self._cell_xy(x0, y0)
        cx1, cy1 = self._cell_xy(x1, y1)
        seen = set()
        for ix in range(cx0, cx1 + 1):
            for iy in range(cy0, cy1 + 1):
                for (k, p) in self.grid.get((ix, iy), []):
                    if k not in seen:
                        seen.add(k)
                        yield k, p


class _SegmentIndex:
    def __init__(self, cell_size):
        self.cell = float(max(1.0, cell_size))
        self.grid = defaultdict(list)
        self.data = {}
        self.active = set()

    def _cells_for_seg(self, a, b, pad=2.0):
        x0 = min(a[0], b[0]) - pad
        y0 = min(a[1], b[1]) - pad
        x1 = max(a[0], b[0]) + pad
        y1 = max(a[1], b[1]) + pad
        ix0 = int(math.floor(x0 / self.cell))
        iy0 = int(math.floor(y0 / self.cell))
        ix1 = int(math.floor(x1 / self.cell))
        iy1 = int(math.floor(y1 / self.cell))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)

    def add(self, seg_id, a, b, u, v):
        a = (float(a[0]), float(a[1]))
        b = (float(b[0]), float(b[1]))
        self.data[seg_id] = (a, b, u, v)
        self.active.add(seg_id)
        for c in self._cells_for_seg(a, b):
            self.grid[c].append(seg_id)

    def candidates(self, a, b):
        a = (float(a[0]), float(a[1]))
        b = (float(b[0]), float(b[1]))  # fixed
        seen = set()
        for c in self._cells_for_seg(a, b):
            for sid in self.grid.get(c, []):
                if sid in self.active and sid not in seen:
                    seen.add(sid)
                    yield sid

    def get(self, seg_id):
        return self.data[seg_id]


class _TriIndex:
    def __init__(self, cell_size):
        self.cell = float(max(1.0, cell_size))
        self.grid = defaultdict(list)
        self.tris = {}

    def _cell_xy(self, x, y):
        return (int(math.floor(float(x) / self.cell)), int(math.floor(float(y) / self.cell)))

    def add(self, tid, tri):
        self.tris[tid] = tri
        x0, y0, x1, y1 = _tri_bbox(tri)
        cx0, cy0 = self._cell_xy(x0, y0)
        cx1, cy1 = self._cell_xy(x1, y1)
        for ix in range(cx0, cx1 + 1):
            for iy in range(cy0, cy1 + 1):
                self.grid[(ix, iy)].append(tid)

    def candidates_bbox(self, x0, y0, x1, y1):
        cx0, cy0 = self._cell_xy(x0, y0)
        cx1, cy1 = self._cell_xy(x1, y1)
        seen = set()
        for ix in range(cx0, cx1 + 1):
            for iy in range(cy0, cy1 + 1):
                for tid in self.grid.get((ix, iy), []):
                    if tid not in seen:
                        seen.add(tid)
                        yield tid

    def get(self, tid):
        return self.tris[tid]


# -----------------------------
# Segment validity
# -----------------------------

def _segment_ok(
    a, b, u, v,
    seg_index,
    boxes,
    ignore_boxes=None,
    min_pt_clear=0.0,
    pts=None,
    min_seg_clear=0.0,
    tri_index=None,
):
    if ignore_boxes is None:
        ignore_boxes = set()

    if boxes and seg_hits_any_box(a, b, boxes, ignore_indices=ignore_boxes):
        return False

    if tri_index is not None:
        x0, y0, x1, y1 = _seg_bbox(a, b)
        for tid in tri_index.candidates_bbox(x0, y0, x1, y1):
            tri = tri_index.get(tid)
            if _segment_crosses_triangle_interior(a, b, tri):
                return False

    for sid in seg_index.candidates(a, b):
        (c, d, u2, v2) = seg_index.get(sid)
        shares = (u == u2) or (u == v2) or (v == u2) or (v == v2)

        if _segments_intersect_or_touch(a, b, c, d):
            if not shares:
                return False
            endpoints = [a, b, c, d]
            segs = [((a, b), (u, v)), ((c, d), (u2, v2))]
            for X in endpoints:
                for (S, T), _uv in segs:
                    if _points_equal(X, S) or _points_equal(X, T):
                        continue
                    if abs(_orient(S, T, X)) <= TOUCH_EPS and _on_segment(S, X, T):
                        return False

        if min_seg_clear > 0.0 and not shares:
            _p1, _p2, dd = _closest_points_segments(a, b, c, d)
            if dd < min_seg_clear:
                return False

    if min_pt_clear > 0.0 and pts:
        for pid, p in pts:
            if pid == u or pid == v:
                continue
            cp, _t = _closest_point_on_segment(p, a, b)
            if _dist(p, cp) < min_pt_clear:
                return False

    return True


# -----------------------------
# Triangulation (left-to-right)
# -----------------------------

def _cache_key(specials, boxes, allow_overflight, seed):
    s = tuple(sorted((str(sp["id"]), round(float(sp["px"]), 3), round(float(sp["py"]), 3)) for sp in specials))
    b = tuple((int(round(x)), int(round(y)), int(round(w)), int(round(h))) for (_a, x, y, w, h) in boxes)
    return (s, b, bool(allow_overflight), int(seed), int(KNN_LOCAL), int(MAX_OUT_EDGES), float(TRI_MIN_AREA_PX2))


def _centroid(tri):
    (A, B, C) = tri
    return ((A[0] + B[0] + C[0]) / 3.0, (A[1] + B[1] + C[1]) / 3.0)


def _barycentric_sample(rnd, A, B, C):
    r1 = rnd.random()
    r2 = rnd.random()
    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2
    P = _add(A, _add(_mul(_sub(B, A), r1), _mul(_sub(C, A), r2)))
    return (float(P[0]), float(P[1]))


def _min_dist_to_points(p, pts):
    dmin = 1e18
    for q in pts:
        d = _dist(p, q)
        if d < dmin:
            dmin = d
    return dmin


def _pick_centroid_or_near(tri, boxes, specials_pts, rnd, tries=32):
    c = _centroid(tri)
    if (not boxes or not point_in_any_box(c, boxes, ignore_indices=set())) and _min_dist_to_points(c, specials_pts) >= MIN_LOGICAL_TO_SPECIAL_PX:
        return (float(c[0]), float(c[1]))
    A, B, C = tri
    best = c
    for _ in range(tries):
        q = _barycentric_sample(rnd, A, B, C)
        q = (0.8 * c[0] + 0.2 * q[0], 0.8 * c[1] + 0.2 * q[1])
        if boxes and point_in_any_box(q, boxes, ignore_indices=set()):
            continue
        if _min_dist_to_points(q, specials_pts) < MIN_LOGICAL_TO_SPECIAL_PX:
            continue
        if _point_in_tri_strict(q, A, B, C):
            best = q
            break
    return (float(best[0]), float(best[1]))


def _triangle_empty_specials(tri, specials_index, ids_in_tri):
    x0, y0, x1, y1 = _tri_bbox(tri)
    A, B, C = tri
    for sid, p in specials_index.query_bbox(x0, y0, x1, y1):
        if sid in ids_in_tri:
            continue
        if _point_in_tri_strict(p, A, B, C):
            return False
    return True


def _build_triangles_left_to_right(specials, boxes, allow_overflight=False, seed=0):
    pts = [(sp["id"], (float(sp["px"]), float(sp["py"])), sp.get("bi", None)) for sp in specials]
    pts.sort(key=lambda it: (it[1][0], it[1][1], str(it[0])))

    pos = {sid: p for (sid, p, _bi) in pts}
    bi = {sid: _bi for (sid, _p, _bi) in pts if _bi is not None}
    ids = [sid for (sid, _p, _bi) in pts]
    idx_of = {sid: i for i, sid in enumerate(ids)}

    specials_index = _GridIndex(cell_size=64.0)
    for sid in ids:
        specials_index.insert(sid, pos[sid])

    seg_index = _SegmentIndex(cell_size=max(24.0, MIN_EDGE_CLEAR_PX))
    tri_index = _TriIndex(cell_size=96.0)

    edges = set()
    adj = defaultdict(set)
    triangles = []

    all_pts_for_clear = [(sid, pos[sid]) for sid in ids]

    def ekey(a, b):
        sa, sb = str(a), str(b)
        return (a, b) if sa < sb else (b, a)

    def try_add_edge(a, b):
        if a == b:
            return False
        ek = ekey(a, b)
        if ek in edges:
            return True

        P = pos[a]
        Q = pos[b]
        ignore = set()
        if a in bi:
            ignore.add(bi[a])
        if b in bi:
            ignore.add(bi[b])

        ok = _segment_ok(
            P, Q, a, b, seg_index,
            boxes=[] if allow_overflight else boxes,
            ignore_boxes=ignore,
            min_pt_clear=MIN_PT_CLEAR_PX,
            pts=all_pts_for_clear,
            min_seg_clear=MIN_EDGE_CLEAR_PX,
            tri_index=tri_index,
        )
        if not ok:
            return False

        seg_index.add(("SS", a, b), P, Q, a, b)
        edges.add(ek)
        adj[a].add(b)
        adj[b].add(a)
        return True

    def try_add_triangle(A, B, C):
        tri = (pos[A], pos[B], pos[C])
        if _triangle_area2(*tri) < TRI_MIN_AREA_PX2:
            return False
        if not _triangle_empty_specials(tri, specials_index, {A, B, C}):
            return False

        x0, y0, x1, y1 = _tri_bbox(tri)
        for tid in tri_index.candidates_bbox(x0, y0, x1, y1):
            if _triangles_overlap_strict(tri, tri_index.get(tid)):
                return False

        tid = ("T", A, B, C, len(triangles))
        tri_index.add(tid, tri)
        triangles.append((A, B, C, tri, _triangle_area2(*tri)))
        return True

    for i, A in enumerate(ids):
        PA = pos[A]
        cand = []
        for j in range(i + 1, len(ids)):
            B = ids[j]
            PB = pos[B]
            cand.append((_dist(PA, PB), B))
        cand.sort(key=lambda t: t[0])
        cand = cand[:KNN_LOCAL]

        out_deg = 0
        for _d, B in cand:
            if out_deg >= MAX_OUT_EDGES:
                break
            if try_add_edge(A, B):
                out_deg += 1

        neigh = [v for v in adj[A] if idx_of[v] > i]
        if len(neigh) < 2:
            continue
        neigh.sort(key=lambda v: math.atan2(pos[v][1] - PA[1], pos[v][0] - PA[0]))

        for k in range(len(neigh) - 1):
            B = neigh[k]
            C = neigh[k + 1]
            if not try_add_edge(B, C):
                continue
            try_add_triangle(A, B, C)

    min_needed = int(math.ceil(len(ids) / float(max(1, LOGICAL_SL_CAP))))
    if min_needed < 1:
        min_needed = 1

    if len(triangles) < min_needed and len(ids) >= 3:
        rnd = random.Random(seed)

        def triple_key(a, b, c):
            ss = sorted([str(a), str(b), str(c)])
            return tuple(ss)

        seen_tris = set()
        for (a, b, c, _t, _a2) in triangles:
            seen_tris.add(triple_key(a, b, c))

        for K in (8, 12, 16, min(KNN_LOCAL, len(ids))):
            if len(triangles) >= min_needed:
                break
            for i, A in enumerate(ids):
                if len(triangles) >= min_needed:
                    break
                PA = pos[A]

                rest = ids[i + 1:]
                if len(rest) < 2:
                    continue

                candB = [(_dist(PA, pos[B]), B) for B in rest]
                candB.sort(key=lambda t: t[0])
                Bs = [B for (_d, B) in candB[:K]]

                triples = []
                for bi_i in range(len(Bs)):
                    for bi_j in range(bi_i + 1, len(Bs)):
                        B = Bs[bi_i]
                        C = Bs[bi_j]
                        per = _dist(pos[A], pos[B]) + _dist(pos[A], pos[C]) + _dist(pos[B], pos[C])
                        triples.append((per, rnd.random(), B, C))
                triples.sort(key=lambda t: (t[0], t[1]))

                for _per, _rr, B, C in triples:
                    if len(triangles) >= min_needed:
                        break
                    tk = triple_key(A, B, C)
                    if tk in seen_tris:
                        continue

                    if not try_add_edge(A, B):
                        continue
                    if not try_add_edge(A, C):
                        continue
                    if not try_add_edge(B, C):
                        continue

                    if try_add_triangle(A, B, C):
                        seen_tris.add(tk)

    triangles.sort(key=lambda it: it[4])
    forbid_tris = [it[3] for it in triangles]
    return triangles, forbid_tris


def _ensure_triangles_and_logicals(specials, pos_skel, boxes, allow_overflight=False, seed=0):
    global _CACHE_KEY, _TRIANGLES, _FORBID_TRIS, _LOGICAL_SIDS, _LOGICAL_POS, _LOGICAL_TRI
    global _TRI_KEY_TO_LOGICAL, _SPECIAL_TO_LOGICALS

    key = _cache_key(specials, boxes, allow_overflight, seed)
    if _CACHE_KEY == key and _LOGICAL_SIDS is not None and _TRIANGLES is not None:
        return

    _CACHE_KEY = key

    tris, forbid = _build_triangles_left_to_right(
        specials,
        boxes=boxes,
        allow_overflight=bool(allow_overflight),
        seed=seed,
    )

    _TRIANGLES = tris
    _FORBID_TRIS = forbid

    specials_pts = [(float(sp["px"]), float(sp["py"])) for sp in specials]
    rnd = random.Random(seed)

    max_existing = max([k for k in pos_skel.keys() if isinstance(k, int)] + [-1])
    next_id = max_existing + 1

    _LOGICAL_SIDS = []
    _LOGICAL_POS = {}
    _LOGICAL_TRI = {}
    _TRI_KEY_TO_LOGICAL = {}
    _SPECIAL_TO_LOGICALS = defaultdict(list)

    def tri_key(A, B, C):
        ss = sorted([str(A), str(B), str(C)])
        return tuple(ss)

    for (A_id, B_id, C_id, tri, _a2) in (_TRIANGLES or []):
        c = _centroid(tri)
        if boxes and point_in_any_box(c, boxes, ignore_indices=set()):
            c = _pick_centroid_or_near(tri, boxes, specials_pts, rnd, tries=24)

        sid = next_id
        next_id += 1

        _LOGICAL_SIDS.append(sid)
        _LOGICAL_POS[sid] = (float(c[0]), float(c[1]))
        _LOGICAL_TRI[sid] = tri
        pos_skel[sid] = _LOGICAL_POS[sid]

        tk = tri_key(A_id, B_id, C_id)
        _TRI_KEY_TO_LOGICAL[tk] = sid

        _SPECIAL_TO_LOGICALS[A_id].append(sid)
        _SPECIAL_TO_LOGICALS[B_id].append(sid)
        _SPECIAL_TO_LOGICALS[C_id].append(sid)


# -----------------------------
# Debug (triangles)
# -----------------------------

def save_triangles_debug_png(out_path, map_bgr, boxes, specials):
    img = map_bgr.copy()

    for _, x, y, w, h in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (70, 70, 70), 1)

    for sp in specials:
        cx, cy = float(sp["px"]), float(sp["py"])
        cv2.circle(img, (int(cx), int(cy)), 7, (0, 0, 255), -1)

    if _TRIANGLES:
        for (_a, _b, _c, (A, B, C), _a2) in _TRIANGLES:
            cv2.line(img, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (255, 255, 0), 2)
            cv2.line(img, (int(B[0]), int(B[1])), (int(C[0]), int(C[1])), (255, 255, 0), 2)
            cv2.line(img, (int(C[0]), int(C[1])), (int(A[0]), int(A[1])), (255, 255, 0), 2)

    if _LOGICAL_POS:
        for _sid, p in _LOGICAL_POS.items():
            cv2.circle(img, (int(p[0]), int(p[1])), 6, (160, 160, 160), -1)

    cv2.imwrite(out_path, img)


def debug_build_and_save_triangles_png(out_path, map_bgr, boxes, specials, pos_skel, allow_overflight=False, seed=0):
    _ensure_logical_globals()
    _ensure_triangles_and_logicals(specials, pos_skel, boxes, allow_overflight=allow_overflight, seed=seed)
    save_triangles_debug_png(out_path, map_bgr, boxes, specials)
    return list(_TRIANGLES or []), list(_LOGICAL_SIDS or [])


# -----------------------------
# Visibility candidates
# -----------------------------

def build_visibility_candidates_for_specials(
    specials,
    candidate_logical_sids,
    pos_skel,
    boxes,
    max_candidates_per_special=40,
    allow_overflight=False,
):
    global _GLOBAL_BOXES, _GLOBAL_ALLOW_OVERFLIGHT
    _GLOBAL_BOXES = boxes
    _GLOBAL_ALLOW_OVERFLIGHT = bool(allow_overflight)

    _ensure_logical_globals()
    _ensure_triangles_and_logicals(specials, pos_skel, boxes, allow_overflight=allow_overflight, seed=0)

    logical_sids = list(_LOGICAL_SIDS) if _LOGICAL_SIDS else []
    allowed = {}

    for sp in specials:
        s_id = sp["id"]
        S = (float(sp["px"]), float(sp["py"]))

        lst = list(_SPECIAL_TO_LOGICALS.get(s_id, [])) if _SPECIAL_TO_LOGICALS else []
        lst2 = []
        for sid in lst:
            P = (float(pos_skel[sid][0]), float(pos_skel[sid][1]))
            if boxes and point_in_any_box(P, boxes, ignore_indices=set()):
                continue
            if (not allow_overflight) and boxes:
                bi = sp.get("bi", None)
                ignore = {bi} if bi is not None else set()
                if seg_hits_any_box(S, P, boxes, ignore_indices=ignore):
                    continue
            lst2.append(sid)

        if not lst2:
            if logical_sids:
                sid0 = min(logical_sids, key=lambda sid: _dist(S, pos_skel[sid]))
                allowed[s_id] = [sid0]
            else:
                allowed[s_id] = []
            continue

        k = int(max(1, max_candidates_per_special))
        lst2 = sorted(lst2, key=lambda sid: _dist(S, pos_skel[sid]))
        allowed[s_id] = lst2[:k]

    return allowed


# -----------------------------
# Logical selection (minimize #logical <= max_nodes)
# -----------------------------

def greedy_select_logical_nodes_randomized(special_ids, allowed_by_special, max_nodes, seed_local=0):
    all_cands = []
    seen = set()
    for s in special_ids:
        for c in allowed_by_special.get(s, []):
            if c not in seen:
                seen.add(c)
                all_cands.append(c)

    all_cands = sorted(all_cands)
    if not all_cands:
        return None

    if max_nodes is None or max_nodes <= 0:
        return all_cands
    if len(all_cands) > int(max_nodes):
        return None
    return all_cands


# -----------------------------
# Special assignment (triangle -> logical center)
# -----------------------------

def assign_special_edges_balanced(
    specials,
    logical_sids_selected,
    allowed_by_special_sid,
    max_deg_total=LOGICAL_MAX_DEG,
    reserve_ll=LOGICAL_LL_TARGET,
    seed=42,
    tries=80,
):
    global _LAST_STATE
    _ensure_logical_globals()

    sp_pos = {sp["id"]: (float(sp["px"]), float(sp["py"])) for sp in specials}
    special_ids = [sp["id"] for sp in specials]

    boxes = _GLOBAL_BOXES if _GLOBAL_BOXES is not None else []
    allow_overflight = bool(_GLOBAL_ALLOW_OVERFLIGHT)

    if _LAST_STATE is None:
        _LAST_STATE = {}
    _LAST_STATE["specials"] = [dict(sp) for sp in specials]

    segi = _SegmentIndex(cell_size=max(24.0, MIN_EDGE_CLEAR_PX))
    assignment = {s: [] for s in special_ids}

    pts = [(sid, _px(sid)) for sid in (logical_sids_selected or [])] + [(s, sp_pos[s]) for s in special_ids]

    for lsid in (logical_sids_selected or []):
        tri = _LOGICAL_TRI.get(lsid, None)
        if tri is None:
            continue

        A, B, C = tri
        verts = [A, B, C]

        vert_specials = []
        for vtx in verts:
            best = None
            best_d = 1e18
            for sid_s in special_ids:
                d = _dist(vtx, sp_pos[sid_s])
                if d < best_d:
                    best_d = d
                    best = sid_s
            if best is not None:
                vert_specials.append(best)

        for sid_s in vert_specials:
            P = sp_pos[sid_s]
            Q = _px(lsid)

            ignore = set()
            if (not allow_overflight) and boxes:
                bi = None
                for sp in specials:
                    if sp["id"] == sid_s:
                        bi = sp.get("bi", None)
                        break
                if bi is not None:
                    ignore = {bi}

            ok = _segment_ok(
                P, Q, sid_s, lsid,
                seg_index=segi,
                boxes=[] if allow_overflight else boxes,
                ignore_boxes=ignore,
                min_pt_clear=MIN_PT_CLEAR_PX,
                pts=pts,
                min_seg_clear=MIN_EDGE_CLEAR_PX,
                tri_index=None,
            )
            if not ok:
                continue

            segi.add(("SL", sid_s, lsid), P, Q, sid_s, lsid)
            assignment[sid_s].append(lsid)

    assignment = {k: tuple(v) for k, v in assignment.items()}

    _LAST_STATE["ll_edges"] = []
    _LAST_STATE["seg_index"] = segi
    _LAST_STATE["assignment"] = dict(assignment)
    _LAST_STATE["logical_sids"] = list(logical_sids_selected or [])

    deg = [0] * len(list(logical_sids_selected or []))
    return assignment, deg


# -----------------------------
# Logical-logical edges (add all feasible, no crossings, clearance)
# -----------------------------

def build_ll_tree_from_caps(ll_caps=None, seed=0):
    global _LAST_STATE
    _ensure_logical_globals()

    sel = []
    if _LAST_STATE and "logical_sids" in _LAST_STATE:
        sel = list(_LAST_STATE["logical_sids"] or [])
    sids = sel if sel else list(_LOGICAL_SIDS or [])

    if len(sids) < 2:
        _LAST_STATE = {
            "ll_edges": [],
            "seg_index": _SegmentIndex(cell_size=max(24.0, MIN_EDGE_CLEAR_PX)),
            "logical_sids": list(sids),
        }
        return []

    boxes = _GLOBAL_BOXES if _GLOBAL_BOXES is not None else []
    allow_overflight = bool(_GLOBAL_ALLOW_OVERFLIGHT)

    segi = _LAST_STATE["seg_index"] if (_LAST_STATE and "seg_index" in _LAST_STATE) else _SegmentIndex(cell_size=max(24.0, MIN_EDGE_CLEAR_PX))

    pts = [(sid, _px(sid)) for sid in sids]
    if _LAST_STATE and "specials" in _LAST_STATE:
        for sp in _LAST_STATE["specials"]:
            pts.append((sp["id"], (float(sp["px"]), float(sp["py"]))))

    pairs = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            u = sids[i]
            v = sids[j]
            pairs.append((_dist(_px(u), _px(v)), u, v))
    pairs.sort(key=lambda t: (t[0], str(t[1]), str(t[2])))

    edges = []
    for _d, u, v in pairs:
        a, b = _px(u), _px(v)
        ok = _segment_ok(
            a, b, u, v,
            seg_index=segi,
            boxes=[] if allow_overflight else boxes,
            ignore_boxes=set(),
            min_pt_clear=MIN_PT_CLEAR_PX,
            pts=pts,
            min_seg_clear=MIN_EDGE_CLEAR_PX,
            tri_index=None,
        )
        if not ok:
            continue
        segi.add(("LL", u, v), a, b, u, v)
        edges.append((u, v))

    _LAST_STATE["ll_edges"] = list(edges)
    _LAST_STATE["seg_index"] = segi
    _LAST_STATE["logical_sids"] = list(sids)

    idx = {sid: i for i, sid in enumerate(sids)}
    return [(idx[u], idx[v]) for (u, v) in edges if u in idx and v in idx]


# -----------------------------
# Optional refine hook (clamp only)
# -----------------------------

def refine_logical_positions_only(
    specials,
    selected_sids,
    assignment_by_special,
    ll_tree_edges,
    boxes,
    W,
    H,
    resolution_m_per_px,
    seed=42,
):
    _ensure_logical_globals()
    out = {}
    for sid in selected_sids:
        p = _px(sid)
        p = (_clamp(p[0], 1.0, float(W - 2)), _clamp(p[1], 1.0, float(H - 2)))
        out[sid] = (float(p[0]), float(p[1]))
        _GLOBAL_POS_SKEL[sid] = out[sid]
        if _LOGICAL_POS and sid in _LOGICAL_POS:
            _LOGICAL_POS[sid] = out[sid]
    return out


# -----------------------------
# Road skeleton (unchanged)
# -----------------------------

def _zs_thin(bin01):
    img = (bin01 > 0).astype(np.uint8).copy()
    changed = True

    def neighbors8(a):
        p2 = np.roll(a, -1, axis=0)
        p3 = np.roll(np.roll(a, -1, axis=0), -1, axis=1)
        p4 = np.roll(a, -1, axis=1)
        p5 = np.roll(np.roll(a, 1, axis=0), -1, axis=1)
        p6 = np.roll(a, 1, axis=0)
        p7 = np.roll(np.roll(a, 1, axis=0), 1, axis=1)
        p8 = np.roll(a, 1, axis=1)
        p9 = np.roll(np.roll(a, -1, axis=0), 1, axis=1)
        return p2, p3, p4, p5, p6, p7, p8, p9

    while changed:
        changed = False
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors8(img)
        A = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
            ((p4 == 0) & (p5 == 1)).astype(np.uint8) + ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
            ((p6 == 0) & (p7 == 1)).astype(np.uint8) + ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
            ((p8 == 0) & (p9 == 1)).astype(np.uint8) + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        m1 = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        if np.any(m1):
            img[m1] = 0
            changed = True

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors8(img)
        A = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
            ((p4 == 0) & (p5 == 1)).astype(np.uint8) + ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
            ((p6 == 0) & (p7 == 1)).astype(np.uint8) + ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
            ((p8 == 0) & (p9 == 1)).astype(np.uint8) + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
        B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

        m2 = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        if np.any(m2):
            img[m2] = 0
            changed = True

        img[0, :] = 0
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0

    return img


def skeletonize_roads(mask_roads_255):
    roads = (mask_roads_255 > 0).astype(np.uint8) * 255
    roads = cv2.morphologyEx(roads, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    roads = cv2.morphologyEx(roads, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bin01 = (roads > 0).astype(np.uint8)
    return _zs_thin(bin01)


def _nbrs8_coords(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def classify_skel_node(skel01, x, y):
    nbrs = []
    for nx, ny in _nbrs8_coords(x, y):
        if skel01[ny, nx]:
            nbrs.append((nx, ny))
    deg = len(nbrs)
    if deg != 2:
        return True
    (x1, y1), (x2, y2) = nbrs[0], nbrs[1]
    vx1, vy1 = x1 - x, y1 - y
    vx2, vy2 = x2 - x, y2 - y
    collinear = (vx1 == -vx2) and (vy1 == -vy2)
    return not collinear


def build_skeleton_graph(skel01):
    H, W = skel01.shape[:2]
    nodes = {}
    for y in range(1, H - 1):
        row = skel01[y]
        if not np.any(row):
            continue
        xs = np.where(row > 0)[0]
        for x in xs:
            if classify_skel_node(skel01, x, y):
                nodes[(x, y)] = None

    node_list = list(nodes.keys())
    node_id = {p: i for i, p in enumerate(node_list)}
    pos = {node_id[p]: p for p in node_list}
    adj = {node_id[p]: [] for p in node_list}

    ys, xs = np.where(skel01 > 0)
    skel_set = set(zip(xs.tolist(), ys.tolist()))

    def step_cost(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(2.0) if (dx != 0 and dy != 0) else 1.0

    visited_dir = set()
    for p in node_list:
        pid = node_id[p]
        for q in _nbrs8_coords(p[0], p[1]):
            if q not in skel_set:
                continue
            key = (p, q)
            if key in visited_dir:
                continue

            prev = p
            curr = q
            length = step_cost(prev, curr)

            while curr not in nodes:
                nbs = [r for r in _nbrs8_coords(curr[0], curr[1]) if r in skel_set and r != prev]
                if len(nbs) == 0:
                    break
                nxt = nbs[0]
                prev, curr = curr, nxt
                length += step_cost(prev, curr)

            if curr in nodes:
                qid = node_id[curr]
                visited_dir.add((p, q))
                adj[pid].append((qid, length))
                adj[qid].append((pid, length))

    return adj, pos


# -----------------------------
# Debug image (final graph)
# -----------------------------

def save_graph_debug_png(
    out_path,
    map_bgr,
    boxes,
    special_nodes,
    logical_sids_selected,
    pos_skel,
    assignment_sid_pairs,
    ll_tree_local=None,
):
    try:
        tri_path = out_path.replace(".png", "_triangles.png")
        save_triangles_debug_png(tri_path, map_bgr, boxes, special_nodes)
    except Exception:
        pass

    img = map_bgr.copy()

    for _, x, y, w, h in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (90, 90, 90), 1)

    CYAN = (255, 255, 0)

    if ll_tree_local is not None:
        logical_sids = list(logical_sids_selected)
        for (i, j) in ll_tree_local:
            if i < 0 or j < 0 or i >= len(logical_sids) or j >= len(logical_sids):
                continue
            a = pos_skel[logical_sids[i]]
            b = pos_skel[logical_sids[j]]
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), CYAN, 2)

    for sp in special_nodes:
        s_id = sp["id"]
        cx, cy = float(sp["px"]), float(sp["py"])
        if s_id in assignment_sid_pairs:
            for sid_k in assignment_sid_pairs[s_id]:
                ax, ay = pos_skel[sid_k]
                cv2.line(img, (int(cx), int(cy)), (int(ax), int(ay)), CYAN, 2)

    for sid in logical_sids_selected:
        x, y = pos_skel[sid]
        cv2.circle(img, (int(x), int(y)), 7, (160, 160, 160), -1)

    role_bgr = {
        "vertiport": (0, 0, 255),
        "supplier": (0, 255, 0),
        "client": (0, 165, 255),
        "charging": (255, 80, 0),
    }
    for sp in special_nodes:
        cx, cy = float(sp["px"]), float(sp["py"])
        role = sp.get("role", "")
        cv2.circle(img, (int(cx), int(cy)), 9, role_bgr.get(role, (0, 0, 255)), -1)

    cv2.imwrite(out_path, img)

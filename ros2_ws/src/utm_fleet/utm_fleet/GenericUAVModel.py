# GenericUAVModel.py

import networkx as nx
from utm_graph import load_graph_data
from ultrades.automata import *


class GenericUAVModel:
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
        self._build_motion_automaton()
        self._build_edge_automata()
        self._build_modes_automaton()
        self._build_support_automata()
        self._build_workflow_spec()
        self._build_map_spec()
        self._build_battery_motion_spec()
        self._build_location_specs()
        self._build_task_complete_specs()
        self._build_charge_exit_spec()

        self.supervisor_mono = None
        self.supervisor_mono = self.compute_monolithic_supervisor()

    def __getstate__(self):
        # copy-by-value state for multiprocessing spawn
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
            rel_uv = f"edge_release::{u}::{v}"
            take_vu = f"edge_take::{v}::{u}"
            rel_vu = f"edge_release::{v}::{u}"

            for nm, ctrl in (
                (take_uv, True), (rel_uv, False),
                (take_vu, True), (rel_vu, False),
            ):
                if nm not in E:
                    E[nm] = event(nm, controllable=ctrl)

            a, b = (u, v) if u <= v else (v, u)
            key = ((a, b), k)

            if key not in self.edge_bundle:
                take_ab = E[f"edge_take::{a}::{b}"]
                take_ba = E[f"edge_take::{b}::{a}"]
                rel_ab = E[f"edge_release::{a}::{b}"]
                rel_ba = E[f"edge_release::{b}::{a}"]
                self.edge_bundle[key] = (take_ab, take_ba, rel_ab, rel_ba)

        for n in self.G.nodes():
            n = str(n)
            knd = self._kind(n)
            if knd == "SUPPLIER":
                ws = f"work_start::{n}::SUPPLIER"
                we = f"work_end::{n}::SUPPLIER"
                if ws not in E:
                    E[ws] = event(ws, controllable=True)
                if we not in E:
                    E[we] = event(we, controllable=False)
            elif knd == "CLIENT":
                ws = f"work_start::{n}::CLIENT"
                we = f"work_end::{n}::CLIENT"
                if ws not in E:
                    E[ws] = event(ws, controllable=True)
                if we not in E:
                    E[we] = event(we, controllable=False)

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) == "STATION":
                cs = f"charge_start::{n}"
                ce = f"charge_end::{n}"
                if cs not in E:
                    E[cs] = event(cs, controllable=True)
                if ce not in E:
                    E[ce] = event(ce, controllable=False)

        if "battery_low" not in E:
            E["battery_low"] = event("battery_low", controllable=False)

        for nm, ctrl in (
            ("task_accept", True),
            ("task_reject", True),
            ("task_done", True),
            ("heartbeat", True),
        ):
            if nm not in E:
                E[nm] = event(nm, controllable=ctrl)

    def _build_motion_automaton(self):
        Idle = state("IDLE", marked=True)
        Moving = state("MOVING")
        trs = []

        for (_ab, _k), (take_ab, take_ba, rel_ab, rel_ba) in self.edge_bundle.items():
            trs.extend([
                (Idle, take_ab, Moving),
                (Moving, rel_ab, Idle),
                (Idle, take_ba, Moving),
                (Moving, rel_ba, Idle),
            ])

        A = dfa(trs, Idle, "motion")
        self.automata["motion"] = A
        self.specs.append(A)

    def _build_edge_automata(self):
        for (ab, k), (take_ab, take_ba, rel_ab, rel_ba) in self.edge_bundle.items():
            a, b = ab

            Free1 = state(f"EDGE_FREE::{a}::{b}", marked=True)
            Occ1 = state(f"EDGE_OCC::{a}::{b}")
            A1 = dfa([(Free1, take_ab, Occ1), (Occ1, rel_ab, Free1)], Free1, f"edge::{a}::{b}::{k}")
            self.plants.append(A1)
            self.automata[f"edge::{a}::{b}::{k}"] = A1

            Free2 = state(f"EDGE_FREE::{b}::{a}", marked=True)
            Occ2 = state(f"EDGE_OCC::{b}::{a}")
            A2 = dfa([(Free2, take_ba, Occ2), (Occ2, rel_ba, Free2)], Free2, f"edge::{b}::{a}::{k}")
            self.plants.append(A2)
            self.automata[f"edge::{b}::{a}::{k}"] = A2

    def _build_modes_automaton(self):
        Normal = state("MODE_NORMAL", marked=True)
        trs = []

        for n in self.G.nodes():
            n = str(n)
            knd = self._kind(n)
            if knd == "SUPPLIER":
                Working = state(f"MODE_WORK_SUPPLIER::{n}")
                ws = self.ev(f"work_start::{n}::SUPPLIER")
                we = self.ev(f"work_end::{n}::SUPPLIER")
                trs.append((Normal, ws, Working))
                trs.append((Working, we, Normal))
            elif knd == "CLIENT":
                Working = state(f"MODE_WORK_CLIENT::{n}")
                ws = self.ev(f"work_start::{n}::CLIENT")
                we = self.ev(f"work_end::{n}::CLIENT")
                trs.append((Normal, ws, Working))
                trs.append((Working, we, Normal))

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) == "STATION":
                Charging = state(f"MODE_CHARGE::{n}")
                cs = self.ev(f"charge_start::{n}")
                ce = self.ev(f"charge_end::{n}")
                trs.append((Normal, cs, Charging))
                trs.append((Charging, ce, Normal))

        A = dfa(trs, Normal, "modes")
        self.automata["modes"] = A
        self.plants.append(A)

    def _build_support_automata(self):
        B = state("BATTERY_SIGNAL", marked=True)
        Ab = dfa([(B, self.ev("battery_low"), B)], B, "battery_low_signal")
        self.automata["battery_low_signal"] = Ab
        self.plants.append(Ab)

        S = state("COMMS_OK", marked=True)
        self.automata["comms"] = dfa([
            (S, self.ev("task_accept"), S),
            (S, self.ev("task_reject"), S),
            (S, self.ev("task_done"), S),
        ], S, "comms")

        H = state("ALIVE", marked=True)
        self.automata["heartbeat"] = dfa([(H, self.ev("heartbeat"), H)], H, "heartbeat")

    def _build_map_spec(self):
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
        for (ab, _k), (take_ab, take_ba, _rel_ab, _rel_ba) in self.edge_bundle.items():
            a, b = ab
            sa = self.node_states[a]
            sb = self.node_states[b]
            trs.append((sa, take_ab, sb))
            trs.append((sb, take_ba, sa))

        A = dfa(trs, initial, "map")
        self.automata["map"] = A
        self.specs.append(A)

    def _build_battery_motion_spec(self):
        Ok = state("BAT_OK", marked=True)
        Low = state("BAT_LOW")
        low_ev = self.ev("battery_low")

        trs = [(Ok, low_ev, Low), (Low, low_ev, Low)]

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) == "STATION":
                cs = self.ev(f"charge_start::{n}")
                trs.append((Low, cs, Ok))
                trs.append((Ok, cs, Ok))

        A = dfa(trs, Ok, "battery_policy")
        self.automata["battery_policy"] = accessible(A)
        self.specs.append(self.automata["battery_policy"])

    def _build_location_specs(self):
        for n in self.G.nodes():
            n = str(n)
            knd = self._kind(n)
            if knd not in ("SUPPLIER", "CLIENT", "STATION"):
                continue

            Inside = state(f"INSIDE::{n}")
            Outside = state(f"OUTSIDE::{n}", marked=True)
            trs = []

            for x in self.G.neighbors(n):
                x = str(x)
                trs.append((Outside, self.ev(f"edge_release::{x}::{n}"), Inside))

            for x in self.G.neighbors(n):
                x = str(x)
                trs.append((Inside, self.ev(f"edge_take::{n}::{x}"), Outside))

            if knd == "SUPPLIER":
                trs.append((Inside, self.ev(f"work_start::{n}::SUPPLIER"), Inside))
            elif knd == "CLIENT":
                trs.append((Inside, self.ev(f"work_start::{n}::CLIENT"), Inside))
            elif knd == "STATION":
                trs.append((Inside, self.ev(f"charge_start::{n}"), Inside))

            A = dfa(trs, Outside, f"loc::{n}")
            self.automata[f"loc::{n}"] = A
            self.specs.append(A)

    def _build_workflow_spec(self):
        Pick = state("WF_PICK")
        Place = state("WF_PLACE")
        Base = state("WF_BASE", marked=True)

        trs = []

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) == "SUPPLIER":
                trs.append((Base, self.ev(f"work_start::{n}::SUPPLIER"), Pick))

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) == "CLIENT":
                trs.append((Pick, self.ev(f"work_start::{n}::CLIENT"), Place))

        for b in [str(n) for n in self.G.nodes() if self._kind(n) == "VERTIPORT"]:
            for u in set(str(x) for x in self.G.predecessors(b)):
                evn = f"edge_take::{u}::{b}"
                if evn in self.events:
                    e = self.ev(evn)
                    trs.append((Place, e, Base))
                    trs.append((Base, e, Base))

        A = dfa(trs, Base, "workflow")
        self.automata["workflow"] = A
        self.specs.append(A)

    def _build_task_complete_specs(self):
        for n in self.G.nodes():
            n = str(n)
            knd = self._kind(n)
            if knd not in ("SUPPLIER", "CLIENT", "STATION", "VERTIPORT"):
                continue

            CanExit = state(f"CAN_EXIT::{n}", marked=True)
            Busy = state(f"BUSY::{n}")
            trs = []

            for x in self.G.neighbors(n):
                x = str(x)
                evn = f"edge_take::{n}::{x}"
                if evn in self.events:
                    trs.append((CanExit, self.ev(evn), CanExit))

            begin = None
            end = None
            if knd == "SUPPLIER":
                begin = self.ev(f"work_start::{n}::SUPPLIER")
                end = self.ev(f"work_end::{n}::SUPPLIER")
            elif knd == "CLIENT":
                begin = self.ev(f"work_start::{n}::CLIENT")
                end = self.ev(f"work_end::{n}::CLIENT")
            elif knd == "STATION":
                begin = self.ev(f"charge_start::{n}")
                end = self.ev(f"charge_end::{n}")

            if begin is not None and end is not None:
                trs.append((CanExit, begin, Busy))
                trs.append((Busy, begin, Busy))
                trs.append((Busy, end, CanExit))

            A = dfa(trs, CanExit, f"task_complete::{n}")
            self.automata[f"task_complete::{n}"] = A
            self.specs.append(A)

    def _build_charge_exit_spec(self):
        Ready = state("CHARGE_READY", marked=True)
        MustExit = state("MUST_EXIT_AFTER_CHARGE")
        trs = []

        for n in self.G.nodes():
            n = str(n)
            if self._kind(n) != "STATION":
                continue

            cs = self.ev(f"charge_start::{n}")
            trs.append((Ready, cs, MustExit))

            for x in self.G.neighbors(n):
                x = str(x)
                evn = f"edge_take::{n}::{x}"
                if evn in self.events:
                    e = self.ev(evn)
                    trs.append((MustExit, e, Ready))
                    trs.append((Ready, e, Ready))

        A = dfa(trs, Ready, "charge_exit")
        self.automata["charge_exit"] = A
        self.specs.append(A)

    def compute_monolithic_supervisor(self, force=False):
        if self.supervisor_mono is None or force:
            self.supervisor_mono = monolithic_supervisor(self.plants, self.specs)
        return self.supervisor_mono
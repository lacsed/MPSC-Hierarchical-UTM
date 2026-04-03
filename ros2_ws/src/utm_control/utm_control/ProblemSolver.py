#!/usr/bin/env python3
from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from ProblemParameters import ProblemParameters


class ProblemSolver:
    """
    Standalone solver for the online UTM MILP.

    This implementation follows the online UTM formulation with:
        - assignment variable y_k
        - stage-indexed routing variables x_ijhk
        - terminal vertiport choice e_dk
        - start times tau_k
        - arc-entry times theta_ijhk
        - node-arrival times u_ihk
        - cumulative energy z_ijhk
        - makespan L
        - binary linearization auxiliaries lambda_ihkq and xi_ijhkq
    """

    def __init__(
        self,
        params: ProblemParameters,
        debug: bool = False,
    ):
        self.p = params
        self.debug = bool(debug)

        self.model = gp.Model(self.p.model_name)
        self.model.setParam("OutputFlag", 1 if self.debug else 0)

        # --------------------------------------------------------------
        # Basic sets
        # --------------------------------------------------------------
        self.K = list(self.p.K)
        self.V = list(self.p.V)
        self.D = list(self.p.D)
        self.S = list(self.p.S)
        self.W = list(self.p.W)
        self.T = list(self.p.T)
        self.T_req = list(self.p.T_req)

        self.H = list(range(1, self.p.H_max + 1))
        self.H0 = list(range(0, self.p.H_max + 1))

        self.num_nodes = self.p.num_nodes
        self.num_vehicles = self.p.num_vehicles

        # --------------------------------------------------------------
        # Valid arcs E = {(i,j): l_ij > 0 and i != j}
        # --------------------------------------------------------------
        self.E = [
            (i, j)
            for i in self.V
            for j in self.V
            if i != j and self.p.distance_matrix[i, j] > 0.0
        ]
        self.E_set = set(self.E)

        self.in_arcs = {j: [] for j in self.V}
        self.out_arcs = {i: [] for i in self.V}
        for i, j in self.E:
            self.out_arcs[i].append((i, j))
            self.in_arcs[j].append((i, j))

        # --------------------------------------------------------------
        # Forbidden sets indexing
        # --------------------------------------------------------------
        self.Q_node = {}
        for i in self.W:
            times = self.p.node_forbidden_time_lists.get(i, [])
            self.Q_node[i] = list(range(len(times)))

        self.Q_arc = {}
        for (i, j) in self.E:
            intervals = self.p.arc_forbidden_interval_lists.get((i, j), [])
            self.Q_arc[(i, j)] = list(range(len(intervals)))

        # --------------------------------------------------------------
        # Variables
        # --------------------------------------------------------------
        self._build_variables()

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------
        self._build_constraints()

        # --------------------------------------------------------------
        # Objective
        # --------------------------------------------------------------
        self._build_objective()

        self.model.update()

    # ------------------------------------------------------------------
    # Variable creation
    # ------------------------------------------------------------------
    def _build_variables(self):
        # y_k
        self.y = self.model.addVars(
            self.K,
            vtype=GRB.BINARY,
            name="y",
        )

        # x_ijhk
        self.x = self.model.addVars(
            [(i, j, h, k) for (i, j) in self.E for h in self.H for k in self.K],
            vtype=GRB.BINARY,
            name="x",
        )

        # e_dk
        self.e = self.model.addVars(
            [(d, k) for d in self.D for k in self.K],
            vtype=GRB.BINARY,
            name="e",
        )

        # tau_k
        self.tau = self.model.addVars(
            self.K,
            vtype=GRB.INTEGER,
            lb=0,
            name="tau",
        )

        # theta_ijhk
        self.theta = self.model.addVars(
            [(i, j, h, k) for (i, j) in self.E for h in self.H for k in self.K],
            vtype=GRB.INTEGER,
            lb=0,
            name="theta",
        )

        # u_ihk, with h = 0,...,Hmax
        self.u = self.model.addVars(
            [(i, h, k) for i in self.V for h in self.H0 for k in self.K],
            vtype=GRB.INTEGER,
            lb=0,
            name="u",
        )

        # z_ijhk
        self.z = self.model.addVars(
            [(i, j, h, k) for (i, j) in self.E for h in self.H for k in self.K],
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="z",
        )

        # L
        self.L = self.model.addVar(
            vtype=GRB.INTEGER,
            lb=0,
            name="L",
        )

        # lambda_ihkq for waypoint forbidden instants
        lambda_keys = []
        for i in self.W:
            for h in self.H:
                for k in self.K:
                    for q in self.Q_node[i]:
                        lambda_keys.append((i, h, k, q))

        self.lambda_var = self.model.addVars(
            lambda_keys,
            vtype=GRB.BINARY,
            name="lambda",
        )

        # xi_ijhkq for forbidden arc intervals
        xi_keys = []
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    for q in self.Q_arc[(i, j)]:
                        xi_keys.append((i, j, h, k, q))

        self.xi = self.model.addVars(
            xi_keys,
            vtype=GRB.BINARY,
            name="xi",
        )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def _build_constraints(self):
        self._add_assignment_constraints()
        self._add_route_constraints()
        self._add_start_end_constraints()
        self._add_time_constraints()
        self._add_energy_constraints()
        self._add_airspace_constraints()

    def _add_assignment_constraints(self):
        # sum_k y_k = 1
        self.model.addConstr(
            gp.quicksum(self.y[k] for k in self.K) == 1,
            name="assign_one_vehicle",
        )

        # y_k <= a_k
        for k in self.K:
            self.model.addConstr(
                self.y[k] <= float(self.p.a_k[k]),
                name=f"vehicle_available_k{k}",
            )

    def _add_route_constraints(self):
        # sum_(i,j in E) x_ijhk <= y_k
        for h in self.H:
            for k in self.K:
                self.model.addConstr(
                    gp.quicksum(self.x[i, j, h, k] for (i, j) in self.E) <= self.y[k],
                    name=f"stage_use_h{h}_k{k}",
                )

        # Flow conservation on non-vertiport nodes, for h=1,...,Hmax-1
        for j in [v for v in self.V if v not in self.D]:
            for h in self.H[:-1]:
                for k in self.K:
                    self.model.addConstr(
                        gp.quicksum(self.x[i, j, h, k] for (i, j2) in self.in_arcs[j] for j2_ in [j] if j2 == j)
                        ==
                        gp.quicksum(self.x[j, ell, h + 1, k] for (_, ell) in self.out_arcs[j]),
                        name=f"flow_j{j}_h{h}_k{k}",
                    )

        # Each required task visited exactly once
        for j in self.T_req:
            self.model.addConstr(
                gp.quicksum(
                    self.x[i, j, h, k]
                    for (i, jj) in self.in_arcs[j]
                    for h in self.H
                    for k in self.K
                    if jj == j
                ) == 1,
                name=f"visit_required_task_j{j}",
            )

    def _add_start_end_constraints(self):
        # sum_(d_k0,j) x_{d_k0,j,1,k} = y_k
        for k in self.K:
            d0 = self.p.initial_vertiport_indices[k]
            arcs_from_d0 = [(i, j) for (i, j) in self.out_arcs[d0]]
            self.model.addConstr(
                gp.quicksum(self.x[i, j, 1, k] for (i, j) in arcs_from_d0) == self.y[k],
                name=f"start_from_initial_vertiport_k{k}",
            )

        # sum_d e_dk = y_k
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.e[d, k] for d in self.D) == self.y[k],
                name=f"one_terminal_vertiport_k{k}",
            )

        # e_dk = 0 for d not in D_k
        for k in self.K:
            allowed = set(self.p.terminal_vertiport_index_sets[k])
            for d in self.D:
                if d not in allowed:
                    self.model.addConstr(
                        self.e[d, k] == 0,
                        name=f"forbid_terminal_d{d}_k{k}",
                    )

        # sum_h sum_(i,d) x_idhk = e_dk
        for d in self.D:
            for k in self.K:
                incoming_to_d = [(i, j) for (i, j) in self.in_arcs[d] if j == d]
                self.model.addConstr(
                    gp.quicksum(
                        self.x[i, d, h, k]
                        for (i, _) in incoming_to_d
                        for h in self.H
                    ) == self.e[d, k],
                    name=f"terminal_link_d{d}_k{k}",
                )

    def _add_time_constraints(self):
        # tau_k >= tau_rel * y_k
        for k in self.K:
            self.model.addConstr(
                self.tau[k] >= self.p.tau_rel * self.y[k],
                name=f"release_time_k{k}",
            )

        # tau_k >= tau_bar_k * y_k
        for k in self.K:
            self.model.addConstr(
                self.tau[k] >= float(self.p.tau_bar_k[k]) * self.y[k],
                name=f"dispatch_time_k{k}",
            )

        # u_{d_k0,0,k} = tau_k
        for k in self.K:
            d0 = self.p.initial_vertiport_indices[k]
            self.model.addConstr(
                self.u[d0, 0, k] == self.tau[k],
                name=f"initial_time_k{k}",
            )

        # If first-stage arc starts at d0, then theta_{d0,j,1,k} >= tau_k - M(1-x)
        for k in self.K:
            d0 = self.p.initial_vertiport_indices[k]
            for (i, j) in self.out_arcs[d0]:
                self.model.addConstr(
                    self.theta[i, j, 1, k] >= self.tau[k] - self.p.M * (1 - self.x[i, j, 1, k]),
                    name=f"theta_start_lb_i{i}_j{j}_k{k}",
                )

        # arrival: u_{j,h,k} >= theta_{i,j,h,k} + t_{i,j,k} - M(1-x)
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    self.model.addConstr(
                        self.u[j, h, k] >= self.theta[i, j, h, k] + float(self.p.travel_time_tensor[i, j, k])
                        - self.p.M * (1 - self.x[i, j, h, k]),
                        name=f"arrival_i{i}_j{j}_h{h}_k{k}",
                    )

        # service/departure linking for h=1,...,Hmax-1
        # theta_{j,l,h+1,k} >= u_{j,h,k} + s_{j,k} - M(1-x_{j,l,h+1,k})
        for (j, ell) in self.E:
            for h in self.H[:-1]:
                for k in self.K:
                    self.model.addConstr(
                        self.theta[j, ell, h + 1, k] >= self.u[j, h, k] + float(self.p.service_time_matrix[j, k])
                        - self.p.M * (1 - self.x[j, ell, h + 1, k]),
                        name=f"service_j{j}_ell{ell}_h{h}_k{k}",
                    )

        # precedence:
        # sum_h u_{q,h,k} >= sum_h u_{p,h,k} + s_{p,k} - M(1-y_k)
        for (p_idx, q_idx) in self.p.precedence_index_pairs:
            for k in self.K:
                self.model.addConstr(
                    gp.quicksum(self.u[q_idx, h, k] for h in self.H)
                    >=
                    gp.quicksum(self.u[p_idx, h, k] for h in self.H)
                    + float(self.p.service_time_matrix[p_idx, k])
                    - self.p.M * (1 - self.y[k]),
                    name=f"precedence_p{p_idx}_q{q_idx}_k{k}",
                )

        # L >= u_{d,h,k}
        for d in self.D:
            for h in self.H:
                for k in self.K:
                    self.model.addConstr(
                        self.L >= self.u[d, h, k],
                        name=f"makespan_d{d}_h{h}_k{k}",
                    )

    def _add_energy_constraints(self):
        # 0 <= z_ijhk <= F_k * x_ijhk
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    self.model.addConstr(
                        self.z[i, j, h, k] <= float(self.p.F_k[k]) * self.x[i, j, h, k],
                        name=f"energy_cap_i{i}_j{j}_h{h}_k{k}",
                    )

        # z_{d0,j,1,k} >= g_{d0,j,k} x_{d0,j,1,k}
        for k in self.K:
            d0 = self.p.initial_vertiport_indices[k]
            for (i, j) in self.out_arcs[d0]:
                self.model.addConstr(
                    self.z[i, j, 1, k] >= float(self.p.energy_tensor[i, j, k]) * self.x[i, j, 1, k],
                    name=f"energy_init_lb_i{i}_j{j}_k{k}",
                )

        # z_{d0,j,1,k} <= Fbar_k
        for k in self.K:
            d0 = self.p.initial_vertiport_indices[k]
            for (i, j) in self.out_arcs[d0]:
                self.model.addConstr(
                    self.z[i, j, 1, k] <= float(self.p.F_bar_k[k]),
                    name=f"energy_init_ub_i{i}_j{j}_k{k}",
                )

        # reset at stations: z_{s,j,h,k} = g_{s,j,k} x_{s,j,h,k}
        for s in self.S:
            for (i, j) in self.out_arcs[s]:
                for h in self.H:
                    for k in self.K:
                        self.model.addConstr(
                            self.z[i, j, h, k] == float(self.p.energy_tensor[i, j, k]) * self.x[i, j, h, k],
                            name=f"energy_reset_s{s}_j{j}_h{h}_k{k}",
                        )

        # propagation for i in W U T, h >= 2
        for i in self.W + self.T:
            incoming = self.in_arcs[i]
            outgoing = self.out_arcs[i]
            for h in self.H[1:]:
                for k in self.K:
                    for (ell, ii) in incoming:
                        for (ii2, j) in outgoing:
                            if ii == i and ii2 == i:
                                self.model.addConstr(
                                    self.z[i, j, h, k] >= self.z[ell, i, h - 1, k]
                                    + float(self.p.energy_tensor[i, j, k])
                                    - self.p.M * (2 - self.x[ell, i, h - 1, k] - self.x[i, j, h, k]),
                                    name=f"energy_flow_l{ell}_i{i}_j{j}_h{h}_k{k}",
                                )

    def _add_airspace_constraints(self):
        # Forbidden node instants only for waypoint nodes
        for i in self.W:
            b_list = self.p.node_forbidden_time_lists.get(i, [])
            if not b_list:
                continue

            in_neighbors = [a for (a, _) in self.in_arcs[i]]

            for h in self.H:
                for k in self.K:
                    visit_expr = gp.quicksum(self.x[a, i, h, k] for a in in_neighbors)

                    for q, b_iq in enumerate(b_list):
                        self.model.addConstr(
                            self.u[i, h, k]
                            <= b_iq - 1
                            + self.p.M * self.lambda_var[i, h, k, q]
                            + self.p.M * (1 - visit_expr),
                            name=f"waypoint_lower_i{i}_h{h}_k{k}_q{q}",
                        )

                        self.model.addConstr(
                            self.u[i, h, k]
                            >= b_iq + 1
                            - self.p.M * (1 - self.lambda_var[i, h, k, q])
                            - self.p.M * (1 - visit_expr),
                            name=f"waypoint_upper_i{i}_h{h}_k{k}_q{q}",
                        )

        # Forbidden arc intervals
        for (i, j) in self.E:
            intervals = self.p.arc_forbidden_interval_lists.get((i, j), [])
            if not intervals:
                continue

            for h in self.H:
                for k in self.K:
                    for q, (b_lo, b_hi) in enumerate(intervals):
                        self.model.addConstr(
                            self.theta[i, j, h, k] + float(self.p.travel_time_tensor[i, j, k])
                            <= b_lo + self.p.M * self.xi[i, j, h, k, q] + self.p.M * (1 - self.x[i, j, h, k]),
                            name=f"arc_lower_i{i}_j{j}_h{h}_k{k}_q{q}",
                        )

                        self.model.addConstr(
                            self.theta[i, j, h, k]
                            >= b_hi + 1
                            - self.p.M * (1 - self.xi[i, j, h, k, q])
                            - self.p.M * (1 - self.x[i, j, h, k]),
                            name=f"arc_upper_i{i}_j{j}_h{h}_k{k}_q{q}",
                        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _build_objective(self):
        dist_expr = gp.quicksum(
            float(self.p.distance_matrix[i, j]) * self.x[i, j, h, k]
            for (i, j) in self.E
            for h in self.H
            for k in self.K
        )

        util_expr = gp.quicksum(
            float(self.p.Psi_req_k[k]) * self.y[k]
            for k in self.K
        )

        self.model.setObjective(
            -float(self.p.omega_util) * util_expr
            + float(self.p.omega_time) * self.L
            + float(self.p.omega_dist) * dist_expr,
            GRB.MINIMIZE,
        )

    # ------------------------------------------------------------------
    # Solve and extract
    # ------------------------------------------------------------------
    def solve(self, time_limit: float | None = None):
        if time_limit is not None:
            self.model.setParam("TimeLimit", float(time_limit))

        self.model.optimize()

        if self.model.SolCount == 0:
            if self.model.status == GRB.INFEASIBLE:
                self.model.computeIIS()
                self.model.write("utm_online_infeasible.ilp")
            return None

        return self.extract_solution()

    def extract_solution(self) -> dict:
        tol = 1e-6

        y_sol = np.zeros(self.num_vehicles, dtype=int)
        for k in self.K:
            if self.y[k].X > 0.5:
                y_sol[k] = 1

        x_sol = np.zeros((self.num_nodes, self.num_nodes, self.p.H_max + 1, self.num_vehicles), dtype=int)
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    if self.x[i, j, h, k].X > 0.5:
                        x_sol[i, j, h, k] = 1

        e_sol = np.zeros((self.num_nodes, self.num_vehicles), dtype=int)
        for d in self.D:
            for k in self.K:
                if self.e[d, k].X > 0.5:
                    e_sol[d, k] = 1

        tau_sol = np.array([int(round(self.tau[k].X)) for k in self.K], dtype=int)

        theta_sol = np.zeros((self.num_nodes, self.num_nodes, self.p.H_max + 1, self.num_vehicles), dtype=float)
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    val = float(self.theta[i, j, h, k].X)
                    if val > tol:
                        theta_sol[i, j, h, k] = val

        u_sol = np.zeros((self.num_nodes, self.p.H_max + 1, self.num_vehicles), dtype=float)
        for i in self.V:
            for h in self.H0:
                for k in self.K:
                    val = float(self.u[i, h, k].X)
                    if val > tol:
                        u_sol[i, h, k] = val

        z_sol = np.zeros((self.num_nodes, self.num_nodes, self.p.H_max + 1, self.num_vehicles), dtype=float)
        for (i, j) in self.E:
            for h in self.H:
                for k in self.K:
                    val = float(self.z[i, j, h, k].X)
                    if val > tol:
                        z_sol[i, j, h, k] = val

        lambda_active = []
        for key in self.lambda_var.keys():
            if self.lambda_var[key].X > 0.5:
                lambda_active.append(tuple(int(v) for v in key))

        xi_active = []
        for key in self.xi.keys():
            if self.xi[key].X > 0.5:
                xi_active.append(tuple(int(v) for v in key))

        chosen_vehicle = None
        for k in self.K:
            if y_sol[k] == 1:
                chosen_vehicle = k
                break

        return {
            "status": int(self.model.status),
            "objective_value": float(self.model.ObjVal),
            "chosen_vehicle": chosen_vehicle,
            "y": y_sol,
            "x": x_sol,
            "e": e_sol,
            "tau": tau_sol,
            "theta": theta_sol,
            "u": u_sol,
            "z": z_sol,
            "L": int(round(self.L.X)),
            "lambda_active": lambda_active,
            "xi_active": xi_active,
            "required_task_labels": list(self.p.required_task_labels),
            "precedence_relations": list(self.p.precedence_relations),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def write_model(self, path: str):
        self.model.write(path)

    def compute_iis(self, path: str = "utm_online_iis.ilp"):
        self.model.computeIIS()
        self.model.write(path)
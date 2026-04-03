#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from .Graph import Graph


class ProblemParameters:
    """
    Simple parameter container for the online UTM MILP.

    Allowed node types:
        - vertiport
        - station
        - waypoint
        - task
    """

    ALLOWED_NODE_TYPES = {"vertiport", "station", "waypoint", "task"}

    def __init__(
        self,
        model_name: str,
        graph: Graph,
        num_vehicles: int,
        required_task_labels: list[str],
        precedence_relations: list[tuple[str, str]] | None,
        tau_rel: int,
        H_max: int,
        distance_matrix: np.ndarray,          # l_ij
        travel_time_tensor: np.ndarray,       # t_ijk
        energy_tensor: np.ndarray,            # g_ijk
        service_time_matrix: np.ndarray,      # s_ik
        F_k: np.ndarray,
        initial_vertiport_labels: list[str],  # d_k^0
        terminal_vertiport_sets: list[list[str]],  # D_k
        psi_matrix: np.ndarray,               # psi_ik
        omega_util: float,
        omega_time: float,
        omega_dist: float,
        a_k: np.ndarray,
        tau_bar_k: np.ndarray,
        F_bar_k: np.ndarray,
        forbidden_node_instants: dict[str, list[int]] | None,          # B_i^N
        forbidden_arc_intervals: dict[tuple[str, str], list[tuple[int, int]]] | None,  # B_ij^E
        M: float,
    ):
        self.model_name = str(model_name)
        self.graph = graph
        self.num_vehicles = int(num_vehicles)
        self.tau_rel = int(tau_rel)
        self.H_max = int(H_max)
        self.M = float(M)

        if self.num_vehicles <= 0:
            raise ValueError("num_vehicles must be > 0.")
        if self.tau_rel < 0:
            raise ValueError("tau_rel must be >= 0.")
        if self.H_max <= 0:
            raise ValueError("H_max must be > 0.")
        if self.M <= 0:
            raise ValueError("M must be > 0.")

        # ------------------------------------------------------------------
        # Graph and node sets
        # ------------------------------------------------------------------
        self.graph_nodes = self.graph.get_node_details()
        self.num_nodes = len(self.graph_nodes)

        if self.num_nodes == 0:
            raise ValueError("The graph is empty.")

        self.node_label_to_index = {}
        self.index_to_node_label = {}
        self.node_type_by_label = {}

        self.vertiport_labels = []
        self.station_labels = []
        self.waypoint_labels = []
        self.task_labels = []

        for i, node in enumerate(self.graph_nodes):
            label = str(node["label"])
            node_type = str(node["type"]).lower()

            if node_type not in self.ALLOWED_NODE_TYPES:
                raise ValueError(
                    f"Invalid node type '{node_type}' for node '{label}'. "
                    f"Allowed types: {sorted(self.ALLOWED_NODE_TYPES)}"
                )

            if label in self.node_label_to_index:
                raise ValueError(f"Duplicate node label: '{label}'")

            self.node_label_to_index[label] = i
            self.index_to_node_label[i] = label
            self.node_type_by_label[label] = node_type

            if node_type == "vertiport":
                self.vertiport_labels.append(label)
            elif node_type == "station":
                self.station_labels.append(label)
            elif node_type == "waypoint":
                self.waypoint_labels.append(label)
            elif node_type == "task":
                self.task_labels.append(label)

        if not self.vertiport_labels:
            raise ValueError("At least one vertiport is required.")

        self.vertiport_indices = [self.node_label_to_index[x] for x in self.vertiport_labels]
        self.station_indices = [self.node_label_to_index[x] for x in self.station_labels]
        self.waypoint_indices = [self.node_label_to_index[x] for x in self.waypoint_labels]
        self.task_indices = [self.node_label_to_index[x] for x in self.task_labels]

        # ------------------------------------------------------------------
        # Required tasks T_req
        # ------------------------------------------------------------------
        if not required_task_labels:
            raise ValueError("required_task_labels cannot be empty.")

        self.required_task_labels = []
        self.required_task_indices = []

        for label in required_task_labels:
            if label not in self.node_label_to_index:
                raise ValueError(f"Required task '{label}' is not in the graph.")
            if self.node_type_by_label[label] != "task":
                raise ValueError(f"Required task '{label}' is not of type 'task'.")
            if label not in self.required_task_labels:
                self.required_task_labels.append(label)
                self.required_task_indices.append(self.node_label_to_index[label])

        # ------------------------------------------------------------------
        # Precedence relations P
        # ------------------------------------------------------------------
        self.precedence_relations = precedence_relations or []
        self.precedence_index_pairs = []

        for p, q in self.precedence_relations:
            if p not in self.required_task_labels or q not in self.required_task_labels:
                raise ValueError(
                    f"Precedence pair ({p}, {q}) must use only required task nodes."
                )
            if p == q:
                raise ValueError("A precedence relation cannot have identical nodes.")
            self.precedence_index_pairs.append(
                (self.node_label_to_index[p], self.node_label_to_index[q])
            )

        # ------------------------------------------------------------------
        # Arrays and tensors
        # Shapes follow the formulation:
        #   l_ij        -> (num_nodes, num_nodes)
        #   t_ijk       -> (num_nodes, num_nodes, num_vehicles)
        #   g_ijk       -> (num_nodes, num_nodes, num_vehicles)
        #   s_ik        -> (num_nodes, num_vehicles)
        #   psi_ik      -> (num_nodes, num_vehicles)
        # ------------------------------------------------------------------
        self.distance_matrix = self._as_array(
            distance_matrix, (self.num_nodes, self.num_nodes), "distance_matrix"
        )
        self.travel_time_tensor = self._as_array(
            travel_time_tensor, (self.num_nodes, self.num_nodes, self.num_vehicles), "travel_time_tensor"
        )
        self.energy_tensor = self._as_array(
            energy_tensor, (self.num_nodes, self.num_nodes, self.num_vehicles), "energy_tensor"
        )
        self.service_time_matrix = self._as_array(
            service_time_matrix, (self.num_nodes, self.num_vehicles), "service_time_matrix"
        )
        self.psi_matrix = self._as_array(
            psi_matrix, (self.num_nodes, self.num_vehicles), "psi_matrix"
        )

        if np.any(self.distance_matrix < 0):
            raise ValueError("distance_matrix must be nonnegative.")
        if np.any(self.travel_time_tensor < 0):
            raise ValueError("travel_time_tensor must be nonnegative.")
        if np.any(self.energy_tensor < 0):
            raise ValueError("energy_tensor must be nonnegative.")
        if np.any(self.service_time_matrix < 0):
            raise ValueError("service_time_matrix must be nonnegative.")
        if np.any(self.psi_matrix < 0):
            raise ValueError("psi_matrix must be nonnegative.")

        # ------------------------------------------------------------------
        # Vehicle parameters
        # ------------------------------------------------------------------
        self.F_k = self._as_vector(F_k, self.num_vehicles, "F_k")
        self.a_k = self._as_vector(a_k, self.num_vehicles, "a_k")
        self.tau_bar_k = self._as_vector(tau_bar_k, self.num_vehicles, "tau_bar_k")
        self.F_bar_k = self._as_vector(F_bar_k, self.num_vehicles, "F_bar_k")

        if np.any(self.F_k <= 0):
            raise ValueError("F_k must be strictly positive.")
        if np.any((self.a_k != 0) & (self.a_k != 1)):
            raise ValueError("a_k must be binary.")
        if np.any(self.tau_bar_k < 0):
            raise ValueError("tau_bar_k must be nonnegative.")
        if np.any(self.F_bar_k < 0):
            raise ValueError("F_bar_k must be nonnegative.")
        if np.any(self.F_bar_k > self.F_k):
            raise ValueError("F_bar_k must satisfy F_bar_k <= F_k.")

        # ------------------------------------------------------------------
        # Initial vertiports d_k^0
        # ------------------------------------------------------------------
        if len(initial_vertiport_labels) != self.num_vehicles:
            raise ValueError("initial_vertiport_labels must have length num_vehicles.")

        self.initial_vertiport_labels = []
        self.initial_vertiport_indices = []

        for label in initial_vertiport_labels:
            if label not in self.node_label_to_index:
                raise ValueError(f"Initial vertiport '{label}' is not in the graph.")
            if self.node_type_by_label[label] != "vertiport":
                raise ValueError(f"Initial node '{label}' is not of type 'vertiport'.")
            self.initial_vertiport_labels.append(label)
            self.initial_vertiport_indices.append(self.node_label_to_index[label])

        # ------------------------------------------------------------------
        # Terminal sets D_k
        # ------------------------------------------------------------------
        if len(terminal_vertiport_sets) != self.num_vehicles:
            raise ValueError("terminal_vertiport_sets must have length num_vehicles.")

        self.terminal_vertiport_sets = []
        self.terminal_vertiport_index_sets = []

        for depot_list in terminal_vertiport_sets:
            if not depot_list:
                raise ValueError("Each terminal vertiport set must be nonempty.")

            labels = []
            indices = []

            for label in depot_list:
                if label not in self.node_label_to_index:
                    raise ValueError(f"Terminal vertiport '{label}' is not in the graph.")
                if self.node_type_by_label[label] != "vertiport":
                    raise ValueError(f"Terminal node '{label}' is not of type 'vertiport'.")
                labels.append(label)
                indices.append(self.node_label_to_index[label])

            self.terminal_vertiport_sets.append(labels)
            self.terminal_vertiport_index_sets.append(indices)

        # ------------------------------------------------------------------
        # Objective weights
        # ------------------------------------------------------------------
        self.omega_util = float(omega_util)
        self.omega_time = float(omega_time)
        self.omega_dist = float(omega_dist)

        if self.omega_util < 0 or self.omega_time < 0 or self.omega_dist < 0:
            raise ValueError("All objective weights must be nonnegative.")

        if abs(self.omega_util + self.omega_time + self.omega_dist - 1.0) > 1e-9:
            raise ValueError(
                "Weights must satisfy omega_util + omega_time + omega_dist = 1."
            )

        # ------------------------------------------------------------------
        # Forbidden node times B_i^N
        # Only defined for waypoint nodes
        # ------------------------------------------------------------------
        forbidden_node_instants = forbidden_node_instants or {}
        self.forbidden_node_instants = {}
        self.node_forbidden_time_lists = {}

        for label, times in forbidden_node_instants.items():
            if label not in self.node_label_to_index:
                raise ValueError(f"Node '{label}' in forbidden_node_instants is not in the graph.")
            if self.node_type_by_label[label] != "waypoint":
                raise ValueError(
                    f"forbidden_node_instants can only be defined for waypoint nodes. Got '{label}'."
                )

            clean_times = sorted({int(t) for t in times})
            if any(t < 0 for t in clean_times):
                raise ValueError(f"Forbidden node instants for '{label}' must be nonnegative.")

            self.forbidden_node_instants[label] = clean_times
            self.node_forbidden_time_lists[self.node_label_to_index[label]] = clean_times

        # ------------------------------------------------------------------
        # Forbidden arc intervals B_ij^E
        # ------------------------------------------------------------------
        forbidden_arc_intervals = forbidden_arc_intervals or {}
        self.forbidden_arc_intervals = {}
        self.arc_forbidden_interval_lists = {}

        for (i_label, j_label), intervals in forbidden_arc_intervals.items():
            if i_label not in self.node_label_to_index or j_label not in self.node_label_to_index:
                raise ValueError(f"Arc ({i_label}, {j_label}) is not valid.")

            clean_intervals = []
            for start, end in intervals:
                start = int(start)
                end = int(end)
                if start < 0 or end < 0:
                    raise ValueError("Forbidden arc interval bounds must be nonnegative.")
                if end < start:
                    raise ValueError("Each forbidden arc interval must satisfy end >= start.")
                clean_intervals.append((start, end))

            self.forbidden_arc_intervals[(i_label, j_label)] = clean_intervals
            self.arc_forbidden_interval_lists[
                (self.node_label_to_index[i_label], self.node_label_to_index[j_label])
            ] = clean_intervals

        # ------------------------------------------------------------------
        # Useful index sets
        # ------------------------------------------------------------------
        self.K = list(range(self.num_vehicles))
        self.V = list(range(self.num_nodes))
        self.D = self.vertiport_indices
        self.S = self.station_indices
        self.W = self.waypoint_indices
        self.T = self.task_indices
        self.T_req = self.required_task_indices
        self.H = list(range(1, self.H_max + 1))

        # Psi_k^req = sum_{i in T_req} psi_ik
        self.Psi_req_k = np.array(
            [np.sum(self.psi_matrix[self.required_task_indices, k]) for k in self.K],
            dtype=float,
        )

    def _as_array(self, arr, shape, name):
        arr = np.asarray(arr, dtype=float)
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
        return arr

    def _as_vector(self, arr, length, name):
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.shape[0] != length:
            raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}.")
        return arr

    def label_to_index(self, label: str) -> int:
        return self.node_label_to_index[label]

    def index_to_label(self, idx: int) -> str:
        return self.index_to_node_label[idx]

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "num_nodes": self.num_nodes,
            "num_vehicles": self.num_vehicles,
            "vertiport_labels": self.vertiport_labels,
            "station_labels": self.station_labels,
            "waypoint_labels": self.waypoint_labels,
            "task_labels": self.task_labels,
            "required_task_labels": self.required_task_labels,
            "precedence_relations": self.precedence_relations,
            "tau_rel": self.tau_rel,
            "H_max": self.H_max,
            "initial_vertiport_labels": self.initial_vertiport_labels,
            "terminal_vertiport_sets": self.terminal_vertiport_sets,
            "omega_util": self.omega_util,
            "omega_time": self.omega_time,
            "omega_dist": self.omega_dist,
            "M": self.M,
        }
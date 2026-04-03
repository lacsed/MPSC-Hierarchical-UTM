import csv
import ast
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
from math import sqrt, dist


class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(dict)
        self.nodes = []
        self.node_indices = {}
        self.node_labels = {}
        self.label_to_coords = {}
        self.label_to_number = {}
        self.node_types = {}
        self.type_colors = {}
        self.next_color_idx = 0
        self.colormap = ListedColormap(
            [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "yellow",
                "brown",
                "pink",
            ]
        )

    def get_node_details(self):
        node_details = []
        for node_label in sorted(self.adjacency_list.keys()):
            if node_label not in self.label_to_coords:
                raise ValueError(
                    f"Node label '{node_label}' not found in label_to_coords."
                )
            coords = self.label_to_coords.get(node_label)
            if coords is None:
                raise ValueError(
                    f"Coordinates for node label '{node_label}' not found."
                )
            if not isinstance(coords, tuple) or len(coords) != 2:
                raise ValueError(
                    f"Invalid coordinates for node label '{node_label}': {coords}"
                )
            if coords not in self.node_labels:
                raise ValueError(f"Node at {coords} not found in the graph.")
            if node_label not in self.node_types:
                raise ValueError(f"Node type for {coords} not found in the graph.")
            node_details.append(
                {
                    "coordinates": coords,
                    "label": self.node_labels.get(coords, str(coords)),
                    "type": self.node_types.get(node_label, "default"),
                }
            )
        return node_details

    def get_node_details_dict(self):
        node_details = dict()
        for node_label in sorted(self.adjacency_list.keys()):
            if node_label not in self.label_to_coords:
                raise ValueError(
                    f"Node label '{node_label}' not found in label_to_coords."
                )
            coords = self.label_to_coords.get(node_label)
            if coords is None:
                raise ValueError(
                    f"Coordinates for node label '{node_label}' not found."
                )
            if not isinstance(coords, tuple) or len(coords) != 2:
                raise ValueError(
                    f"Invalid coordinates for node label '{node_label}': {coords}"
                )
            if coords not in self.node_labels:
                raise ValueError(f"Node at {coords} not found in the graph.")
            if node_label not in self.node_types:
                raise ValueError(f"Node type for {coords} not found in the graph.")
            node_details[node_label] = {
                "coordinates": coords,
                "label": self.node_labels.get(coords, str(coords)),
                "type": self.node_types.get(node_label, "default"),
            }

        return node_details

    def get_edges(self):
        edges = []
        for from_node, neighbors in self.adjacency_list.items():
            for to_node, weight in neighbors.items():
                edges.append((from_node, to_node, weight))
        return edges

    def get_node_label(self, coords: tuple) -> str:
        if not isinstance(coords, tuple) or len(coords) != 2:
            raise ValueError("Coordinates must be a tuple of (x, y)")
        if coords not in self.node_labels:
            raise ValueError(f"Node at {coords} not found in the graph.")
        return self.node_labels.get(coords, "")

    def get_node_number(self, label: str) -> int:
        if label not in self.label_to_number:
            raise ValueError(f"Node with label '{label}' not found in the graph.")
        return self.label_to_number[label]

    def get_edges_with_node_labels(self):
        edges = []
        for from_node, neighbors in self.adjacency_list.items():
            from_label = self.node_labels.get(from_node, str(from_node))
            for to_node, weight in neighbors.items():
                to_label = self.node_labels.get(to_node, str(to_node))
                edges.append((from_label, to_label, weight))
        return edges

    def read_from_csv(self, filename: str):
        self.__init__()

        rows = []
        try:
            with open(filename, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file, delimiter=";")
                rows = [row for row in reader]
        except FileNotFoundError:
            print(f"Erro: Arquivo '{filename}' não encontrado.")
            return
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")
            return

        for row in rows:
            try:
                name = row["name"]
                coords = (float(row["x"]), float(row["y"]))
                node_type = row["type"]

                if name in self.label_to_coords:
                    print(
                        f"Aviso: Rótulo de nó duplicado '{name}' encontrado. Sobrescrevendo."
                    )

                self.label_to_coords[name] = coords
                self.node_labels[coords] = name
                self.node_types[name] = node_type

                if node_type not in self.type_colors:
                    color = self.colormap(
                        self.next_color_idx % len(self.colormap.colors)
                    )
                    self.type_colors[node_type] = color
                    self.next_color_idx += 1
            except KeyError as e:
                print(
                    f"Erro: Coluna ausente no CSV: {e}. Verifique o cabeçalho (name;x;y;type;neighbors)."
                )
                return
            except ValueError as e:
                print(f"Erro: Valor inválido na linha {row}: {e}")
                return

        for row in rows:
            from_label = row["name"]

            try:
                neighbor_names = ast.literal_eval(row["neighbors"])
            except (ValueError, SyntaxError):
                print(
                    f"Aviso: Formato de vizinho inválido para '{from_label}'. Pulando arestas."
                )
                continue

            if not isinstance(neighbor_names, list):
                print(
                    f"Aviso: 'neighbors' para '{from_label}' não é uma lista. Pulando."
                )
                continue

            for to_label in neighbor_names:
                if to_label not in self.label_to_coords:
                    print(
                        f"Aviso: Nó vizinho '{to_label}' (de '{from_label}') não foi encontrado. Pulando aresta."
                    )
                    continue

                from_coords = self.label_to_coords[from_label]
                to_coords = self.label_to_coords[to_label]
                weight = dist(from_coords, to_coords)
                self.adjacency_list[from_label][to_label] = weight

        self._update_indices()

    def read_from_virtual_graph_csv(self, filename: str):
        self.__init__()

        rows = []
        try:
            with open(filename, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file, delimiter=";")
                rows = [row for row in reader]
        except Exception as e:
            print(f"Erro ao ler arquivo: {e}")
            return

        print(f"Number of rows read from {filename}: {len(rows)}")
        if not rows:
            print(f"Erro: Nenhum dado encontrado no arquivo '{filename}'.")
            raise ValueError("Arquivo CSV está vazio.")

        print("(Debug) Rows of the graph CSV:")
        for row in rows:
            print(row)

        for row in rows:
            name = row["name"]
            coords = (float(row["x"]), float(row["y"]))
            node_type = row["type"]

            self.label_to_coords[name] = coords
            self.node_labels[coords] = name
            self.node_types[name] = node_type

            if node_type not in self.type_colors:
                color = self.colormap(self.next_color_idx % len(self.colormap.colors))
                self.type_colors[node_type] = color
                self.next_color_idx += 1

        for row in rows:
            from_label = row["name"]
            try:
                neighbors_data = ast.literal_eval(row["neighbors"])
            except Exception:
                continue

            if not isinstance(neighbors_data, list):
                continue

            for item in neighbors_data:
                if isinstance(item, tuple) and len(item) == 2:
                    to_label, weight = item
                    self.adjacency_list[from_label][to_label] = float(weight)
                elif isinstance(item, str):
                    to_label = item
                    if to_label in self.label_to_coords:
                        from_node = self.label_to_coords[from_label]
                        to_node = self.label_to_coords[to_label]
                        self.adjacency_list[from_label][to_label] = dist(
                            from_node, to_node
                        )

        self._update_indices()

    def read_from_node_edge_csv(
        self,
        nodes_csv: str,
        edges_csv: str,
        make_bidirectional: bool = True,
        type_mapping: dict | None = None,
        use_z_in_distance: bool = False,
    ):
        """
        Lê o par:
            - graph_nodes.csv  -> id,type,x,y,z
            - graph_edges.csv  -> src,dst

        e popula o grafo interno no formato da classe.

        Parameters
        ----------
        nodes_csv : str
            Caminho para o CSV de nós.
        edges_csv : str
            Caminho para o CSV de arestas.
        make_bidirectional : bool
            Se True, adiciona também a aresta reversa dst->src.
            Isso é útil quando o arquivo de arestas representa conectividade
            não direcionada usando apenas uma linha por ligação.
        type_mapping : dict | None
            Mapeamento opcional dos tipos de nó.
            Exemplo:
                {
                    "vertiport": "Depot",
                    "charging": "station",
                    "supplier": "target",
                    "client": "target",
                    "logical": "waypoint",
                }
        use_z_in_distance : bool
            Se True, calcula peso pela distância 3D usando x,y,z.
            Se False, usa apenas x,y.
        """
        self.__init__()

        if type_mapping is None:
            type_mapping = {
                "vertiport": "Depot",
                "charging": "station",
                "supplier": "target",
                "client": "target",
                "logical": "waypoint",
            }

        z_by_label = {}

        # ---------- leitura dos nós ----------
        with open(nodes_csv, mode="r", newline="", encoding="utf-8") as f_nodes:
            reader = csv.DictReader(f_nodes)
            required_cols = {"id", "type", "x", "y"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"graph_nodes.csv sem colunas obrigatórias: {sorted(missing)}"
                )

            for row in reader:
                label = row["id"].strip()
                raw_type = row["type"].strip()
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"]) if "z" in row and row["z"] not in (None, "") else 0.0

                mapped_type = type_mapping.get(raw_type, raw_type)

                coords = (x, y)

                self.label_to_coords[label] = coords
                self.node_labels[coords] = label
                self.node_types[label] = mapped_type
                z_by_label[label] = z

                if mapped_type not in self.type_colors:
                    color = self.colormap(self.next_color_idx % len(self.colormap.colors))
                    self.type_colors[mapped_type] = color
                    self.next_color_idx += 1

                # garante que o nó apareça mesmo sem aresta ainda
                _ = self.adjacency_list[label]

        # ---------- leitura das arestas ----------
        with open(edges_csv, mode="r", newline="", encoding="utf-8") as f_edges:
            reader = csv.DictReader(f_edges)
            required_cols = {"src", "dst"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"graph_edges.csv sem colunas obrigatórias: {sorted(missing)}"
                )

            for row in reader:
                src = row["src"].strip()
                dst = row["dst"].strip()

                if src not in self.label_to_coords:
                    raise ValueError(f"Nó de origem '{src}' não existe em {nodes_csv}.")
                if dst not in self.label_to_coords:
                    raise ValueError(f"Nó de destino '{dst}' não existe em {nodes_csv}.")

                x1, y1 = self.label_to_coords[src]
                x2, y2 = self.label_to_coords[dst]

                if use_z_in_distance:
                    z1 = z_by_label[src]
                    z2 = z_by_label[dst]
                    weight = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                else:
                    weight = dist((x1, y1), (x2, y2))

                self.adjacency_list[src][dst] = weight

                if make_bidirectional:
                    self.adjacency_list[dst][src] = weight

        self._update_indices()

    def write_semicolon_graph_csv(self, output_csv: str, sort_neighbors: bool = True):
        """
        Exporta o grafo para o formato:

            name;x;y;type;neighbors

        onde neighbors é uma lista Python serializada, por exemplo:
            ['LOGICAL_001', 'CLIENT_000', 'SUPPLIER_000']
        """
        self._update_indices()

        node_names = sorted(self.label_to_coords.keys())

        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["name", "x", "y", "type", "neighbors"])

            for name in node_names:
                x, y = self.label_to_coords[name]
                node_type = self.node_types.get(name, "default")

                neighbors = list(self.adjacency_list.get(name, {}).keys())
                if sort_neighbors:
                    neighbors = sorted(neighbors)

                writer.writerow([name, x, y, node_type, repr(neighbors)])

    @classmethod
    def from_node_edge_csv(
        cls,
        nodes_csv: str,
        edges_csv: str,
        make_bidirectional: bool = True,
        type_mapping: dict | None = None,
        use_z_in_distance: bool = False,
    ):
        g = cls()
        g.read_from_node_edge_csv(
            nodes_csv=nodes_csv,
            edges_csv=edges_csv,
            make_bidirectional=make_bidirectional,
            type_mapping=type_mapping,
            use_z_in_distance=use_z_in_distance,
        )
        return g

    def convert_node_edge_csv_to_semicolon_csv(
        self,
        nodes_csv: str,
        edges_csv: str,
        output_csv: str,
        make_bidirectional: bool = True,
        type_mapping: dict | None = None,
        use_z_in_distance: bool = False,
        sort_neighbors: bool = True,
    ):
        self.read_from_node_edge_csv(
            nodes_csv=nodes_csv,
            edges_csv=edges_csv,
            make_bidirectional=make_bidirectional,
            type_mapping=type_mapping,
            use_z_in_distance=use_z_in_distance,
        )
        self.write_semicolon_graph_csv(
            output_csv=output_csv,
            sort_neighbors=sort_neighbors,
        )

    def _update_indices(self):
        all_nodes = set(self.label_to_coords.keys()) | set(self.adjacency_list.keys())
        self.nodes = sorted(all_nodes)
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}

    def to_adjacency_matrix(self):
        self._update_indices()
        size = len(self.nodes)
        matrix = [[0.0] * size for _ in range(size)]

        for i, node1 in enumerate(self.nodes):
            self.label_to_number[node1] = i
            for j, node2 in enumerate(self.nodes):
                matrix[i][j] = self.adjacency_list[node1].get(node2, 0.0)

        return matrix, self.nodes

    def print_adjacency_matrix(self):
        matrix, nodes = self.to_adjacency_matrix()

        def format_node(n):
            node_type = self.node_types.get(n, "default")
            return f"{n}({node_type[:1]})"

        print("     ", "  ".join(f"{format_node(n):<16}" for n in nodes))
        for i, row in enumerate(matrix):
            formatted_row = " ".join(
                f"{val:6.1f}" if val != 0 else "   0  " for val in row
            )
            print(f"{format_node(nodes[i]):<16} [{formatted_row}]")

    def add_edge(self, from_node, to_node, weight: float = 1.0):
        if isinstance(from_node, str):
            from_label = from_node
        else:
            from_label = self.node_labels[from_node]

        if isinstance(to_node, str):
            to_label = to_node
        else:
            to_label = self.node_labels[to_node]

        self.adjacency_list[from_label][to_label] = weight
        self._update_indices()

    def get_edge_weight(self, from_node, to_node) -> float:
        if isinstance(from_node, str):
            from_label = from_node
        else:
            from_label = self.node_labels[from_node]

        if isinstance(to_node, str):
            to_label = to_node
        else:
            to_label = self.node_labels[to_node]

        return self.adjacency_list[from_label].get(to_label, 0.0)
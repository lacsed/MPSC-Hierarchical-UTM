
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from .node_types import NodeType


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    node_type: NodeType
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SpawnRecord:
    vehicle_id: str
    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass(frozen=True)
class GraphData:
    """
    Output used by the movement simulation:
    - graph: directed MultiDiGraph 
    - nodes: only DES nodes 
    - spawns: only vehicles
    - positions: convenience lookup (x,y,z) 
    """
    graph: nx.MultiDiGraph
    nodes: Dict[str, NodeRecord]
    spawns: List[SpawnRecord]
    positions: Dict[str, Tuple[float, float, float]]

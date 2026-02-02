from enum import Enum


class NodeType(str, Enum):
    # Canonical naming (international)
    SUPPLIER = "supplier"
    CLIENT = "client"
    CHARGING = "charging"
    VERTIPORT = "vertiport"
    LOGICAL = "logical"
    VEHICLE = "vehicle"

    @staticmethod
    def parse(value: str) -> "NodeType":
        """
        Robust parser:
        - accepts lowercase/uppercase
        """
        if value is None:
            raise ValueError("Node type is None")

        v = str(value).strip()
        if not v:
            raise ValueError("Empty node type")

        lower = v.lower()

        # direct canonical
        for t in NodeType:
            if lower == t.value:
                return t

        # infer from prefix before '_' 
        prefix = v.split("_", 1)[0].lower()
        for t in NodeType:
            if prefix == t.value:
                return t

        raise ValueError(f"Unknown node type: '{value}'")


DES_NODE_TYPES = {
    NodeType.SUPPLIER,
    NodeType.CLIENT,
    NodeType.CHARGING,
    NodeType.VERTIPORT,
    NodeType.LOGICAL,
}

SPAWN_TYPES = {NodeType.VEHICLE}

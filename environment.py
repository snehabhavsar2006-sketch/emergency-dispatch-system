"""
Simulation Environment Module (environment.py)
----------------------------------------------
Design Rationale & Viva Defense Notes:
- Grid Model: Representing the urban emergency environment as a discrete 10x10 grid 
  (coordinates 0..9) provides a clean, computationally efficient spatial model 
  well-suited for grid-based pathfinding (A*) and hyperdimensional spatial encoding (HDC).
- Terrain & Hazard Costs: Each grid cell maintains a terrain traversal multiplier 
  (default 1.0) and a boolean 'blocked' flag. This allows modeling real-world 
  road closures, traffic congestion, and disaster zone hazards.
- Dataclass Schemas: Structured dataclasses for Incident and Responder maintain strict 
  type signatures and field naming, eliminating ambiguous key lookups across the pipeline.
- Seeded Generation: Generator functions accept explicit random seeds to ensure 
  reproducibility across experimental benchmark runs and demonstration scenarios.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


INCIDENT_TYPES = ["fire", "accident", "medical", "structural"]
RESPONDER_TYPES = ["fire_truck", "ambulance", "police", "rescue_unit"]


@dataclass
class Incident:
    incident_id: str
    type: str  # Normalized vocabulary: "fire", "accident", "medical", "structural"
    location: Tuple[int, int]  # (x, y) where x, y in [0, 9]
    severity: int  # 1 (low) to 5 (critical)
    timestamp: float = field(default_factory=time.time)
    status: str = "active"  # "active", "resolved"
    source: str = "DEMO SIMULATION"
    message: str = ""
    location_name: str = ""
    assigned_responder_id: Optional[str] = None
    responder_status: str = "UNASSIGNED"
    latitude: float = 21.3400
    longitude: float = 74.8800
    location_source: str = "SIMULATION"
    extracted_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "type": self.type,
            "location": list(self.location),
            "severity": self.severity,
            "timestamp": self.timestamp,
            "status": self.status,
            "source": self.source,
            "message": self.message,
            "location_name": self.location_name,
            "assigned_responder_id": self.assigned_responder_id,
            "responder_status": self.responder_status,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "extracted_info": self.extracted_info,
        }


@dataclass
class Responder:
    responder_id: str
    type: str  # Normalized vocabulary: "fire_truck", "ambulance", "police", "rescue_unit"
    location: Tuple[int, int]  # (x, y) where x, y in [0, 9]
    capacity: int  # Integer unit capacity (e.g. 1 to 5)
    available: bool = True
    status: str = "AVAILABLE"
    current_assignment: Optional[str] = None
    latitude: float = 21.3400
    longitude: float = 74.8800
    location_source: str = "SIMULATION"
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "responder_id": self.responder_id,
            "type": self.type,
            "location": list(self.location),
            "capacity": self.capacity,
            "available": self.available,
            "status": self.status,
            "current_assignment": self.current_assignment,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
        }


class GridCity:
    """
    10x10 Grid environment representing city terrain, road blocks, and coordinates.
    """
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        # terrain_cost[x][y] defaults to 1.0 per movement step
        self.terrain_cost = np.ones((width, height), dtype=np.float64)
        # blocked[x][y] represents impassable road closures / hazards
        self.blocked = np.zeros((width, height), dtype=bool)

    def is_valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def block_cell(self, x: int, y: int) -> None:
        if not self.is_valid_coord(x, y):
            raise ValueError(f"Coordinate ({x}, {y}) out of grid bounds.")
        self.blocked[x, y] = True

    def unblock_cell(self, x: int, y: int) -> None:
        if not self.is_valid_coord(x, y):
            raise ValueError(f"Coordinate ({x}, {y}) out of grid bounds.")
        self.blocked[x, y] = False

    def set_terrain_cost(self, x: int, y: int, cost: float) -> None:
        if not self.is_valid_coord(x, y):
            raise ValueError(f"Coordinate ({x}, {y}) out of grid bounds.")
        if cost <= 0:
            raise ValueError("Terrain cost must be strictly positive.")
        self.terrain_cost[x, y] = float(cost)


def generate_incidents(n: int, seed: Optional[int] = None) -> List[Incident]:
    """
    Generate n random active incidents across the 10x10 city grid.
    """
    rng = np.random.default_rng(seed)
    incidents = []
    now = time.time()
    for i in range(n):
        inc_type = rng.choice(INCIDENT_TYPES)
        x = int(rng.integers(0, 10))
        y = int(rng.integers(0, 10))
        severity = int(rng.integers(1, 6))  # 1 to 5 inclusive
        inc = Incident(
            incident_id=f"INC-{i+1:03d}",
            type=str(inc_type),
            location=(x, y),
            severity=severity,
            timestamp=now - float(rng.uniform(0, 300)),  # reported 0 to 5 mins ago
            status="active",
        )
        incidents.append(inc)
    return incidents


def generate_responders(n: int, seed: Optional[int] = None) -> List[Responder]:
    """
    Generate n random available responders across the 10x10 city grid.
    """
    rng = np.random.default_rng(seed)
    responders = []
    for i in range(n):
        resp_type = rng.choice(RESPONDER_TYPES)
        x = int(rng.integers(0, 10))
        y = int(rng.integers(0, 10))
        capacity = int(rng.integers(1, 6))  # capacity 1 to 5
        resp = Responder(
            responder_id=f"RESP-{i+1:03d}",
            type=str(resp_type),
            location=(x, y),
            capacity=capacity,
            available=True,
        )
        responders.append(resp)
    return responders

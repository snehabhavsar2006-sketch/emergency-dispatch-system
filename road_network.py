import math
from typing import Dict, List, Optional, Tuple

import networkx as nx


RADIUS_EARTH_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * RADIUS_EARTH_KM * math.asin(math.sqrt(a))


def _build_road_graph() -> nx.Graph:
    graph = nx.Graph()
    nodes = {
        "N1": (21.3335, 74.8675),
        "N2": (21.3364, 74.8707),
        "N3": (21.3392, 74.8744),
        "N4": (21.3428, 74.8789),
        "N5": (21.3456, 74.8833),
        "N6": (21.3486, 74.8882),
        "N7": (21.3410, 74.8695),
        "N8": (21.3461, 74.8723),
        "N9": (21.3501, 74.8781),
        "N10": (21.3369, 74.8810),
        "N11": (21.3404, 74.8866),
        "N12": (21.3347, 74.8901),
    }
    for node_id, (lat, lon) in nodes.items():
        graph.add_node(node_id, latitude=lat, longitude=lon)

    edges = [
        ("N1", "N2", 0.35),
        ("N2", "N3", 0.38),
        ("N3", "N4", 0.43),
        ("N4", "N5", 0.40),
        ("N5", "N6", 0.44),
        ("N2", "N7", 0.46),
        ("N3", "N7", 0.31),
        ("N7", "N4", 0.35),
        ("N4", "N8", 0.33),
        ("N8", "N5", 0.38),
        ("N7", "N8", 0.48),
        ("N8", "N9", 0.47),
        ("N1", "N10", 0.29),
        ("N10", "N3", 0.36),
        ("N10", "N11", 0.42),
        ("N11", "N5", 0.41),
        ("N10", "N12", 0.35),
        ("N12", "N6", 0.43),
        ("N12", "N1", 0.39),
        ("N9", "N11", 0.44),
        ("N3", "N8", 0.51),
        ("N7", "N10", 0.39),
    ]
    for a, b, distance in edges:
        graph.add_edge(a, b, weight=distance)
    return graph


def _nearest_node(graph: nx.Graph, latitude: float, longitude: float) -> str:
    closest_id = None
    closest_distance = float("inf")
    for node_id, data in graph.nodes(data=True):
        node_lat = data["latitude"]
        node_lon = data["longitude"]
        dist = haversine_km(latitude, longitude, node_lat, node_lon)
        if dist < closest_distance:
            closest_distance = dist
            closest_id = node_id
    if closest_id is None:
        raise ValueError("Road graph is empty.")
    return closest_id


class RealWorldAStarRouter:
    """Road-network A* over a lightweight geographic graph that remains compatible with NetworkX A*."""

    def __init__(self, graph: Optional[nx.Graph] = None):
        self.graph = graph if graph is not None else _build_road_graph()

    def _heuristic(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return haversine_km(lat1, lon1, lat2, lon2)

    def route_between(self, origin_lat: float, origin_lon: float, destination_lat: float, destination_lon: float) -> Dict[str, object]:
        if origin_lat is None or origin_lon is None or destination_lat is None or destination_lon is None:
            return {
                "status": "unavailable",
                "distance_km": 0.0,
                "eta_minutes": 0.0,
                "path": [],
                "route": "",
            }

        start_node = _nearest_node(self.graph, origin_lat, origin_lon)
        end_node = _nearest_node(self.graph, destination_lat, destination_lon)

        if start_node == end_node:
            path = [start_node]
            distance_km = 0.0
        else:
            try:
                path = nx.astar_path(
                    self.graph,
                    start_node,
                    end_node,
                    heuristic=lambda current, goal: self._heuristic(
                        self.graph.nodes[current]["latitude"],
                        self.graph.nodes[current]["longitude"],
                        self.graph.nodes[goal]["latitude"],
                        self.graph.nodes[goal]["longitude"],
                    ),
                    weight="weight",
                )
                distance_km = nx.path_weight(self.graph, path, weight="weight")
            except nx.NetworkXNoPath:
                return {
                    "status": "no_path",
                    "distance_km": float("inf"),
                    "eta_minutes": float("inf"),
                    "path": [],
                    "route": "",
                }

        route_points = [
            (self.graph.nodes[node_id]["latitude"], self.graph.nodes[node_id]["longitude"]) for node_id in path
        ]
        eta_minutes = max(1.0, (distance_km / 30.0) * 60.0)
        return {
            "status": "ok",
            "distance_km": round(distance_km, 2),
            "eta_minutes": round(eta_minutes, 1),
            "path": route_points,
            "route": " -> ".join([f"{lat:.4f},{lon:.4f}" for lat, lon in route_points]),
        }

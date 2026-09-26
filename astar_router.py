"""
A* Route Planning Module (astar_router.py)
------------------------------------------
Design Rationale & Viva Defense Notes:
- Role of A*: A* calculates exact spatial trajectories and traversal costs for candidates 
  that pass CSP constraint validation.
- Pure Traversal Cost Rationale:
  - Route cost is computed purely from grid cell terrain traversal costs and step distance.
  - Incident severity is NOT multiplied into route cost. Severity is used exclusively 
    for incident processing order in the IncidentDetector & Coordinator. Keeping route cost 
    unpolluted ensures defensible, physical time/distance routing metrics.
- Heuristic: Manhattan distance h((x1, y1), (x2, y2)) = |x1 - x2| + |y1 - y2|. 
  Since movements are strictly 4-directional (Up, Down, Left, Right) on a grid, Manhattan 
  distance is admissible (never overestimates) and consistent.
- Impassability: Grid cells marked as 'blocked' (e.g. road closures) are treated as 
  impassable obstacles (cost = infinity).
"""

import heapq
from typing import Tuple, List, Optional, Dict
from environment import GridCity, Incident, Responder


class AStarRouter:
    """
    Grid-based A* route planning engine.
    """

    @staticmethod
    def manhattan_distance(coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional grid movement."""
        return float(abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]))

    def find_route(
        self, start: Tuple[int, int], goal: Tuple[int, int], grid: GridCity
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        Executes A* search from start (x, y) to goal (x, y) on grid.
        
        Returns:
        - path: List of (x, y) grid coordinates from start to goal (inclusive), or None if unroutable.
        - total_cost: Cumulative traversal cost, or float('inf') if unroutable.
        """
        if not (grid.is_valid_coord(start[0], start[1]) and grid.is_valid_coord(goal[0], goal[1])):
            return None, float("inf")

        if grid.blocked[start[0], start[1]] or grid.blocked[goal[0], goal[1]]:
            return None, float("inf")

        if start == goal:
            return [start], 0.0

        # Priority queue stores tuples: (f_score, g_score, current_node)
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self.manhattan_distance(start, goal), 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        # 4-directional movement vectors: Up, Down, Left, Right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g_score[goal]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not grid.is_valid_coord(neighbor[0], neighbor[1]):
                    continue

                if grid.blocked[neighbor[0], neighbor[1]]:
                    continue

                step_cost = grid.terrain_cost[neighbor[0], neighbor[1]]
                tentative_g = current_g + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # Destination unroutable
        return None, float("inf")

    def rank_candidates(
        self, incident: Incident, valid_responders: List[Responder], grid: GridCity
    ) -> List[Tuple[Responder, List[Tuple[int, int]], float]]:
        """
        Calculates route and cost for each CSP-valid responder.
        
        Returns list of (Responder, path, cost) tuples sorted ascending by cost.
        Filters out candidates with no valid path (cost == inf).
        """
        ranked_results: List[Tuple[Responder, List[Tuple[int, int]], float]] = []

        for resp in valid_responders:
            path, cost = self.find_route(resp.location, incident.location, grid)
            if path is not None and cost < float("inf"):
                ranked_results.append((resp, path, cost))

        # Sort ascending by route cost (shortest path first)
        ranked_results.sort(key=lambda item: item[2])
        return ranked_results

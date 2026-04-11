"""
SaferoadAI – A* Ambulance Routing Engine
==========================================
Finds the fastest ambulance route from an accident location to
the nearest hospital using an A* algorithm over an in-memory road graph.

Falls back to Google Directions API if GOOGLE_MAPS_API_KEY is set in env.

City graph is loaded from `api/city_graph.json` (bidirectional edges).

Usage:
    router = AmbulanceRouter()
    result = router.route(accident_lat=28.5439, accident_lon=77.3305)
    print(result)
    # {
    #   "status": "ok",
    #   "from_node": "H5",
    #   "to_hospital": "HOSP1",
    #   "hospital_name": "Max Super Speciality Hospital",
    #   "waypoints": [...],
    #   "distance_km": 3.2,
    #   "eta_minutes": 8,
    #   "algorithm": "astar"
    # }
"""

from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Config ───────────────────────────────────────────────────────────────────

GRAPH_PATH         = Path(__file__).parent / "city_graph.json"
AMBULANCE_SPEED_KMH = 50.0   # average ambulance speed in city traffic


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    id: str
    lat: float
    lon: float
    name: str


@dataclass(order=True)
class _PQItem:
    f_score: float
    node_id: str = field(compare=False)


# ─── AmbulanceRouter ──────────────────────────────────────────────────────────

class AmbulanceRouter:
    """
    Routes an ambulance from the nearest road node to the nearest hospital
    using the A* algorithm over a pre-loaded city road graph.

    Parameters
    ----------
    graph_path : Path to city_graph.json.
    speed_kmh  : Assumed ambulance speed for ETA calculation.
    """

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        speed_kmh: float = AMBULANCE_SPEED_KMH,
    ) -> None:
        self.speed_kmh = speed_kmh
        self._nodes: Dict[str, GraphNode] = {}
        self._adj: Dict[str, List[Tuple[str, float]]] = {}   # node_id → [(neighbor_id, km)]
        self._hospital_ids: List[str] = []
        self._load_graph(graph_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        accident_lat: float,
        accident_lon: float,
        target_hospital_id: Optional[str] = None,
    ) -> dict:
        """
        Compute the optimal ambulance route.

        Parameters
        ----------
        accident_lat, accident_lon : GPS coordinates of the accident.
        target_hospital_id        : Force routing to a specific hospital (optional).

        Returns
        -------
        dict with keys: status, from_node, to_hospital, hospital_name,
                        waypoints, distance_km, eta_minutes, algorithm.
        """
        # Check for Google Maps fallback
        gmaps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if gmaps_key:
            return self._gmaps_route(accident_lat, accident_lon, gmaps_key)

        # ── Snap accident to nearest node ─────────────────────────────────
        start_id = self._nearest_node(accident_lat, accident_lon)
        if not start_id:
            return {"status": "error", "message": "Empty graph"}

        # ── Route to each hospital, pick shortest ─────────────────────────
        hospitals_to_try = [target_hospital_id] if target_hospital_id else self._hospital_ids
        best: Optional[dict] = None

        for hosp_id in hospitals_to_try:
            if hosp_id not in self._nodes:
                continue
            result = self._astar(start_id, hosp_id)
            if result and (best is None or result["distance_km"] < best["distance_km"]):
                best = result
                best["from_node"] = start_id
                best["to_hospital"] = hosp_id
                best["hospital_name"] = self._nodes[hosp_id].name

        if best is None:
            return {"status": "error", "message": "No path found to any hospital"}

        dist = best["distance_km"]
        eta = (dist / self.speed_kmh) * 60  # minutes

        return {
            "status":        "ok",
            "from_node":     best["from_node"],
            "to_hospital":   best["to_hospital"],
            "hospital_name": best["hospital_name"],
            "waypoints":     best["waypoints"],
            "distance_km":   round(dist, 2),
            "eta_minutes":   round(eta, 1),
            "algorithm":     "astar",
        }

    # ── A* Implementation ─────────────────────────────────────────────────────

    def _astar(self, start: str, goal: str) -> Optional[dict]:
        """Standard A* over the city road graph. Returns path dict or None."""
        open_heap: List[_PQItem] = []
        g_score: Dict[str, float] = {start: 0.0}
        came_from: Dict[str, Optional[str]] = {start: None}

        heapq.heappush(open_heap, _PQItem(
            f_score=self._heuristic(start, goal),
            node_id=start,
        ))

        while open_heap:
            current = heapq.heappop(open_heap).node_id

            if current == goal:
                return self._reconstruct(came_from, current, g_score[current])

            for neighbor, edge_km in self._adj.get(current, []):
                tentative_g = g_score.get(current, math.inf) + edge_km
                if tentative_g < g_score.get(neighbor, math.inf):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, _PQItem(f_score=f, node_id=neighbor))

        return None  # no path

    def _heuristic(self, a_id: str, b_id: str) -> float:
        """Haversine distance between two node IDs as admissible heuristic."""
        a, b = self._nodes.get(a_id), self._nodes.get(b_id)
        if not a or not b:
            return 0.0
        return _haversine(a.lat, a.lon, b.lat, b.lon)

    def _reconstruct(
        self, came_from: dict, current: str, total_km: float
    ) -> dict:
        path: List[str] = []
        node = current
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        waypoints = [
            {"node_id": nid,
             "lat": self._nodes[nid].lat,
             "lon": self._nodes[nid].lon,
             "name": self._nodes[nid].name}
            for nid in path if nid in self._nodes
        ]
        return {"waypoints": waypoints, "distance_km": total_km}

    # ── Graph Loading ─────────────────────────────────────────────────────────

    def _load_graph(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)

        for n in data["nodes"]:
            self._nodes[n["id"]] = GraphNode(
                id=n["id"], lat=n["lat"], lon=n["lon"], name=n["name"]
            )
            self._adj[n["id"]] = []

        for e in data["edges"]:
            km = e["weight_km"]
            # Bidirectional
            self._adj[e["from"]].append((e["to"], km))
            self._adj[e["to"]].append((e["from"], km))

        self._hospital_ids = data.get("hospitals", [])

    def _nearest_node(self, lat: float, lon: float) -> Optional[str]:
        best_id, best_dist = None, math.inf
        for nid, node in self._nodes.items():
            d = _haversine(lat, lon, node.lat, node.lon)
            if d < best_dist:
                best_dist, best_id = d, nid
        return best_id

    # ── Google Maps fallback ──────────────────────────────────────────────────

    def _gmaps_route(self, lat: float, lon: float, api_key: str) -> dict:
        """Use Google Directions API when API key is configured."""
        try:
            import requests
            origin = f"{lat},{lon}"
            hosp = self._nodes.get(self._hospital_ids[0]) if self._hospital_ids else None
            if not hosp:
                return {"status": "error", "message": "No hospital configured"}
            dest = f"{hosp.lat},{hosp.lon}"

            url = (
                f"https://maps.googleapis.com/maps/api/directions/json"
                f"?origin={origin}&destination={dest}&mode=driving&key={api_key}"
            )
            r = requests.get(url, timeout=5)
            data = r.json()
            if data.get("status") != "OK":
                return {"status": "error", "message": data.get("status")}

            leg = data["routes"][0]["legs"][0]
            return {
                "status":        "ok",
                "from_node":     origin,
                "to_hospital":   hosp.id,
                "hospital_name": hosp.name,
                "waypoints":     [{"lat": s["end_location"]["lat"], "lon": s["end_location"]["lng"]}
                                  for s in leg["steps"]],
                "distance_km":   round(leg["distance"]["value"] / 1000, 2),
                "eta_minutes":   round(leg["duration"]["value"] / 60, 1),
                "algorithm":     "google_maps",
            }
        except Exception as ex:
            return {"status": "error", "message": str(ex)}


# ─── Haversine ────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

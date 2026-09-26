"""
HDC Candidate Screening Module (hdc_screener.py)
-----------------------------------------------
Design Rationale & Viva Defense Notes:
- Role of HDC: Hyperdimensional Computing (HDC) acts strictly as a fast candidate 
  screener. It narrows down a large responder pool to a shortlist (default top_n=3).
- Bipolar Vectors: Dimension D = 10,000 bipolar vectors (+1 / -1) are used. 
  High dimension guarantees quasi-orthogonality between random vectors.
- Item Memory: Categorical values (types, severity, capacity) are assigned fixed random 
  bipolar hypervectors once at initialization.
- Spatial Locality Encoding:
  - Coordinate hypervectors (Vx for x, Vy for y) are generated using continuous angle 
    interpolation between two orthogonal basis hypervectors:
    v_x = cos((pi/2) * (x/9)) * basis_1 + sin((pi/2) * (x/9)) * basis_2
    V_Xx = sign(v_x)
  - This guarantees that cosine similarity between coordinate hypervectors degrades 
    smoothly as grid distance |x1 - x2| increases, without introducing arbitrary constants.
- Role-Filler Binding & Bundling:
  - Role-Filler Binding: Element-wise multiplication (V_role * V_filler).
  - Bundling: Summing bound role-filler vectors and taking sign() majority rule.
- Cosine Similarity: Standard dot product divided by vector dimensions, ranging from -1 to +1.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from environment import Incident, Responder, INCIDENT_TYPES, RESPONDER_TYPES


class HDCScreener:
    """
    Hyperdimensional Computing Screener for fast candidate search space narrowing.
    """

    def __init__(self, dimension: int = 10000, seed: int = 42):
        self.D = dimension
        self.rng = np.random.default_rng(seed)
        self.item_memory: Dict[str, np.ndarray] = {}
        self._init_item_memory()

    def _generate_random_bipolar(self) -> np.ndarray:
        """Generate a single D-dimensional bipolar hypervector (+1/-1)."""
        vec = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.D)
        return vec

    def _init_item_memory(self) -> None:
        """Initialize item memory for roles, incident types, responder types, and capacities."""
        # Roles
        roles = ["ROLE_TYPE", "ROLE_SEVERITY", "ROLE_CAPACITY", "ROLE_LOC_X", "ROLE_LOC_Y"]
        for role in roles:
            self.item_memory[role] = self._generate_random_bipolar()

        # Incident types
        for inc_type in INCIDENT_TYPES:
            self.item_memory[f"INC_TYPE_{inc_type}"] = self._generate_random_bipolar()

        # Responder types
        for resp_type in RESPONDER_TYPES:
            self.item_memory[f"RESP_TYPE_{resp_type}"] = self._generate_random_bipolar()

        # Cross-domain type affinity mappings (Incident type -> Ideal Responder type filler)
        # Allows HDC similarity to reflect functional compatibility in vector space
        self.type_affinity: Dict[str, str] = {
            "fire": "fire_truck",
            "accident": "ambulance",
            "medical": "ambulance",
            "structural": "rescue_unit",
        }

        # Severity levels (1 to 5)
        for s in range(1, 6):
            self.item_memory[f"SEV_{s}"] = self._generate_random_bipolar()

        # Capacity levels (1 to 5)
        for c in range(1, 6):
            self.item_memory[f"CAP_{c}"] = self._generate_random_bipolar()

        # 2D Spatial Coordinate Encodings (smooth angle interpolation)
        # X-axis basis vectors
        u_x = self.rng.standard_normal(self.D)
        v_x = self.rng.standard_normal(self.D)
        for x in range(10):
            theta = (np.pi / 2.0) * (x / 9.0)
            interp = np.cos(theta) * u_x + np.sin(theta) * v_x
            bipolar = np.where(interp >= 0, 1, -1).astype(np.int8)
            self.item_memory[f"LOC_X_{x}"] = bipolar

        # Y-axis basis vectors
        u_y = self.rng.standard_normal(self.D)
        v_y = self.rng.standard_normal(self.D)
        for y in range(10):
            theta = (np.pi / 2.0) * (y / 9.0)
            interp = np.cos(theta) * u_y + np.sin(theta) * v_y
            bipolar = np.where(interp >= 0, 1, -1).astype(np.int8)
            self.item_memory[f"LOC_Y_{y}"] = bipolar

    def encode_incident(self, incident: Incident) -> np.ndarray:
        """
        Encode an Incident into a D-dimensional bipolar hypervector.
        Binds role & filler vectors, then bundles via majority sum.
        """
        # Type filler: use ideal responder type for cross-domain matching
        ideal_resp_type = self.type_affinity.get(incident.type, incident.type)
        type_filler = self.item_memory.get(f"RESP_TYPE_{ideal_resp_type}", self.item_memory[f"INC_TYPE_{incident.type}"])
        v_type = self.item_memory["ROLE_TYPE"] * type_filler

        # Severity / Demand filler
        v_sev = self.item_memory["ROLE_SEVERITY"] * self.item_memory[f"SEV_{incident.severity}"]

        # Location fillers
        x, y = incident.location
        v_loc_x = self.item_memory["ROLE_LOC_X"] * self.item_memory[f"LOC_X_{x}"]
        v_loc_y = self.item_memory["ROLE_LOC_Y"] * self.item_memory[f"LOC_Y_{y}"]

        # Bundling via majority sum
        bound_sum = (v_type.astype(np.int32) + 
                     v_sev.astype(np.int32) + 
                     v_loc_x.astype(np.int32) + 
                     v_loc_y.astype(np.int32))
        
        # Threshold (break ties deterministically)
        bundled = np.where(bound_sum >= 0, 1, -1).astype(np.int8)
        return bundled

    def encode_responder(self, responder: Responder) -> np.ndarray:
        """
        Encode a Responder into a D-dimensional bipolar hypervector.
        """
        # Type filler
        v_type = self.item_memory["ROLE_TYPE"] * self.item_memory[f"RESP_TYPE_{responder.type}"]

        # Capacity filler (clip to 1..5)
        cap = max(1, min(5, responder.capacity))
        v_cap = self.item_memory["ROLE_CAPACITY"] * self.item_memory[f"CAP_{cap}"]

        # Location fillers
        x, y = responder.location
        v_loc_x = self.item_memory["ROLE_LOC_X"] * self.item_memory[f"LOC_X_{x}"]
        v_loc_y = self.item_memory["ROLE_LOC_Y"] * self.item_memory[f"LOC_Y_{y}"]

        # Bundling via majority sum
        bound_sum = (v_type.astype(np.int32) + 
                     v_cap.astype(np.int32) + 
                     v_loc_x.astype(np.int32) + 
                     v_loc_y.astype(np.int32))

        bundled = np.where(bound_sum >= 0, 1, -1).astype(np.int8)
        return bundled

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two bipolar hypervectors."""
        dot = float(np.dot(v1.astype(np.float64), v2.astype(np.float64)))
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return dot / denom

    def screen(
        self, incident: Incident, responders: List[Responder], top_n: int = 3
    ) -> Tuple[List[Tuple[Responder, float]], Dict[str, float]]:
        """
        Screen responder pool against incident vector.
        
        Returns:
        - shortlist: List of top_n (Responder, similarity_score) pairs sorted descending.
        - all_scores: Dict mapping responder_id -> similarity_score (for all responders).
        """
        inc_vec = self.encode_incident(incident)
        all_scores: Dict[str, float] = {}
        scored_pairs: List[Tuple[Responder, float]] = []

        for resp in responders:
            resp_vec = self.encode_responder(resp)
            sim = self.cosine_similarity(inc_vec, resp_vec)
            all_scores[resp.responder_id] = sim
            scored_pairs.append((resp, sim))

        # Sort descending by similarity score
        scored_pairs.sort(key=lambda item: item[1], reverse=True)
        shortlist = scored_pairs[:top_n]
        return shortlist, all_scores

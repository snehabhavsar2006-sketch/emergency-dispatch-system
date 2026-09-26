"""
CSP Constraint Validation Module (csp_validator.py)
---------------------------------------------------
Design Rationale & Viva Defense Notes:
- Role of CSP: While HDC rapidly screens candidate similarity, HDC is a probabilistic 
  vector method that cannot guarantee 100% hard constraint compliance. The CSP stage 
  serves as the deterministic eligibility validator.
- Explicit Constraint Functions: Each business rule is implemented as an isolated, 
  testable function for clear viva defense and modular unit testing.
- Explainable Rejection Labels: When a candidate fails a constraint, an explicit 
  label ("unavailable", "type mismatch", "insufficient capacity") is recorded 
  in the DecisionTrace for the explainer module and UI dashboard.
- Prototype Rule Disclaimer:
  - Capacity Rule: responder.capacity >= incident.severity.
  - This is a simplified prototype rule for simulation and must be documented 
    as such; it does not represent a real-world emergency capacity standard.
"""

from typing import List, Tuple, Dict, Set
from environment import Incident, Responder


# Compatible responder types for each incident category
COMPATIBILITY_MATRIX: Dict[str, Set[str]] = {
    "fire": {"fire_truck", "rescue_unit"},
    "accident": {"ambulance", "police", "rescue_unit"},
    "medical": {"ambulance"},
    "structural": {"rescue_unit", "fire_truck"},
}


class CSPValidator:
    """
    Validates screened candidate responders against hard operational constraints.
    """

    @staticmethod
    def is_available(responder: Responder) -> bool:
        """Constraint 1: Responder must be available for assignment."""
        return bool(responder.available)

    @staticmethod
    def type_compatible(incident: Incident, responder: Responder) -> bool:
        """Constraint 2: Responder type must match incident category capabilities."""
        allowed_types = COMPATIBILITY_MATRIX.get(incident.type, set())
        return responder.type in allowed_types

    @staticmethod
    def capacity_sufficient(incident: Incident, responder: Responder) -> bool:
        """
        Constraint 3: Responder unit capacity must satisfy incident severity demand.
        
        Disclaimer: This is a simplified prototype rule for simulation and must be 
        documented as such; it does not represent a real-world emergency capacity standard.
        """
        return responder.capacity >= incident.severity

    def validate(
        self, incident: Incident, candidates: List[Responder]
    ) -> Tuple[List[Responder], List[Tuple[Responder, str]]]:
        """
        Validates candidate list against constraints sequentially.
        
        Returns:
        - valid_responders: List of candidates passing ALL constraints.
        - rejected_candidates: List of (Responder, rejection_reason) tuples.
        """
        valid_responders: List[Responder] = []
        rejected_candidates: List[Tuple[Responder, str]] = []

        for resp in candidates:
            # Check 1: Availability
            if not self.is_available(resp):
                rejected_candidates.append((resp, "unavailable"))
                continue

            # Check 2: Type Compatibility
            if not self.type_compatible(incident, resp):
                rejected_candidates.append((resp, "type mismatch"))
                continue

            # Check 3: Capacity Sufficiency
            if not self.capacity_sufficient(incident, resp):
                rejected_candidates.append((resp, "insufficient capacity"))
                continue

            # Passed all constraints
            valid_responders.append(resp)

        return valid_responders, rejected_candidates

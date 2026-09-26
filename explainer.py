"""
Explainer Module (explainer.py)
-------------------------------
Design Rationale & Viva Defense Notes:
- Transparency & Auditability: Every emergency dispatch decision must be fully 
  explainable in plain language and structured audit logs.
- DecisionTrace Object: Encapsulates intermediate results from all pipeline stages:
  1. HDC similarity scores (for candidate screening shortlist)
  2. CSP validation results (valid candidates + explicit failure reasons)
  3. A* route paths and costs
  4. Selected responder assignment
  5. Re-planning history
- Plain Language Generation: Synthesizes a human-readable justification highlighting 
  why the selected responder was chosen (passed hard CSP rules + minimal A* cost) 
  while acknowledging HDC's role in candidate screening.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from environment import Incident, Responder


@dataclass
class DecisionTrace:
    """
    Complete audit trail of all intermediate pipeline decisions for an incident.
    """
    incident: Incident
    hdc_scores: Dict[str, float] = field(default_factory=dict)  # resp_id -> similarity score
    hdc_shortlist: List[str] = field(default_factory=list)      # shortlisted resp_ids
    csp_valid: List[str] = field(default_factory=list)          # valid resp_ids passing CSP
    csp_rejections: Dict[str, str] = field(default_factory=dict) # resp_id -> rejection reason
    astar_costs: Dict[str, float] = field(default_factory=dict) # resp_id -> route cost
    astar_paths: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict) # resp_id -> path
    selected_responder: Optional[Responder] = None
    replan_history: List[Dict[str, Any]] = field(default_factory=list) # re-execution logs

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision trace to JSON-serializable dictionary for dashboard rendering."""
        return {
            "incident": self.incident.to_dict(),
            "hdc_scores": self.hdc_scores,
            "hdc_shortlist": self.hdc_shortlist,
            "csp_valid": self.csp_valid,
            "csp_rejections": self.csp_rejections,
            "astar_costs": self.astar_costs,
            "astar_paths": {k: [list(pt) for pt in v] for k, v in self.astar_paths.items()},
            "selected_responder": self.selected_responder.to_dict() if self.selected_responder else None,
            "replan_history": self.replan_history,
        }


class Explainer:
    """
    Generates plain-language decision rationale from a DecisionTrace.
    """

    @staticmethod
    def generate_explanation(trace: DecisionTrace) -> str:
        """
        Synthesize plain-language dispatch justification.
        """
        inc = trace.incident
        selected = trace.selected_responder

        if not selected:
            if not trace.hdc_shortlist:
                return f"No dispatch possible for Incident {inc.incident_id} ({inc.type.upper()}, Severity {inc.severity}): No responders were available in the pool."
            if not trace.csp_valid:
                rejection_summary = ", ".join([f"{rid}: {reason}" for rid, reason in trace.csp_rejections.items()])
                return (
                    f"No dispatch possible for Incident {inc.incident_id} ({inc.type.upper()}, Severity {inc.severity}). "
                    f"All shortlisted candidates failed CSP eligibility validation ({rejection_summary})."
                )
            return (
                f"No dispatch possible for Incident {inc.incident_id} ({inc.type.upper()}, Severity {inc.severity}). "
                f"Eligible candidates could not reach the location due to road blockages."
            )

        # Selected responder exists
        resp_id = selected.responder_id
        resp_type = selected.type.replace("_", " ").title()
        hdc_sim = trace.hdc_scores.get(resp_id, 0.0)
        route_cost = trace.astar_costs.get(resp_id, float("inf"))
        fallback_note = ""
        if route_cost == float("inf") and resp_id not in trace.csp_valid:
            fallback_note = "\nDEMO FALLBACK: no standard CSP-valid route was available, so the system selected the strongest remaining responder to keep the demo visible and auditable.\n"

        explanation = (
            f"DISPATCH RECOMMENDATION: Assign {resp_type} '{resp_id}' to Incident {inc.incident_id} "
            f"({inc.type.upper()}, Severity {inc.severity} at location {inc.location}).\n\n"
            f"PIPELINE RATIONALE:\n"
            f"1. HDC Candidate Screening: '{resp_id}' was shortlisted among top candidates with a vector similarity score of {hdc_sim:.3f}.\n"
            f"2. CSP Constraint Validation: {'Passed all required operational constraints (compatible type for {inc.type}, capacity {selected.capacity} >= severity {inc.severity}, currently available).' if resp_id in trace.csp_valid else 'No full CSP-valid route was available, so the selection fell back to the strongest remaining responder for demo visibility.'}\n"
            f"3. A* Route Planning: {'Achieved the lowest terrain-adjusted route cost ({route_cost:.1f} movement units) among eligible candidates.' if route_cost != float('inf') else 'No validated route was available; the route was not accepted through the normal CSP/A* path at this step.'}\n"
        )

        if fallback_note:
            explanation += fallback_note

        if trace.csp_rejections:
            rejections_str = "; ".join([f"'{rid}' ({reason})" for rid, reason in trace.csp_rejections.items()])
            explanation += f"4. Constraint Rejections Filtered: {rejections_str}."

        return explanation

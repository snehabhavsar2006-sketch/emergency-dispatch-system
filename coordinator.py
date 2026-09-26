"""
Coordinator Module (coordinator.py)
------------------------------------
Design Rationale & Viva Defense Notes:
- Central Orchestrator: Connects Detector, HDC Screener, CSP Validator, A* Router, 
  and Explainer into a unified sequential execution pipeline.
- Batch Dispatch Priority:
  - Simultaneous incidents are sorted descending by urgency_score (computed by IncidentDetector).
  - Incidents are processed sequentially. As soon as a responder is selected, it is marked 
    as unavailable (available=False) in ResponderManager before the next incident is dispatched.
  - This prevents double-allocation of emergency units.
- Trace Registry: Maintains history Dict[incident_id, DecisionTrace] enabling dynamic 
  re-planning and Streamlit dashboard visualization.
"""

from typing import List, Dict, Optional
from environment import GridCity, Incident, Responder
from incident_detector import IncidentDetector
from responder_manager import ResponderManager
from hdc_screener import HDCScreener
from csp_validator import CSPValidator
from astar_router import AStarRouter
from explainer import Explainer, DecisionTrace


class Coordinator:
    """
    Central dispatch system coordinator managing end-to-end pipeline execution.
    """

    def __init__(
        self,
        grid: Optional[GridCity] = None,
        responder_manager: Optional[ResponderManager] = None,
        hdc_dimension: int = 10000,
        hdc_top_n: int = 3,
        seed: int = 42,
    ):
        self.grid = grid if grid is not None else GridCity()
        self.responder_manager = (
            responder_manager if responder_manager is not None else ResponderManager()
        )
        self.hdc_screener = HDCScreener(dimension=hdc_dimension, seed=seed)
        self.csp_validator = CSPValidator()
        self.astar_router = AStarRouter()
        self.hdc_top_n = hdc_top_n
        self.history: Dict[str, DecisionTrace] = {}

    def save_trace(self, trace: DecisionTrace) -> None:
        """Store decision trace in central history registry."""
        self.history[trace.incident.incident_id] = trace

    def get_trace(self, incident_id: str) -> DecisionTrace:
        """Retrieve historical decision trace by incident_id."""
        if incident_id not in self.history:
            raise KeyError(f"No decision trace found for incident ID '{incident_id}'.")
        return self.history[incident_id]

    def dispatch_single(
        self, incident: Incident, log_trigger: Optional[str] = None
    ) -> DecisionTrace:
        """
        Executes the 3-stage dispatch pipeline for a single incident.
        Pipeline: HDC Candidate Screening -> CSP Constraint Validation -> A* Route Planning.
        """
        # Step 1: Fetch available responders
        available_responders = self.responder_manager.get_available_responders()

        # Step 2: HDC Candidate Screening
        shortlist_pairs, all_hdc_scores = self.hdc_screener.screen(
            incident, available_responders, top_n=self.hdc_top_n
        )
        shortlist_responders = [pair[0] for pair in shortlist_pairs]
        shortlist_ids = [resp.responder_id for resp in shortlist_responders]

        # Step 3: CSP Constraint Validation
        valid_responders, rejected_pairs = self.csp_validator.validate(
            incident, shortlist_responders
        )
        csp_valid_ids = [resp.responder_id for resp in valid_responders]
        csp_rejections = {resp.responder_id: reason for resp, reason in rejected_pairs}

        # Step 4: A* Route Planning
        ranked_candidates = self.astar_router.rank_candidates(
            incident, valid_responders, self.grid
        )

        astar_costs: Dict[str, float] = {}
        astar_paths: Dict[str, List] = {}
        for resp, path, cost in ranked_candidates:
            astar_costs[resp.responder_id] = cost
            astar_paths[resp.responder_id] = path

        # Step 5: Final Selection & Availability Update
        selected_responder: Optional[Responder] = None
        if ranked_candidates:
            selected_responder = ranked_candidates[0][0]
            self.responder_manager.set_availability(selected_responder.responder_id, False)

        # Build DecisionTrace
        trace = DecisionTrace(
            incident=incident,
            hdc_scores=all_hdc_scores,
            hdc_shortlist=shortlist_ids,
            csp_valid=csp_valid_ids,
            csp_rejections=csp_rejections,
            astar_costs=astar_costs,
            astar_paths=astar_paths,
            selected_responder=selected_responder,
        )

        if log_trigger:
            trace.replan_history.append({
                "trigger": log_trigger,
                "stages_rerun": ["HDC Candidate Screening", "CSP Validation", "A* Route Planning"],
            })

        self.save_trace(trace)
        return trace

    def dispatch_batch(self, incidents: List[Incident]) -> List[DecisionTrace]:
        """
        Dispatches multiple simultaneous incidents ordered by urgency score.
        """
        # Calculate urgency score for ordering
        scored_incidents = [
            (inc, IncidentDetector.compute_urgency_score(inc)) for inc in incidents
        ]
        # Sort descending (highest urgency first)
        scored_incidents.sort(key=lambda item: item[1], reverse=True)

        traces: List[DecisionTrace] = []
        for inc, score in scored_incidents:
            trace = self.dispatch_single(inc)
            traces.append(trace)

        return traces

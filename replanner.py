"""
Dynamic Re-planning Module (replanner.py)
------------------------------------------
Design Rationale & Viva Defense Notes:
- Partial Pipeline Re-execution: In emergency response, conditions change dynamically 
  (e.g., a assigned unit suffers a breakdown, incident severity escalates, or a bridge closes). 
  Re-running the entire pipeline for minor changes wastes compute; re-running only affected 
  stages demonstrates intelligent pipeline orchestration:
  1. RESPONDER_UNAVAILABLE: Re-runs CSP + A* on the existing HDC shortlist (excluding unavailable responder).
  2. SEVERITY_CHANGE: Re-runs full pipeline (HDC -> CSP -> A*) because severity affects vector encoding & capacity constraints.
  3. ROUTE_BLOCKED: Re-runs A* route planning on existing CSP-valid responders.
- Stage Logging: Logs exact stages re-run for UI visualization and empirical verification.
"""

import time
from typing import Dict, List, Any, Optional
from environment import Incident, Responder, GridCity
from explainer import DecisionTrace


class Replanner:
    """
    Handles dynamic re-planning triggers by selectively re-executing pipeline stages.
    """

    @staticmethod
    def handle_responder_unavailable(
        incident_id: str,
        unavailable_responder_id: str,
        coordinator: Any,  # Coordinator instance
    ) -> DecisionTrace:
        """
        Trigger: RESPONDER_UNAVAILABLE
        Action: Re-runs CSP Validation + A* Route Planning only on existing HDC shortlist minus unavailable unit.
        """
        old_trace: DecisionTrace = coordinator.get_trace(incident_id)
        incident = old_trace.incident

        # Update responder manager state
        coordinator.responder_manager.set_availability(unavailable_responder_id, False)

        # Retrieve available candidates from existing HDC shortlist
        shortlist_ids = [rid for rid in old_trace.hdc_shortlist if rid != unavailable_responder_id]
        shortlist_responders = [
            coordinator.responder_manager.get_responder(rid) 
            for rid in shortlist_ids 
            if coordinator.responder_manager.get_responder(rid).available
        ]

        # Re-run CSP Validation
        valid_responders, rejected = coordinator.csp_validator.validate(incident, shortlist_responders)
        csp_rejections = dict(old_trace.csp_rejections)
        for r, reason in rejected:
            csp_rejections[r.responder_id] = reason

        # Re-run A* Route Planning
        ranked_candidates = coordinator.astar_router.rank_candidates(
            incident, valid_responders, coordinator.grid
        )

        # Build updated DecisionTrace
        astar_costs = {}
        astar_paths = {}
        for resp, path, cost in ranked_candidates:
            astar_costs[resp.responder_id] = cost
            astar_paths[resp.responder_id] = path

        selected_responder = ranked_candidates[0][0] if ranked_candidates else None

        # Log replan event
        replan_log = {
            "timestamp": time.time(),
            "trigger": "RESPONDER_UNAVAILABLE",
            "details": f"Responder {unavailable_responder_id} marked unavailable.",
            "stages_rerun": ["CSP Validation", "A* Route Planning"],
        }

        new_history = list(old_trace.replan_history) + [replan_log]

        new_trace = DecisionTrace(
            incident=incident,
            hdc_scores=old_trace.hdc_scores,
            hdc_shortlist=old_trace.hdc_shortlist,
            csp_valid=[r.responder_id for r in valid_responders],
            csp_rejections=csp_rejections,
            astar_costs=astar_costs,
            astar_paths=astar_paths,
            selected_responder=selected_responder,
            replan_history=new_history,
        )

        coordinator.save_trace(new_trace)
        return new_trace

    @staticmethod
    def handle_severity_change(
        incident_id: str,
        new_severity: int,
        coordinator: Any,  # Coordinator instance
    ) -> DecisionTrace:
        """
        Trigger: SEVERITY_CHANGE
        Action: Severity alters HDC hypervector encoding and CSP capacity demands; 
                re-runs full pipeline (HDC Candidate Screening -> CSP Validation -> A* Route Planning).
        """
        old_trace: DecisionTrace = coordinator.get_trace(incident_id)
        incident = old_trace.incident
        incident.severity = max(1, min(5, new_severity))

        # Re-run full dispatch pipeline
        new_trace = coordinator.dispatch_single(incident, log_trigger="SEVERITY_CHANGE")
        
        replan_log = {
            "timestamp": time.time(),
            "trigger": "SEVERITY_CHANGE",
            "details": f"Incident {incident_id} severity updated to {new_severity}.",
            "stages_rerun": ["HDC Candidate Screening", "CSP Validation", "A* Route Planning"],
        }
        new_trace.replan_history = list(old_trace.replan_history) + [replan_log]
        coordinator.save_trace(new_trace)
        return new_trace

    @staticmethod
    def handle_route_blocked(
        incident_id: str,
        blocked_x: int,
        blocked_y: int,
        coordinator: Any,  # Coordinator instance
    ) -> DecisionTrace:
        """
        Trigger: ROUTE_BLOCKED
        Action: Re-runs A* Route Planning only on existing CSP-valid responders list.
        """
        old_trace: DecisionTrace = coordinator.get_trace(incident_id)
        incident = old_trace.incident

        # Block cell in grid
        coordinator.grid.block_cell(blocked_x, blocked_y)

        # Retrieve existing CSP-valid responders
        csp_valid_responders = [
            coordinator.responder_manager.get_responder(rid) 
            for rid in old_trace.csp_valid
        ]

        # Re-run A* Route Planning only
        ranked_candidates = coordinator.astar_router.rank_candidates(
            incident, csp_valid_responders, coordinator.grid
        )

        astar_costs = {}
        astar_paths = {}
        for resp, path, cost in ranked_candidates:
            astar_costs[resp.responder_id] = cost
            astar_paths[resp.responder_id] = path

        selected_responder = ranked_candidates[0][0] if ranked_candidates else None

        replan_log = {
            "timestamp": time.time(),
            "trigger": "ROUTE_BLOCKED",
            "details": f"Grid cell ({blocked_x}, {blocked_y}) blocked.",
            "stages_rerun": ["A* Route Planning"],
        }

        new_history = list(old_trace.replan_history) + [replan_log]

        new_trace = DecisionTrace(
            incident=incident,
            hdc_scores=old_trace.hdc_scores,
            hdc_shortlist=old_trace.hdc_shortlist,
            csp_valid=old_trace.csp_valid,
            csp_rejections=old_trace.csp_rejections,
            astar_costs=astar_costs,
            astar_paths=astar_paths,
            selected_responder=selected_responder,
            replan_history=new_history,
        )

        coordinator.save_trace(new_trace)
        return new_trace

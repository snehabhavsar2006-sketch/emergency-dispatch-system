"""
Demo Scenario Script (demo.py)
------------------------------
Design Rationale & Viva Defense Notes:
- Realistic & Reproducible Test Scenario: Runs 3 simultaneous incidents across a 
  10x10 city grid with 5 generated responders.
- Fixed Seed Verification: Uses generate_responders(5, seed=FIXED_SEED) to guarantee 
  that at least one feasible responder exists for each incident while leaving 
  unsuitable candidates to demonstrate CSP rejections and HDC screening.
- Comprehensive Console Logging: Prints intermediate outputs for all 3 pipeline stages:
  1. Urgency Score Processing Order
  2. HDC Candidate Similarity Scores & Shortlisting (top_n=3)
  3. CSP Hard Constraint Validation & Rejection Reasons
  4. A* Route Paths & Terrain-Aware Traversal Costs
  5. Explainer Plain-Language Output
- Dashboard Integration: Serves as a standalone console validator and launches 
  or pre-loads state for the Streamlit dashboard.
"""

import os
import sys
import time
from typing import List
from environment import GridCity, Incident, Responder, generate_responders
from responder_manager import ResponderManager
from coordinator import Coordinator
from explainer import Explainer


def find_feasible_demo_seed(num_responders: int = 5) -> int:
    """
    Finds a deterministic random seed for generate_responders that guarantees 
    at least one CSP-valid, capacity-sufficient responder for each demo incident 
    while retaining unsuitable candidates for screening demonstration.
    """
    from csp_validator import CSPValidator
    csp = CSPValidator()

    demo_incidents = [
        Incident(incident_id="INC-001", type="fire", location=(2, 8), severity=5, timestamp=time.time()),
        Incident(incident_id="INC-002", type="accident", location=(7, 1), severity=3, timestamp=time.time() - 60),
        Incident(incident_id="INC-003", type="medical", location=(5, 5), severity=2, timestamp=time.time() - 120),
    ]

    for seed in range(1, 1000):
        responders = generate_responders(num_responders, seed=seed)
        # Check if every incident has at least one CSP-valid candidate in pool
        feasible = True
        for inc in demo_incidents:
            valid, _ = csp.validate(inc, responders)
            if not valid:
                feasible = False
                break
        if feasible:
            return seed
    return 42  # Fallback seed


def run_demo():
    print("==================================================================")
    print("  INTELLIGENT EMERGENCY DISPATCH SYSTEM — DEMONSTRATION RUNNER  ")
    print("==================================================================\n")

    # 1. Initialize Grid City
    grid = GridCity(width=10, height=10)
    # Set a congested terrain area for demonstration
    grid.set_terrain_cost(4, 5, 3.0)
    grid.set_terrain_cost(5, 6, 2.5)

    # 2. Define 3 Simultaneous Demo Incidents
    now = time.time()
    incidents = [
        Incident(
            incident_id="INC-001",
            type="fire",
            location=(2, 8),
            severity=5,
            timestamp=now,  # Reported just now
        ),
        Incident(
            incident_id="INC-002",
            type="accident",
            location=(7, 1),
            severity=3,
            timestamp=now - 60,  # Reported 1 min ago
        ),
        Incident(
            incident_id="INC-003",
            type="medical",
            location=(5, 5),
            severity=2,
            timestamp=now - 180,  # Reported 3 mins ago
        ),
    ]

    # 3. Generate 5 Responders using Feasible Seed
    demo_seed = find_feasible_demo_seed(num_responders=5)
    responders = generate_responders(n=5, seed=demo_seed)
    responder_mgr = ResponderManager(responders)

    print(f"[ENVIRONMENT SETUP] Initialized 10x10 GridCity.")
    print(f"[ENVIRONMENT SETUP] Generated 5 Responders (Seed={demo_seed}):")
    for resp in responders:
        print(f"  - {resp.responder_id}: Type={resp.type:<12} Loc={resp.location} Capacity={resp.capacity} Available={resp.available}")
    print("\n------------------------------------------------------------------")

    # 4. Initialize Coordinator
    coordinator = Coordinator(grid=grid, responder_manager=responder_mgr, hdc_top_n=3, seed=42)

    # 5. Execute Batch Dispatch (Prioritized by Urgency Score)
    print("\n[STAGE 1: INCIDENT DETECTION & URGENCY PRIORITIZATION]")
    for inc in incidents:
        from incident_detector import IncidentDetector
        score = IncidentDetector.compute_urgency_score(inc, current_time=now)
        print(f"  - {inc.incident_id} ({inc.type.upper()}, Severity {inc.severity}, Loc {inc.location}) -> Urgency Score = {score:.2f}")

    print("\n[EXECUTING SEQUENTIAL DISPATCH PIPELINE...]\n")
    traces = coordinator.dispatch_batch(incidents)

    # 6. Display Pipeline Traces for All Incidents
    for idx, trace in enumerate(traces, 1):
        inc = trace.incident
        print(f"==================================================================")
        print(f" INCIDENT #{idx}: {inc.incident_id} ({inc.type.upper()} at {inc.location}, Severity {inc.severity})")
        print(f"==================================================================")

        # Stage 1: HDC Screening Scores
        print("\n--- STAGE 1: HDC Candidate Screening (D=10,000 Bipolar Vector Similarity) ---")
        for rid, sim in sorted(trace.hdc_scores.items(), key=lambda x: x[1], reverse=True):
            shortlisted_flag = " [SHORTLISTED]" if rid in trace.hdc_shortlist else ""
            print(f"  Responder {rid:<8} Vector Cosine Similarity = {sim:+.4f}{shortlisted_flag}")

        # Stage 2: CSP Validation
        print("\n--- STAGE 2: CSP Constraint Validation ---")
        print(f"  Valid Candidates Passing All Rules : {trace.csp_valid}")
        if trace.csp_rejections:
            print("  Rejected Candidates & Reasons:")
            for rid, reason in trace.csp_rejections.items():
                print(f"    - {rid}: Rejection Reason = '{reason}'")

        # Stage 3: A* Route Planning
        print("\n--- STAGE 3: A* Route Planning ---")
        if trace.astar_costs:
            for rid, cost in trace.astar_costs.items():
                path = trace.astar_paths.get(rid, [])
                print(f"  Responder {rid:<8} Path = {path} | Total Terrain Cost = {cost:.1f} units")
        else:
            print("  No routable candidate available.")

        # Stage 4: Explainer Output
        print("\n--- STAGE 4: Plain-Language Explanation ---")
        explanation = Explainer.generate_explanation(trace)
        print(explanation)
        print("\n")

    print("==================================================================")
    print("  DEMO COMPLETED SUCCESSFULLY.")
    print("  To launch the interactive dashboard, run:")
    print("  streamlit run dashboard.py")
    print("==================================================================\n")


if __name__ == "__main__":
    run_demo()

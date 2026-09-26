"""
Evaluation Suite (evaluate.py)
------------------------------
Design Rationale & Viva Defense Notes:
- Empirical Integrity Requirement: No metric is hardcoded or fabricated. 
  All measurements (timing, candidate counts, retention rates, memory bytes) 
  are computed dynamically at runtime across simulation runs.
- Baseline Reference & Retention Metric:
  - HDC is a candidate screener (shortlisting top_n=3 out of N responders).
  - The 'Baseline Pipeline' runs CSP validation + A* route planning on the ENTIRE 
    responder pool (bypassing HDC candidate screening).
  - Retention measures whether the baseline-selected optimal responder was present 
    in HDC's shortlisted top_n:
    Retention (%) = (cases where baseline-selected responder is in HDC shortlist / total evaluated cases) * 100
- Direct Memory Measurement: Item memory array bytes are queried directly via 
  NumPy's `.nbytes` attribute, providing exact RAM footprint metrics.
"""

import time
import sys
import numpy as np
from typing import Dict, Any, List
from environment import GridCity, generate_incidents, generate_responders
from responder_manager import ResponderManager
from hdc_screener import HDCScreener
from csp_validator import CSPValidator
from astar_router import AStarRouter
from coordinator import Coordinator


def run_evaluation(
    num_runs: int = 50,
    num_incidents_per_run: int = 3,
    num_responders_per_run: int = 10,
    top_n: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs empirical evaluation across multiple randomized simulation iterations.
    Returns aggregate metric statistics dictionary.
    """
    rng = np.random.default_rng(seed)

    hdc_screener = HDCScreener(dimension=10000, seed=seed)
    csp_validator = CSPValidator()
    astar_router = AStarRouter()

    total_matching_time_sec = 0.0
    total_incidents_evaluated = 0
    total_candidates_scored = 0
    total_pool_responders = 0
    valid_assignments_count = 0
    constraint_violations_caught = 0
    baseline_retention_hits = 0
    evaluable_baseline_cases = 0

    for run_idx in range(num_runs):
        run_seed = int(rng.integers(1, 100000))
        grid = GridCity()
        
        incidents = generate_incidents(num_incidents_per_run, seed=run_seed)
        responders = generate_responders(num_responders_per_run, seed=run_seed + 1)
        resp_mgr = ResponderManager(responders)

        for inc in incidents:
            avail_responders = resp_mgr.get_available_responders()
            if not avail_responders:
                continue

            total_incidents_evaluated += 1
            total_pool_responders += len(avail_responders)

            # 1. Measure HDC Screening Wall-Clock Time
            t0 = time.perf_counter()
            shortlist_pairs, all_hdc_scores = hdc_screener.screen(
                inc, avail_responders, top_n=top_n
            )
            t1 = time.perf_counter()
            total_matching_time_sec += (t1 - t0)
            total_candidates_scored += len(all_hdc_scores)

            shortlist_responders = [pair[0] for pair in shortlist_pairs]
            shortlist_ids = {resp.responder_id for resp in shortlist_responders}

            # 2. HDC -> CSP -> A* Pipeline Execution
            valid_screened, rejected_screened = csp_validator.validate(inc, shortlist_responders)
            constraint_violations_caught += len(rejected_screened)

            ranked_screened = astar_router.rank_candidates(inc, valid_screened, grid)
            if ranked_screened:
                valid_assignments_count += 1
                assigned_resp = ranked_screened[0][0]
                # Mark assigned responder unavailable for subsequent incidents in run
                resp_mgr.set_availability(assigned_resp.responder_id, False)

            # 3. Baseline Pipeline Reference (Bypassing HDC, evaluating ENTIRE pool)
            valid_full, _ = csp_validator.validate(inc, avail_responders)
            ranked_full = astar_router.rank_candidates(inc, valid_full, grid)

            if ranked_full:
                baseline_optimal_resp = ranked_full[0][0]
                evaluable_baseline_cases += 1
                if baseline_optimal_resp.responder_id in shortlist_ids:
                    baseline_retention_hits += 1

    # Aggregate metric calculations
    avg_matching_time_ms = (
        (total_matching_time_sec / max(1, total_incidents_evaluated)) * 1000.0
    )
    valid_assignment_pct = (
        (valid_assignments_count / max(1, total_incidents_evaluated)) * 100.0
    )
    retention_pct = (
        (baseline_retention_hits / max(1, evaluable_baseline_cases)) * 100.0
        if evaluable_baseline_cases > 0
        else 0.0
    )
    candidates_screened_ratio = (
        total_candidates_scored / max(1, total_pool_responders)
    )

    # 4. Direct Memory Measurement of Item Memory Hypervectors
    item_mem_bytes = sum(vec.nbytes for vec in hdc_screener.item_memory.values())

    # Measure dynamic entity encoding size for 1 Incident + 1 Responder
    sample_inc = generate_incidents(1, seed=1)[0]
    sample_resp = generate_responders(1, seed=1)[0]
    inc_vec = hdc_screener.encode_incident(sample_inc)
    resp_vec = hdc_screener.encode_responder(sample_resp)
    per_entity_vector_bytes = inc_vec.nbytes

    results = {
        "num_runs": num_runs,
        "total_incidents_evaluated": total_incidents_evaluated,
        "avg_matching_time_ms": avg_matching_time_ms,
        "valid_assignments_pct": valid_assignment_pct,
        "constraint_violations_caught": constraint_violations_caught,
        "baseline_retention_pct": retention_pct,
        "evaluable_baseline_cases": evaluable_baseline_cases,
        "baseline_retention_hits": baseline_retention_hits,
        "candidates_evaluated_avg": total_candidates_scored / max(1, total_incidents_evaluated),
        "total_pool_avg": total_pool_responders / max(1, total_incidents_evaluated),
        "item_memory_bytes": item_mem_bytes,
        "per_entity_vector_bytes": per_entity_vector_bytes,
    }
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print formatted Markdown evaluation table to stdout."""
    print("\n===========================================================")
    print("       INTELLIGENT EMERGENCY DISPATCH SYSTEM               ")
    print("              EMPIRICAL EVALUATION REPORT                  ")
    print("===========================================================\n")
    print(f"Total Simulation Runs        : {results['num_runs']}")
    print(f"Total Incidents Evaluated    : {results['total_incidents_evaluated']}\n")
    
    print("| Metric                             | Empirical Value              |")
    print("|------------------------------------|------------------------------|")
    print(f"| HDC Matching Wall-Clock Time       | {results['avg_matching_time_ms']:.3f} ms / incident        |")
    print(f"| Candidates Evaluated (HDC / Pool)  | {results['candidates_evaluated_avg']:.1f} / {results['total_pool_avg']:.1f} candidates    |")
    print(f"| Valid Dispatch Assignments         | {results['valid_assignments_pct']:.1f}%                        |")
    print(f"| CSP Constraint Violations Caught   | {results['constraint_violations_caught']} rejections               |")
    print(f"| Baseline-Selection Retention       | {results['baseline_retention_pct']:.1f}% ({results['baseline_retention_hits']}/{results['evaluable_baseline_cases']} cases)       |")
    print(f"| Item Memory RAM Usage              | {results['item_memory_bytes'] / 1024:.2f} KB                    |")
    print(f"| Per-Entity Vector Size             | {results['per_entity_vector_bytes'] / 1024:.2f} KB (D=10,000 int8)   |")
    print("\n===========================================================\n")


if __name__ == "__main__":
    metrics = run_evaluation(num_runs=50, num_incidents_per_run=3, num_responders_per_run=10, top_n=3, seed=42)
    print_evaluation_report(metrics)

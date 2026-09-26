# Intelligent Emergency Dispatch System

An academic prototype for emergency responder dispatching built using a **3-stage sequential architecture** in Python.

> **Project Framing & Core Philosophy**
> This is an *intelligent emergency dispatch system* made of three complementary stages solving three different sub-problems — it is not "an HDC project." HDC narrows the candidate search space, CSP validates which of those candidates are actually eligible, and A* determines route and cost for the eligible ones. No single stage is "the main algorithm." The stages work in tandem to support responder screening, constraint validation, route planning, and explainable dispatch decisions.
>
> - **HDC — Candidate Screening:** *"Which responders are potentially suitable?"*
> - **CSP — Constraint Validation:** *"Which of those responders are actually eligible?"*
> - **A\* — Route Planning:** *"How should the eligible responder reach the incident, and at what cost?"*

---

## 1. System Architecture & Module Overview

The codebase is organized into isolated, modular files to maintain clean separation of concerns:

```
emergency_dispatch_system/
├── environment.py         # GridCity (10x10), Incident & Responder schemas, seed-controlled generators
├── incident_detector.py   # Raw data validation, type normalization, urgency score calculation
├── responder_manager.py   # Registry for active responders & availability state management
├── hdc_screener.py        # Hyperdimensional computing candidate screener (D=10000 bipolar vectors)
├── csp_validator.py       # Constraint Satisfaction Problem validator (type, capacity, availability)
├── astar_router.py        # Grid A* shortest path & cost calculator
├── explainer.py          # DecisionTrace data structure & plain-language explanation generator
├── replanner.py          # Trigger-based dynamic re-planning (partial pipeline re-execution)
├── coordinator.py        # Central orchestrator & batch processing by urgency score
├── evaluate.py           # Empirical performance benchmarking & retention metric computation
├── dashboard.py          # Interactive Streamlit dashboard
├── demo.py               # Pre-loaded 3-incident scenario console runner & dashboard launcher
└── requirements.txt      # Python dependencies
```

---

## 2. Research Integrity Statement

All evaluation metrics, similarity scores, CSP validation outcomes, A* route costs, timing measurements, baseline retention rates, and memory bytes are **generated dynamically from the implemented algorithms at runtime**. No benchmark results or algorithm outputs are hardcoded, precomputed, or fabricated.

---

## 3. Pipeline Workflow

```
[Raw Incident Payload]
         │
         ▼
 1. Incident Detector  ───> Urgency Score (Determines processing order ONLY)
         │
         ▼
 2. HDC Screener       ───> Candidate Screening (D=10,000 hypervector similarity shortlist)
         │
         ▼
 3. CSP Validator      ───> Constraint Validation (Hard rule compliance & rejection reasons)
         │
         ▼
 4. A* Router          ───> Route Planning (Terrain-aware grid shortest path & cost)
         │
         ▼
 5. Explainer          ───> Plain-Language Justification & DecisionTrace Audit Log
```

### Dynamic Re-Planning Semantics
When environment or operational conditions change, the **Replanner** re-executes only the necessary pipeline stages:
- **`RESPONDER_UNAVAILABLE`**: Re-runs **CSP Validation** $\rightarrow$ **A\* Route Planning** on existing HDC shortlist.
- **`SEVERITY_CHANGE`**: Re-runs full pipeline (**HDC Screener** $\rightarrow$ **CSP Validator** $\rightarrow$ **A\* Router**) because severity alters vector encodings and capacity demands.
- **`ROUTE_BLOCKED`**: Re-runs **A\* Route Planning** only on existing CSP-valid candidates.

---

## 4. How to Run

### Prerequisite Installation
```bash
pip install -r requirements.txt
```

### 1. Run Demo Scenario (Console Output)
Runs a 3-incident, 5-responder simulation scenario, printing intermediate stage logs and plain-language justifications:
```bash
python demo.py
```

### 2. Run Empirical Evaluation Benchmarks
Executes 50 randomized simulation runs and outputs empirical timing, retention, and memory metrics:
```bash
python evaluate.py
```

### 3. Launch Interactive Streamlit Dashboard
Launches the web UI for grid map visualization, candidate ranking tables, dynamic re-planning controls, and live metrics:
```bash
streamlit run dashboard.py
```

---

## 5. SETUP GOOGLE MAPS API

For the role-based application, the map navigation links use Google Maps directions when available. This is used for real-world navigation support, not to replace the A* route-planning research model.

1. Create a Google Cloud project and enable the required Maps APIs.
2. Generate a Google Maps API key.
3. Store it in Streamlit secrets or an environment variable.
4. Example `.streamlit/secrets.toml`:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"
```

5. Run the app with:

```bash
streamlit run dashboard.py
```

> Do not hardcode the key in Python source code. Never print the key in logs or UI output.

---

## 6. Viva Defense Preparation Guide

### Q1: Why use a 3-stage architecture instead of applying A* directly to all responders?
**Answer:** In large urban emergency networks with thousands of units, running full A* graph search for every unit against every incident is computationally expensive. HDC acts as a ultra-fast vector screener ($O(D)$ dot product) to filter down candidates. CSP then eliminates ineligible candidates in $O(1)$ rule checks. Finally, A* runs only on the small remaining eligible pool, minimizing compute while guaranteeing hard constraint safety and physical route optimality.

### Q2: Is HDC "the main algorithm" in this project?
**Answer:** No. HDC is strictly a candidate screener. HDC alone cannot guarantee 100% hard constraint compliance because vector similarity is probabilistic. CSP enforces hard constraint compliance, and A* determines the exact physical path and traversal cost. No single stage is "the main algorithm"—they form a complementary pipeline.

### Q3: Why is incident severity NOT multiplied into A* route cost?
**Answer:** A* route cost represents physical travel cost (distance and terrain traversal time). Multiplying route cost by severity would create an arbitrary mathematical scaling factor with no real-world physical justification. Instead, incident severity is used by `IncidentDetector` to compute `urgency_score`, which dictates the **batch processing order** of simultaneous incidents.

### Q4: How does HDC encode spatial coordinates without arbitrary parameters?
**Answer:** Coordinate hypervectors ($V_X$ for $x$, $V_Y$ for $y$) are generated using continuous angle interpolation between two orthogonal basis hypervectors:
$$\mathbf{v}_x = \cos\left(\frac{\pi}{2} \cdot \frac{x}{9}\right) \mathbf{u}_x + \sin\left(\frac{\pi}{2} \cdot \frac{x}{9}\right) \mathbf{v}_x, \quad V_{Xx} = \text{sign}(\mathbf{v}_x)$$
This guarantees that cosine similarity degrades smoothly as physical distance $|x_1 - x_2|$ increases, preserving 2D spatial locality without introducing ad-hoc thresholds.

### Q5: What is the Baseline-Selection Retention Metric?
**Answer:** Baseline retention measures whether the optimal responder selected by running CSP+A* over the *entire* responder pool (bypassing HDC) is present in HDC's top-N shortlist:
$$\text{Retention (\%)} = \left(\frac{\text{Cases where baseline-selected responder is in HDC shortlist}}{\text{Total evaluated cases}}\right) \times 100$$
This evaluates HDC's recall accuracy as a candidate screener.

### Q6: Are the capacity rules and urgency formulas real-world emergency standards?
**Answer:** No. Formulas like `responder.capacity >= incident.severity` and `urgency_score = severity * 10 + waiting_time` are simplified prototype simulation rules designed to model unit capacity and queuing priority in a controlled environment.

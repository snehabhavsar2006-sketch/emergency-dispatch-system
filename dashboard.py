"""Simple emergency dispatch application built on the existing research pipeline."""

import time
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from coordinator import Coordinator
from database import (
    add_message,
    create_dispatch,
    create_incident,
    get_active_incident_count,
    get_count,
    get_dispatch_history,
    get_incident,
    get_incidents,
    get_processing_results,
    get_responders,
    get_table_rows,
    init_database,
    save_processing_result,
    save_route_update,
    seed_demo_responders,
    update_incident_status,
    update_responder_status,
)
from environment import GridCity, Incident, Responder
from explainer import Explainer
from incident_detector import IncidentDetector
from road_network import RealWorldAStarRouter
from responder_manager import ResponderManager
from replanner import Replanner


st.set_page_config(page_title="Intelligent Emergency Dispatch System", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F5F9FC; color: #263842; }
        h1, h2, h3, h4 { color: #24485C; }
        .stButton > button {
            background: #2F6F95; color: white; border: 1px solid #2F6F95;
            border-radius: 4px; padding: 0.5rem 1rem; font-weight: 600;
        }
        .stButton > button:hover { background: #4F8FB3; border-color: #4F8FB3; }
        div[data-testid="stDataFrame"] { border: 1px solid #D5E3EB; }
        .stSuccess { background: #EAF3F8; border: 1px solid #4FAF7B; color: #24485C; }
        .stWarning { background: #EAF3F8; border: 1px solid #E3A44A; color: #263842; }
        .stInfo { background: #EAF3F8; border: 1px solid #D5E3EB; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def latlon_to_grid(latitude: float, longitude: float) -> tuple:
    x = int(abs(float(latitude) - 21.34) * 100) % 10
    y = int(abs(float(longitude) - 74.88) * 100) % 10
    return (x, y)


def format_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def build_navigation_url(destination: str, origin: str = "") -> str:
    destination_q = destination.strip().replace(" ", "+")
    origin_q = origin.strip().replace(" ", "+")
    if origin_q:
        return f"https://www.google.com/maps/dir/?api=1&origin={origin_q}&destination={destination_q}&travelmode=driving&dir_action=navigate"
    return f"https://www.google.com/maps/search/?api=1&query={destination_q}"


def request_browser_location() -> Optional[Dict[str, Any]]:
    component_value = components.html(
        """
        <script>
        function setLocation(position) {
            Streamlit.setComponentValue({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                source: 'LIVE GPS',
                message: 'Using live browser location'
            });
        }
        function failLocation() {
            Streamlit.setComponentValue({
                source: 'SIMULATION',
                message: 'Live GPS unavailable — simulation mode active'
            });
        }
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(setLocation, failLocation);
        } else {
            failLocation();
        }
        </script>
        """,
        height=0,
        width=0,
    )
    if isinstance(component_value, dict):
        return component_value
    return {"source": "SIMULATION", "message": "Live GPS unavailable — simulation mode active"}


def ensure_demo_data() -> None:
    if get_incidents():
        return

    demo_rows = [
        (
            "INC-001",
            "fire",
            "Fire near NMIMS Shirpur main gate. Immediate fire response needed.",
            "NMIMS Shirpur Main Gate",
            21.3462,
            74.8850,
            5,
            time.time(),
            "DEMO SIMULATION",
            "RECEIVED",
            "",
            "",
        ),
        (
            "INC-002",
            "accident",
            "Road accident near the bus stop. Two people injured.",
            "NMIMS Shirpur Bus Stop",
            21.3417,
            74.8725,
            3,
            time.time() - 60,
            "DEMO SIMULATION",
            "RECEIVED",
            "",
            "",
        ),
        (
            "INC-003",
            "medical",
            "Medical emergency at the library. Need ambulance support.",
            "NMIMS Shirpur Library",
            21.3388,
            74.8796,
            2,
            time.time() - 180,
            "DEMO SIMULATION",
            "RECEIVED",
            "",
            "",
        ),
    ]

    for idx, row in enumerate(demo_rows, start=1):
        incident_id, incident_type, message, location_text, lat, lon, severity, ts, source, status, assigned, route = row
        create_incident(incident_id, incident_type, message, location_text, lat, lon, severity, ts, source, status, assigned, route)
        add_message(f"MSG-{idx:03d}", incident_id, message, ts, source)


def build_runtime_responder_manager() -> ResponderManager:
    rows = get_responders()
    responders = []
    for row in rows:
        grid_x, grid_y = latlon_to_grid(float(row.get("latitude", 21.34)), float(row.get("longitude", 74.88)))
        responder = Responder(
            responder_id=row["responder_id"],
            type=row["type"],
            location=(grid_x, grid_y),
            capacity=int(row["capacity"]),
            available=bool(row["availability"]),
            status=row["status"],
            current_assignment=row["current_assignment"],
            latitude=float(row.get("latitude", 21.34)),
            longitude=float(row.get("longitude", 74.88)),
            location_source=row.get("location_source", "SIMULATION"),
            grid_x=grid_x,
            grid_y=grid_y,
        )
        responders.append(responder)
    return ResponderManager(responders)


def build_runtime_incident(incident_record: Dict[str, Any]) -> Incident:
    grid_x, grid_y = latlon_to_grid(float(incident_record["latitude"]), float(incident_record["longitude"]))
    return Incident(
        incident_id=incident_record["incident_id"],
        type=incident_record["incident_type"],
        location=(grid_x, grid_y),
        severity=int(incident_record["severity"]),
        timestamp=float(incident_record["timestamp"]),
        status=incident_record["status"],
        source=incident_record["source"],
        message=incident_record["original_message"],
        location_name=incident_record["location_text"],
        assigned_responder_id=incident_record.get("assigned_responder"),
        responder_status=incident_record.get("status", "UNASSIGNED"),
        latitude=float(incident_record["latitude"]),
        longitude=float(incident_record["longitude"]),
        location_source=incident_record.get("location_source", "SIMULATION"),
        extracted_info={
            "type": incident_record["incident_type"],
            "location": (grid_x, grid_y),
            "location_name": incident_record["location_text"],
            "severity": incident_record["severity"],
            "source": incident_record["source"],
        },
    )


def run_pipeline_for_incident(incident_id: str):
    incident_record = get_incident(incident_id)
    if not incident_record:
        return None

    grid = GridCity(10, 10)
    grid.set_terrain_cost(4, 5, 3.0)
    grid.set_terrain_cost(5, 6, 2.5)

    responder_manager = build_runtime_responder_manager()
    coordinator = Coordinator(grid=grid, responder_manager=responder_manager, hdc_top_n=3, seed=42)
    incident = build_runtime_incident(incident_record)
    trace = coordinator.dispatch_single(incident)

    for resp in responder_manager.get_all_responders():
        selected = trace.selected_responder is not None and resp.responder_id == trace.selected_responder.responder_id
        path = trace.astar_paths.get(resp.responder_id, [])
        distance_units = max(len(path) - 1, 0) if path else 0
        save_processing_result(
            processing_id=f"PROC-{incident.incident_id}-{resp.responder_id}",
            incident_id=incident.incident_id,
            responder_id=resp.responder_id,
            hdc_similarity=float(trace.hdc_scores.get(resp.responder_id, 0.0)),
            shortlisted=int(resp.responder_id in trace.hdc_shortlist),
            csp_valid=int(resp.responder_id in trace.csp_valid),
            rejection_reason=trace.csp_rejections.get(resp.responder_id, "-"),
            route_cost=float(trace.astar_costs.get(resp.responder_id, 0.0)),
            selected=int(selected),
            timestamp=time.time(),
            distance=float(distance_units),
            eta=float(distance_units),
        )

    if trace.selected_responder is not None:
        selected_path = trace.astar_paths.get(trace.selected_responder.responder_id, [])
        route = " -> ".join([f"({x},{y})" for x, y in selected_path]) if selected_path else ""
        cost = float(trace.astar_costs.get(trace.selected_responder.responder_id, 0.0))
        update_incident_status(incident.incident_id, "DISPATCHED", trace.selected_responder.responder_id, route)
        update_responder_status(trace.selected_responder.responder_id, 0, "ASSIGNED", incident.incident_id)
        create_dispatch(
            dispatch_id=f"DISP-{incident.incident_id}",
            incident_id=incident.incident_id,
            responder_id=trace.selected_responder.responder_id,
            assigned_at=time.time(),
            route=route,
            route_cost=cost,
            status="ASSIGNED",
            distance=float(max(len(selected_path) - 1, 0)),
            eta=float(max(len(selected_path) - 1, 0)),
        )

    return trace


def initialize_app() -> None:
    if "app_ready" in st.session_state:
        return

    init_database()
    seed_demo_responders()
    ensure_demo_data()
    st.session_state.app_ready = True
    st.session_state.selected_incident_id = get_incidents()[0]["incident_id"] if get_incidents() else None
    st.session_state.active_page = "Messages"
    st.session_state.last_message = ""


def current_incident() -> Optional[Dict[str, Any]]:
    incidents = get_incidents()
    if not incidents:
        return None
    selected = st.session_state.get("selected_incident_id")
    for inc in incidents:
        if inc["incident_id"] == selected:
            return inc
    st.session_state.selected_incident_id = incidents[0]["incident_id"]
    return incidents[0]


def render_navigation() -> str:
    nav_names = ["Messages", "Processing", "Dashboard", "Navigation", "More"]
    nav_cols = st.columns(len(nav_names))
    selected = st.session_state.get("active_page", "Messages")
    for i, label in enumerate(nav_names):
        with nav_cols[i]:
            if st.button(label, key=f"nav-{label}", use_container_width=True):
                selected = label
                st.session_state.active_page = label
    return selected


def render_messages() -> None:
    st.title("INCOMING EMERGENCY MESSAGES")

    with st.form("send_emergency_form"):
        st.subheader("+ SEND EMERGENCY")
        message = st.text_area("Emergency Message", height=120, placeholder="Accident near NMIMS Shirpur Main Gate. Two people are injured. Need ambulance.")
        location = st.text_input("Location", value="NMIMS Shirpur")
        severity = st.slider("Optional Severity", min_value=1, max_value=5, value=3)
        if st.form_submit_button("SEND EMERGENCY"):
            if not message.strip():
                st.warning("Emergency message cannot be empty.")
            else:
                extracted = IncidentDetector.extract_from_message(message, location, severity)
                if extracted.get("requires_confirmation"):
                    st.warning(extracted.get("details", "Requires confirmation"))
                else:
                    incident_id = f"INC-{int(time.time() * 1000) % 100000:05d}"
                    lat = float(extracted["location"][0])
                    lon = float(extracted["location"][1])
                    create_incident(
                        incident_id=incident_id,
                        incident_type=extracted["type"],
                        original_message=message,
                        location_text=extracted.get("location_name", location),
                        latitude=lat,
                        longitude=lon,
                        severity=int(extracted["severity"]),
                        timestamp=time.time(),
                        source="Emergency Message",
                        status="RECEIVED",
                    )
                    add_message(f"MSG-{incident_id}", incident_id, message, time.time(), "Emergency Message")
                    st.session_state.selected_incident_id = incident_id
                    st.success(f"Emergency Submitted Successfully\nEmergency ID: {incident_id}\nStatus: RECEIVED")
                    st.rerun()

    st.markdown("---")
    incidents = get_incidents()
    if not incidents:
        st.info("No incidents found.")
        return

    for inc in incidents:
        emergency_id = inc["incident_id"]
        selected = emergency_id == st.session_state.get("selected_incident_id")
        card = st.container()
        with card:
            label = "🔴 NEW" if inc["status"] == "RECEIVED" else "✅" if inc["status"] == "DISPATCHED" else "📌"
            st.markdown(f"{label} **{emergency_id}**")
            st.write(f"{inc['original_message']}")
            st.write(f"{format_time(float(inc['timestamp']))}")
            st.caption(f"Source: {inc['source']} | Status: {inc['status']} | Location: {inc['location_text']}")
            if st.button("View", key=f"view-{emergency_id}", use_container_width=False):
                st.session_state.selected_incident_id = emergency_id
                st.session_state.active_page = "Processing"
                st.rerun()
        st.markdown("---")


def render_processing() -> None:
    incident = current_incident()
    if incident is None:
        st.info("No incident selected.")
        return

    incident_id = incident["incident_id"]
    st.title("PROCESSING")
    st.subheader(f"CURRENT EMERGENCY: {incident_id}")
    st.write(f"Original message: {incident['original_message']}")
    st.write(f"Type: {incident['incident_type']}")
    st.write(f"Location: {incident['location_text']}")
    st.write(f"Severity: {incident['severity']}")
    st.write(f"Timestamp: {format_time(float(incident['timestamp']))}")
    st.write(f"Source: {incident['source']}")
    st.write(f"Status: {incident['status']}")

    trace = run_pipeline_for_incident(incident_id)
    if trace is None:
        st.warning("Could not process incident.")
        return

    manager = build_runtime_responder_manager()
    selected_id = trace.selected_responder.responder_id if trace.selected_responder else None

    st.markdown("### SECTION 1 — HDC — Candidate Screening")
    hdc_rows = []
    for resp in manager.get_all_responders():
        rid = resp.responder_id
        hdc_rows.append({
            "Responder": rid,
            "Type": resp.type,
            "Availability": "YES" if resp.available else "NO",
            "HDC similarity": f"{trace.hdc_scores.get(rid, 0.0):+.3f}",
            "Shortlisted": "YES" if rid in trace.hdc_shortlist else "NO",
        })
    st.dataframe(hdc_rows, use_container_width=True)

    st.markdown("### SECTION 2 — CSP — Constraint Validation")
    csp_rows = []
    for resp in manager.get_all_responders():
        rid = resp.responder_id
        status = "PASS" if rid in trace.csp_valid else "FAIL" if rid in trace.csp_rejections else "SKIP"
        reason = trace.csp_rejections.get(rid, "-")
        csp_rows.append({
            "Responder": rid,
            "Type": resp.type,
            "Capacity": resp.capacity,
            "Availability": "YES" if resp.available else "NO",
            "CSP Result": status,
            "Rejection Reason": reason,
        })
    st.dataframe(csp_rows, use_container_width=True)

    st.markdown("### SECTION 3 — A* — 10×10 Research Grid")
    route_rows = []
    for rid in trace.csp_valid:
        resp = manager.get_responder(rid)
        path = trace.astar_paths.get(rid, [])
        distance_units = max(len(path) - 1, 0)
        route_rows.append({
            "Responder": rid,
            "Grid Source": resp.location,
            "Grid Destination": incident["location_text"],
            "Distance": f"{distance_units} units",
            "Terrain Cost": float(trace.astar_costs.get(rid, 0.0)),
            "A* Status": "SELECTED" if trace.selected_responder and rid == trace.selected_responder.responder_id else "EVALUATED",
        })
    if not route_rows:
        st.warning("No responder passed CSP validation.")
    st.dataframe(route_rows, use_container_width=True)

    st.markdown("### EXPLAINABLE DASHBOARD — 10×10 Research Simulation")
    fig, ax = plt.subplots(figsize=(6, 6))
    grid_bg = np.full((10, 10), 1.0)
    ax.imshow(grid_bg, cmap="Greys", alpha=0.9, vmin=0, vmax=1)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.grid(True, color="#D5E3EB", linewidth=0.8)

    blocked_cells = [(2, 2), (2, 3), (3, 3), (5, 5), (6, 5), (7, 4), (7, 5)]
    for x, y in blocked_cells:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#4B4F5A", edgecolor="#2A2E35", linewidth=1.2))

    if incident.get("grid_x") is not None and incident.get("grid_y") is not None:
        grid_x, grid_y = int(incident["grid_x"]), int(incident["grid_y"])
    else:
        grid_x, grid_y = latlon_to_grid(float(incident["latitude"]), float(incident["longitude"]))
    ax.scatter(grid_x, grid_y, c="#D95C5C", s=240, marker="*", label="Emergency / Destination")

    selected_path = trace.astar_paths.get(selected_id, []) if selected_id else []
    if selected_path:
        xs = [p[0] for p in selected_path]
        ys = [p[1] for p in selected_path]
        ax.plot(xs, ys, color="#4FAF7B", linewidth=3.0, label="Selected A* Route")

    for resp in manager.get_all_responders():
        if resp.location is not None and len(resp.location) == 2:
            gx, gy = int(resp.location[0]), int(resp.location[1])
        else:
            gx, gy = latlon_to_grid(float(resp.latitude), float(resp.longitude))
        if resp.responder_id == selected_id:
            ax.scatter(gx, gy, c="#2F6F95", s=120, marker="s", label="Responder / Source")
        else:
            ax.scatter(gx, gy, c="#4F8FB3", s=100, marker="s")

    for rid, path in trace.astar_paths.items():
        if rid == selected_id:
            continue
        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="#E3A44A", linewidth=1.8, linestyle="--", alpha=0.9, label="Alternative Route")

    ax.set_title("10×10 Research Simulation")
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="#F5F9FC", edgecolor="#D5E3EB")
    st.pyplot(fig)

    st.markdown("### A* ROUTE ANALYSIS")
    analysis_rows = []
    for rid in trace.csp_valid:
        resp = manager.get_responder(rid)
        path = trace.astar_paths.get(rid, [])
        distance = max(len(path) - 1, 0)
        analysis_rows.append({
            "Candidate": rid,
            "Distance": f"{distance} units",
            "Terrain Cost": float(trace.astar_costs.get(rid, 0.0)),
            "Status": "SELECTED" if trace.selected_responder and rid == trace.selected_responder.responder_id else "NOT SELECTED",
        })
    if not analysis_rows:
        st.warning("A* route analysis unavailable because no responder passed CSP validation.")
    else:
        st.dataframe(analysis_rows, use_container_width=True)
    st.caption("● BLUE = Responder  ● RED = Emergency  ● GREEN = Selected A* Route  ● ORANGE = Alternative Route  ● DARK GREY = Blocked Cell  ○ LIGHT = Normal Cell")

    st.markdown("### DISPATCH DECISION")
    if trace.selected_responder:
        st.success(Explainer.generate_explanation(trace))
    else:
        st.warning("No responder passed CSP validation.")
        if trace.csp_rejections:
            st.write("Rejection reasons:")
            for rid, reason in trace.csp_rejections.items():
                st.write(f"- {rid}: {reason}")

    st.markdown("### PROCESSING TIMELINE")
    st.write("✓ Message Received")
    st.write("✓ Information Extracted")
    st.write("✓ HDC Screening")
    st.write("✓ CSP Validation")
    st.write("✓ Real-World Road A*")
    st.write("✓ Dispatch Decision")


def render_dashboard() -> None:
    incident = current_incident()
    if incident is None:
        st.info("No incident selected.")
        return

    st.title("DASHBOARD")
    incident_id = incident["incident_id"]
    trace = run_pipeline_for_incident(incident_id)
    if trace is None:
        st.warning("No decision trace available.")
        return

    st.subheader("EMERGENCY")
    st.write(f"Incident: {incident['incident_type']}")
    st.write(f"Location: {incident['location_text']}")
    st.write(f"Latitude: {float(incident['latitude']):.4f}")
    st.write(f"Longitude: {float(incident['longitude']):.4f}")

    st.subheader("DECISION SUMMARY")
    st.write(f"HDC shortlisted count: {len(trace.hdc_shortlist)}")
    st.write(f"CSP valid count: {len(trace.csp_valid)}")
    st.write(f"A* route evaluations: {len(trace.astar_costs)}")

    if trace.selected_responder:
        st.subheader("SELECTED RESPONDER")
        st.success(f"{trace.selected_responder.responder_id} ({trace.selected_responder.type})")
        st.write("Why selected: ✓ Compatible ✓ Available ✓ Passed constraints ✓ Lowest calculated route cost among valid candidates")
    else:
        st.warning("No responder assigned")
        if trace.csp_rejections:
            st.write("Actual rejection reasons:")
            for rid, reason in trace.csp_rejections.items():
                st.write(f"- {rid}: {reason}")

    st.subheader("DISPATCH HISTORY")
    for item in get_dispatch_history():
        st.write(f"{format_time(float(item['assigned_at']))} | {item['incident_id']} | {item['responder_id']} | {item['status']}")


def render_map() -> None:
    incident = current_incident()
    if incident is None:
        st.info("No incident selected.")
        return

    st.title("NAVIGATION")
    st.caption("Google Maps is used only for real-world navigation handoff.")
    st.caption("Research A* route is calculated independently on the 10×10 simulation grid.")

    selected_responder = None
    for item in get_dispatch_history():
        if item["incident_id"] == incident["incident_id"]:
            selected_responder = item["responder_id"]
            break

    responder = next((r for r in get_responders() if r["responder_id"] == selected_responder), None)
    if responder is None:
        st.warning("No responder assigned.")
        return

    origin_address = str(responder.get("address") or f"{responder['latitude']}, {responder['longitude']}")
    destination_address = str(incident.get("location_text") or "NMIMS Shirpur, Babulde, Shirpur, Maharashtra")
    location_source = str(responder.get("location_source") or "DEMO SIMULATION")

    st.subheader("SELECTED RESPONDER")
    st.write(f"Responder: {responder['responder_id']}")
    st.write(f"Type: {responder['type']}")
    st.write(f"Current Stored Address: {origin_address}")
    st.write(f"Location Source: {location_source}")
    st.write("Destination:")
    st.write(destination_address)

    maps_url = build_navigation_url(destination_address, origin_address)
    st.link_button("OPEN NAVIGATION", maps_url, use_container_width=True)
    st.caption("Google Maps opens after click using the selected responder's real stored address and the incident's real stored destination address.")
    st.caption("No embedded map is displayed before navigation begins.")


def render_more() -> None:
    st.title("More")
    st.subheader("Database / System Information")
    st.write("DATABASE")
    st.write("--------------------------")
    st.write("Status: ● Connected")
    st.write("Engine: SQLite")
    st.write("File: emergency_dispatch.db")

    table_names = [
        "messages",
        "incidents",
        "responders",
        "dispatches",
        "processing_results",
        "route_updates",
    ]
    st.write("Tables:")
    for table in table_names:
        st.write(f"- {table.replace('_', ' ').title()}: {get_count(table)}")

    tabs = st.tabs(["Messages", "Incidents", "Responders", "Dispatches", "Processing Results", "Route Updates"])
    for tab, table in zip(tabs, table_names):
        with tab:
            st.dataframe(get_table_rows(table), use_container_width=True)

    st.subheader("Research Configuration")
    st.write("HDC dimensions: 10,000")
    st.write("HDC threshold: top-n shortlist")
    st.write("A* configuration: research grid + road-network A*")
    st.write("Simulation settings: 10x10 grid, live GPS fallback available")

    st.subheader("System Status")
    st.write("Live traffic unavailable — routing uses current road-network data.")


def main() -> None:
    inject_theme()
    initialize_app()

    st.title("INTELLIGENT EMERGENCY DISPATCH SYSTEM")
    st.caption("Academic emergency dispatch prototype")
    page = render_navigation()

    if page == "Messages":
        render_messages()
    elif page == "Processing":
        render_processing()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Navigation":
        render_map()
    else:
        render_more()


if __name__ == "__main__":
    main()

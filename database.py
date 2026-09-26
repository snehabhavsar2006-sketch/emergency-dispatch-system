import os
import sqlite3
import time
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "emergency_dispatch.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _sync_demo_responder_addresses(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    existing = {row[1] for row in cur.execute("PRAGMA table_info(responders)").fetchall()}
    if "address" not in existing:
        return

    address_map = {
        "RESP-001": "Near Shirpur Bus Stand, Shirpur, Maharashtra",
        "RESP-002": "Near Shirpur Police Station, Shirpur, Maharashtra",
        "RESP-003": "Near Shirpur Railway Station, Shirpur, Maharashtra",
        "RESP-004": "Near Shirpur Market, Shirpur, Maharashtra",
        "RESP-005": "Near Thalner Road, Shirpur, Maharashtra",
    }
    for responder_id, address in address_map.items():
        cur.execute(
            "UPDATE responders SET address = ?, city = 'Shirpur, Maharashtra', location_source = 'DEMO SIMULATION' WHERE responder_id = ?",
            (address, responder_id),
        )

    cur.execute(
        "UPDATE responders SET address = COALESCE(address, 'Sample address for ' || responder_id), city = COALESCE(city, 'Shirpur, Maharashtra'), location_source = 'DEMO SIMULATION' WHERE address IS NULL OR city IS NULL OR location_source IS NULL OR location_source = 'SIMULATION'"
    )
    conn.commit()


def init_database() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            incident_id TEXT,
            message_text TEXT,
            timestamp REAL,
            source TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id TEXT PRIMARY KEY,
            incident_type TEXT,
            original_message TEXT,
            location_text TEXT,
            latitude REAL,
            longitude REAL,
            severity INTEGER,
            timestamp REAL,
            source TEXT,
            status TEXT,
            assigned_responder TEXT,
            selected_route TEXT,
            location_source TEXT DEFAULT 'SIMULATION',
            grid_x INTEGER,
            grid_y INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS responders (
            responder_id TEXT PRIMARY KEY,
            type TEXT,
            address TEXT,
            city TEXT,
            latitude REAL,
            longitude REAL,
            grid_x INTEGER,
            grid_y INTEGER,
            capacity INTEGER,
            availability INTEGER,
            status TEXT,
            current_assignment TEXT,
            location_source TEXT DEFAULT 'DEMO SIMULATION',
            last_updated REAL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dispatches (
            dispatch_id TEXT PRIMARY KEY,
            incident_id TEXT,
            responder_id TEXT,
            assigned_at REAL,
            route TEXT,
            route_cost REAL,
            distance REAL,
            eta REAL,
            status TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents(incident_id),
            FOREIGN KEY (responder_id) REFERENCES responders(responder_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS processing_results (
            processing_id TEXT PRIMARY KEY,
            incident_id TEXT,
            responder_id TEXT,
            hdc_similarity REAL,
            shortlisted INTEGER,
            csp_valid INTEGER,
            rejection_reason TEXT,
            route_cost REAL,
            distance REAL,
            eta REAL,
            selected INTEGER,
            timestamp REAL,
            FOREIGN KEY (incident_id) REFERENCES incidents(incident_id),
            FOREIGN KEY (responder_id) REFERENCES responders(responder_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS route_updates (
            update_id TEXT PRIMARY KEY,
            incident_id TEXT,
            old_route TEXT,
            new_route TEXT,
            reason TEXT,
            old_cost REAL,
            new_cost REAL,
            timestamp REAL,
            FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
        )
        """
    )

    _ensure_columns(conn, "incidents", {
        "location_source": "TEXT DEFAULT 'SIMULATION'",
        "grid_x": "INTEGER",
        "grid_y": "INTEGER",
    })
    _ensure_columns(conn, "responders", {
        "address": "TEXT",
        "city": "TEXT",
        "grid_x": "INTEGER",
        "grid_y": "INTEGER",
        "location_source": "TEXT DEFAULT 'DEMO SIMULATION'",
        "last_updated": "REAL",
    })
    _ensure_columns(conn, "dispatches", {
        "distance": "REAL",
        "eta": "REAL",
    })
    _ensure_columns(conn, "processing_results", {
        "distance": "REAL",
        "eta": "REAL",
    })

    _sync_demo_responder_addresses(conn)
    conn.commit()
    conn.close()


def add_message(message_id: str, incident_id: str, message_text: str, timestamp: float, source: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO messages (message_id, incident_id, message_text, timestamp, source) VALUES (?, ?, ?, ?, ?)",
        (message_id, incident_id, message_text, timestamp, source),
    )
    conn.commit()
    conn.close()


def create_incident(
    incident_id: str,
    incident_type: str,
    original_message: str,
    location_text: str,
    latitude: float,
    longitude: float,
    severity: int,
    timestamp: float,
    source: str,
    status: str = "RECEIVED",
    assigned_responder: Optional[str] = None,
    selected_route: Optional[str] = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO incidents (
            incident_id, incident_type, original_message, location_text, latitude, longitude,
            severity, timestamp, source, status, assigned_responder, selected_route
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            incident_type,
            original_message,
            location_text,
            float(latitude),
            float(longitude),
            int(severity),
            float(timestamp),
            source,
            status,
            assigned_responder,
            selected_route,
        ),
    )
    conn.commit()
    conn.close()


def get_incidents() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM incidents ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_incident(incident_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def add_responder(
    responder_id: str,
    responder_type: str,
    latitude: float,
    longitude: float,
    capacity: int,
    availability: int = 1,
    status: str = "AVAILABLE",
    current_assignment: Optional[str] = None,
    grid_x: Optional[int] = None,
    grid_y: Optional[int] = None,
    location_source: str = "DEMO SIMULATION",
    last_updated: Optional[float] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO responders (
            responder_id, type, address, city, latitude, longitude, grid_x, grid_y, capacity, availability, status,
            current_assignment, location_source, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            responder_id,
            responder_type,
            address or f"Sample address for {responder_id}",
            city or "Shirpur, Maharashtra",
            float(latitude),
            float(longitude),
            grid_x,
            grid_y,
            int(capacity),
            int(availability),
            status,
            current_assignment,
            location_source,
            float(last_updated) if last_updated is not None else time.time(),
        ),
    )
    conn.commit()
    conn.close()


def get_responders() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM responders ORDER BY responder_id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_available_responders() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM responders WHERE availability = 1 ORDER BY responder_id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_responder_status(responder_id: str, availability: int, status: str, current_assignment: Optional[str] = None) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE responders SET availability = ?, status = ?, current_assignment = ? WHERE responder_id = ?",
        (int(availability), status, current_assignment, responder_id),
    )
    conn.commit()
    conn.close()


def get_dispatch_history() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM dispatches ORDER BY assigned_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_dispatch(
    dispatch_id: str,
    incident_id: str,
    responder_id: str,
    assigned_at: float,
    route: str,
    route_cost: float,
    status: str,
    distance: Optional[float] = None,
    eta: Optional[float] = None,
) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO dispatches (dispatch_id, incident_id, responder_id, assigned_at, route, route_cost, distance, eta, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (dispatch_id, incident_id, responder_id, float(assigned_at), route, float(route_cost), float(distance) if distance is not None else None, float(eta) if eta is not None else None, status),
    )
    conn.commit()
    conn.close()


def save_processing_result(
    processing_id: str,
    incident_id: str,
    responder_id: str,
    hdc_similarity: float,
    shortlisted: int,
    csp_valid: int,
    rejection_reason: str,
    route_cost: float,
    selected: int,
    timestamp: float,
    distance: Optional[float] = None,
    eta: Optional[float] = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO processing_results (
            processing_id, incident_id, responder_id, hdc_similarity, shortlisted,
            csp_valid, rejection_reason, route_cost, distance, eta, selected, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            processing_id,
            incident_id,
            responder_id,
            float(hdc_similarity),
            int(shortlisted),
            int(csp_valid),
            rejection_reason,
            float(route_cost),
            float(distance) if distance is not None else None,
            float(eta) if eta is not None else None,
            int(selected),
            float(timestamp),
        ),
    )
    conn.commit()
    conn.close()


def get_processing_results(incident_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM processing_results WHERE incident_id = ? ORDER BY hdc_similarity DESC",
        (incident_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_route_update(
    update_id: str,
    incident_id: str,
    old_route: str,
    new_route: str,
    reason: str,
    old_cost: float,
    new_cost: float,
    timestamp: float,
) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO route_updates (update_id, incident_id, old_route, new_route, reason, old_cost, new_cost, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (update_id, incident_id, old_route, new_route, reason, float(old_cost), float(new_cost), float(timestamp)),
    )
    conn.commit()
    conn.close()


def update_incident_status(incident_id: str, status: str, assigned_responder: Optional[str] = None, selected_route: Optional[str] = None) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE incidents SET status = ?, assigned_responder = ?, selected_route = ? WHERE incident_id = ?",
        (status, assigned_responder, selected_route, incident_id),
    )
    conn.commit()
    conn.close()


def get_count(table: str) -> int:
    conn = get_connection()
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return count


def get_table_rows(table: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_active_incident_count() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('COMPLETED', 'RESOLVED')").fetchone()[0]
    conn.close()
    return count


def seed_demo_responders() -> None:
    demo_rows = [
        ("RESP-001", "ambulance", 21.3442, 74.8760, 5, 1, "AVAILABLE", None, 5, 7, "DEMO SIMULATION", time.time(), "Near Shirpur Bus Stand, Shirpur, Maharashtra", "Shirpur, Maharashtra"),
        ("RESP-002", "fire_truck", 21.3498, 74.8855, 5, 1, "AVAILABLE", None, 1, 8, "DEMO SIMULATION", time.time(), "Near Shirpur Police Station, Shirpur, Maharashtra", "Shirpur, Maharashtra"),
        ("RESP-003", "rescue_unit", 21.3524, 74.8821, 3, 1, "AVAILABLE", None, 3, 8, "DEMO SIMULATION", time.time(), "Near Shirpur Railway Station, Shirpur, Maharashtra", "Shirpur, Maharashtra"),
        ("RESP-004", "police", 21.3384, 74.8703, 3, 1, "AVAILABLE", None, 8, 2, "DEMO SIMULATION", time.time(), "Near Shirpur Market, Shirpur, Maharashtra", "Shirpur, Maharashtra"),
        ("RESP-005", "ambulance", 21.3337, 74.8779, 1, 1, "AVAILABLE", None, 5, 0, "DEMO SIMULATION", time.time(), "Near Thalner Road, Shirpur, Maharashtra", "Shirpur, Maharashtra"),
    ]
    existing = {r["responder_id"] for r in get_responders()}
    for row in demo_rows:
        if row[0] in existing:
            conn = get_connection()
            conn.execute(
                "UPDATE responders SET type=?, latitude=?, longitude=?, capacity=?, availability=?, status=?, current_assignment=?, grid_x=?, grid_y=?, location_source=?, last_updated=?, address=?, city=? WHERE responder_id=?",
                (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[0]),
            )
            conn.commit(); conn.close()
        else:
            add_responder(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13]
            )

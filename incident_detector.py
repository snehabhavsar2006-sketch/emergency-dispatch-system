"""
Incident Detector Module (incident_detector.py)
-----------------------------------------------
Design Rationale & Viva Defense Notes:
- Input Parsing & Robustness: Real-world emergency dispatch feeds often contain messy 
  or unstructured input. The IncidentDetector validates field presence, coerces types, 
  and normalizes incident strings to a canonical vocabulary.
- Vocabulary Normalization: Synonym mapping (e.g., "Fire Outbreak" -> "fire") prevents 
  downstream HDC item memory misses and CSP validation errors.
- Processing Order vs Route Cost:
  - Urgency Score is used ONLY by the Coordinator to determine processing order 
    when multiple incidents arrive simultaneously.
  - It is NOT included in A* route cost calculations. Distance/terrain traversal cost 
    remains an unpolluted metric.
  - Note: The urgency formula (severity * 10 + waiting_time) is a simplified simulation 
    rule used only to establish processing order; it does not represent a real emergency-dispatch 
    prioritization standard.
"""

import time
from typing import Dict, Any, Optional
from environment import Incident, INCIDENT_TYPES


TYPE_ALIASES: Dict[str, str] = {
    "fire": "fire",
    "fire_truck": "fire",
    "structure_fire": "fire",
    "blaze": "fire",
    "accident": "accident",
    "traffic_accident": "accident",
    "car_crash": "accident",
    "crash": "accident",
    "medical": "medical",
    "medical_emergency": "medical",
    "patient": "medical",
    "elderly_fall": "medical",
    "structural": "structural",
    "building_collapse": "structural",
    "collapse": "structural",
}


class IncidentDetector:
    """
    Validates, normalizes, and scores incoming incident reports.
    """

    @staticmethod
    def resolve_location(location_text: Optional[str]) -> Optional[tuple]:
        if location_text is None:
            return None
        cleaned = location_text.strip().lower()
        if not cleaned:
            return None
        known_locations = {
            "nmims shirpur": (5, 5),
            "nmims shirpur main gate": (2, 8),
            "shirpur main gate": (2, 8),
            "nmims shirpur bus stop": (7, 1),
            "bus stop": (7, 1),
            "library": (5, 5),
            "main gate": (2, 8),
            "campus gate": (2, 8),
        }
        if cleaned in known_locations:
            return known_locations[cleaned]
        for key, value in known_locations.items():
            if key in cleaned:
                return value
        return None

    @staticmethod
    def extract_from_message(message: str, location_text: Optional[str] = None, severity_input: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Emergency message cannot be empty.")

        text = message.strip().lower()
        incident_type = None
        if any(keyword in text for keyword in ["fire", "blaze", "smoke", "explosion"]):
            incident_type = "fire"
        elif any(keyword in text for keyword in ["accident", "crash", "collision", "vehicle", "road accident"]):
            incident_type = "accident"
        elif any(keyword in text for keyword in ["injured", "injury", "medical", "ambulance", "patient", "hospital", "emergency treatment"]):
            incident_type = "medical"
        elif any(keyword in text for keyword in ["collapse", "building", "structural", "fallen wall", "roof"]):
            incident_type = "structural"

        if incident_type is None:
            return {
                "requires_confirmation": True,
                "details": "The message does not clearly identify the incident type. Information requires dispatcher confirmation.",
            }

        resolved_location = IncidentDetector.resolve_location(location_text)
        if resolved_location is None:
            resolved_location = IncidentDetector.resolve_location(message)
        if resolved_location is None:
            return {
                "requires_confirmation": True,
                "details": "The location could not be confidently mapped to a grid coordinate. Information requires dispatcher confirmation.",
            }

        if severity_input is None:
            if any(word in text for word in ["critical", "severe", "fatal", "emergency"]):
                severity = 5
            elif any(word in text for word in ["serious", "major", "injured", "burning"]):
                severity = 4
            elif any(word in text for word in ["moderate", "accident", "medical"]):
                severity = 3
            else:
                severity = 2
        else:
            severity = int(severity_input)

        location_name = (location_text or message).strip() or "Unknown location"
        return {
            "type": incident_type,
            "location": resolved_location,
            "location_name": location_name,
            "severity": max(1, min(5, severity)),
            "source": "Emergency User",
            "message": message,
            "requires_confirmation": False,
        }

    @staticmethod
    def normalize_type(raw_type: str) -> str:
        if not isinstance(raw_type, str):
            raise ValueError(f"Incident type must be a string, got {type(raw_type).__name__}")
        cleaned = raw_type.strip().lower().replace(" ", "_")
        if cleaned in TYPE_ALIASES:
            return TYPE_ALIASES[cleaned]
        if cleaned in INCIDENT_TYPES:
            return cleaned
        raise ValueError(
            f"Invalid incident type '{raw_type}'. Expected one of {INCIDENT_TYPES} or known aliases."
        )

    @classmethod
    def parse_incident(cls, raw_data: Dict[str, Any]) -> Incident:
        """
        Parses raw dictionary payload into a validated Incident object.
        Rejects malformed input with explicit ValueError.
        """
        if not isinstance(raw_data, dict):
            raise ValueError("Raw incident data must be a dictionary.")

        required_keys = {"incident_id", "type", "location", "severity"}
        missing = required_keys - set(raw_data.keys())
        if missing:
            raise ValueError(f"Missing required incident fields: {sorted(list(missing))}")

        inc_id = str(raw_data["incident_id"]).strip()
        if not inc_id:
            raise ValueError("incident_id cannot be empty.")

        norm_type = cls.normalize_type(raw_data["type"])

        # Validate location
        loc = raw_data["location"]
        if not (isinstance(loc, (list, tuple)) and len(loc) == 2):
            raise ValueError(f"Location must be a 2-tuple (x, y), got {loc}")
        try:
            x, y = int(loc[0]), int(loc[1])
        except (ValueError, TypeError):
            raise ValueError(f"Location coordinates must be integers, got {loc}")

        if not (0 <= x < 10 and 0 <= y < 10):
            raise ValueError(f"Location coordinates ({x}, {y}) out of 10x10 bounds [0..9].")

        # Validate severity
        try:
            severity = int(raw_data["severity"])
        except (ValueError, TypeError):
            raise ValueError(f"Severity must be an integer, got {raw_data['severity']}")
        if not (1 <= severity <= 5):
            raise ValueError(f"Severity must be an integer between 1 and 5, got {severity}")

        timestamp = float(raw_data.get("timestamp", time.time()))
        status = str(raw_data.get("status", "active")).strip().lower()

        return Incident(
            incident_id=inc_id,
            type=norm_type,
            location=(x, y),
            severity=severity,
            timestamp=timestamp,
            status=status,
        )

    @staticmethod
    def compute_urgency_score(incident: Incident, current_time: Optional[float] = None) -> float:
        """
        Computes urgency score to prioritize simultaneous incidents.
        Higher score = processed earlier.
        
        Formula Rationale:
        - Severity carries major weight (severity * 10.0).
        - Elapsed waiting time adds incremental priority (time_since_report).
        - Disclaimer: The urgency formula is a simplified simulation rule used only 
          to establish processing order; it does not represent a real emergency-dispatch 
          prioritization standard.
        """
        if current_time is None:
            current_time = time.time()
        elapsed_seconds = max(0.0, current_time - incident.timestamp)
        return float(incident.severity * 10.0 + elapsed_seconds)

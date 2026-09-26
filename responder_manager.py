"""
Responder Manager Module (responder_manager.py)
-----------------------------------------------
Design Rationale & Viva Defense Notes:
- State Encapsulation: Responders are central entities whose availability and 
  spatial coordinates change dynamically during simulation runs. Centralizing 
  this in a Manager prevents state desynchronization across modules.
- Exception Safety: Operating on an unregistered responder ID explicitly raises 
  a custom ResponderNotFoundError, preventing silent state corruption.
- Thread/Batch Safety: The registry allows the Coordinator to mark responders 
  as assigned (available=False) during batch dispatch of simultaneous incidents.
"""

from typing import Dict, List, Tuple, Optional
from environment import Responder


class ResponderNotFoundError(KeyError):
    """Raised when operating on a responder ID not present in the registry."""
    pass


class ResponderManager:
    """
    Registry and state manager for all active emergency responders.
    """

    def __init__(self, responders: Optional[List[Responder]] = None):
        self._registry: Dict[str, Responder] = {}
        if responders:
            for resp in responders:
                self.register_responder(resp)

    def register_responder(self, responder: Responder) -> None:
        """Registers or overwrites a responder in the registry."""
        if not isinstance(responder, Responder):
            raise TypeError("Expected a Responder object.")
        self._registry[responder.responder_id] = responder

    def get_responder(self, responder_id: str) -> Responder:
        """Retrieve responder by ID, raising ResponderNotFoundError if missing."""
        if responder_id not in self._registry:
            raise ResponderNotFoundError(f"Responder ID '{responder_id}' is not registered.")
        return self._registry[responder_id]

    def get_all_responders(self) -> List[Responder]:
        """Returns list of all registered responders."""
        return list(self._registry.values())

    def get_available_responders(self) -> List[Responder]:
        """Returns list of responders currently marked as available."""
        return [resp for resp in self._registry.values() if resp.available]

    def set_availability(self, responder_id: str, available: bool) -> None:
        """Updates availability state for a specific responder."""
        resp = self.get_responder(responder_id)  # Raises error if missing
        resp.available = bool(available)

    def update_location(self, responder_id: str, new_location: Tuple[int, int]) -> None:
        """Updates grid location for a specific responder."""
        if not (isinstance(new_location, (list, tuple)) and len(new_location) == 2):
            raise ValueError(f"Location must be a 2-tuple (x, y), got {new_location}")
        x, y = int(new_location[0]), int(new_location[1])
        if not (0 <= x < 10 and 0 <= y < 10):
            raise ValueError(f"Coordinate ({x}, {y}) out of 10x10 grid bounds.")

        resp = self.get_responder(responder_id)  # Raises error if missing
        resp.location = (x, y)

    def clear(self) -> None:
        """Clear all registered responders."""
        self._registry.clear()

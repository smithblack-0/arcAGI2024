"""
Events and the event bus are used
to pass important information around
through various parts of gui.
"""

from typing import Callable
from enum import Enum

class Events(Enum):
    # Intercell and zone events
    PALETTEREQUEST = "ask_for_palette_color"
    PALETTE_CHANGED = "palette_changed"
    SHAPE_CHANGED = "shape_changed"
    EDITMADE = "edit_made"

    # Case selection mainly
    UNSAVED_CHANGES = "unsaved_changes"


class EventBus:
    """Centralized event bus to manage event dispatching and listening."""
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to a specific event with a callback."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event."""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)

    def publish(self, event_type: str, *args, **kwargs):
        """Publish an event to all subscribers."""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                callback(*args, **kwargs)



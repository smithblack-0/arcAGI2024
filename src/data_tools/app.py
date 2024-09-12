"""
The entire app
"""
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from typing import Callable, List, Any, Dict

from src.data_tools.events import Events, EventBus


class BlockableButton(QPushButton):
    """
    A button that will emit its associated event when pressed, or emit a blocked
    button event, depending on whether pressing is blocked or not. It will also
    automatically set itself to its blocked/unblocked display when the event
    goes by
    """

    def set_as_blocked(self):
        """Blocks the button from being pressed."""
        self.blocked = True
        self.setStyleSheet(f"background-color: {self.button_styles['blocked']};")

    def set_as_unblocked(self):
        """Unblocks the button, allowing it to be pressed."""
        self.blocked = False
        self.setStyleSheet(f"background-color: {self.button_styles['default']};")

    def mousePressEvent(self, event):
        """Handle the press event."""
        if self.blocked:
            self.event_bus.publish(Events.BLOCKEDCLICK.value, self)
        else:
            self.event_bus.publish(self.event_str, self)

    def __init__(self,
                 button_text: str,
                 event_str: str,
                 button_bus: EventBus,
                 config: dict,
                 parent=None):
        super().__init__(parent)
        self.event_str = event_str
        self.event_bus = button_bus
        self.blocked = False
        self.button_styles = config["button_styles"]

        # Set default button appearance
        self.setStyleSheet(f"background-color: {self.button_styles['default']};")
        self.setText(button_text)

        # Listen for block/unblock events
        self.event_bus.subscribe(Events.BLOCK.value, self.set_as_blocked)
        self.event_bus.subscribe(Events.UNBLOCK.value, self.set_as_unblocked)



class BlockableButton(QPushButton):
    """
    A button that can be blocked from being pressed. It supports callbacks for blocked and unblocked states.
    """

    def on_click(self):
        """Handles the button press and invokes the appropriate callback based on the blocked state."""
        if self.blocked:
            self.blocked_callback()
        else:
            self.press_callback()

    def enterEvent(self, event):
        """Change button color when hovered over if unblocked."""
        if not self.blocked:
            self.setStyleSheet(f"background-color: {self.config['buttons_colors']['hover']};")

    def leaveEvent(self, event):
        """Reset button color when no longer hovered over."""
        if not self.blocked:
            self.setStyleSheet(f"background-color: {self.config['buttons_colors']['default']};")

    def __init__(self,
                 press_callback: Callable[[], None],
                 blocked_callback: Callable[[], None],
                 button_text: str,
                 config: Dict[str, Any],
                 parent=None):
        super().__init__(parent)

        # Store the callbacks
        self.press_callback = press_callback
        self.blocked_callback = blocked_callback

        # Store the config and initialize as unblocked
        self.config = config
        self.blocked = False

        # Set button text
        self.setText(button_text)

        # Set default style
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['default']};")

        # Connect the button's click event to the on_click method
        self.clicked.connect(self.on_click)

        # Enable hover effect
        self.setMouseTracking(True)

    def set_as_blocked(self):
        """Sets the button as blocked, preventing it from being pressed."""
        self.blocked = True
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['blocked']};")

    def set_as_unblocked(self):
        """Unblocks the button, allowing it to be pressed."""
        self.blocked = False
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['default']};")


from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton

class UnblockableButton(QPushButton):
    """
    A button that cannot be blocked. It will flash if a blocked click event is invoked.
    """
    def recolor_flashed(self):
        """Changes the button color to the flash color."""
        self.setStyleSheet(f"background-color: {self.flash_color};")

    def recolor_default(self):
        """Changes the button color back to the default color."""
        self.setStyleSheet(f"background-color: {self.default_color};")

    def flash(self, cycle_time, num_flashes):
        """
        Initiates a flash sequence that lasts for `num_flashes` cycles.
        Each cycle lasts for `cycle_time` seconds.
        """
        if self.is_flashing:
            return  # If already flashing, ignore further flash requests

        self.is_flashing = True
        total_flash_duration = cycle_time * 1000 * num_flashes  # Convert seconds to milliseconds
        time_accumulator = 0

        # We calculate and dump the timers on the stack to toggle the colors in sequence
        for i in range(num_flashes):
            # Schedule flash color
            QTimer.singleShot(time_accumulator, self.recolor_flashed)
            time_accumulator += (cycle_time * 1000) // 2  # Convert half-cycle to milliseconds

            # Schedule default color halfway through the cycle
            QTimer.singleShot(time_accumulator, self.recolor_default)
            time_accumulator += (cycle_time * 1000) // 2  # Convert half-cycle to milliseconds

        # Set is_flashing to False at the end of the last flash
        QTimer.singleShot(total_flash_duration, lambda: setattr(self, 'is_flashing', False))

    def emit_press_event(self):
        """Emits the provided event string when the button is pressed."""
        self.event_bus.publish(self.event_str)

    def on_blocked_click(self):
        """Initiates flashing when a BLOCKEDCLICK event is detected."""
        if not self.is_flashing:
            self.flash(self.config["flash_duty"], 3)  # Flash 3 times

    def __init__(self,
                 button_text: str,
                 event_str: str,
                 event_bus,
                 config: dict,
                 parent=None):

        super().__init__(parent)

        self.event_str = event_str
        self.event_bus = event_bus
        self.config = config
        self.is_flashing = False  # Tracks if a flash sequence is in progress

        # Button's default and flashed colors from config
        self.default_color = config["buttons_colors"]["default"]
        self.flash_color = config["buttons_colors"]["flash"]
        self.setText(button_text)

        # Setup button appearance
        self.setStyleSheet(f"background-color: {self.default_color};")
        self.clicked.connect(self.emit_press_event)

        # Subscribe to BLOCKEDCLICK event to initiate the flash
        self.event_bus.subscribe(Events.BLOCKEDCLICK.value, self.on_blocked_click)

class UnblockableButton(QPushButton):
    """
    A button that cannot be blocked. It always invokes the press_callback when pressed.
    """

    def on_click(self):
        """Handles the button press and always invokes the press callback."""
        self.press_callback()

    def enterEvent(self, event):
        """Change button color when hovered over."""
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['hover']};")

    def leaveEvent(self, event):
        """Reset button color when no longer hovered over."""
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['default']};")

    def __init__(self,
                 button_text: str,
                 press_callback: Callable[[], None],
                 config: Dict[str, Any],
                 parent=None):
        super().__init__(parent)

        # Store the press callback
        self.press_callback = press_callback

        # Store config and apply the "default" color from the config
        self.config = config

        # Set button text
        self.setText(button_text)

        # Set default style
        self.setStyleSheet(f"background-color: {self.config['buttons_colors']['default']};")

        # Connect the button's click event to the on_click method
        self.clicked.connect(self.on_click)

        # Enable hover effect
        self.setMouseTracking(True)




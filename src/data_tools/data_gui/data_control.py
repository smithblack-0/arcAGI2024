"""
The data control and cases selection module.
"""
import copy
import json
import numpy as np

from PyQt5.QtWidgets import QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from typing import Any, Dict

from src.data_tools.data_gui.events import DataSelectionSignals, Events, EventBus

class IOAdapter:
    """
    A small adapter for performing io operations
    """

    def walk_and_convert_ndarray(self, data):
        if isinstance(data, dict):
            # If it's a dictionary, recursively walk through its values
            return {k: self.walk_and_convert_ndarray(v) for k, v in data.items()}
        elif isinstance(data, list):
            # If it's a list, recursively walk through its elements
            return [self.walk_and_convert_ndarray(item) for item in data]
        elif isinstance(data, np.ndarray):
            # If it's an ndarray, convert to a list of lists
            return data.tolist()
        else:
            # If it's not a dict, list, or ndarray, return the data as is
            return data
    def load(self)->Any:
        with open(self.file, 'r') as f:
            data = json.load(f)
        return data

    def save(self, data: Any):
        print("save requested")
        # Convert ndarrays to lists
        data = self.walk_and_convert_ndarray(data)
        with open(self.file, 'w') as f:
            json.dump(data, f)

    def __init__(self, file: str):
        self.file = file

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
            self.event_bus.publish(DataSelectionSignals.BLOCKEDCLICK.value)
        else:
            self.event_bus.publish(self.event_str)

    def __init__(self,
                 button_text: str,
                 event_str: str,
                 event_bus: EventBus,
                 config: dict,
                 parent=None):
        super().__init__(parent)
        self.event_str = event_str
        self.event_bus = event_bus
        self.blocked = False
        self.button_styles = config["button_styles"]

        # Set default button appearance
        self.setStyleSheet(f"background-color: {self.button_styles['default']};")
        self.setText(button_text)

        # Listen for block/unblock events
        self.event_bus.subscribe(DataSelectionSignals.BLOCK.value, self.set_as_blocked)
        self.event_bus.subscribe(DataSelectionSignals.UNBLOCK.value, self.set_as_unblocked)

class UnblockableButton(QPushButton):
    """
    An UnblockableButton that emits a specific event when clicked.

    It is also capable of flashing to bring deep_memories to itself when
    a user tries to click a blocked button.
    """

    def recolor_flashed(self):
        """Set the button's color to the flash color."""
        self.setStyleSheet(f"background-color: {self.button_styles['flash']}")

    def recolor_default(self):
        """Set the button's color back to the default."""
        self.setStyleSheet(f"background-color: {self.button_styles['default']}")

    def emit_event(self):
        """Emit the event string when the button is clicked."""
        self.event_bus.publish(self.event_str)

    def flash(self):
        """Flash the button by alternating between default and flash colors."""
        cycle_time = self.cycle_time
        num_flashes = self.num_flashes

        if self.is_flashing:
            return

        self.is_flashing = True
        for i in range(num_flashes):
            QTimer.singleShot(int(cycle_time * 1000 * (i * 2)), self.recolor_flashed)
            QTimer.singleShot(int(cycle_time * 1000 * (i * 2 + 1)), self.recolor_default)

        # Reset the flashing state after the last flash.
        QTimer.singleShot(int(cycle_time * 1000 * num_flashes * 2), lambda: setattr(self, 'is_flashing', False))

    def __init__(self,
                 button_text: str,
                 event_str: str,
                 event_bus: EventBus,
                 config: Dict[str, Any],
                 parent=None):
        super().__init__(button_text, parent)
        self.event_str = event_str
        self.event_bus = event_bus
        self.button_styles = config["button_styles"]
        self.cycle_time = config["flash_cycle_time"]
        self.num_flashes = config["num_flashes"]
        self.is_flashing = False

        # Recolor to default
        self.recolor_default()

        # Define connections
        self.setMouseTracking(True)
        self.clicked.connect(self.emit_event)  # Emit the event when clicked.

        # Define flashing response
        self.event_bus.subscribe(DataSelectionSignals.BLOCKEDCLICK.value, self.flash)

class DataControlDisplay(QFrame):
    """
    Displays the currently selected case and provides buttons
    to navigate between cases, commit changes, or save the data.
    Emits signals for getting the next case, the last case, committing,
    or saving. Also displays the currently selected case.
    """

    def set_edited_case(self, name: str):
        """Sets the edited case in the text display."""
        self.case_display_label.setText(f"Current case: {name}")

    def publish_unblock_event(self):
        self.event_bus.publish(DataSelectionSignals.UNBLOCK.value)

    def create_unblockable_button(self, button_text: str, event_str: str):
        """Creates and appends an unblockable button into the button layout"""
        # Note that running one of these buttons makes the zone status unambigous,
        # and thus unlocks the buttons
        button = UnblockableButton(button_text, event_str, self.event_bus, self.config, self)
        self.button_layout.addWidget(button)

    def create_blockable_button(self, button_text: str, event_str: str):
        """Creates and appends a blockable button into the button layout"""
        button = BlockableButton(button_text, event_str, self.event_bus, self.config, self)
        self.button_layout.addWidget(button)

    def __init__(self,
                 config: Dict[str, Any],
                 event_bus: EventBus,
                 parent=None):
        super().__init__(parent)

        # Setup static attributes
        self.config = config
        self.event_bus = event_bus

        # Create layout
        self.main_layout = QHBoxLayout(self)
        self.setLayout(self.main_layout)

        # Display label for the currently selected case
        self.case_display_label = QLabel("Current case: None", self)
        self.case_display_label.setFont(QFont('Arial', 12))
        self.main_layout.addWidget(self.case_display_label)
        self.event_bus.subscribe(DataSelectionSignals.NOWEDITING.value, self.set_edited_case)

        # Create the control buttons
        self.button_layout = QHBoxLayout()
        self.create_blockable_button("Previous", DataSelectionSignals.PREVIOUS.value)
        self.create_blockable_button("Save", DataSelectionSignals.SAVE.value)
        self.create_unblockable_button("Commit", DataSelectionSignals.COMMIT.value)
        self.create_unblockable_button("Revert", DataSelectionSignals.REVERT.value)
        self.create_blockable_button("Next", DataSelectionSignals.NEXT.value)

        # Add the button layout to the main layout
        self.main_layout.addLayout(self.button_layout)


class DataManager(QFrame):
    """
    The backend data controller and interface between the
    actions that can be performed and the data that is being
    modified
    """

    # Define helper functions related to allowing the definition
    # and editing of the various zone collections, or
    # "example cases", in terms of their keys and the next or
    # previous key.

    def edit_case_with_key(self, name: str):
        """Sets up a single example to be edited"""
        assert name in self.data

        # Define fields
        self.case_cache = copy.deepcopy(self.data[name])
        self.editing_key = name

        # Publish events
        self.event_bus.publish(Events.DISPLAY_ZONES.value, self.case_cache)
        self.button_bus.publish(DataSelectionSignals.NOWEDITING.value, name)

    def get_next_key(self) -> str:
        """Gets the next key based on the current key.
        If there is no next key, returns the current key instead.
        """
        assert not self.case_change_is_blocked
        keys = list(self.data._keys())
        index = keys.index(self.editing_key)
        if index == len(keys) - 1:
            return self.editing_key
        return keys[index + 1]

    def get_last_key(self) -> str:
        """
        Gets the last case key based on the current key.
        If there is no last key, returns the current key instead.
        :return: The last key
        """
        assert not self.case_change_is_blocked
        keys = list(self.data._keys())
        index = keys.index(self.editing_key)
        if index == 0:
            return self.editing_key
        return keys[index - 1]

    ##
    # Handles blocking and unblocking
    ##
    def block_change_case(self):
        """Blocks case change and updates UI buttons accordingly."""
        self.case_change_is_blocked = True
        self.button_bus.publish(DataSelectionSignals.BLOCK.value)

    def unblock_change_case(self):
        """Unblocks case change and updates UI buttons."""
        self.case_change_is_blocked = False
        self.button_bus.publish(DataSelectionSignals.UNBLOCK.value)

    ##
    # Define the primary manipulation callbacks that will be utilized
    ##
    def next(self):
        """Moves to the next case for editing."""
        assert not self.case_change_is_blocked
        next_key = self.get_next_key()
        self.edit_case_with_key(next_key)

    def previous(self):
        """Moves to the previous case for editing."""
        assert not self.case_change_is_blocked
        key = self.get_last_key()
        self.edit_case_with_key(key)

    def commit(self):
        """Commits the current edits to the case."""
        self.data[self.editing_key] = self.case_cache
        self.unblock_change_case()

    def revert(self):
        """Reverts to the last committed state of the case."""
        self.edit_case_with_key(self.editing_key)
        self.unblock_change_case()

    def save(self):
        """Saves the current data to the file."""
        assert not self.case_change_is_blocked
        self.io.save(self.data)

    def __init__(self,
                 io: IOAdapter,
                 master_bus: EventBus,
                 config: Dict[str, Any],
                 parent=None):
        super().__init__(parent)

        assert "button_styles" in config and all(
            k in config["button_styles"] for k in ["blocked", "default", "flash"]), \
            "Missing button styles in config"

        # Setup static and fields
        self.io = io
        self.config = config
        self.button_bus = EventBus()
        self.event_bus = master_bus
        self.case_change_is_blocked = False

        # Get data to be edited
        self.data = self.io.load()

        # Setup GUI
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Initialize the DataControlDisplay and add it to the layout
        self.data_control_display = DataControlDisplay(config, self.button_bus, self)
        self.main_layout.addWidget(self.data_control_display)

        # Setup the default editing case
        first_key = next(iter(self.data._keys()))
        self.edit_case_with_key(first_key)

        # Attach unsaved changes button blocking
        self.event_bus.subscribe(Events.UNSAVED_CHANGES.value, self.block_change_case)

        # Attach button listeners for navigation and actions
        self.button_bus.subscribe(DataSelectionSignals.PREVIOUS.value, self.previous)
        self.button_bus.subscribe(DataSelectionSignals.SAVE.value, self.save)
        self.button_bus.subscribe(DataSelectionSignals.COMMIT.value, self.commit)
        self.button_bus.subscribe(DataSelectionSignals.NEXT.value, self.next)
        self.button_bus.subscribe(DataSelectionSignals.REVERT.value, self.revert)

class MockIOAdapter:
    """
    Mock adapter to simulate file I/O operations for testing purposes.
    Instead of interacting with the file system, it uses an in-memory dictionary.
    """
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def load(self) -> Dict[str, Any]:
        """Simulate loading data."""
        return self.data

    def save(self, data: Dict[str, Any]):
        """Simulate saving data."""
        print("Data saved:")
        print(json.dumps(data, indent=4))

def main():
    import sys
    from PyQt5.QtWidgets import QApplication, QPushButton

    app = QApplication(sys.argv)

    # Simulated data for testing
    mock_data = {
        "case1": {"zone": "A1", "description": "Case 1 details"},
        "case2": {"zone": "B2", "description": "Case 2 details"},
        "case3": {"zone": "C3", "description": "Case 3 details"}
    }

    # Mock IO Adapter instance
    mock_io = MockIOAdapter(mock_data)

    # Event bus instance
    event_bus = EventBus()

    # Config for button styles and flashing
    config = {
        "button_styles": {
            "default": "#D4D0C8",  # Windows grey (default button color)
            "blocked": "#A9A9A9",  # Faded light grey for blocked state
            "flash": "#FFFF99"  # Light yellow for flashing
        },
        "flash_cycle_time": 0.2,
        "num_flashes": 3
    }

    # Create the DataManager instance
    manager = DataManager(io=mock_io, master_bus=event_bus, config=config)

    # Add a button to emit the UNSAVED_CHANGES event for testing locking behavior


    unsaved_changes_button = QPushButton("Emit Unsaved Changes", manager)
    unsaved_changes_button.clicked.connect(lambda: event_bus.publish(Events.UNSAVED_CHANGES.value))

    # Add the test button to the layout
    manager.main_layout.addWidget(unsaved_changes_button)

    # Show the manager window
    manager.setWindowTitle("Data Manager Test")
    manager.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
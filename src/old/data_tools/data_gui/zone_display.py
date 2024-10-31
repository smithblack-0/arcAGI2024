"""
The main data case editing display module.

If it involves editing the components of a
particular arg-agi data case, it is managed
here.

Processes like choosing which data case to
manipulate, and saving, loading, and locking,
are elsewhere

For the purposes of this module a "data case"
is a string of zones each with blocks of content
- basically, a training example and answers.
"""


import sys
import numpy as np
import copy
from typing import Dict, List, Any, Callable

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFrame,
                             QLabel, QButtonGroup, QTabWidget
                             )
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from src.old.data_tools.data_gui.events import Events, EventBus
from src.old.data_tools.data_gui.cell_editors import BlockCellEditor
from src.old.data_tools.data_gui.mytypes import Zones


class PaletteWidget(QFrame):
    """
    A widget that displays a horizontal list of color buttons based on the color map.
    Only one button can be selected at a time. When a color is selected, we
    emit an event indicating the selected palette color.
    """

    def __init__(self,
                 color_map: Dict[int, str],
                 event_bus: EventBus,
                 title_size: int,
                 parent=None):
        super().__init__(parent)
        self.color_map = color_map
        self.event_bus = event_bus
        self.palette = 0

        # Layout for color buttons (horizontal)
        self.main_layout = QVBoxLayout(self)

        # Define the title
        self.title = QLabel("Palette Color Control")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont('Arial', title_size))

        self.button_layout = QHBoxLayout()

        self.main_layout.addWidget(self.title)
        self.main_layout.addLayout(self.button_layout)
        self.button_group = QButtonGroup(self)

        # Create buttons for each color
        for color_id, color_value in self.color_map.items():
            button = QPushButton(f"Color {color_id}")
            button.setStyleSheet(f"background-color: {color_value}")
            button.setCheckable(True)  # Allow the button to stay selected
            button.setMinimumSize(20, 20)  # Set a visible size for the buttons

            self.button_layout.addWidget(button)
            self.button_group.addButton(button, color_id)

        # Connect signal when a button is clicked
        self.button_group.buttonClicked[int].connect(self.select_color)

        # Connect palette request signal.
        self.event_bus.subscribe(Events.PALETTEREQUEST.value, self.palette_request)

        # Set frame style for the dotted border
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

    def palette_request(self, callback: Callable[[int], None]):
        """Returns the current palette value"""
        callback(self.palette)
    def select_color(self, color_id: int):
        """Emits the selected color ID when a button is clicked."""
        self.palette = color_id
        self.event_bus.publish(Events.PALETTE_CHANGED.value, color_id)

class Clipboard:
    """
    Contains the clipboard.

    The clipboard stores, acccepts, and releases
    information to run the copy mechanism. This means
    listening for copy events and storing the provided
    state, and listening for paste events then invoking
    the callback.
    """

    # Define the important callbacks
    def on_paste_event(self,
                       callback: Callable[[Dict[str, Any]], None]
                       ):
        """Handles the emission of the block to the provided callback when a paste event is passed"""
        if self.clipboard is not None:
            if "debug" in self.config and self.config["debug"]:
                print("Clipboard is Emitting paste event in clipboard")

            block = copy.deepcopy(self.clipboard)
            callback(block)
        else:
            print("nothing to paste yet")

    def on_copy_event(self, block: Dict[str, Any]):
        """Handles the reception of new copy data"""
        if "debug" in self.config and self.config["debug"]:
            print("Clipboard Received block to store")
        self.clipboard = block

    # Setup the clipboard and its singular paste button.
    def __init__(self,
                 event_bus: EventBus,
                 config: Dict[str, Any],
                 parent=None):



        # Initialize the event bus and internal clipboard state
        self.event_bus = event_bus
        self.clipboard = None
        self.config = config

        # Add the listeners.
        self.event_bus.subscribe(Events.COPY_CELL.value, self.on_copy_event)
        self.event_bus.subscribe(Events.PASTE_CELL.value, self.on_paste_event)

class ZoneSelector(QFrame):
    """
    The Zone Selector lets the user switch between
    the various zones within a data case. These zones
    are in turn populated by BlockCellEditors, related
    to each zone being edited.

    A tab-based structure is used to display this information.
    """

    def setup_block_editors(self,
                            zones: Dict[str, List[Dict[str, Any]]]
                            ) -> Dict[str, 'BlockCellEditor']:
        """Sets up and returns a dictionary of block cell editors for each zone."""
        zone_editors = {}
        for zone_name, blocks in zones.items():
            zone_editors[zone_name] = BlockCellEditor(
                self.config,
                blocks,
                self.event_bus,
                self
            )
        return zone_editors

    def __init__(self,
                 zones: Dict[str, List[Dict[str, Any]]],
                 config: Dict[str, Any],
                 datacase_bus: EventBus,
                 parent=None):
        super().__init__(parent)

        # Set static attributes
        self.config = config
        self.event_bus = datacase_bus

        # Setup BlockCellEditors bound to self
        self.block_editors = self.setup_block_editors(zones)

        # Setup tab layout
        self.tab_widget = QTabWidget(self)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.tab_widget)
        self.setLayout(self.main_layout)

        # Insert each block editor as a tab
        for zone_name, editor in self.block_editors.items():
            self.tab_widget.addTab(editor, zone_name)

class ZoneDisplay(QFrame):
    """
    This is the primary cell editing display.
    It manages the zones and allows the user to copy blocks,
    edit zones, and control the palette of colors for the grid.
    """

    def forward_edit_events(self):
        """Lets the master bus know there are uncommitted changes"""
        self.master_bus.publish(Events.UNSAVED_CHANGES.value)
    def __init__(self,
                 zones: Dict[str, List[Dict[str, Any]]],
                 config: Dict[str, Any],
                 master_bus: EventBus,
                 parent=None):
        super().__init__(parent)

        # Setup the event bus for communication between the zones components,
        # and the master bus for communication at a higher level
        self.master_bus = master_bus
        self.event_bus = EventBus()

        # Initialize the clipboard for handling copy operations
        self.clipboard = Clipboard(self.event_bus, config, self)

        # Set up the layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Create the zone selector to manage zone-specific editors
        self.zone_selector = ZoneSelector(zones, config, self.event_bus, self)
        self.main_layout.addWidget(self.zone_selector)

        # Create the palette widget for selecting colors
        self.palette_widget = PaletteWidget(config["color_map"], self.event_bus, 14, self)
        self.main_layout.addWidget(self.palette_widget)

        # Listener
        self.event_bus.subscribe(Events.EDITMADE.value, self.forward_edit_events)

class ZonesManager(QFrame):
    """
    The Zone Manager is responsible for observing DISPLAY_ZONES
    messages and pulling up the appropriate zone. It transparently
    pretends to be the top-level controller for all zone displays.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 manager_event_bus: EventBus,
                 parent=None):
        super().__init__(parent)

        # Store the configuration and event bus
        self.config = config
        self.event_bus = manager_event_bus

        # Initialize the current zone display placeholder
        self.current_zone_display = None

        # Listen for display event requests.
        self.event_bus.subscribe(Events.DISPLAY_ZONES.value, self.display_new_zones)

        # Set up the layout for the ZonesManager
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

    def display_new_zones(self, zones: Zones):
        """
        A callback that will display a zone collection for editing.
        Called when a DISPLAY_ZONES event is received.
        """
        # Create a new ZoneDisplay with the passed zones
        zone_display = ZoneDisplay(zones, self.config, self.event_bus, self)

        # Replace the current zone display with the new one
        self.set_display(zone_display)

    def set_display(self, zone_display: ZoneDisplay):
        """
        Updates the internal GUI to show the new zone display.
        Removes the old one if present.
        """
        # If there is a current display, remove it
        if self.current_zone_display:
            self.main_layout.removeWidget(self.current_zone_display)
            self.current_zone_display.deleteLater()

        # Set the new zone display and add it to the layout
        self.current_zone_display = zone_display
        self.main_layout.addWidget(zone_display)



def main():
    app = QApplication(sys.argv)

    # Create the main window
    main_window = QWidget()
    main_layout = QVBoxLayout(main_window)

    # Create the EventBus object for handling events
    event_bus = EventBus()

    # Configuration for the grid editor
    config = {
        "debug": True,
        "grid_element_size": 20,
        "color_map": {0: "#FFFFFF", 1: "#000000", 2: "#FF0000"},  # White, Black, Red
        "cell_size": 50,  # Cell size for GridEditorCell
    }

    # Create the ZonesManager, which will listen for DISPLAY_ZONES events
    zone_manager = ZonesManager(config=config, manager_event_bus=event_bus, parent=main_window)

    # Add the ZonesManager to the main layout
    main_layout.addWidget(zone_manager)

    # Simulate the first display event
    zones = {
        "Zone 1": [
            {"mode": "text", "payload": "Sample text in TextCell for Zone 1."},  # TextCell block
            {"mode": "intgrid", "payload": np.zeros((5, 5), dtype=int)},  # GridEditorCell block for Zone 1
        ]
    }

    # Trigger the event to display the first zone (this would be handled by the system normally)
    event_bus.publish(Events.DISPLAY_ZONES.value, zones)

    # Configure the main window
    main_window.setWindowTitle("Zone Manager Example")
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


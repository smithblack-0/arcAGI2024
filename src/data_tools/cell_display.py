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
from typing import Dict, List, Any, Optional, Callable

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QScrollArea, QFrame,
                             QGridLayout, QLabel, QLineEdit, QSizePolicy,
                             QButtonGroup, QTabWidget
                             )
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QEvent, Qt
from src.data_tools.events import Events, EventBus
from src.data_tools.cell_editors import BlockCellEditor



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

        # Set frame style for the dotted border
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

    def select_color(self, color_id: int):
        """Emits the selected color ID when a button is clicked."""
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

class CellDisplay(QFrame):
    """
    This is the primary cell editing display.
    It manages the zones and allows the user to copy blocks,
    edit zones, and control the palette of colors for the grid.
    """

    def __init__(self,
                 zones: Dict[str, List[Dict[str, Any]]],
                 config: Dict[str, Any],
                 parent=None):
        super().__init__(parent)

        # Initialize the event bus for communication between components
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

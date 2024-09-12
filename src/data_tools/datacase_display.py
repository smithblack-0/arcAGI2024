import copy
import json
from typing import Dict, List, Any


from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QScrollArea, QSizePolicy, QWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor


from PyQt5.QtWidgets import (QFrame, QLabel, QVBoxLayout, QScrollArea, QPushButton,
                             QSizePolicy, QWidget, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from .events import Events, EventBus


from .events import Events, EventBus
from .mytypes import Examples, Zones
class IOAdapter:
    """
    A backend piece responsible for knowing how to load from,
    or save to, a particular file. At the moment it is set to
    json.
    """
    def __init__(self,
                 file: str,
                 ):
        self.file = file
    def load(self)->Examples:
        """
        Returns a dictionary defining examples, then zones
        filled with blocks.
        """
        with open(self.file) as f:
            return json.load(f)

    def save(self, data: Examples):
        """
        Saves a dictionary defined in terms of examples
        :param data:
        :return:
        """
        with open(self.file, 'w') as f:
            json.dump(data, f)
class DataSourceControl:
    """
    A manager for the json data feature being edited, the
    data source control:

    - Controls access to and from the data file actually
      being edited
    - Allows saving a new file with a different name
    - Allows an example to be checked out for editing, and will
      not let more than one example be checked out at a time.

    """

    # Define some very important callbacks that
    # we will need to bind events to in order for
    # everything to work.
    def attempt_commit_changes(self):
        """Commits changes to in memory data structure, and allows checking something else out"""
        if self.checked_out_name is not None:
            name = self.checked_out_name
            self.data[name] = self.checked_out_data


            # Free up slots
            self.checked_out_data = None
            self.checked_out_key = None

            # Emit checked in signal
            self.event_bus.publish(Events.CHECKIN.value, name)

    def attempt_revert_changes(self):
        """Reverts changes to in memory data structure, and allows checking something else out"""
        if self.checked_out_name is not None:
            # Check this is a sane key
            name = self.checked_out_name

            # Free up checked out slots
            self.checked_out_data = None
            self.checked_out_key = None

            # Emit checked in signal
            self.event_bus.publish(Events.CHECKIN.value, name)


    def attempt_checkout(self, name: str):
        """
        Attempts to check out an example case for editing within the gui.

        Checkout will only be allowed to proceed if nothing else is currently
        checked out. Otherwise, it will fail.
        :param name: The name of the entity being selected for check out
        """

        if self.checked_out_name is None:
            assert name in self.data

            # Check it out and store it
            self.checked_out_key = name
            self.checked_out_data = copy.deepcopy(self.data[name])

            # Publish it
            self.event_bus.publish(Events.CHECKOUT.value, name, self.checked_out_data)
        else:
            self.event_bus.publish(Events.CHECKOUT_REJECTED.value, self.checked_out_key)

    def save_data(self):
        """Saves the contents of the structure in memory."""
        self.io.save(self.data)

    ## Some useful communication functions
    def get_zones_names(self)->List[str]:
        """Gets a list of the names of all the zone collections out there"""
        return list(self.data.keys())

    def get_is_checked_out(self)->bool:
        """Returns a bool indicating whether or not something is checked out"""
        return self.checked_out_name is not None

    # Init

    def __init__(self,
                 io: IOAdapter,
                 config: Dict[str, Any],
                 event_bus: EventBus,
                 ):

        self.checked_out_data = None
        self.checked_out_name = None

        self.io = io
        self.data = self.io.load()
        self.config = config
        self.event_bus = event_bus

        self.event_bus.subscribe(Events.ATTEMPT_COMMIT.value, self.attempt_commit_changes)
        self.event_bus.subscribe(Events.ATTEMPT_REVERT.value, self.attempt_revert_changes)
        self.event_bus.subscribe(Events.ATTEMPT_CHECKOUT.value, self.attempt_checkout)
        self.event_bus.subscribe(Events.ASK_ZONES_NAMES.value, self.get_zones_names)
        self.event_bus.subscribe(Events.ASK_IS_ZONES_CHECKED_OUT.value, self.get_is_checked_out)




class ZonesKeyTextWidget(QFrame):
    """
    A widget that represents a single selectable
    zone. This can be selected, made inactive,
    or otherwise modified.
    """
    # Emits the associated key when clicked.
    clicked = pyqtSignal(str)

    def __init__(self, config, key: str, selection_bus: EventBus, parent=None):
        """
        Initialize the widget representing a key that is selectable.

        :param config: The configuration dictionary containing color settings.
        :param key: The string key representing the zone.
        :param selection_bus: An EventBus instance for broadcasting and listening to events.
        :param parent: Parent widget (if any).
        """
        super().__init__(parent)
        self.key_colors = config["key_colors"]  # Dict with color configurations for states
        self.key = key  # Store the key
        self.active = True  # By default, keys are active and selectable
        self.selection_bus = selection_bus  # Event bus for communicating selections

        # Define widget appearance
        self.label = QLabel(key, self)
        self.label.setFont(QFont('Arial', 12))  # Set font
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.set_active()  # Start in active state

        # Layout setup
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Listening for key-related events (activating, checking out keys)
        self.selection_bus.subscribe(Events.ACTIVATE_ALL_KEYS.value, self.set_active)
        self.selection_bus.subscribe(Events.CHECKOUT_KEY.value, self.checkout_key)

    # Define some important recoloring functions based on state
    def set_selected(self):
        """
        Updates the key's color to the 'selected' color when it's selected for checkout.
        """
        color = self.key_colors["selected"]
        self.label.setStyleSheet(f"color: {color}")

    def set_inactive(self):
        """
        Updates the key's color to the 'inactive' color when it's inactive (unavailable).
        """
        color = self.key_colors["inactive"]
        self.label.setStyleSheet(f"color: {color}")

    def checkout_key(self, name: str):
        """
        Handles the visual state when a key is checked out. If the key is checked out,
        it will be marked as selected, and other keys become inactive.
        """
        self.active = False  # Checked-out keys become inactive for selection
        if name == self.key:
            self.set_selected()
        else:
            self.set_inactive()

    def set_active(self):
        """
        Updates the key's color to the 'active' color when it is selectable.
        """
        color = self.key_colors["active"]
        self.active = True
        self.label.setStyleSheet(f"color: {color}")

    def mousePressEvent(self, event):
        """
        Emit the clicked signal when the element is clicked, allowing the parent to decide
        what to do with the key click event.
        """
        if self.active:
            self.clicked.emit(self.key)  # Emit the key name when clicked


class ScrollableKeysDisplay(QFrame):
    """
    A display to show all the zone keys in a scrollable area. This widget manages
    the display of keys and handles user interaction for selecting keys, checking
    them out, and updating their visual states.
    """

    def __init__(self, config, event_bus: EventBus, parent=None):
        """
        Initialize the scrollable display of zone keys.

        :param config: Configuration for color settings and other UI parameters.
        :param event_bus: The EventBus for managing global communication of events.
        :param parent: The parent widget (if any).
        """
        super().__init__(parent)
        self.event_bus = event_bus
        self.selection_bus = EventBus()  # Separate bus for handling key selections
        self.config = config

        # Get list of keys from the data source control
        self.keys = self.get_zone_names()
        self.current_selected_key = None  # Keep track of the currently selected key

        # Setup scroll area to make the key display scrollable
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        # Create a container widget for all the key text elements
        self.container = QWidget()
        self.scroll_layout = QVBoxLayout(self.container)

        # Add each key as a ZonesKeyTextWidget
        for key in self.keys:
            element = ZonesKeyTextWidget(config, key, self.selection_bus)
            element.clicked.connect(self.on_element_clicked)  # Handle key clicks
            self.scroll_layout.addWidget(element)

        self.scroll_area.setWidget(self.container)

        # Main layout for this frame
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)

    def get_zone_names(self) -> List[str]:
        """
        Retrieve the list of zone names from the event bus by asking the backend.
        :return: List of zone names.
        """
        reference = []
        self.event_bus.publish(Events.ASK_ZONES_NAMES.value, lambda x: reference.append(x))
        return reference.pop()

    def on_element_clicked(self, key: str):
        """
        Handle what happens when an element (key) is clicked. This will select the key
        if nothing is checked out or will forward the event to attempt a checkout.

        :param key: The string key of the element that was clicked.
        """
        # Forward the checkout event if nothing is checked out
        self.current_selected_key = key
        print(f"Selected: {key}")

    def forward_checkout_event(self, name: str, _):
        """
        Forward the checkout event from other parts of the model to the local responsibilities.
        :param name: The name of the key being checked out.
        :param _: Additional arguments (not used here).
        """
        self.selection_bus.publish(Events.CHECKOUT_KEY.value, name)


class KeysAndButtonsPanel(QFrame):
    """
    This widget contains the scrollable list of keys and the "save", "commit", and "revert"
    buttons at the bottom of the panel. It listens for and emits events as needed.
    """

    def __init__(self, config, event_bus: EventBus, parent=None):
        """
        Initialize the widget containing the scrollable key list and the control buttons.

        :param config: Configuration settings for colors and other UI elements.
        :param event_bus: The EventBus for communication with the data source and other UI elements.
        :param parent: Parent widget (if any).
        """
        super().__init__(parent)
        self.event_bus = event_bus

        # Scrollable display for keys
        self.keys_display = ScrollableKeysDisplay(config, event_bus, self)

        # Layout setup for keys and buttons
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.keys_display)

        # Buttons: save, commit, and revert
        self.button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save", self)
        self.commit_button = QPushButton("Commit", self)
        self.revert_button = QPushButton("Revert", self)

        # Connect buttons to emit corresponding events
        self.save_button.clicked.connect(lambda: self.event_bus.publish(Events.ATTEMPT_SAVE.value))
        self.commit_button.clicked.connect(lambda: self.event_bus.publish(Events.ATTEMPT_COMMIT.value))
        self.revert_button.clicked.connect(lambda: self.event_bus.publish(Events.ATTEMPT_REVERT.value))

        # Add buttons to the layout
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.commit_button)
        self.button_layout.addWidget(self.revert_button)

        # Add buttons layout to the main layout
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)

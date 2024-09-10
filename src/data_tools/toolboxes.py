"""
Toolboxes are collections of tools that are assigned
per data type we are attempting to edit, and which
can be used to modify how such data is being manipulated

"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTabWidget, QButtonGroup, QPushButton, \
    QHBoxLayout, QLineEdit, QSizePolicy, QFrame
from typing import Dict, Any, Optional
from PyQt5.QtCore import QObject, pyqtSignal

from src.data_tools.events import EventBus, Events


class AbstractTool(QWidget):
    """
    AbstractTool serves as a base class for creating custom tools.
    Tools are generally expected to make changes through the event
    bus.
    """

    def __init__(self, parent: QWidget, config: Dict[str, Any], events_bus: QObject):
        super().__init__(parent)

        self.parent = parent
        self.config = config
        self.events_bus = events_bus



# Toolbox class, similar to CellEditorZone
class Toolbox(QWidget):
    """
    Toolbox is a container for tools, with tabs for each tool.

    Attributes:
    - config: The configuration dictionary passed to the tools.
    - events_bus: The events bus used to subscribe to or emit events.
    """

    def __init__(self, config: Dict[str, Any], events_bus: QObject, parent: QWidget = None):
        super().__init__(parent)

        self.config = config
        self.events_bus = events_bus

        # Set up the main layout for the toolbox
        self.layout = QVBoxLayout(self)

        # Create a QTabWidget to hold tools as tabs
        self.tabs = QTabWidget(self)

        # Add the QTabWidget to the main layout
        self.layout.addWidget(self.tabs)

    def add_tool(self, tool_name: str, tool_widget: AbstractTool):
        """
        Adds a tool as a new tab in the toolbox.
        :param tool_name: The label for the tool's tab.
        :param tool_widget: An instance of a tool that inherits from AbstractTool.
        """
        self.tabs.addTab(tool_widget, tool_name)



# The text associated toolbox
class TextTool(AbstractTool):
    """
    A simple tool that displays some text. Inherits from AbstractTool.
    """

    def __init__(self, parent: QWidget, config: Dict[str, Any], events_bus: QObject):
        super().__init__(parent, config, events_bus)

        # Example tool content
        self.layout = QVBoxLayout(self)
        self.label = QLabel("There are no tools available for text boxes at this time", self)
        self.layout.addWidget(self.label)

# Define the tool for int grid editing. This includes the palette and shape widget.


class ArcIntGridTool(AbstractTool):
    """
    A tool for editing ArcInt grids, containing the PaletteWidget and ShapeWidget.
    This tool allows the user to interact with the grid by changing colors and resizing the grid.
    """

    def __init__(self, parent=None, config=None, events_bus=None):
        super().__init__(parent, config, events_bus)

        # Extract configuration details
        color_map = config['color_map']
        grid_shape = config.get("default_grid_shape", (5, 5))  # Default shape if not provided
        title_size = config["font_sizes"]["title"]

        # Initialize and attach widgets
        self.palette_frame = PaletteWidget(color_map, self.events_bus, title_size, self)
        self.shape_frame = ShapeWidget(grid_shape[0], grid_shape[1], events_bus, self)

        # Main layout (vertical) for the tool
        self.main_layout = QVBoxLayout(self)

        # Create a horizontal layout to align PaletteWidget and ShapeWidget side by side
        self.tool_layout = QVBoxLayout()

        # Add PaletteWidget and ShapeWidget to the horizontal layout
        self.tool_layout.addWidget(self.palette_frame)
        self.tool_layout.addWidget(self.shape_frame)

        # Add the tool layout (horizontal) to the main vertical layout
        self.main_layout.addLayout(self.tool_layout)

        # Apply the layout
        self.setLayout(self.main_layout)

def main():
    app = QApplication(sys.argv)

    # Create an event bus
    event_bus = EventBus()

    # Create the main window
    main_window = QWidget()
    main_layout = QVBoxLayout(main_window)

    # Configuration for tools
    config = {
        "font_sizes" : {"title" : 10},
        "color_map": {0: "#FFFFFF", 1: "#000000", 2: "#FF0000"},  # White, Black, Red
        "grid_shape": (5, 5)  # Default grid shape
    }

    # Create the toolbox
    toolbox = Toolbox(config=config, events_bus=event_bus)

    # Add tools to the toolbox
    toolbox.add_tool("Text Tool", TextTool(toolbox, config, event_bus))
    toolbox.add_tool("ArcIntGrid Tool", ArcIntGridTool(toolbox, config, event_bus))

    # Add toolbox to the main layout
    main_layout.addWidget(toolbox)

    # Set up the main window
    main_window.setWindowTitle("Toolbox Example with PyQt5")
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()

    sys.exit(app.exec_())


def test_palette_widget():
    app = QApplication(sys.argv)

    # Sample color map for testing
    color_map = {0: "#FFFFFF", 1: "#000000", 2: "#FF0000"}  # White, Black, Red

    # Create a simple window to display the PaletteWidget
    window = QWidget()
    layout = QVBoxLayout(window)

    event_bus = EventBus()

    # Create the PaletteWidget
    palette_widget = PaletteWidget(color_map, event_bus)

    # Add the palette to the layout
    layout.addWidget(palette_widget)

    # Add a label to show the selected color
    selected_color_label = QLabel("Selected Color: None")
    layout.addWidget(selected_color_label)

    # Subscribe to the event bus for color changes
    def on_palette_change(color_id):
        selected_color_label.setText(f"Selected Color: {color_id}")

    event_bus.subscribe(Events.PALETTE_CHANGED, on_palette_change)

    # Setup window details and start application
    window.setWindowTitle("Palette Widget Test")
    window.setGeometry(100, 100, 400, 100)
    window.show()

    sys.exit(app.exec_())


def test_shape_widget():
    app = QApplication(sys.argv)

    # Create a simple window to display the ShapeWidget
    window = QWidget()
    layout = QVBoxLayout(window)

    event_bus = EventBus()

    # Create the ShapeWidget
    shape_widget = ShapeWidget(5, 5, event_bus)  # Default to 5x5 shape

    # Add the shape widget to the layout
    layout.addWidget(shape_widget)

    # Add a label to show the selected shape
    selected_shape_label = QLabel("Selected Shape: 5 X 5")
    layout.addWidget(selected_shape_label)

    # Subscribe to the event bus for shape changes
    def on_shape_change(rows, cols):
        selected_shape_label.setText(f"Selected Shape: {rows} X {cols}")

    event_bus.subscribe(Events.SHAPE_CHANGED, on_shape_change)

    # Setup window details and start application
    window.setWindowTitle("Shape Widget Test")
    window.setGeometry(100, 100, 400, 200)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

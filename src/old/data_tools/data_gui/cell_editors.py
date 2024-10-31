import copy
import sys
import numpy as np
from typing import Optional, Callable

from PyQt5.QtWidgets import (QApplication, QWidget, QTextEdit, QScrollArea, QGridLayout, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from src.old.data_tools.data_gui.mytypes import Block, Blocks

#import copy
from typing import Dict, Any
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QPushButton, QHBoxLayout
from src.old.data_tools.data_gui.events import Events, EventBus


class AbstractCell(QFrame):
    """
    The abstract cell class is a QFrame designed to be docked into a CellEditor.
    It provides core functionality for managing payload data of a given block,
    emitting an event when an edit is made.

    ---- Public Attributes ----
    config: Dict[str, Any]
        Configuration dictionary for the cell.
    event_bus: EventBus
        Event bus used for publishing and subscribing to global events.

    ---- Public Methods ----
    - `delete_self()`: Deletes this cell from the parent.
    - `promote_self()`: Moves this cell higher in the list.
    - `demote_self()`: Moves this cell lower in the list.
    - `copy_self()`: Copies this cell into the clipboard.
    - `paste_above()`: Pastes the clipboard content above this cell.
    - `paste_below()`: Pastes the clipboard content below this cell.

    ---- Properties ----
    - `payload`: Getter and setter for the payload content, emitting an event on edit.
    - `mode`: Getter for the cell's mode.
    """

    # ----- Getter and Setter for Payload -----
    def get_payload(self) -> Any:
        """Returns a deepcopy of the current payload."""
        return copy.deepcopy(self._block["payload"])

    def set_payload(self, new_payload: Any):
        """Sets a new payload and emits the EDITMADE event."""
        if "debug" in self.config and self.config["debug"]:
            print("Setting new payload")
            print(new_payload)
        self._block["payload"] = new_payload
        self.event_bus.publish(Events.EDITMADE.value)

    @property
    def mode(self) -> str:
        """Returns the mode of the cell."""
        return self._block["mode"]

    # ----- Payload Initialization -----
    def default_payload(self) -> Any:
        """
        Returns a default payload if none is provided in the block.
        Must be implemented by subclasses to define the default behavior.
        """
        raise NotImplementedError("Subclasses must implement default_payload")

    # ----- Initialization and Layout -----
    def __init__(self,
                 parent: QFrame,
                 config: Dict[str, Any],
                 block: Dict[str, Any],
                 event_bus: EventBus):
        """
        Initializes the abstract cell with configuration, block data, and event bus.

        :param parent: The parent widget to attach to.
        :param config: The configuration dictionary.
        :param block: The block dictionary reference for the cell data.
        :param event_bus: EventBus used for emitting and subscribing to events.
        """
        super().__init__(parent)

        # Store the block privately
        self._block = block

        # Set static parameters
        self.config = config
        self.event_bus = event_bus
        self.parent = parent

        # If the payload has not been set, assign a default payload
        if "payload" not in self._block or self._block["payload"] is None:
            self.set_payload(self.default_payload())



        # Setup the layout for buttons and cell management
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.layout = QVBoxLayout(self)
        self.button_layout = QHBoxLayout()

        # Add buttons for managing the cell
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.clicked.connect(self.delete_self)
        self.button_layout.addWidget(self.delete_button)

        self.promote_button = QPushButton("Promote", self)
        self.promote_button.clicked.connect(self.promote_self)
        self.button_layout.addWidget(self.promote_button)

        self.demote_button = QPushButton("Demote", self)
        self.demote_button.clicked.connect(self.demote_self)
        self.button_layout.addWidget(self.demote_button)

        self.copy_button = QPushButton("Copy", self)
        self.copy_button.clicked.connect(self.copy_self)
        self.button_layout.addWidget(self.copy_button)

        self.paste_above_button = QPushButton("Paste Above", self)
        self.paste_above_button.clicked.connect(self.paste_above)
        self.button_layout.addWidget(self.paste_above_button)

        self.paste_below_button = QPushButton("Paste Below", self)
        self.paste_below_button.clicked.connect(self.paste_below)
        self.button_layout.addWidget(self.paste_below_button)

        self.layout.addLayout(self.button_layout)

    # ----- Cell Management Methods -----
    def delete_self(self):
        """Deletes this cell by calling the parent delete function."""
        self.parent.delete_cell(self)

    def promote_self(self):
        """Promotes this cell by invoking the parent promote function."""
        self.parent.promote_cell(self)

    def demote_self(self):
        """Demotes this cell by invoking the parent demote function."""
        self.parent.demote_cell(self)

    def copy_self(self):
        """Copies this cell into the clipboard."""
        self.parent.copy_cell(self)

    def paste_above(self):
        """Pastes the clipboard content above this cell."""
        self.parent.paste_cell_above(self)

    def paste_below(self):
        """Pastes the clipboard content below this cell."""
        self.parent.paste_cell_below(self)

class BlockCellEditor(QWidget):
    """
    A cell editor widget designed to allow for the
    creation and deletion of cells.

    It is designed to allow the seamless editing of data
    blocks in a manner reminiscent of ipynb cells. In particular,
    individual blocks act as cells in which edits can be made.
    Cells can be added, or deleted, using provided utilities.
    """

    ##
    # Define the class level registry for cell types, and
    # the registration function. This will track what
    # cells are active and such
    ##

    cells = {}  # Class-level registry for cell types
    @classmethod
    def register_cell(cls, cell_type, cell_class):
        """Register a new cell type and create a corresponding button in the UI."""
        cls.cells[cell_type] = cell_class

    @classmethod
    def register_decorator(cls, cell_type) -> Callable[[AbstractCell], AbstractCell]:
        def callback(cell: AbstractCell):
            cls.register_cell(cell_type, cell)
            return cell
        return callback

    ##
    # Define some helper functions
    ##

    def get_index_based_on_cell(self, cell: AbstractCell)->int:
        """Gets the index based on the cell"""
        # Unfortunately, list.index uses == for tracking down
        # the index, rather than "is". This causes it to choke on
        # numpy arrays. So we implement our own flavor.
        target_block = cell._block
        for i, block in enumerate(self.blocks):
            if target_block is block:
                return i
        raise KeyError("Cound not find index")
    ##
    # Define cell manipulator functions for promotion, demotion,
    # addition, and deletion, plus whatever other abstract cell callbacks are needed.
    ##

    def promote_cell(self, cell: AbstractCell):
        """Promote cell to live higher on the list"""
        index = self.get_index_based_on_cell(cell)
        if index == 0:
            # Do nothing. Cannot promote past zero.
            return
        block = self.blocks.pop(index)
        self.blocks.insert(index-1, block)
        self.render_cells()
    def demote_cell(self, cell: AbstractCell):
        """Demote cell to live lower on the list"""
        index = self.get_index_based_on_cell(cell)
        if index == len(self.blocks) - 1:
            # Do nothing. Cannot demote any further
            return

        # Move the cell and the block around
        block = self.blocks.pop(index)
        self.blocks.insert(index+1, block)

        # render
        self.render_cells()

    def delete_cell(self, cell):
        """Removes a block and its cell"""
        index = self.get_index_based_on_cell(cell)
        self.blocks.pop(index)
        cell.deleteLater()
        self.render_cells()

    def copy_cell(self, cell: AbstractCell):
        """Code to invoke a copy of a cell at a particular index"""
        index = self.get_index_based_on_cell(cell)
        block = self.blocks[index]
        block = copy.deepcopy(block)
        self.event_bus.publish(Events.COPY_CELL.value, block)

    def get_contents_of_clipboard(self)->Dict[str, Any]:
        # Get the current paste value in the clipboard.
        #
        # We have to do some nasty reference work since python
        # does not sanely support dereferencing pointers. So we isolate it
        # in here

        block_response = []
        append_block = lambda block: block_response.append(block)
        self.event_bus.publish(Events.PASTE_CELL.value, append_block)
        if len(block_response) != 0:
            return block_response.pop()
        return None

    def paste_cell_above(self, cell: AbstractCell):
        """Paste the contents of the clipboard above the indicated cell"""
        index = self.get_index_based_on_cell(cell)
        block = self.get_contents_of_clipboard()
        if block is not None:
            # Insert it.
            self.blocks.insert(index, block)
            self.render_cells()
    def paste_cell_below(self, cell: AbstractCell):
        """Paste the contents of the clipboard below the current cell"""
        index = self.get_index_based_on_cell(cell)
        block = self.get_contents_of_clipboard()
        if block is not None:
            # Insert it. Or append if appropriate
            target_index = index + 1
            if target_index == len(self.blocks):
                self.blocks.append(block)
            else:
                self.blocks.insert(index, block)
            self.render_cells()

    def new_cell(self, cell_type: str):
        block = {"mode" : cell_type}
        self.blocks.append(block)
        self.render_cells()

    ###
    # Rendering mechanism. Matches blocks to cells, when
    # invoked
    ##
    def render_cells(self):
        """ Renders all cells to be synchronized with cell instance"""
        # Remove all the widgets that are currently present
        while self.scroll_area_layout.count() > 0:
            item = self.scroll_area_layout.itemAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()  # Schedule widget for deletion
                self.scroll_area_layout.removeWidget(widget)

        # Construct and populate layout based on block
        for block in self.blocks:
            mode = block["mode"]
            cell_class = BlockCellEditor.cells[mode]
            cell_instance = cell_class(self, self.config, block, self.event_bus)
            self.scroll_area_layout.addWidget(cell_instance)

    def __init__(self,
                 config: Dict[str, Any],
                 blocks: Blocks,
                 event_bus: EventBus,
                 parent: Optional[Any] = None):

        super().__init__(parent)

        # Store the config dictionary.
        #
        # This will be passed along when creating cells.
        # Also, setup selected cell to none

        self.config = config
        self.blocks = blocks
        self.event_bus = event_bus

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Scroll area to hold the cells
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.scroll_area.setWidget(self.scroll_area_widget)

        # Button layout
        self.button_layout = QHBoxLayout()

        # Add the widgets to the main layout
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addLayout(self.button_layout)

        # Setup the cell buttons for the first time
        self.setup_cell_buttons()

        # Render for the first time
        self.render_cells()

    def setup_cell_buttons(self):
        """Creates buttons dynamically for all registered cells."""
        for cell_type in self.cells:
            button = QPushButton(f"Add {cell_type.capitalize()} Cell")
            button.clicked.connect(lambda _, c=cell_type: self.new_cell(c))
            self.button_layout.addWidget(button)


@BlockCellEditor.register_decorator("text")
class TextCell(AbstractCell):
    """
    A text cell editor

    Allows the editing of text in terms of a cell.
    """

    def default_payload(self):
        return ""

    def communicate_update(self):
        self.set_payload(self.text_edit.toPlainText())
    def __init__(self,
                 parent: BlockCellEditor,
                 config: Dict[str, Any],
                 block: Block,
                 event_bus: EventBus):
        super().__init__(parent, config, block, event_bus)

        # Text editor specific to the TextCell
        self.text_edit = QTextEdit(self)
        self.text_edit.setText(self.get_payload())
        self.text_edit.textChanged.connect(self.communicate_update)

        # Add the text editor to the layout
        self.layout.insertWidget(0, self.text_edit)  # Insert at the top before the delete button


# Define the grid editor cell, and all of the subwidgets

class ShapeFrame(QFrame):
    """
    A widget that allows the user to set the shape of the grid (rows x columns).
    Emits the new shape when the "Resize" button is pressed.
    """
    def __init__(self,
                 reshape_bus: EventBus,
                 parent=None):

        super().__init__(parent)
        # bus
        self.event_bus = reshape_bus

        # Layout for the widget
        self.main_layout = QVBoxLayout(self)

        # Title
        self.title = QLabel("Grid Shape Editing")
        self.title.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title)

        # Row and Column input fields
        self.shape_layout = QHBoxLayout()
        self.row_input = QLineEdit("5")  # Default value
        self.row_input.setMinimumWidth(50)  # Ensure the input fields have some width
        self.col_input = QLineEdit("5")  # Default value
        self.col_input.setMinimumWidth(50)
        self.shape_layout.addWidget(self.row_input)
        self.shape_layout.addWidget(QLabel("X"))
        self.shape_layout.addWidget(self.col_input)
        self.main_layout.addLayout(self.shape_layout)

        # Resize button
        self.resize_button = QPushButton("Resize")
        self.resize_button.clicked.connect(self.emit_resize_signal)
        self.main_layout.addWidget(self.resize_button)

        # Set frame style for the dotted border
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

    def emit_resize_signal(self):
        """Emits the new shape (rows, cols) when the Resize button is pressed."""
        try:
            rows = int(self.row_input.text())
            cols = int(self.col_input.text())
            self.event_bus.publish(Events.SHAPE_CHANGED.value, rows, cols)  # Emit the signal with the new shape
        except ValueError:
            pass  # Ignore invalid input

    def set_shape(self, rows: int, cols: int):
        """Sets the current shape in the input fields."""
        self.row_input.setText(str(rows))
        self.col_input.setText(str(cols))
class ColorSquare(QFrame):
    """
    Represents a single square in the grid. It changes color based on the current palette and
    updates the grid's payload when clicked.
    """
    def __init__(self,
                 grid: "Grid",
                 row: int,
                 col: int,
                 color_id: int,
                 color_map: Dict[int, str],
                 square_size: int):
        super().__init__(grid)
        self.grid = grid
        self.row = row
        self.col = col
        self.color_id = color_id
        self.color_map = color_map
        self.square_size = square_size
        self.setFixedSize(square_size, square_size)
        self.setStyleSheet(f"background-color: {self.color_map[color_id]};")
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

    def update_color(self, new_color_id: int):
        """Updates the square's color."""
        self.color_id = new_color_id
        self.setStyleSheet(f"background-color: {self.color_map[self.color_id]};")

    def mousePressEvent(self, event):
        """Handles mouse click events to change the color of the square."""
        new_color_id = self.grid.palette
        self.grid.update_square_color(self.row, self.col, new_color_id)

class Grid(QFrame):
    """
    A frame containing clickable elements that can be recolored
    according to the palette. It interacts directly with the parent cell's payload,
    fetching and updating the grid state via the parent.
    """

    def __init__(self,
                 reshape_bus: EventBus,
                 palette_bus: EventBus,
                 color_map: Dict[int, str],
                 square_size: int,
                 parent: AbstractCell):
        super().__init__(parent)

        # Save configuration and parent reference
        self.palette = 0  # Default color selection
        self.color_map = color_map
        self.square_size = square_size
        self.parent = parent
        self.squares = {}  # To store ColorSquare instances by (row, col)

        # Set up signals
        reshape_bus.subscribe(Events.SHAPE_CHANGED.value, self.on_shape_changed)
        palette_bus.subscribe(Events.PALETTE_CHANGED.value, self.on_palette_changed)

        # Initialize palette
        palette_bus.publish(Events.PALETTEREQUEST.value, self.on_palette_changed)

        # Set up the layout to display the grid
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(1)  # Adjust spacing between squares
        self.setLayout(self.grid_layout)

        # Ensure that the aspect ratio is maintained and the grid is not resizable independently
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Draw the grid for the first time
        self.repaint_grid()

        # Set frame style
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

    def sizeHint(self):
        """Override sizeHint to ensure a fixed aspect ratio."""
        rows, cols = self.parent.get_payload().shape
        total_width = cols * self.square_size
        total_height = rows * self.square_size
        return QSize(total_width, total_height)

    def repaint_grid(self):
        """Rebuild the grid of squares based on the current state of the parent's payload."""
        # Clear the current layout
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get the current grid dimensions from the parent's payload
        grid_data = self.parent.get_payload()
        try:
            rows, cols = grid_data.shape
        except Exception as err:
            print(err)
            raise err

        # Create grid squares and store them
        self.squares.clear()
        for row in range(rows):
            for col in range(cols):
                color_id = grid_data[row, col]
                square = ColorSquare(self, row, col, color_id, self.color_map, self.square_size)
                self.squares[(row, col)] = square
                self.grid_layout.addWidget(square, row, col)

    def on_shape_changed(self, rows: int, cols: int):
        """Resizes the grid and updates the parent's payload."""
        new_grid = np.zeros((rows, cols), dtype=int)
        old_grid = self.parent.get_payload()

        # Copy over the old grid values to the new grid, adding padding if needed
        min_rows = min(rows, old_grid.shape[0])
        min_cols = min(cols, old_grid.shape[1])
        new_grid[:min_rows, :min_cols] = old_grid[:min_rows, :min_cols]

        # Update the parent's payload with the new grid
        self.parent.set_payload(new_grid)
        self.repaint_grid()

    def on_palette_changed(self, palette: int):
        """Updates the currently selected palette color."""
        self.palette = palette

    def update_square_color(self, row: int, col: int, new_color_id: int):
        """Updates the selected square with the current palette color."""
        grid_data = self.parent.get_payload()  # Get the current grid from the parent
        grid_data[row, col] = new_color_id

        # Update the parent's payload to trigger further updates
        self.parent.set_payload(grid_data)

        # Update the specific square color without redrawing the entire grid
        square = self.squares[(row, col)]
        square.update_color(new_color_id)


@BlockCellEditor.register_decorator("intgrid")
class GridEditorCell(AbstractCell):
    """
    An editor for arc-agi grid data. This allows the viewing of
    a grid of ints in terms of colors provided in the config.

    This matches up to how arc-agi stores its data. The
    user can edit the colors to be the current color palette
    color, or reshape the grid inside.
    """

    def get_payload(self) -> np.ndarray:
        """Reimplimentation of get paylaod to avoid deepcopy hit"""
        return np.copy(self._block["payload"])

    def default_payload(self) -> Any:

        return np.zeros([5, 5], dtype=int)

    def __init__(self,
                 parent: QWidget,
                 config: Dict[str, Any],
                 block: Block,
                 globals_bus: EventBus):
        super().__init__(parent, config, block, globals_bus)

        # Setup layout
        self.display_layout = QHBoxLayout()
        self.setLayout(self.layout)

        # Get buses ready, and items out of config
        reshape_bus = EventBus()
        color_map = config["color_map"]
        button_size = config["grid_element_size"]

        # Create grid display
        self.grid_display = Grid(reshape_bus=reshape_bus,
                                 palette_bus=globals_bus,
                                 color_map=color_map,
                                 square_size=button_size,
                                 parent=self)
        self.display_layout.addWidget(self.grid_display)

        # Create reshape frame
        self.reshape_frame = ShapeFrame(reshape_bus, self)
        self.display_layout.addWidget(self.reshape_frame)

        # Add the layouts
        self.layout.insertLayout(0, self.display_layout)

def main():
    app = QApplication(sys.argv)

    # Create the main window
    main_window = QWidget()
    main_layout = QVBoxLayout(main_window)

    # Create the EventBus object for handling events
    event_bus = EventBus()

    # Configuration for the grid editor
    config = {
        "debug" : True,
        "grid_element_size" : 20,
        "color_map": {0: "#FFFFFF", 1: "#000000", 2: "#FF0000"},  # White, Black, Red
        "cell_size": 50,  # Cell size for GridEditorCell
    }

    # Initialize a list of blocks for testing
    blocks = [
        {"mode": "text", "payload": "Sample text in TextCell."},  # TextCell block
        {"mode": "intgrid", "payload": np.zeros((5, 5), dtype=int)},  # GridEditorCell block with a 5x5 grid
    ]

    # Create the CellEditorZone with the config, block list, and event bus
    cell_editor = BlockCellEditor(config=config, blocks=blocks, event_bus=event_bus)

    # Add the CellEditorZone to the main layout
    main_layout.addWidget(cell_editor)

    # Configure the main window
    main_window.setWindowTitle("Cell Editor Zone Example")
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
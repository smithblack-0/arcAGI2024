import copy
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Callable

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QScrollArea, QFrame,
                             QGridLayout, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QEvent, Qt
from src.data_tools.events import Events, EventBus


#
class AbstractCell(QFrame):
    """
    The abstract cell class is a QFrame that is
    designed to be docked into a CellEditor. Think
    along the lines of ipynb cells, and you will
    not be too wrong.

    ---- user attributes ----

    config: The config items
    event_bus:
        - The EventBus object, which allows various global events.
    """

    # Setup the two subclass functions
    # designed to handle payload manipulation
    # and default assignment

    def default_payload(self) -> Any:
        """Sets up a default payload for a new block."""
        raise NotImplementedError("Need to implement default payload to add cells.")

    def commit_payload(self, payload: Any):
        """Commits a fresh payload to the block"""
        if "debug" in self.config and self.config["debug"]:
            print("change in payload follows")
            print(payload)
        self.block['payload'] = payload

    # Setup some important cell callbacks
    # used to interact with the broader
    # gui.
    def delete_self(self):
        """Deletes this cell by calling the parent delete function."""
        self.parent.delete_cell(self)

    def promote_self(self):
        """Promotes cell in cell list by invoking parent promote function."""
        self.parent.promote_cell(self)

    def demote_self(self):
        """Demotes cell in cell list by invoking parent demote function"""
        self.parent.demote_cell(self)

    def copy_self(self):
        """Invokes the parent's copy function to move cell block into clipboard."""
        self.parent.copy_cell(self)

    def paste_above(self):
        """Pastes the contents of the clipboard above the cell"""
        self.parent.paste_cell_above(self)

    def paste_below(self):
        """Pastes the contents of the clipboard below the cell"""
        self.parent.paste_cell_below(self)
    def __init__(self,
                 parent: QWidget,
                 config: Dict[str, Any],
                 block: Dict[str, Any],
                 event_bus: EventBus):
        """
        Abstract cells must be initialized with several
        things. These things are:

        :param parent: The parent to attach to.
        :param config: The dictionary with any needed config info
        :param block: The block dictionary reference. This will always exist,
                      but may not always have a payload reference.
        :param event_bus: The EventBus for communication.
        """

        super().__init__(parent)

        # If the payload has not been set up, set a default payload
        if "payload" not in block:
            block["payload"] = self.default_payload()

        # Store the parameters for later use
        self.config = config
        self.parent = parent
        self.block = block
        self.event_bus = event_bus

        # Layout for the abstract cell
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.layout = QVBoxLayout(self)

        # Create a horizontal layout for the buttons
        self.button_layout = QHBoxLayout()

        # Delete button common to all cell types
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.clicked.connect(self.delete_self)
        self.button_layout.addWidget(self.delete_button)

        # Promote button to move higher in general layout
        self.promote_button = QPushButton("Promote", self)
        self.promote_button.clicked.connect(self.promote_self)
        self.button_layout.addWidget(self.promote_button)

        # Demote button to move lower in general layout
        self.demote_button = QPushButton("Demote", self)
        self.demote_button.clicked.connect(self.demote_self)
        self.button_layout.addWidget(self.demote_button)

        # Copy button to move a copy of this cell into the clipboard
        self.copy_button = QPushButton("Copy", self)
        self.copy_button.clicked.connect(self.copy_self)
        self.button_layout.addWidget(self.copy_button)

        # Paste above button. Does what it says
        self.paste_above_button = QPushButton("Paste Above", self)
        self.paste_above_button.clicked.connect(self.paste_above)
        self.button_layout.addWidget(self.paste_above_button)

        # Paste below button.
        self.paste_below_button = QPushButton("Paste Below", self)
        self.paste_below_button.clicked.connect(self.paste_below)
        self.button_layout.addWidget(self.paste_below_button)

        # Add the button layout (horizontal) to the main layout (vertical)
        self.layout.addLayout(self.button_layout)


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

    def get_index_based_on_cell(self, cell: Dict[str, Any])->int:
        return self.blocks.index(cell.block)

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
        if index == len(self.cells_instance) - 1:
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
        block = block_response.pop()
        return block

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
            if target_index == len(self.cells_instance):
                self.blocks.append(block)
            else:
                self.blocks.insert(block)
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
                 blocks: List[Dict[str, Any]],
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
        self.commit_payload(self.text_edit.toPlainText())

    def __init__(self, parent: BlockCellEditor, config: Dict[str, Any], block: Dict[str, Any], event_bus: EventBus):
        super().__init__(parent, config, block, event_bus)

        # Text editor specific to the TextCell
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Enter some text...")
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


class Grid(QFrame):
    """
    A frame containing clickable elements that can be recolored
    according to the palette. It is synchronized to a numpy
    backend, and listens for palette change or shape change events.
    """

    def on_shape_changed(self, rows: int, cols: int):
        """Resizes the grid and updates the grid collection."""
        new_grid = np.zeros((rows, cols), dtype=int)
        old_grid = self.block['payload']

        # Copy over the old grid values to the new grid, adding padding if needed
        min_rows = min(rows, old_grid.shape[0])
        min_cols = min(cols, old_grid.shape[1])
        new_grid[:min_rows, :min_cols] = old_grid[:min_rows, :min_cols]

        self.block['payload'] = new_grid
        self.parent.commit_payload(self.block["payload"])
        self.repaint()  # Redraw the grid with the new shape

    def on_palette_changed(self, palette: int):
        """Updates the currently selected palette color."""
        self.palette = palette

    def update_square_color(self, row: int, col: int):
        """Updates the selected square with the current palette color."""
        self.block["payload"][row, col] = self.palette
        self.repaint()  # Redraw the grid with the updated colors
        self.parent.commit_payload(self.block["payload"])

    def __init__(self,
                 block: Dict[str, Any],
                 reshape_bus: EventBus,
                 palette_bus: EventBus,
                 color_map: Dict[int, str],
                 button_size: int,
                 parent: AbstractCell,
                 ):
        super().__init__(parent)

        # Save configuration and block state
        self.block = block
        self.palette = 0  # Default color selection
        self.color_map = color_map
        self.button_size = button_size
        self.parent = parent

        # Set up signals
        reshape_bus.subscribe(Events.SHAPE_CHANGED.value, self.on_shape_changed)
        palette_bus.subscribe(Events.PALETTE_CHANGED.value, self.on_palette_changed)

        # Set up the layout to display the grid
        self.grid_layout = QGridLayout(self)
        self.setLayout(self.grid_layout)

        # Draw the grid for the first time
        self.repaint()

        # Set frame style
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

    def repaint(self):
        """Redraw the grid of squares based on the current state of block['payload']."""
        # Clear the current layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Get the current grid dimensions from the block's payload
        grid_data = self.block["payload"]
        rows, cols = grid_data.shape

        # Create grid buttons
        for row in range(rows):
            for col in range(cols):
                button = QPushButton(self)
                button.setFixedSize(self.button_size, self.button_size)  # Set the size of each square
                color_id = grid_data[row, col]
                button.setStyleSheet(f"background-color: {self.color_map[color_id]};")

                # Connect the click event to update the corresponding square
                button.clicked.connect(lambda _, r=row, c=col: self.update_square_color(r, c))

                # Add the button to the grid layout
                self.grid_layout.addWidget(button, row, col)

@BlockCellEditor.register_decorator("intgrid")
class GridEditorCell(AbstractCell):
    """
    An editor for arc-agi grid data. This allows the viewing of
    a grid of ints in terms of colors provided in the config.

    This matches up to how arc-agi stores its data. The
    user can edit the colors to be the current color palette
    color, or reshape the grid inside.
    """
    def default_payload(self) -> Any:
        return np.zeros([1, 1])
    def __init__(self,
                 parent: QWidget,
                 config: Dict[str, Any],
                 block: Dict[str, Any],
                 globals_bus: EventBus
                 ):

        super().__init__(parent, config, block, globals_bus)

        # Setup layout
        self.display_layout = QHBoxLayout()
        self.setLayout(self.layout)

        # Get buses ready, and items out of config
        reshape_bus = EventBus()
        color_map = config["color_map"]
        button_size = config["grid_element_size"]

        # Create display. Hopefully working
        self.grid_display = Grid(block,
                                 reshape_bus,
                                 globals_bus,
                                 color_map,
                                 button_size,
                                 self)
        self.display_layout.addWidget(self.grid_display)

        # Create reshape display
        self.reshape_frame = ShapeFrame(reshape_bus, self)
        self.display_layout.addWidget(self.reshape_frame)

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
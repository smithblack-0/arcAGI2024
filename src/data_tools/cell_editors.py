import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QScrollArea, QFrame,
                             QButtonGroup, QLineEdit, QLabel)
from typing import Dict, List, Any, Callable, Optional
from PyQt5.QtCore import pyqtSlot, pyqtSignal

#
class AbstractCell(QFrame):
    """
    The abstract cell class is a QFrame that is
    designed to be docked into a CellEditor. Think
    along the lines of ipynb cells, and you will
    not be too wrong.

    ---- attributes ----

    parent: The parent widget
    config: The config items
    payload: The provided payload
    payload_callback: Invoke this with the new payload whenever changes happen.
    """

    def default_payload(self) -> Any:
        """Sets up a default payload for a new block."""
        raise NotImplementedError("Need to implement default payload to add cells.")

    def commit_payload(self, payload: Any):
        """Commits a fresh payload to the block"""
        self.block['payload'] = payload

    def delete_self(self):
        """Deletes this cell by calling the parent delete function."""
        self.parent.delete_cell(self)

    def promote_self(self):
        """Promotes cell in cell list by invoking parent promote function."""
        self.parent.promote_cell(self)

    def demote_self(self):
        """Demotes cell in cell list by invoking parent demote function"""
        self.parent.demote_cell(self)

    def __init__(self,
                 parent: QWidget,
                 config: Dict[str, Any],
                 block: Dict[str, Any]
                 ):
        """
        Abstract cells must be initialized with three
        things. These things are:

        :param parent: The parent to attach to.
        :param config: The dictionary with any needed config info
        :param block: The block dictionary reference. This will always exist,
                      but may not always have a payload reference.
        """

        super().__init__(parent)

        # If the payload has not been set up, set a default payload
        if "payload" not in block:
            block["payload"] = self.default_payload()

        # Store the parameters for later use
        self.config = config
        self.parent = parent
        self.block = block

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

        # Add the button layout (horizontal) to the main layout (vertical)
        self.layout.addLayout(self.button_layout)


class CellEditorZone(QWidget):
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

    ##
    # Define cell manipulator functions for promotion, demotion,
    # addition, and deletion.
    ##

    def rerender_cells(self):
        """ rerenders all cells to be syncronized with cell instance"""
        # Remove all the widgets then add the widgets back
        # in the proper order.
        for widget in self.cells_instance:
            self.scroll_area_layout.removeWidget(widget)
        for widget in self.cells_instance:
            self.scroll_area_layout.addWidget(widget)
    def promote_cell(self, cell: AbstractCell):
        """Promote cell to live higher on the list"""
        index = self.cells_instance.index(cell)
        if index == 0:
            # Do nothing. Cannot promote past zero.
            return

        # Move the cell and the block around
        cell = self.cells_instance.pop(index)
        block = self.blocks.pop(index)

        self.cells_instance.insert(index-1, cell)
        self.blocks.insert(index-1, block)

        # Rerender everything to reflect the new structure
        self.rerender_cells()

    def demote_cell(self, cell: AbstractCell):
        """Demote cell to live lower on the list"""
        index = self.cells_instance.index(cell)
        if index == len(self.cells_instance) - 1:
            # Do nothing. Cannot demote any further
            return

        # Move the cell and the block around
        cell = self.cells_instance.pop(index)
        block = self.blocks.pop(index)

        self.blocks.insert(index+1, block)
        self.cells_instance.insert(index+1, cell)

        # Re render everything
        self.rerender_cells()

    def delete_cell(self, cell):
        """Removes a block and its cell"""
        cell.deleteLater()
        index = self.cells_instance.index(cell)
        self.cells_instance.remove(cell)
        self.blocks.pop(index)
        self.rerender_cells()

    def add_cell(self, cell_type: str, block: Optional[Any] = None):
        """Adds a new cell based on the registered type."""
        if block is None:
            block = {"mode" : cell_type}

        if cell_type in CellEditorZone.cells:
            cell_class = CellEditorZone.cells[cell_type]
            cell_instance = cell_class(self, self.config, block)

            self.cells_instance.append(cell_instance)
            self.blocks.append(block)

            self.rerender_cells()
        else:
            raise KeyError(f"Cell of type {cell_type} does not exist")
    def __init__(self,
                 config: Dict[str, Any],
                 blocks: List[Dict[str, Any]],
                 parent: Optional[Any] = None,
                 ):

        super().__init__(parent)

        # Store the config dictionary.
        #
        # This will be passed along when creating cells.
        # Also, setup selected cell to none

        self.config = config
        self.blocks = blocks

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

        # Setup cell tracking
        self.cells_instance: List[AbstractCell] = []

        # Initialize cells based on blocks
        for block in blocks:
           self.add_cell(block["mode"], block)

    def setup_cell_buttons(self):
        """Creates buttons dynamically for all registered cells."""
        for cell_type in self.cells:
            button = QPushButton(f"Add {cell_type.capitalize()} Cell")
            button.clicked.connect(lambda _, c=cell_type: self.add_cell(c))
            self.button_layout.addWidget(button)



# TextCell is a concrete subclass of AbstractCell
class TextCell(AbstractCell):
    """
    A text cell editor

    Allows the editing of text in terms of a cell.
    """

    def default_payload(self):
        return ""
    def communicate_update(self):
        self.commit_payload(self.text_edit.toPlainText())
    def __init__(self, parent=None, config=None, block=None):
        super().__init__(parent, config, block)

        # Text editor specific to the TextCell
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Enter some text...")
        self.text_edit.textChanged.connect(self.communicate_update)

        # Add the text editor to the layout
        self.layout.insertWidget(0, self.text_edit)  # Insert at the top before the delete button


# Define the grid editor cell, and all of the subwidgets
class PaletteWidget(QWidget):
    """
    A widget that displays a vertical list of color buttons based on the color map.
    Only one button can be selected at a time.
    Emits the selected color when a button is clicked.
    """

    color_selected = pyqtSignal(int)

    def __init__(self, color_map: Dict[int, str], parent=None):
        super().__init__(parent)
        self.color_map = color_map

        # Layout for color buttons
        self.layout = QVBoxLayout(self)
        self.button_group = QButtonGroup(self)

        # Create buttons for each color
        for color_id, color_value in self.color_map.items():
            button = QPushButton(f"Color {color_id}")
            button.setStyleSheet(f"background-color: {color_value}")
            button.setCheckable(True)  # Allow the button to stay selected
            self.layout.addWidget(button)
            self.button_group.addButton(button, color_id)

        # Connect signal when a button is clicked
        self.button_group.buttonClicked[int].connect(self.select_color)

    def select_color(self, color_id: int):
        """Emits the selected color ID when a button is clicked."""
        self.color_selected.emit(color_id)

class ShapeWidget(QWidget):
    """
    A widget that allows the user to set the shape of the grid (rows x columns).
    Emits the new shape when it changes.
    """

    shape_changed = pyqtSignal(int, int)

    def __init__(self, rows: int, cols: int, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols

        # Layout for the widget
        self.layout = QVBoxLayout(self)

        # Shape label
        self.layout.addWidget(QLabel("Shape"))

        # Row and Column input fields
        self.shape_layout = QHBoxLayout()
        self.row_input = QLineEdit(str(self.rows))
        self.col_input = QLineEdit(str(self.cols))
        self.shape_layout.addWidget(self.row_input)
        self.shape_layout.addWidget(QLabel("X"))
        self.shape_layout.addWidget(self.col_input)

        self.layout.addLayout(self.shape_layout)

        # Connect input changes to emit the signal
        self.row_input.textChanged.connect(self.on_shape_changed)
        self.col_input.textChanged.connect(self.on_shape_changed)

    def on_shape_changed(self):
        """Handles changes in the shape inputs."""
        try:
            rows = int(self.row_input.text())
            cols = int(self.col_input.text())
            self.shape_changed.emit(rows, cols)
        except ValueError:
            pass  # Ignore invalid input

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt5.QtCore import pyqtSignal

class GridEditWidget(QWidget):
    """
    A widget to display and edit a grid of cells (based on np.ndarray).
    Each cell's color corresponds to a value in the grid (mapped via color_map).
    Clicking a cell sets its color to the selected one from the palette.
    """

    cell_updated = pyqtSignal(int, int, int)  # x, y, new color value

    def __init__(self, grid_data: np.ndarray, color_map: Dict[int, str], cell_size: int, palette_widget: PaletteWidget, parent=None):
        super().__init__(parent)
        self.grid_data = grid_data
        self.color_map = color_map
        self.cell_size = cell_size
        self.palette_widget = palette_widget

        # Layout for the grid
        self.layout = QGridLayout(self)

        # Create buttons for each cell in the grid
        self.buttons = {}
        for row in range(self.grid_data.shape[0]):
            for col in range(self.grid_data.shape[1]):
                button = QPushButton()
                button.setFixedSize(cell_size, cell_size)
                self.update_button_color(button, self.grid_data[row, col])
                button.clicked.connect(lambda _, r=row, c=col: self.on_cell_clicked(r, c))
                self.layout.addWidget(button, row, col)
                self.buttons[(row, col)] = button

        # Connect the palette's color selection signal to track selected color
        self.selected_color = 0
        self.palette_widget.color_selected.connect(self.on_color_selected)

    def update_button_color(self, button: QPushButton, color_id: int):
        """Updates the button's background color based on the color map."""
        button.setStyleSheet(f"background-color: {self.color_map[color_id]}")

    def on_color_selected(self, color_id: int):
        """Tracks the currently selected color from the palette."""
        self.selected_color = color_id

    def on_cell_clicked(self, row: int, col: int):
        """Handles cell clicks and updates the grid."""
        self.grid_data[row, col] = self.selected_color
        self.update_button_color(self.buttons[(row, col)], self.selected_color)
        self.cell_updated.emit(row, col, self.selected_color)

class ArcIntGridCell(AbstractCell):
    """
    ArcIntGridCell is a custom cell widget that allows the editing of a grid of integers,
    where each integer is mapped to a color.
    """

    # Define default payload.
    #
    # Otherwise, we get an error
    def default_payload(self) -> Any:
        return np.zeros([2, 2], dtype=int)

    def __init__(self, parent=None, config=None, block=None):
        super().__init__(parent, config, block)

        # Extract configuration details
        color_map = config['color_map']
        cell_size = config['cell_size']
        subwidget_room = config['subwidget_room']
        grid_data = block['payload']

        # Initialize Palette, Shape, and GridEdit subwidgets
        self.palette_widget = PaletteWidget(color_map, self)
        self.shape_widget = ShapeWidget(grid_data.shape[0], grid_data.shape[1], self)
        self.grid_edit_widget = GridEditWidget(grid_data, color_map, cell_size, self.palette_widget, self)

        # Arrange the layout
        self.layout = QHBoxLayout(self)

        # Left side: GridEdit
        self.layout.addWidget(self.grid_edit_widget)

        # Right side: Palette and Shape in a vertical layout
        self.right_side_layout = QVBoxLayout()
        self.right_side_layout.addWidget(self.palette_widget)
        self.right_side_layout.addWidget(self.shape_widget)

        # Create a fixed-width container for the right-side layout
        self.right_side_widget = QWidget()
        self.right_side_widget.setFixedWidth(subwidget_room)
        self.right_side_widget.setLayout(self.right_side_layout)
        self.layout.addWidget(self.right_side_widget)

        # Set the layout for the ArcIntGridCell to apply it
        self.setLayout(self.layout)

        # Connect signals for shape changes      and grid cell updates
        self.shape_widget.shape_changed.connect(self.on_shape_changed)
        self.grid_edit_widget.cell_updated.connect(self.commit_grid_update)

    def on_shape_changed(self, rows: int, cols: int):
        """Resizes the grid and updates the GridEdit widget."""
        new_grid = np.zeros((rows, cols), dtype=int)
        old_grid = self.block['payload']

        # Copy over the old grid values to the new grid, adding padding if needed
        min_rows = min(rows, old_grid.shape[0])
        min_cols = min(cols, old_grid.shape[1])
        new_grid[:min_rows, :min_cols] = old_grid[:min_rows, :min_cols]

        self.block['payload'] = new_grid
        self.grid_edit_widget.grid_data = new_grid
        self.grid_edit_widget.repaint()  # Refresh the grid

    def commit_grid_update(self, row: int, col: int, new_value: int):
        """Commits the grid updates to the block."""
        self.block['payload'][row, col] = new_value

# Main function to run the application
def main():
    app = QApplication(sys.argv)

    # Register the TextCell and ArcIntGridCell as cell types
    CellEditorZone.register_cell("text", TextCell)
    CellEditorZone.register_cell("arcintgrid", ArcIntGridCell)

    # Config dictionary passed into CellZone for the ArcIntGridCell
    config = {
        "color_map": {0: "#FFFFFF", 1: "#000000", 2: "#FF0000"},  # White, Black, Red
        "cell_size": 50,  # Size of each grid cell
        "subwidget_room": 150  # Fixed width for the right-side widgets
    }

    # Use the provided blocks as is
    blocks = []

    # Create the main window with the CellEditorZone
    window = CellEditorZone(config=config, blocks=blocks)
    window.setup_cell_buttons()  # Call this after all cells are registered
    window.setWindowTitle("ARC-AGI Cell Editor with PyQt5")
    window.setGeometry(100, 100, 800, 600)  # Window size to accommodate larger grid
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSplitter


def prototype_

def create_gui_layout(hierarchical_display, control_window, editor_window):
    """
    Creates and displays the GUI layout with the provided subwindows.

    Parameters:
    hierarchical_display (QWidget): The widget to be used as the hierarchical display.
    control_window (QWidget): The widget to be used as the control window.
    editor_window (QWidget): The widget to be used as the editor window.
    """
    # Create the main application
    app = QApplication(sys.argv)

    # Main window
    main_window = QMainWindow()
    main_window.setWindowTitle("Config Editor")
    main_window.setGeometry(100, 100, 1000, 600)

    # Main widget and layout
    main_widget = QWidget()
    main_layout = QHBoxLayout(main_widget)

    # Splitter to separate the left and right sections
    main_splitter = QSplitter()

    # Add the hierarchical display (left side) to the main splitter
    main_splitter.addWidget(hierarchical_display)

    # Vertical splitter on the right side for control and editor windows
    right_splitter = QSplitter()
    right_splitter.setOrientation(1)  # 1 means vertical orientation
    right_splitter.addWidget(control_window)
    right_splitter.addWidget(editor_window)
    right_splitter.setStretchFactor(1, 4)  # Editor window is larger

    # Add the right-side splitter to the main splitter
    main_splitter.addWidget(right_splitter)

    # Add the main splitter to the main layout
    main_layout.addWidget(main_splitter)

    # Set the central widget
    main_window.setCentralWidget(main_widget)

    # Show the main window
    main_window.show()

    # Run the application's event loop
    sys.exit(app.exec_())


# Example usage
if __name__ == "__main__":
    from PyQt5.QtWidgets import QTreeView, QLabel, QTextEdit

    # Define the three subwindows
    hierarchical_display = QTreeView()
    hierarchical_display.setHeaderLabel("Configuration Structure")

    control_window = QLabel("Control Window")
    control_window.setStyleSheet("background-color: lightgray; padding: 10px;")

    editor_window = QTextEdit("Editor Window")

    # Pass them to the layout function
    create_gui_layout(hierarchical_display, control_window, editor_window)

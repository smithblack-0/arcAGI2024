import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from zone_display import ZonesManager
from data_control import IOAdapter, DataManager
from events import EventBus, Events


# Define the existing config for the tool
config = {
"debug": True,
"button_styles": {
    "default": "#D4D0C8",  # Windows grey (default button color)
    "blocked": "#A9A9A9",  # Faded light grey for blocked state
    "flash": "#FFFF99"  # Light yellow for flashing
},
"flash_cycle_time": 0.2,
"num_flashes": 3,
"grid_element_size": 30,
"color_map": {
    0: "#000000",  # Black
    1: "#0000FF",  # Blue
    2: "#FF0000",  # Red
    3: "#008000",  # Green
    4: "#FFFF00",  # Yellow
    5: "#808080",  # Grey
    6: "#FFC0CB",  # Pink
    7: "#FFA500",  # Orange
    8: "#ADD8E6",  # Light Blue
    9: "#8B0000"   # Dark Red
},  # White, Black, Red
"cell_size": 50,  # Cell size for GridEditorCell
}

# Define a main function

def edit_file(file: str):
    # Setup the io
    io = IOAdapter(file)

    # Setup the master event bus
    master_bus = EventBus()

    # Setup the application and the main window
    app = QtWidgets.QApplication(sys.argv)

    # Create the main window
    main_window = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(main_window)

    # Create and attach the zone editor and the data
    # control

    zone_manager = ZonesManager(config, master_bus, main_window)
    main_layout.addWidget(zone_manager)

    data_manager = DataManager(io, master_bus, config, main_window)
    main_layout.addWidget(data_manager)

    # Configure the main window
    main_window.setWindowTitle("Zones Editor Application")
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()

    # Run the app

    sys.exit(app.exec_())

edit_file(r"C:\Users\chris\PycharmProjects\arcAGI2024\data\block_zone\converted_data.json")

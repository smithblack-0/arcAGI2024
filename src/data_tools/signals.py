from PyQt5.QtCore import pyqtSignal

class Signals:
    """
    A place signals can be put for long distance signal
    communication between major modules.
    """
    def __init__(self):
        self.palette_change_signal = pyqtSignal()
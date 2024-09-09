"""
Tree display functions and nodes

This includes the specialized display nodes such as the leaf,
dictionary, etc nodes.
"""

from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QBrush, QColor
from typing import Optional, Callable, Tuple
class CustomDisplayNode(QTreeWidgetItem):
    def __init__(self, text, status="unfinished"):
        super().__init__([text])
        self.set_color_based_on_status(status)

    def set_color_based_on_status(self, status):
        """
        Set the text color of the node based on its status.

        Parameters:
        - status (str): The status of the node, which can be "unfinished", "finished", "locked", or "generic".
        """
        if status == "unfinished":
            self.setForeground(0, QBrush(QColor("red")))
        elif status == "finished":
            self.setForeground(0, QBrush(QColor("green")))
        elif status == "locked":
            self.setForeground(0, QBrush(QColor("grey")))
        elif status == "generic":
            self.setForeground(0, QBrush(QColor("yellow")))class PrimitiveDisplayNode(QTreeWidgetItem):

class PrimitiveNode(CustomDisplayNode):
    """
    Displays a primitive tree node, and the data structure
    position, for the user to review. Sets the appropriate colors
    """
    def __init__(self,
                 text: str,
                 status: str,
                 callback: Optional[Callable]):
        super().__init__(text, status)
        self.callback = callback
    def on_item_clicked(self, item: QTreeWidgetItem):
        callback

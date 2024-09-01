from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QBrush, QColor


class HierarchicalDisplay(QTreeWidget):
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Configuration Structure")

    def add_item(self, parent, key, entry_type, status="unfinished"):
        """
        Adds an item to the tree. If a parent is provided, it adds it as a child of that parent.
        If no parent is provided, it adds it as a top-level item.
        """
        item_text = f"{key} ({entry_type})"
        item = QTreeWidgetItem([item_text])

        # Set the color based on status
        if status == "unfinished":
            item.setForeground(0, QBrush(QColor("red")))
        elif status == "finished":
            item.setForeground(0, QBrush(QColor("green")))
        elif status == "locked":
            item.setForeground(0, QBrush(QColor("grey")))
        elif entry_type.startswith("generic"):
            item.setForeground(0, QBrush(QColor("yellow")))

        if parent:
            parent.addChild(item)
        else:
            self.addTopLevelItem(item)
        return item  # Return the item so that children can be attached to it


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Config Editor")

        # Main layout
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Hierarchical display
        self.tree = HierarchicalDisplay()
        layout.addWidget(self.tree)

        # Example usage: Creating a nested structure
        # Top-level dict
        root_item = self.tree.add_item(None, "root", "dict", status="unfinished")

        # Child of root: List with unfinished entries
        list_item = self.tree.add_item(root_item, "my_list", "generic_list", status="unfinished")

        # Children of list_item: Individual elements in the list
        self.tree.add_item(list_item, "index 0", "int", status="unfinished")
        self.tree.add_item(list_item, "index 1", "str", status="unfinished")

        # Another top-level dict with a finished status
        another_root = self.tree.add_item(None, "another_root", "dict", status="finished")

        # Nested structure under another_root
        nested_dict = self.tree.add_item(another_root, "nested_dict", "dict", status="unfinished")
        self.tree.add_item(nested_dict, "key 'a'", "float", status="unfinished")
        self.tree.add_item(nested_dict, "key 'b'", "int", status="finished")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

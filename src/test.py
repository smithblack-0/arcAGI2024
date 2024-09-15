import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTabWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the main layout for the window
        self.layout = QVBoxLayout()

        # Create a QTabWidget to hold the tabs
        self.tabs = QTabWidget()

        # Create the first tab with some text
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.tab1_label = QLabel("This is the content of the first tab.")
        self.tab1_layout.addWidget(self.tab1_label)
        self.tab1.setLayout(self.tab1_layout)

        # Create the second tab with different text
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        self.tab2_label = QLabel("This is the content of the second tab.")
        self.tab2_layout.addWidget(self.tab2_label)
        self.tab2.setLayout(self.tab2_layout)

        # Add tabs to the QTabWidget
        self.tabs.addTab(self.tab1, "Tab 1")
        self.tabs.addTab(self.tab2, "Tab 2")

        # Add the QTabWidget to the main layout
        self.layout.addWidget(self.tabs)

        # Set the layout for the main window
        self.setLayout(self.layout)

        # Set window properties
        self.setWindowTitle("Tabbed Interface Example")
        self.setGeometry(100, 100, 400, 300)


# Main function to run the application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

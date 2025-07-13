import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel)
from PyQt5.QtCore import Qt
from modules.trace_inspection import TraceInspectionModule
from modules.dataframe_inspection import DataFrameInspectionModule
from modules.trace_selection import TraceSelectionModule

CONFIG_FILE = "config.json"

class MainWindow(QMainWindow):
    """Main window for the ROI processing application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Processing")
        self.setGeometry(100, 100, 1200, 800)
        
        # Store references to GUI windows
        self.trace_inspection_gui = None
        self.dataframe_inspection_gui = None
        self.trace_selection_gui = None
        
        # Load last used data folder from config, default to './DATA'
        self.data_folder = self.load_config().get("data_folder", "./DATA")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create channel selection buttons
        channel_layout = QHBoxLayout()
        self.chan_a_btn = QPushButton("ChanA")
        self.chan_b_btn = QPushButton("ChanB")
        self.chan_a_btn.setCheckable(True)
        self.chan_b_btn.setCheckable(True)
        self.chan_a_btn.setChecked(True)  # Default to ChanA
        
        self.chan_a_btn.clicked.connect(lambda: self.select_channel("ChanA"))
        self.chan_b_btn.clicked.connect(lambda: self.select_channel("ChanB"))
        
        channel_layout.addWidget(self.chan_a_btn)
        channel_layout.addWidget(self.chan_b_btn)
        layout.addLayout(channel_layout)
        
        # Create module buttons
        self.trace_inspect_btn = QPushButton("Trace Inspect")
        self.trace_inspect_btn.clicked.connect(self.launch_trace_inspection)
        layout.addWidget(self.trace_inspect_btn)
        
        # Add 'Trace Selection' button
        self.trace_selection_btn = QPushButton("Trace Selection")
        self.trace_selection_btn.clicked.connect(self.launch_trace_selection)
        layout.addWidget(self.trace_selection_btn)
        
        # Add 'Locate DATA Folder' button
        self.locate_data_btn = QPushButton("Locate DATA Folder")
        self.locate_data_btn.clicked.connect(self.locate_data_folder)
        layout.addWidget(self.locate_data_btn)
        
        # Add 'DataFrame Inspection' button
        self.dataframe_inspect_btn = QPushButton("DataFrame Inspection")
        self.dataframe_inspect_btn.clicked.connect(self.launch_dataframe_inspection)
        layout.addWidget(self.dataframe_inspect_btn)
        
        # Add label to display current data folder
        self.data_folder_label = QLabel(f"Current DATA Folder: {self.data_folder}")
        layout.addWidget(self.data_folder_label)
        
        # Initialize current channel
        self.current_channel = "ChanA"
        
    def load_config(self):
        """Load configuration from config.json."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        return {}
        
    def save_config(self):
        """Save configuration to config.json."""
        config = {"data_folder": self.data_folder}
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")
        
    def select_channel(self, channel):
        """Select the current channel."""
        self.current_channel = channel
        self.chan_a_btn.setChecked(channel == "ChanA")
        self.chan_b_btn.setChecked(channel == "ChanB")
        
    def locate_data_folder(self):
        """Open a folder selection dialog to locate the DATA folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select DATA Folder", self.data_folder)
        if folder:
            self.data_folder = folder
            self.save_config()
            self.data_folder_label.setText(f"Current DATA Folder: {self.data_folder}")
        
    def launch_trace_inspection(self):
        """Launch the trace inspection module."""
        module = TraceInspectionModule(self.current_channel, self.data_folder)
        if module.process():
            self.trace_inspection_gui = module.create_gui()
            self.trace_inspection_gui.show()
            
    def launch_trace_selection(self):
        """Launch the trace selection module."""
        module = TraceSelectionModule(self.current_channel, self.data_folder)
        if module.process():
            self.trace_selection_gui = module.create_gui()
            self.trace_selection_gui.show()
            
    def launch_dataframe_inspection(self):
        """Launch the DataFrame inspection module."""
        module = DataFrameInspectionModule(self.current_channel, self.data_folder)
        if module.process():
            self.dataframe_inspection_gui = module.create_gui()
            self.dataframe_inspection_gui.show()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
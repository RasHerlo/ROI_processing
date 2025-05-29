import pandas as pd
from pathlib import Path
from ..base_module import BaseModule

class DataFrameInspectionModule(BaseModule):
    """Module for inspecting processed DataFrame data."""
    
    def __init__(self, channel: str, data_folder: str = "./DATA"):
        super().__init__(channel, data_folder)
        self.processed_data = None
        
    def process(self):
        """Load the processed DataFrame data."""
        processed_file = Path(self.data_path) / "processed_traces.pkl"
        
        if not processed_file.exists():
            from PyQt5.QtWidgets import QMessageBox, QPushButton
            
            # Check if selected_traces.pkl exists first
            selected_file = Path(self.data_path) / "selected_traces.pkl"
            
            if not selected_file.exists():
                QMessageBox.critical(None, "Error", "selected_traces.pkl file does not exist")
                return False
            
            # Create message box asking to create processed data
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Processed Data Not Found")
            msg.setText("The processed_traces.pkl file was not found.")
            msg.setInformativeText("Would you like to create it now using the pickle file expansion module?")
            
            create_btn = QPushButton("Create Now")
            cancel_btn = QPushButton("Cancel")
            msg.addButton(create_btn, QMessageBox.AcceptRole)
            msg.addButton(cancel_btn, QMessageBox.RejectRole)
            
            if msg.exec_() == QMessageBox.AcceptRole:
                # Import and run the pickle file expansion
                from .. import pickle_file_expansion
                try:
                    pickle_file_expansion.process_pickle_file(str(selected_file))
                    # Try loading again
                    if processed_file.exists():
                        self.processed_data = pd.read_pickle(processed_file)
                        return True
                    else:
                        QMessageBox.critical(None, "Error", "Failed to create processed_traces.pkl")
                        return False
                except Exception as e:
                    QMessageBox.critical(None, "Error", f"Failed to create processed data: {str(e)}")
                    return False
            else:
                return False
        else:
            # Load existing processed data
            try:
                self.processed_data = pd.read_pickle(processed_file)
                return True
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Error", f"Failed to load processed data: {str(e)}")
                return False
    
    def create_gui(self):
        """Create and return the DataFrame inspection GUI."""
        from .gui import DataFrameInspectionGUI
        return DataFrameInspectionGUI(self) 
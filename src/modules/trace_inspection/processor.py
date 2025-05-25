import os
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from ..base_module import BaseModule
from .. import pickle_file_expansion

class TraceInspectionModule(BaseModule):
    """Module for inspecting traces with interactive visualization."""
    
    def __init__(self, channel: str, data_folder: str = "./DATA"):
        super().__init__(channel, data_folder)
        self.processed_data = None
    
    def process(self) -> bool:
        """
        Process the data for trace inspection.
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        processed_path = os.path.join(self.data_path, 'processed_traces.pkl')
        
        if not os.path.exists(processed_path):
            selected_path = os.path.join(self.data_path, 'selected_traces.pkl')
            
            if not os.path.exists(selected_path):
                QMessageBox.critical(None, "Error", "selected_traces.pkl file does not exist")
                return False
            
            reply = QMessageBox.question(None, 'Process Traces',
                                       'Want to process selected_traces.pkl?',
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                try:
                    pickle_file_expansion.process_pickle_file(selected_path)
                    # Verify that processed_traces.pkl was created
                    if not os.path.exists(processed_path):
                        QMessageBox.critical(None, "Error", "Processing completed but output file was not created")
                        return False
                except Exception as e:
                    QMessageBox.critical(None, "Error", f"Failed to process file: {str(e)}")
                    return False
            else:
                return False
        
        # Only try to load if we're sure the file exists
        if os.path.exists(processed_path):
            try:
                self.processed_data = pd.read_pickle(processed_path)
                return True
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load processed data: {str(e)}")
                return False
        return False
    
    def create_gui(self):
        """Create the GUI components for trace inspection."""
        from .gui import TraceInspectionGUI
        return TraceInspectionGUI(self) 
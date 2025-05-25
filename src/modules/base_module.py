from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os

class BaseModule(ABC):
    """Base class for all modules in the ROI processing system."""
    
    def __init__(self, channel: str, data_folder: str = "./DATA"):
        """
        Initialize the module.
        
        Args:
            channel (str): The channel to use ('ChanA' or 'ChanB')
            data_folder (str): The base folder for data, defaults to './DATA'
        """
        self.channel = channel
        self.data_folder = data_folder
        self.data_path = os.path.join(self.data_folder, f"SUPPORT_{channel}", "derippled")
    
    @abstractmethod
    def process(self) -> None:
        """Process the data for this module."""
        pass
    
    @abstractmethod
    def create_gui(self) -> None:
        """Create the GUI components for this module."""
        pass
    
    def get_data_path(self) -> str:
        """Get the data path for the current channel."""
        return self.data_path 
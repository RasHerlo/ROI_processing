import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from ..base_module import BaseModule
from .. import pickle_file_expansion
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TraceSelectionModule(BaseModule):
    """Module for manual trace selection with ROI and trace visualization."""
    
    def __init__(self, channel: str, data_folder: str = "./DATA"):
        super().__init__(channel, data_folder)
        self.processed_data = None
        self.ops = None
        self.stat = None
        self.processed_path = None
        self.background_png = None
    
    def process(self) -> bool:
        """
        Process the data for trace selection.
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        self.processed_path = os.path.join(self.data_path, 'processed_traces.pkl')
        
        if not os.path.exists(self.processed_path):
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
                    if not os.path.exists(self.processed_path):
                        QMessageBox.critical(None, "Error", "Processing completed but output file was not created")
                        return False
                except Exception as e:
                    QMessageBox.critical(None, "Error", f"Failed to process file: {str(e)}")
                    return False
            else:
                return False
        
        # Load processed data
        if os.path.exists(self.processed_path):
            try:
                self.processed_data = pd.read_pickle(self.processed_path)
                
                # Add manual_selection column if it doesn't exist
                if 'manual_selection' not in self.processed_data.columns:
                    # Initialize all as True (checked)
                    self.processed_data['manual_selection'] = True
                    self.save_data()
                    print("Added manual_selection column, initialized all as True")
                
                # Load additional data needed for ROI visualization
                suite2p_path = os.path.join(self.data_path, 'suite2p', 'plane0')
                ops_path = os.path.join(suite2p_path, 'ops.npy')
                stat_path = os.path.join(suite2p_path, 'stat.npy')
                
                # Try to load ops and stat files with fallback
                self.ops = self._load_ops_with_fallback(ops_path)
                self.stat = self._load_stat_with_fallback(stat_path)
                
                if self.ops is None or self.stat is None:
                    return False
                
                # Try to load background PNG image from parent folder
                self.background_png = self._load_background_png()
                
                return True
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load processed data: {str(e)}")
                return False
        return False
    
    def _load_ops_with_fallback(self, ops_path):
        """Load ops.npy with fallback to suite2p_registered."""
        # Try loading from suite2p first
        if os.path.exists(ops_path):
            try:
                ops = np.load(ops_path, allow_pickle=True).item()
                print("Loaded ops.npy from suite2p folder")
                
                # Check if MeanImg or MeanImgE are missing and try to load from suite2p_registered
                if 'meanImg' not in ops or 'meanImgE' not in ops:
                    ops = self._supplement_ops_from_registered(ops)
                
                return ops
            except Exception as e:
                print(f"Failed to load ops.npy from suite2p: {str(e)}")
        
        # Try fallback to suite2p_registered
        suite2p_registered_path = os.path.join(self.data_path, 'suite2p_registered', 'plane0')
        ops_registered_path = os.path.join(suite2p_registered_path, 'ops.npy')
        
        if os.path.exists(ops_registered_path):
            try:
                ops = np.load(ops_registered_path, allow_pickle=True).item()
                print("Loaded ops.npy from suite2p_registered folder (fallback)")
                return ops
            except Exception as e:
                print(f"Failed to load ops.npy from suite2p_registered: {str(e)}")
        
        QMessageBox.critical(None, "Error", "ops.npy not found in suite2p or suite2p_registered folders")
        return None
    
    def _load_stat_with_fallback(self, stat_path):
        """Load stat.npy with fallback to suite2p_registered."""
        # Try loading from suite2p first
        if os.path.exists(stat_path):
            try:
                stat = np.load(stat_path, allow_pickle=True)
                print("Loaded stat.npy from suite2p folder")
                return stat
            except Exception as e:
                print(f"Failed to load stat.npy from suite2p: {str(e)}")
        
        # Try fallback to suite2p_registered
        suite2p_registered_path = os.path.join(self.data_path, 'suite2p_registered', 'plane0')
        stat_registered_path = os.path.join(suite2p_registered_path, 'stat.npy')
        
        if os.path.exists(stat_registered_path):
            try:
                stat = np.load(stat_registered_path, allow_pickle=True)
                print("Loaded stat.npy from suite2p_registered folder (fallback)")
                return stat
            except Exception as e:
                print(f"Failed to load stat.npy from suite2p_registered: {str(e)}")
        
        QMessageBox.critical(None, "Error", "stat.npy not found in suite2p or suite2p_registered folders")
        return None
    
    def _supplement_ops_from_registered(self, ops):
        """Supplement ops with missing images from suite2p_registered."""
        try:
            suite2p_registered_path = os.path.join(self.data_path, 'suite2p_registered', 'plane0')
            ops_registered_path = os.path.join(suite2p_registered_path, 'ops.npy')
            
            if os.path.exists(ops_registered_path):
                ops_registered = np.load(ops_registered_path, allow_pickle=True).item()
                
                # Add missing images from the registered ops
                if 'meanImg' not in ops and 'meanImg' in ops_registered:
                    ops['meanImg'] = ops_registered['meanImg']
                    print("Loaded meanImg from suite2p_registered folder")
                
                if 'meanImgE' not in ops and 'meanImgE' in ops_registered:
                    ops['meanImgE'] = ops_registered['meanImgE']
                    print("Loaded meanImgE from suite2p_registered folder")
            else:
                print("Warning: suite2p_registered/plane0/ops.npy not found")
        except Exception as e:
            print(f"Warning: Failed to load from suite2p_registered: {str(e)}")
        
        return ops
    
    def _load_background_png(self):
        """Load background PNG image from parent folder."""
        try:
            # Get parent folder (SUPPORT_ChanX)
            parent_folder = os.path.dirname(self.data_path)
            
            # Look for PNG files in the parent folder
            png_files = [f for f in os.listdir(parent_folder) if f.lower().endswith('.png')]
            
            if not png_files:
                print("No PNG files found in parent folder")
                return None
            
            # Use the first PNG file found
            png_path = os.path.join(parent_folder, png_files[0])
            
            # Load the PNG image
            img = mpimg.imread(png_path)
            
            # Get target dimensions from meanImg
            if self.ops and 'meanImg' in self.ops:
                target_shape = self.ops['meanImg'].shape
                
                # Handle different image formats
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA
                        img = img[:, :, :3]  # Remove alpha channel
                    # Keep as RGB color image
                elif len(img.shape) == 2:
                    # Already grayscale
                    pass
                
                # Resize PNG to match Suite2p dimensions if needed
                if img.shape[:2] != target_shape:
                    try:
                        from scipy.ndimage import zoom
                        if len(img.shape) == 3:  # Color image
                            zoom_factors = (target_shape[0] / img.shape[0], 
                                          target_shape[1] / img.shape[1], 1)
                        else:  # Grayscale
                            zoom_factors = (target_shape[0] / img.shape[0], 
                                          target_shape[1] / img.shape[1])
                        
                        img = zoom(img, zoom_factors, order=1)  # Linear interpolation
                        print(f"Resized PNG from {png_files[0]} to match Suite2p dimensions: {target_shape}")
                    except ImportError:
                        print("Warning: scipy not available for image resizing. PNG may not align properly with ROIs.")
                else:
                    print(f"PNG dimensions match Suite2p: {target_shape}")
            
            print(f"Loaded background PNG: {png_files[0]}")
            return img
            
        except Exception as e:
            print(f"Failed to load background PNG: {str(e)}")
            return None
    
    def save_data(self):
        """Save the processed data back to pickle file."""
        if self.processed_data is not None and self.processed_path is not None:
            try:
                self.processed_data.to_pickle(self.processed_path)
                print("Saved updated processed_traces.pkl")
            except Exception as e:
                print(f"Failed to save processed_traces.pkl: {str(e)}")
                QMessageBox.warning(None, "Warning", f"Failed to save data: {str(e)}")
    
    def update_manual_selection(self, row_index, checked):
        """Update the manual_selection status for a specific row."""
        if self.processed_data is not None:
            self.processed_data.loc[row_index, 'manual_selection'] = checked
            self.save_data()
    
    def create_gui(self):
        """Create the GUI components for trace selection."""
        from .gui import TraceSelectionGUI
        return TraceSelectionGUI(self) 
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QPushButton, QLabel)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TraceInspectionGUI(QWidget):
    """GUI for trace inspection module."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.current_trace_idx = 0
        # Store sorted indices for navigation
        self.sorted_idx = np.argsort(self.module.processed_data['rastermap_indices'])
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Create figure for rasterplot and trace plot
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots
        self.raster_ax = self.fig.add_subplot(211)
        self.trace_ax = self.fig.add_subplot(212)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Slider
        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.module.processed_data) - 1)
        self.slider.valueChanged.connect(self.update_trace)
        
        # Up/Down buttons
        up_btn = QPushButton("↑")
        down_btn = QPushButton("↓")
        up_btn.clicked.connect(self.move_up)
        down_btn.clicked.connect(self.move_down)
        
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(up_btn)
        controls_layout.addWidget(down_btn)
        
        # Add widgets to layout
        layout.addWidget(self.canvas)
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
        # Initial plot
        self.update_plots()
        
    def update_plots(self):
        """Update both the rasterplot and trace plot."""
        # Clear previous plots
        self.raster_ax.clear()
        self.trace_ax.clear()
        
        # Get sorted traces
        traces_norm = self.module.processed_data['traces_norm'].iloc[self.sorted_idx]
        
        # Convert list of arrays to 2D array for visualization
        traces_norm_array = np.array([trace for trace in traces_norm])
        
        # Plot rasterplot
        self.raster_ax.imshow(traces_norm_array, aspect='auto', cmap='jet')
        
        # Highlight current trace
        self.raster_ax.axhline(y=self.current_trace_idx, color='red', linestyle='--', alpha=0.5)
        
        # Plot selected trace (using original ROI index)
        original_idx = self.sorted_idx[self.current_trace_idx]
        trace = self.module.processed_data['traces'].iloc[original_idx]
        self.trace_ax.plot(trace)
        
        # Get post-stim range for highlighting
        poststim_range = self.module.processed_data['poststim_80s'].iloc[original_idx]
        
        # Add red rectangle to highlight post-stim range
        ymin, ymax = self.trace_ax.get_ylim()
        rect = plt.Rectangle((poststim_range[0], ymin), 
                           poststim_range[1] - poststim_range[0], 
                           ymax - ymin,
                           facecolor='red', alpha=0.5)
        self.trace_ax.add_patch(rect)
        
        # Set title with indices and bPAC_dFF value
        raster_idx = self.module.processed_data['rastermap_indices'].iloc[original_idx]
        roi_idx = self.module.processed_data['roi_indices'].iloc[original_idx]
        bpac_dff = self.module.processed_data['bPAC_dFF'].iloc[original_idx]
        self.trace_ax.set_title(f'Rastermap Index: {raster_idx}, ROI Index: {roi_idx}, bPAC_dFF: {bpac_dff:.4f}')
        
        self.canvas.draw()
        
    def update_trace(self, value):
        """Update the current trace index and plots."""
        self.current_trace_idx = value
        self.update_plots()
        
    def move_up(self):
        """Move to the previous trace."""
        if self.current_trace_idx > 0:
            self.current_trace_idx -= 1
            self.slider.setValue(self.current_trace_idx)
            
    def move_down(self):
        """Move to the next trace."""
        if self.current_trace_idx < len(self.module.processed_data) - 1:
            self.current_trace_idx += 1
            self.slider.setValue(self.current_trace_idx) 
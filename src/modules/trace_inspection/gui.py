import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QPushButton, QLabel)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation

class TraceInspectionGUI(QWidget):
    """GUI for trace inspection module."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.current_trace_idx = 0
        self.sort_by_bpac = False  # Track sorting method
        # Use rastermap_indices directly for sorting
        self.sorted_idx = self.module.processed_data['rastermap_indices']
        
        # ROI visualization settings
        self.show_mean_img = True  # Toggle between meanImg and meanImgE
        self.color_by_bpac = False  # Toggle between random colors and bPAC_dFF colors
        self.roi_colors = None  # Will store ROI colors
        self.generate_roi_colors()  # Generate initial random colors
        
        self.setup_ui()
    
    def generate_roi_colors(self):
        """Generate colors for ROIs based on current mode."""
        n_rois = len(self.module.processed_data)
        
        if self.color_by_bpac:
            # Get bPAC_dFF values
            bpac_values = self.module.processed_data['bPAC_dFF'].to_numpy()
            # Create normalized values for colormapping (ignoring nans)
            valid_vals = bpac_values[~np.isnan(bpac_values)]
            vmin, vmax = np.min(valid_vals), np.max(valid_vals)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            # Create colormap
            cmap = plt.cm.bwr
            # Generate colors with 80% transparency
            self.roi_colors = np.zeros((n_rois, 4))
            for i, val in enumerate(bpac_values):
                if np.isnan(val):
                    self.roi_colors[i] = [0.5, 0.5, 0.5, 0.2]  # Gray for nan values
                else:
                    color = list(cmap(norm(val)))
                    color[3] = 0.2  # 80% transparency
                    self.roi_colors[i] = color
        else:
            # Generate random colors with 80% transparency
            self.roi_colors = np.random.rand(n_rois, 4)
            self.roi_colors[:, 3] = 0.2  # Set alpha to 0.2 (80% transparent)
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        
        # Create figure with wider size to accommodate ROI plot
        self.fig = Figure(figsize=(15, 8))  # Increased width
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots - raster and trace on left, ROI on right
        self.raster_ax = self.fig.add_subplot(221)  # Top left
        self.trace_ax = self.fig.add_subplot(223)   # Bottom left
        self.roi_ax = self.fig.add_subplot(122)     # Right (full height)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Left controls (existing)
        left_controls = QHBoxLayout()
        
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
        
        # Add sorting toggle button
        self.sort_toggle = QPushButton("Sort by bPAC_dFF")
        self.sort_toggle.setCheckable(True)
        self.sort_toggle.clicked.connect(self.toggle_sorting)
        
        left_controls.addWidget(self.slider)
        left_controls.addWidget(up_btn)
        left_controls.addWidget(down_btn)
        left_controls.addWidget(self.sort_toggle)
        
        # Right controls (new)
        right_controls = QVBoxLayout()
        
        # Add BG-IMAGE label and toggle
        bg_label = QLabel("BG-IMAGE")
        bg_label.setAlignment(Qt.AlignCenter)
        self.bg_toggle = QPushButton("Switch to MeanImgE" if self.show_mean_img else "Switch to MeanImg")
        self.bg_toggle.clicked.connect(self.toggle_background)
        
        # Add ROI-COLOR label and toggle
        color_label = QLabel("ROI-COLOR")
        color_label.setAlignment(Qt.AlignCenter)
        self.color_toggle = QPushButton("Color by bPAC_dFF")
        self.color_toggle.setCheckable(True)
        self.color_toggle.clicked.connect(self.toggle_roi_color)
        
        right_controls.addWidget(bg_label)
        right_controls.addWidget(self.bg_toggle)
        right_controls.addWidget(color_label)
        right_controls.addWidget(self.color_toggle)
        
        # Add controls to main controls layout
        controls_layout.addLayout(left_controls)
        controls_layout.addLayout(right_controls)
        
        # Add widgets to layout
        layout.addWidget(self.canvas)
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
        # Initial plot
        self.update_plots()
    
    def toggle_background(self):
        """Toggle between meanImg and meanImgE background."""
        self.show_mean_img = not self.show_mean_img
        self.bg_toggle.setText("Switch to MeanImgE" if self.show_mean_img else "Switch to MeanImg")
        self.update_roi_plot()
        
    def update_roi_plot(self):
        """Update the ROI visualization plot."""
        self.roi_ax.clear()
        
        # Get background image
        bg_img = self.module.ops['meanImg'] if self.show_mean_img else self.module.ops['meanImgE']
        
        # Display background image
        self.roi_ax.imshow(bg_img, cmap='gray')
        
        # Get current ROI index
        original_idx = self.sorted_idx[self.current_trace_idx]
        current_roi_idx = self.module.processed_data['roi_indices'].iloc[original_idx]
        
        # Plot all ROIs
        for i, roi_idx in enumerate(self.module.processed_data['roi_indices']):
            roi_stat = self.module.stat[roi_idx]
            xpix, ypix = roi_stat['xpix'], roi_stat['ypix']
            
            if roi_idx == current_roi_idx:
                # Selected ROI: white fill
                self.roi_ax.scatter(xpix, ypix, c='white', s=1)
                
                # Create outline by finding boundary pixels
                # Create a mask of the ROI
                mask = np.zeros_like(bg_img, dtype=bool)
                mask[ypix, xpix] = True
                
                # Find boundary pixels using binary dilation
                dilated = binary_dilation(mask)
                boundary = dilated & ~mask
                
                # Get boundary coordinates
                boundary_y, boundary_x = np.where(boundary)
                # Plot boundary in red
                self.roi_ax.scatter(boundary_x, boundary_y, c='red', s=1)
            else:
                # Other ROIs: random color with high transparency
                self.roi_ax.scatter(xpix, ypix, c=[self.roi_colors[i]], s=1)
        
        # Set title with ROI info
        current_stat = self.module.stat[current_roi_idx]
        self.roi_ax.set_title(f'ROI {current_roi_idx} (Size: {current_stat["npix"]} pixels)')
        
        # Remove axes ticks
        self.roi_ax.set_xticks([])
        self.roi_ax.set_yticks([])
        
        self.canvas.draw()
    
    def update_plots(self):
        """Update both the rasterplot and trace plot."""
        # Clear previous plots
        self.raster_ax.clear()
        self.trace_ax.clear()
        
        # Update sorting indices based on current mode
        if self.sort_by_bpac:
            pre_idc = self.module.processed_data['bPAC_dFF'].to_numpy().argsort()
            self.sorted_idx = pre_idc[::-1]
            cmap = 'bwr'  # Blue-white-red colormap for bPAC_dFF sorting
        else:
            # Use original rastermap sorting
            self.sorted_idx = self.module.processed_data['rastermap_indices']
            cmap = 'jet'  # Original jet colormap for rastermap sorting
        
        # Get sorted traces
        traces_norm = self.module.processed_data['traces_norm'].iloc[self.sorted_idx]
        
        # Convert list of arrays to 2D array for visualization
        traces_norm_array = np.array([trace for trace in traces_norm])
        
        # Plot rasterplot
        self.raster_ax.imshow(traces_norm_array, aspect='auto', cmap=cmap)
        
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
        rasmap_num = self.module.processed_data['rasmap_nums'].iloc[original_idx]
        roi_idx = self.module.processed_data['roi_indices'].iloc[original_idx]
        bpac_dff = self.module.processed_data['bPAC_dFF'].iloc[original_idx]
        self.trace_ax.set_title(f'Rastermap Position: {rasmap_num}, ROI Index: {roi_idx}, bPAC_dFF: {bpac_dff:.4f}')
        
        # Update ROI plot
        self.update_roi_plot()
        
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
        
    def toggle_sorting(self):
        """Toggle between rastermap and bPAC_dFF sorting."""
        self.sort_by_bpac = self.sort_toggle.isChecked()
        self.sort_toggle.setText("Sort by Rastermap" if self.sort_by_bpac else "Sort by bPAC_dFF")
        self.update_plots()
        
    def toggle_roi_color(self):
        """Toggle between random colors and bPAC_dFF-based colors for ROIs."""
        self.color_by_bpac = self.color_toggle.isChecked()
        self.color_toggle.setText("Use Random Colors" if self.color_by_bpac else "Color by bPAC_dFF")
        self.generate_roi_colors()
        self.update_roi_plot() 
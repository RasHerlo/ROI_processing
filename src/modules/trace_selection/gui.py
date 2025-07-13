import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                            QTableWidgetItem, QCheckBox, QPushButton, QHeaderView, 
                            QSplitter, QLabel, QFrame)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import binary_dilation

class TraceSelectionGUI(QWidget):
    """GUI for trace selection module."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.current_row = 0
        self.background_mode = 0  # 0: meanImg, 1: meanImgE, 2: PNG image
        
        self.setWindowTitle("Trace Selection")
        self.setGeometry(100, 100, 1400, 800)
        
        self.setup_ui()
        self.populate_table()
        self.setup_roi_click_handler()
        self.update_visualizations()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['ROI Index', 'Manual Selection'])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        
        # Connect table signals
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Enable keyboard navigation
        self.table.setFocusPolicy(Qt.StrongFocus)
        
        left_layout.addWidget(self.table)
        
        # Right panel - Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Background image toggle
        bg_label = QLabel("BG-IMAGE")
        bg_label.setAlignment(Qt.AlignCenter)
        self.bg_toggle = QPushButton(self.get_toggle_button_text())
        self.bg_toggle.clicked.connect(self.toggle_background)
        
        # Current background indicator
        self.bg_indicator = QLabel(self.get_background_indicator_text())
        self.bg_indicator.setAlignment(Qt.AlignCenter)
        self.bg_indicator.setStyleSheet("color: blue; font-weight: bold;")
        
        controls_layout.addWidget(bg_label)
        controls_layout.addWidget(self.bg_toggle)
        controls_layout.addWidget(self.bg_indicator)
        controls_layout.addStretch()  # Push controls to the left
        
        right_layout.addLayout(controls_layout)
        
        # Visualization area
        self.fig = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.fig)
        
        # Create subplots - ROI on top (bigger), trace on bottom
        # Using height_ratios to make ROI plot bigger
        gs = self.fig.add_gridspec(2, 1, height_ratios=[4, 1])
        self.roi_ax = self.fig.add_subplot(gs[0])
        self.trace_ax = self.fig.add_subplot(gs[1])
        
        right_layout.addWidget(self.canvas)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (30% table, 70% visualization)
        splitter.setSizes([300, 700])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
        
    def populate_table(self):
        """Populate the table with data."""
        df = self.module.processed_data
        
        # Set table dimensions
        self.table.setRowCount(len(df))
        
        # Populate cells
        for row in range(len(df)):
            # ROI Index column
            roi_idx = df.iloc[row]['roi_indices']
            roi_item = QTableWidgetItem(str(roi_idx))
            roi_item.setFlags(roi_item.flags() & ~Qt.ItemIsEditable)  # Read-only
            self.table.setItem(row, 0, roi_item)
            
            # Manual Selection column (checkbox)
            checkbox = QCheckBox()
            checkbox.setChecked(df.iloc[row]['manual_selection'])
            checkbox.stateChanged.connect(lambda state, r=row: self.on_checkbox_changed(r, state))
            
            # Center the checkbox in the cell
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            
            self.table.setCellWidget(row, 1, checkbox_widget)
        
        # Adjust column widths
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 120)
        
        # Select first row by default
        if len(df) > 0:
            self.table.selectRow(0)
            
    def setup_roi_click_handler(self):
        """Set up click handler for ROI plot."""
        self.canvas.mpl_connect('button_press_event', self.on_roi_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_roi_hover)
            
    def on_cell_clicked(self, row, col):
        """Handle cell clicks - select the row."""
        self.current_row = row
        self.table.selectRow(row)
        self.update_visualizations()
        
    def on_selection_changed(self):
        """Handle selection changes."""
        selected_ranges = self.table.selectedRanges()
        if selected_ranges:
            self.current_row = selected_ranges[0].topRow()
            self.update_visualizations()
            
    def on_checkbox_changed(self, row, state):
        """Handle checkbox state changes."""
        checked = state == Qt.Checked
        self.module.update_manual_selection(row, checked)
        # Don't change selection when checkbox is clicked
        
    def on_roi_click(self, event):
        """Handle clicks on ROI plot to navigate to corresponding row."""
        # Only handle clicks on the ROI axes
        if event.inaxes != self.roi_ax:
            return
        
        # Only handle left mouse button clicks
        if event.button != 1:
            return
        
        # Get click coordinates
        if event.xdata is None or event.ydata is None:
            return
        
        click_x, click_y = int(event.xdata), int(event.ydata)
        
        # Find which ROI was clicked
        clicked_roi_idx = self.find_roi_at_position(click_x, click_y)
        
        if clicked_roi_idx is not None:
            # Find corresponding row in dataframe
            row_idx = self.find_row_for_roi(clicked_roi_idx)
            if row_idx is not None:
                self.current_row = row_idx
                self.table.selectRow(row_idx)
                self.update_visualizations()
                
    def find_roi_at_position(self, x, y):
        """Find which ROI contains the given position."""
        # Check bounds - use meanImg for bounds checking since all images should have same dimensions
        bg_img = self.module.ops['meanImg']
        if x < 0 or x >= bg_img.shape[1] or y < 0 or y >= bg_img.shape[0]:
            return None
        
        # Check each ROI to see if it contains the click point
        closest_roi = None
        min_distance = float('inf')
        
        for roi_idx in self.module.processed_data['roi_indices']:
            roi_stat = self.module.stat[roi_idx]
            xpix, ypix = roi_stat['xpix'], roi_stat['ypix']
            
            # Check if click point is within this ROI
            for i in range(len(xpix)):
                if xpix[i] == x and ypix[i] == y:
                    return roi_idx
            
            # If no exact match, find the closest ROI (for near-misses)
            # Calculate distance to ROI centroid
            centroid_x = np.mean(xpix)
            centroid_y = np.mean(ypix)
            distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            
            # Check if click is reasonably close to this ROI
            if distance < 20:  # Within 20 pixels
                if distance < min_distance:
                    min_distance = distance
                    closest_roi = roi_idx
        
        return closest_roi
        
    def find_row_for_roi(self, roi_idx):
        """Find the dataframe row corresponding to the given ROI index."""
        df = self.module.processed_data
        for i, row_roi_idx in enumerate(df['roi_indices']):
            if row_roi_idx == roi_idx:
                return i
        return None
        
    def on_roi_hover(self, event):
        """Handle hover events to show cursor feedback."""
        if event.inaxes != self.roi_ax:
            return
        
        # Get hover coordinates
        if event.xdata is None or event.ydata is None:
            return
        
        hover_x, hover_y = int(event.xdata), int(event.ydata)
        
        # Check if we're hovering over an ROI
        hovered_roi = self.find_roi_at_position(hover_x, hover_y)
        
        if hovered_roi is not None:
            # Change cursor to pointer to indicate clickable area
            self.canvas.setCursor(Qt.PointingHandCursor)
            
            # Show tooltip with ROI information
            row_idx = self.find_row_for_roi(hovered_roi)
            if row_idx is not None:
                bpac_dff = self.module.processed_data.iloc[row_idx]['bPAC_dFF']
                tooltip_text = f"ROI {hovered_roi} - bPAC_dFF: {bpac_dff:.4f}\nClick to select"
                self.canvas.setToolTip(tooltip_text)
        else:
            # Reset cursor to default
            self.canvas.setCursor(Qt.ArrowCursor)
            self.canvas.setToolTip("")
        
    def get_toggle_button_text(self):
        """Get the text for the toggle button based on current mode."""
        if self.background_mode == 0:
            return "Switch to MeanImgE"
        elif self.background_mode == 1:
            if self.module.background_png is not None:
                return "Switch to PNG Image"
            else:
                return "Switch to MeanImg"
        else:  # mode == 2
            return "Switch to MeanImg"
    
    def get_background_indicator_text(self):
        """Get the text showing current background mode."""
        if self.background_mode == 0:
            return "Current: MeanImg"
        elif self.background_mode == 1:
            return "Current: MeanImgE"
        else:  # mode == 2
            return "Current: PNG Image"
    
    def toggle_background(self):
        """Toggle between meanImg, meanImgE, and PNG image background."""
        if self.background_mode == 0:
            self.background_mode = 1
        elif self.background_mode == 1:
            if self.module.background_png is not None:
                self.background_mode = 2
            else:
                self.background_mode = 0
        else:  # mode == 2
            self.background_mode = 0
        
        self.bg_toggle.setText(self.get_toggle_button_text())
        self.bg_indicator.setText(self.get_background_indicator_text())
        self.update_roi_plot()
        
    def update_visualizations(self):
        """Update both ROI and trace visualizations."""
        self.update_roi_plot()
        self.update_trace_plot()
        
    def update_roi_plot(self):
        """Update the ROI visualization plot."""
        self.roi_ax.clear()
        
        if self.current_row >= len(self.module.processed_data):
            return
        
        # Get background image based on current mode
        if self.background_mode == 0:
            bg_img = self.module.ops['meanImg']
        elif self.background_mode == 1:
            bg_img = self.module.ops['meanImgE']
        else:  # mode == 2
            bg_img = self.module.background_png
            if bg_img is None:
                # Fallback to meanImg if PNG not available
                bg_img = self.module.ops['meanImg']
        
        # Display background image with appropriate colormap
        if self.background_mode == 2 and self.module.background_png is not None:
            # For PNG image, display as-is without colormap enhancement
            if len(bg_img.shape) == 3:
                # Color image - display without colormap
                self.roi_ax.imshow(bg_img)
            else:
                # Grayscale image - use gray colormap
                self.roi_ax.imshow(bg_img, cmap='gray')
        else:
            # For Suite2p images, use jet colormap
            self.roi_ax.imshow(bg_img, cmap='jet')
        
        # Get current ROI info
        current_roi_idx = self.module.processed_data.iloc[self.current_row]['roi_indices']
        current_stat = self.module.stat[current_roi_idx]
        
        # Plot all ROIs with low opacity
        for i, roi_idx in enumerate(self.module.processed_data['roi_indices']):
            roi_stat = self.module.stat[roi_idx]
            xpix, ypix = roi_stat['xpix'], roi_stat['ypix']
            
            if roi_idx == current_roi_idx:
                # Selected ROI: white fill
                self.roi_ax.scatter(xpix, ypix, c='white', s=1)
                
                # Create outline by finding boundary pixels
                # Create 2D mask regardless of bg_img dimensions
                if len(bg_img.shape) == 3:
                    # For color images, use the first 2 dimensions
                    mask_shape = bg_img.shape[:2]
                else:
                    # For grayscale images
                    mask_shape = bg_img.shape
                
                mask = np.zeros(mask_shape, dtype=bool)
                mask[ypix, xpix] = True
                
                # Find boundary pixels using binary dilation
                dilated = binary_dilation(mask)
                boundary = dilated & ~mask
                
                # Get boundary coordinates
                boundary_y, boundary_x = np.where(boundary)
                # Plot boundary in red
                self.roi_ax.scatter(boundary_x, boundary_y, c='red', s=1)
            else:
                # Other ROIs: low opacity gray
                self.roi_ax.scatter(xpix, ypix, c='gray', s=1, alpha=0.1)
        
        # Set title with ROI info
        self.roi_ax.set_title(f'ROI {current_roi_idx} (Size: {current_stat["npix"]} pixels)')
        
        # Remove axes ticks
        self.roi_ax.set_xticks([])
        self.roi_ax.set_yticks([])
        
        self.canvas.draw()
        
    def update_trace_plot(self):
        """Update the trace plot."""
        self.trace_ax.clear()
        
        if self.current_row >= len(self.module.processed_data):
            return
        
        # Get trace data
        trace = self.module.processed_data.iloc[self.current_row]['traces']
        roi_idx = self.module.processed_data.iloc[self.current_row]['roi_indices']
        
        # Plot trace
        self.trace_ax.plot(trace)
        
        # Get post-stim range for highlighting
        poststim_range = self.module.processed_data.iloc[self.current_row]['poststim_80s']
        
        # Add red rectangle to highlight post-stim range
        ymin, ymax = self.trace_ax.get_ylim()
        rect = plt.Rectangle((poststim_range[0], ymin), 
                           poststim_range[1] - poststim_range[0], 
                           ymax - ymin,
                           facecolor='red', alpha=0.5)
        self.trace_ax.add_patch(rect)
        
        # Get additional info
        rasmap_num = self.module.processed_data.iloc[self.current_row]['rasmap_nums']
        bpac_dff = self.module.processed_data.iloc[self.current_row]['bPAC_dFF']
        
        # Set title with trace info
        self.trace_ax.set_title(f'Trace - ROI: {roi_idx}, Rastermap: {rasmap_num}, bPAC_dFF: {bpac_dff:.4f}')
        self.trace_ax.set_xlabel('Time')
        self.trace_ax.set_ylabel('Intensity')
        
        self.canvas.draw()
        
    def keyPressEvent(self, event):
        """Handle keyboard events for navigation."""
        if event.key() == Qt.Key_Up:
            if self.current_row > 0:
                self.current_row -= 1
                self.table.selectRow(self.current_row)
                self.update_visualizations()
        elif event.key() == Qt.Key_Down:
            if self.current_row < len(self.module.processed_data) - 1:
                self.current_row += 1
                self.table.selectRow(self.current_row)
                self.update_visualizations()
        else:
            super().keyPressEvent(event) 
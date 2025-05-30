import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QLabel,
                            QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt

class PlotWindow(QMainWindow):
    """Separate window for plotting vector traces."""
    
    def __init__(self, vector_data, title, poststim_range=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Plot the data
        self.plot_vector(vector_data, poststim_range)
        
    def plot_vector(self, vector_data, poststim_range):
        """Plot the vector data with optional poststim range overlay."""
        ax = self.figure.add_subplot(111)
        
        # Plot the main trace
        x_data = np.arange(len(vector_data))
        ax.plot(x_data, vector_data, 'b-', linewidth=1, label='Trace')
        
        # Add red transparent rectangle for poststim_80s range if provided
        if poststim_range is not None and len(poststim_range) >= 2:
            start_idx = int(poststim_range[0]) if poststim_range[0] < len(vector_data) else 0
            end_idx = int(poststim_range[1]) if poststim_range[1] < len(vector_data) else len(vector_data)
            
            # Add transparent red rectangle
            ax.axvspan(start_idx, end_idx, color='red', alpha=0.5, label='PostStim 80s')
        
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.canvas.draw()

class PlotWindowManager:
    """Manages multiple plot windows."""
    
    def __init__(self):
        self.plot_windows = []
    
    def create_plot_window(self, vector_data, title, poststim_range=None):
        """Create and show a new plot window."""
        plot_window = PlotWindow(vector_data, title, poststim_range)
        plot_window.show()
        
        # Keep reference to prevent garbage collection
        self.plot_windows.append(plot_window)
        
        # Clean up closed windows
        self.cleanup_closed_windows()
        
        return plot_window
    
    def cleanup_closed_windows(self):
        """Remove references to closed windows."""
        self.plot_windows = [window for window in self.plot_windows if window.isVisible()]

class CellActionHandler:
    """Base class for handling cell actions."""
    
    def can_handle(self, value, row_data, column_name):
        """Check if this handler can process the cell."""
        return False
    
    def execute(self, value, row_data, column_name, context):
        """Execute the action for this cell."""
        pass

class VectorPlotAction(CellActionHandler):
    """Handler for plotting vector data."""
    
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
    
    def can_handle(self, value, row_data, column_name):
        """Check if the cell contains a plottable vector."""
        return isinstance(value, np.ndarray) and value.ndim == 1
    
    def execute(self, value, row_data, column_name, context):
        """Create a plot window for the vector."""
        # Get roi_indices for the title
        roi_index = row_data.get('roi_indicies', 'Unknown')  # Note: keeping original spelling
        title = f"{column_name} - ROI: {roi_index}"
        
        # Get poststim_80s range if available
        poststim_range = row_data.get('poststim_80s', None)
        
        # Create plot window
        self.plot_manager.create_plot_window(value, title, poststim_range)

class DataFrameInspectionGUI(QMainWindow):
    """GUI for DataFrame inspection."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.setWindowTitle(f"DataFrame Inspection - {self.module.channel}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize action system
        self.plot_manager = PlotWindowManager()
        self.action_handlers = [
            VectorPlotAction(self.plot_manager)
        ]
        
        self.setup_ui()
        self.populate_table()
        self.setup_interactions()
        
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add info label
        info_label = QLabel(f"DataFrame Shape: {self.module.processed_data.shape}")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # Add instruction label
        instruction_label = QLabel("Click on cells containing vectors to plot them")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instruction_label)
        
        # Create tab widget for future extensibility
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create main DataFrame view tab
        self.create_dataframe_tab()
        
    def create_dataframe_tab(self):
        """Create the main DataFrame viewing tab."""
        df_widget = QWidget()
        layout = QVBoxLayout(df_widget)
        
        # Create table widget
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # Add tab
        self.tab_widget.addTab(df_widget, "DataFrame View")
    
    def setup_interactions(self):
        """Set up interactive behavior for the table."""
        self.table.cellClicked.connect(self.on_cell_clicked)
        
        # Change cursor for clickable cells (will be set during population)
        self.table.setMouseTracking(True)
        
    def on_cell_clicked(self, row, col):
        """Handle cell click events."""
        # Get the value and row data
        df = self.module.processed_data
        value = df.iloc[row, col]
        row_data = df.iloc[row].to_dict()
        column_name = df.columns[col]
        
        # Find a handler that can process this cell
        for handler in self.action_handlers:
            if handler.can_handle(value, row_data, column_name):
                try:
                    handler.execute(value, row_data, column_name, {'table': self.table})
                    return
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to process cell: {str(e)}")
                    return
        
        # No handler found - could add a message or just ignore
        pass
        
    def populate_table(self):
        """Populate the table with DataFrame data."""
        df = self.module.processed_data
        
        # Set table dimensions
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        
        # Set headers
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        self.table.setVerticalHeaderLabels([str(i) for i in df.index])
        
        # Populate cells
        for row in range(len(df)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                display_text = self.format_cell_value(value)
                
                item = QTableWidgetItem(display_text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                
                # Check if this cell is clickable (contains vector)
                row_data = df.iloc[row].to_dict()
                column_name = df.columns[col]
                is_clickable = any(handler.can_handle(value, row_data, column_name) 
                                 for handler in self.action_handlers)
                
                if is_clickable:
                    # Make clickable cells visually distinct
                    item.setBackground(Qt.lightGray)
                    item.setToolTip("Click to plot this vector")
                
                self.table.setItem(row, col, item)
        
        # Adjust column widths
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
    def format_cell_value(self, value):
        """Format cell value for display."""
        # Check for arrays/lists FIRST (before NaN check)
        if isinstance(value, (list, tuple)):
            return f"({len(value)},)"
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                return f"({value.shape[0]},)"
            else:
                return str(value.shape)
        # NOW safe to check for NaN on scalar values
        elif pd.isna(value):
            return "NaN"
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.4f}" if abs(value) < 1000 else f"{value:.2e}"
            else:
                return str(value)
        elif isinstance(value, str):
            # Truncate very long strings
            if len(value) > 50:
                return value[:47] + "..."
            return value
        else:
            # For any other type, convert to string
            str_val = str(value)
            if len(str_val) > 50:
                return str_val[:47] + "..."
            return str_val 
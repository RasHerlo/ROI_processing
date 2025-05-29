import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QLabel,
                            QHeaderView)
from PyQt5.QtCore import Qt

class DataFrameInspectionGUI(QMainWindow):
    """GUI for DataFrame inspection."""
    
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.setWindowTitle(f"DataFrame Inspection - {self.module.channel}")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setup_ui()
        self.populate_table()
        
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add info label
        info_label = QLabel(f"DataFrame Shape: {self.module.processed_data.shape}")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
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
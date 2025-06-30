#!/usr/bin/env python3
"""
bPAC_metrics_collect_eval.py

Interactive GUI for evaluating bPAC_metrics_collect.pkl files.
Displays dataframe on the left with sortable columns and clickable rows.
Shows corresponding figures on the right panel.
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk


class bPACMetricsEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("bPAC Metrics Evaluator")
        self.root.geometry("1400x800")
        
        # Data storage
        self.df = None
        self.current_sort_column = None
        self.sort_ascending = {}  # Track sort direction for each column
        
        # Load configuration
        self.config = self.load_config()
        
        # Setup GUI
        self.setup_gui()
        
        # Load file on startup
        self.load_file()
    
    def load_config(self):
        """Load configuration from config.json file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config
            else:
                return {}
        except Exception as e:
            return {}
    
    def get_initial_directory(self):
        """Get initial directory for file picker from config or default."""
        if self.config and 'data_folder' in self.config:
            config_dir = self.config['data_folder']
            if os.path.exists(config_dir):
                return config_dir
        
        # Fallback to current working directory
        return os.getcwd()
    
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main paned window (resizable panels)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Dataframe
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Right panel - Figure
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=1)
        
        self.setup_dataframe_panel()
        self.setup_figure_panel()
    
    def setup_dataframe_panel(self):
        """Setup the left panel with dataframe display."""
        # Title
        title_label = ttk.Label(self.left_frame, text="bPAC Metrics Data", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(self.left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, 
                                yscrollcommand=v_scrollbar.set,
                                xscrollcommand=h_scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.tree.yview)
        h_scrollbar.config(command=self.tree.xview)
        
        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self.on_row_select)
        self.tree.bind('<Button-1>', self.on_header_click)
    
    def setup_figure_panel(self):
        """Setup the right panel for figure display."""
        # Title
        title_label = ttk.Label(self.right_frame, text="bPAC Figure Display", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.show_message("Select a file to begin analysis.")
    
    def load_file(self):
        """Load a bPAC_metrics_collect.pkl file."""
        initial_dir = self.get_initial_directory()
        file_path = filedialog.askopenfilename(
            title="Select bPAC_metrics_collect.pkl file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        
        if not file_path:
            self.root.quit()
            return
        
        try:
            # Load the dataframe
            self.df = pd.read_pickle(file_path)
            
            # Validate dataframe structure
            required_columns = ['dir', 'Celltype', 'Tot_Counts', 'Mean bPAC_dFF fit']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                messagebox.showerror("Invalid File", 
                                   f"File is missing required columns: {missing_columns}")
                self.root.quit()
                return
            
            # Setup the treeview with data
            self.setup_treeview()
            
            # Select first row by default
            if len(self.df) > 0:
                first_item = self.tree.get_children()[0]
                self.tree.selection_set(first_item)
                self.tree.focus(first_item)
                self.on_row_select(None)
            
            self.show_message("Data loaded successfully. Select a row to view its figure.")
            
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Failed to load file:\n{str(e)}")
            self.root.quit()
    
    def setup_treeview(self):
        """Setup treeview columns and populate with data."""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Setup columns
        columns = list(self.df.columns)
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'
        
        # Configure columns with equal width and make them resizable
        total_width = 1200  # Approximate width for left panel
        col_width = total_width // len(columns)
        
        for col in columns:
            self.tree.heading(col, text=col, anchor='center')
            self.tree.column(col, width=col_width, minwidth=50, anchor='center')
            # Initialize sort direction
            self.sort_ascending[col] = None
        
        # Populate with data
        self.populate_treeview()
    
    def populate_treeview(self):
        """Populate treeview with current dataframe data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add rows
        for index, row in self.df.iterrows():
            # Format values for display
            values = []
            for col in self.df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("NaN")
                elif isinstance(val, float):
                    if np.isinf(val):
                        values.append("inf" if val > 0 else "-inf")
                    else:
                        values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            
            self.tree.insert('', 'end', values=values, tags=(index,))
    
    def on_header_click(self, event):
        """Handle column header clicks for sorting."""
        region = self.tree.identify_region(event.x, event.y)
        
        if region == "heading":
            column = self.tree.identify_column(event.x)
            column_index = int(column.replace('#', '')) - 1
            
            if 0 <= column_index < len(self.df.columns):
                column_name = self.df.columns[column_index]
                self.sort_by_column(column_name)
    
    def sort_by_column(self, column_name):
        """Sort dataframe by specified column."""
        # Determine sort direction
        if self.sort_ascending.get(column_name) is None:
            # First click - descending
            ascending = False
            self.sort_ascending[column_name] = False
            sort_symbol = " ▼"
        elif self.sort_ascending[column_name] == False:
            # Second click - ascending
            ascending = True
            self.sort_ascending[column_name] = True
            sort_symbol = " ▲"
        else:
            # Third click - descending again
            ascending = False
            self.sort_ascending[column_name] = False
            sort_symbol = " ▼"
        
        # Reset other column sort indicators
        for col in self.df.columns:
            if col != column_name:
                self.sort_ascending[col] = None
                self.tree.heading(col, text=col)
        
        # Update header with sort indicator
        self.tree.heading(column_name, text=column_name + sort_symbol)
        
        # Perform sorting based on data type
        try:
            # Check if column contains numeric data
            if self.df[column_name].dtype in ['int64', 'float64'] or \
               pd.api.types.is_numeric_dtype(self.df[column_name]):
                # Numeric sorting
                self.df = self.df.sort_values(column_name, ascending=ascending, na_position='last')
            else:
                # String/alphabetical sorting
                self.df = self.df.sort_values(column_name, ascending=ascending, na_position='last')
        except Exception as e:
            print(f"Sorting error: {e}")
            # Fallback to string sorting
            self.df = self.df.sort_values(column_name, ascending=ascending, na_position='last', key=str)
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        # Repopulate treeview
        self.populate_treeview()
        
        # Maintain selection if possible
        if len(self.df) > 0:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
    
    def on_row_select(self, event):
        """Handle row selection to display corresponding figure."""
        selection = self.tree.selection()
        if not selection:
            return
        
        # Get selected row data
        item = selection[0]
        values = self.tree.item(item, 'values')
        
        if not values:
            return
        
        # Find the row in dataframe
        dir_path = values[0]  # 'dir' is the first column
        
        # Load and display the figure
        self.load_figure(dir_path)
    
    def load_figure(self, dir_path):
        """Load and display figure based on directory path."""
        try:
            # Determine figure filename based on directory path
            if 'SUPPORT_ChanA' in dir_path:
                figure_name = 'bPAC_dFF_evaluation_ChanA.png'
            else:
                figure_name = 'bPAC_dFF_evaluation_ChanB.png'
            
            # Add 'derippled' subfolder to the path
            derippled_path = os.path.join(dir_path, 'derippled')
            figure_path = os.path.join(derippled_path, figure_name)
            
            # Check if base directory exists
            if not os.path.exists(dir_path):
                self.show_error(f"Base directory not found:\n{dir_path}")
                return
            
            # Check if derippled directory exists
            if not os.path.exists(derippled_path):
                self.show_error(f"Derippled directory not found:\n{derippled_path}\n\nExpected 'derippled' subfolder in:\n{dir_path}")
                return
            
            # Check if figure file exists
            if not os.path.exists(figure_path):
                self.show_error(f"Figure file not found:\n{figure_path}\n\nExpected: {figure_name}")
                return
            
            # Load and display the image
            self.display_image(figure_path)
            
        except Exception as e:
            self.show_error(f"Error loading figure:\n{str(e)}")
    
    def display_image(self, image_path):
        """Display image in the figure panel."""
        try:
            # Clear previous plot
            self.fig.clear()
            
            # Load image
            img = Image.open(image_path)
            
            # Create subplot and display image
            ax = self.fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            
            # Set title
            ax.set_title(os.path.basename(image_path), fontsize=10, pad=10)
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying image:\n{str(e)}")
    
    def show_message(self, message):
        """Show a text message in the figure panel."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', 
                fontsize=12, wrap=True, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
    
    def show_error(self, error_message):
        """Show an error message in the figure panel."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, f"ERROR:\n\n{error_message}", ha='center', va='center', 
                fontsize=11, color='red', wrap=True, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = bPACMetricsEvaluator(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()


if __name__ == "__main__":
    main() 
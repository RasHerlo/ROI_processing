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
        self.original_df = None  # Keep original for filtering
        self.current_sort_column = None
        self.sort_ascending = {}  # Track sort direction for each column
        self.file_path = None  # Store file path for saving
        
        # Checkbox columns and their states
        self.checkbox_columns = ['Derippling', 'Segmentation', 'HQ Exp', 'bPAC pos', 'TBD', 'To fix']
        self.checkbox_symbols = {0: '☐', 1: '⊡', 2: '☑'}  # Empty, Maybe, Good
        self.column_filters = {col: False for col in self.checkbox_columns}  # Filter states
        
        # Dropdown column options
        self.dropdown_column = 'ex-type'
        self.dropdown_options = ['LED only', 'LED+APs', 'AC Endfeet', 'Other']
        
        # Comments column
        self.comments_column = 'comments'
        
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
        self.tree.bind('<Button-1>', self.on_treeview_click)
    
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
            # Store file path for saving
            self.file_path = file_path
            
            # Load the dataframe
            self.df = pd.read_pickle(file_path)
            
            # Add ex-type dropdown column if it doesn't exist
            if self.dropdown_column not in self.df.columns:
                self.df[self.dropdown_column] = ''  # Initialize as empty string
            
            # Add comments column if it doesn't exist
            if self.comments_column not in self.df.columns:
                self.df[self.comments_column] = ''  # Initialize as empty string
            
            # Add checkbox columns if they don't exist
            for col in self.checkbox_columns:
                if col not in self.df.columns:
                    if col == 'TBD':
                        self.df[col] = 2  # Initialize TBD as checked (Good)
                    else:
                        self.df[col] = 0  # Initialize other checkboxes as empty
            
            # Reorder columns to put ex-type between dir and Celltype
            self.reorder_columns()
            
            # Validate dataframe structure
            required_columns = ['dir', 'Celltype', 'Tot_Counts', 'Mean bPAC_dFF fit']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                messagebox.showerror("Invalid File", 
                                   f"File is missing required columns: {missing_columns}")
                self.root.quit()
                return
            
            # Save original dataframe for filtering
            self.original_df = self.df.copy()
            
            # Setup the treeview with data
            self.setup_treeview()
            
            # Apply initial filtering
            self.apply_filters()
            
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
    
    def save_data(self):
        """Save the current dataframe back to the pkl file."""
        if self.file_path and self.original_df is not None:
            try:
                self.original_df.to_pickle(self.file_path)
                print(f"Data successfully saved to {self.file_path}")
            except Exception as e:
                print(f"Error saving data: {str(e)}")
        else:
            print("Warning: Cannot save data - file_path or original_df is None")
    
    def reorder_columns(self):
        """Reorder dataframe columns to put ex-type between dir and Celltype."""
        if 'dir' in self.df.columns and 'Celltype' in self.df.columns and self.dropdown_column in self.df.columns:
            # Get all columns
            cols = list(self.df.columns)
            
            # Remove ex-type from its current position
            if self.dropdown_column in cols:
                cols.remove(self.dropdown_column)
            
            # Find the position of Celltype
            celltype_idx = cols.index('Celltype')
            
            # Insert ex-type before Celltype
            cols.insert(celltype_idx, self.dropdown_column)
            
            # Reorder the dataframe
            self.df = self.df[cols]
    
    def setup_treeview(self):
        """Setup treeview columns and populate with data."""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Setup columns
        columns = list(self.original_df.columns)
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'
        
        # Configure columns with equal width and make them resizable
        total_width = 1200  # Approximate width for left panel
        col_width = total_width // len(columns)
        
        for col in columns:
            # Create header text with filter indicator for checkbox columns
            header_text = col
            if col in self.checkbox_columns:
                filter_symbol = '☑' if self.column_filters[col] else '☐'
                header_text = f"{filter_symbol} {col}"
            
            self.tree.heading(col, text=header_text, anchor='center')
            self.tree.column(col, width=col_width, minwidth=50, anchor='center')
            # Initialize sort direction
            self.sort_ascending[col] = None
    
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
                if col in self.checkbox_columns:
                    # Display checkbox symbol
                    values.append(self.checkbox_symbols.get(val, '☐'))
                elif col == self.dropdown_column:
                    # Display dropdown value or "Not Set"
                    values.append(str(val) if val else "Not Set")
                elif col == self.comments_column:
                    # Display comments status
                    status = "In use" if val else "Empty"
                    values.append(status)
                elif pd.isna(val):
                    values.append("NaN")
                elif isinstance(val, float):
                    if np.isinf(val):
                        values.append("inf" if val > 0 else "-inf")
                    else:
                        values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            
            # Store original dataframe index as tag
            original_index = self.original_df.index[self.original_df['dir'] == row['dir']].tolist()[0]
            self.tree.insert('', 'end', values=values, tags=(original_index,))
    
    def on_treeview_click(self, event):
        """Handle clicks on treeview (both headers and cells)."""
        region = self.tree.identify_region(event.x, event.y)
        
        if region == "heading":
            # Header click - handle sorting or filtering
            column = self.tree.identify_column(event.x)
            column_index = int(column.replace('#', '')) - 1
            
            if 0 <= column_index < len(self.original_df.columns):
                column_name = self.original_df.columns[column_index]
                
                if column_name in self.checkbox_columns:
                    # Toggle filter
                    self.toggle_column_filter(column_name)
                else:
                    # Sort by column
                    self.sort_by_column(column_name)
        
        elif region == "cell":
            # Cell click - handle checkbox toggle
            column = self.tree.identify_column(event.x)
            column_index = int(column.replace('#', '')) - 1
            
            if 0 <= column_index < len(self.original_df.columns):
                column_name = self.original_df.columns[column_index]
                
                if column_name in self.checkbox_columns:
                    # Toggle checkbox
                    item = self.tree.identify_row(event.y)
                    if item:
                        self.toggle_checkbox(item, column_name)
                elif column_name == self.dropdown_column:
                    # Open dropdown selection
                    item = self.tree.identify_row(event.y)
                    if item:
                        self.show_dropdown_selection(item, column_name)
                elif column_name == self.comments_column:
                    # Open comments text editor
                    item = self.tree.identify_row(event.y)
                    if item:
                        self.show_comments_editor(item, column_name)
    
    def toggle_column_filter(self, column_name):
        """Toggle filter for a checkbox column."""
        self.column_filters[column_name] = not self.column_filters[column_name]
        
        # Update header display
        filter_symbol = '☑' if self.column_filters[column_name] else '☐'
        header_text = f"{filter_symbol} {column_name}"
        self.tree.heading(column_name, text=header_text)
        
        # Apply filters
        self.apply_filters()
    
    def toggle_checkbox(self, item, column_name):
        """Toggle checkbox state for a specific item and column."""
        # Get original dataframe index from item tags
        tags = self.tree.item(item, 'tags')
        if not tags:
            return
        
        original_index = int(tags[0])
        
        # Get current value and cycle through states (0 -> 1 -> 2 -> 0)
        current_value = self.original_df.loc[original_index, column_name]
        new_value = (current_value + 1) % 3
        
        # Update original dataframe
        self.original_df.loc[original_index, column_name] = new_value
        
        # Save data
        self.save_data()
        
        # Refresh display
        self.apply_filters()
    
    def apply_filters(self):
        """Apply active filters to create filtered dataframe."""
        # Start with original dataframe
        filtered_df = self.original_df.copy()
        
        # Apply each active filter
        for col, is_active in self.column_filters.items():
            if is_active:
                # Show rows where checkbox is 1 (Maybe) or 2 (Good)
                filtered_df = filtered_df[filtered_df[col].isin([1, 2])]
        
        # Update current dataframe and display
        self.df = filtered_df.reset_index(drop=True)
        self.populate_treeview()
        
        # Try to maintain selection
        if len(self.df) > 0:
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
    
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
        
        # Reset other column sort indicators (except checkbox columns)
        for col in self.original_df.columns:
            if col != column_name and col not in self.checkbox_columns:
                self.sort_ascending[col] = None
                self.tree.heading(col, text=col)
        
        # Update header with sort indicator (if not checkbox column)
        if column_name not in self.checkbox_columns:
            self.tree.heading(column_name, text=column_name + sort_symbol)
        
        # Perform sorting on original dataframe
        try:
            # Check if column contains numeric data
            if self.original_df[column_name].dtype in ['int64', 'float64'] or \
               pd.api.types.is_numeric_dtype(self.original_df[column_name]):
                # Numeric sorting
                self.original_df = self.original_df.sort_values(column_name, ascending=ascending, na_position='last')
            else:
                # String/alphabetical sorting
                self.original_df = self.original_df.sort_values(column_name, ascending=ascending, na_position='last')
        except Exception as e:
            print(f"Sorting error: {e}")
            # Fallback to string sorting
            self.original_df = self.original_df.sort_values(column_name, ascending=ascending, na_position='last', key=str)
        
        # Reset index
        self.original_df = self.original_df.reset_index(drop=True)
        
        # Apply filters to update display
        self.apply_filters()
    
    def show_dropdown_selection(self, item, column_name):
        """Show dropdown selection dialog for ex-type column."""
        # Get original dataframe index from item tags
        tags = self.tree.item(item, 'tags')
        if not tags:
            return
        
        original_index = int(tags[0])
        
        # Get current value
        current_value = self.original_df.loc[original_index, column_name]
        
        # Create popup dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Experiment Type")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create label
        label = ttk.Label(dialog, text="Select experiment type:", font=('Arial', 10))
        label.pack(pady=10)
        
        # Create combobox
        selected_value = tk.StringVar(value=current_value)
        options = ['Clear'] + self.dropdown_options
        
        combo = ttk.Combobox(dialog, textvariable=selected_value, 
                            values=options, state='readonly', width=20)
        combo.pack(pady=10)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def on_ok():
            new_value = selected_value.get()
            if new_value == 'Clear':
                new_value = ''
            
            # Update original dataframe
            self.original_df.loc[original_index, column_name] = new_value
            
            # Save data
            self.save_data()
            
            # Refresh display
            self.apply_filters()
            
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
        ok_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Focus on combobox
        combo.focus()
    
    def show_comments_editor(self, item, column_name):
        """Show text editor dialog for comments column."""
        # Get original dataframe index from item tags
        tags = self.tree.item(item, 'tags')
        if not tags:
            return
        
        original_index = int(tags[0])
        
        # Get current value
        current_value = self.original_df.loc[original_index, column_name]
        
        # Create popup dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Comments")
        dialog.geometry("500x400")
        dialog.resizable(True, True)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create label
        label = ttk.Label(dialog, text="Enter comments:", font=('Arial', 10))
        label.pack(pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text widget
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Arial', 10))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert current value
        text_widget.insert(tk.END, current_value)
        
        # Character counter
        char_count_var = tk.StringVar()
        char_count_label = ttk.Label(dialog, textvariable=char_count_var, font=('Arial', 9))
        char_count_label.pack(pady=2)
        
        def update_char_count():
            """Update character count display."""
            content = text_widget.get(1.0, tk.END)[:-1]  # Remove trailing newline
            char_count = len(content)
            char_count_var.set(f"Characters: {char_count}")
        
        # Update counter initially and on key events
        update_char_count()
        text_widget.bind('<KeyRelease>', lambda e: update_char_count())
        text_widget.bind('<Button-1>', lambda e: dialog.after(1, update_char_count))
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_save():
            """Save the comments text."""
            new_value = text_widget.get(1.0, tk.END)[:-1]  # Remove trailing newline
            
            # Debug output
            print(f"Saving comment: '{new_value}' (length: {len(new_value)})")
            
            # Update original dataframe
            self.original_df.loc[original_index, column_name] = new_value
            
            # Save data
            self.save_data()
            
            # Refresh display
            self.apply_filters()
            
            dialog.destroy()
        
        def on_cancel():
            """Cancel without saving."""
            dialog.destroy()
        
        # Buttons
        save_button = ttk.Button(button_frame, text="Save", command=on_save)
        save_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Focus on text widget
        text_widget.focus()
        
        # Select all text for easy editing
        text_widget.tag_add("sel", "1.0", "end")
    
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
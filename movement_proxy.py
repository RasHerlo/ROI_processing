#!/usr/bin/env python3
"""
movement_proxy.py

Interactive GUI for inspecting suite2p ChanA data availability
for Astrocyte rows flagged as bPAC positive (bPAC pos = 1 or 2).

Layout mirrors bPAC_metrics_collect_eval.py:
  - Left panel: filtered dataframe with sortable columns
  - Right panel: visualization for the selected row
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELLTYPE_FILTER = "Astrocytes"
BPAC_POS_VALUES = [1, 2]

CHECKBOX_SYMBOLS = {True: "☑", False: "☐"}

# Columns shown in the treeview (in order)
DISPLAY_COLUMNS = ["AC dir", "ex-type", "bPAC pos", "s2p ChanA", "Fneu"]


# ---------------------------------------------------------------------------
# Helper functions (pure, no GUI dependency)
# ---------------------------------------------------------------------------

def derive_chanA_path(dir_path: str) -> str:
    """
    Given a dir path ending in …/SUPPORT_ChanB, return the corresponding
    suite2p plane0 folder under SUPPORT_ChanA.

    Example:
        …/SUPPORT_ChanB  →  …/SUPPORT_ChanA/derippled/suite2p/plane0
    """
    clean = dir_path.rstrip("/\\")
    parent = os.path.dirname(clean)
    return os.path.join(parent, "SUPPORT_ChanA", "derippled", "suite2p", "plane0")


def check_fneu(chanA_path: str) -> bool | None:
    """
    Return True  if Fneu.npy exists inside chanA_path,
           False if the folder exists but Fneu.npy is missing,
           None  if the folder itself does not exist.
    """
    if not os.path.isdir(chanA_path):
        return None
    return os.path.isfile(os.path.join(chanA_path, "Fneu.npy"))


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class MovementProxyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Movement Proxy")
        self.root.geometry("1400x800")

        # ── Data ──────────────────────────────────────────────────────────
        self.source_df: pd.DataFrame | None = None   # filtered rows from pkl
        self.display_df: pd.DataFrame | None = None  # what is currently shown
        self.file_path: str | None = None

        # Sort state: column → True (asc) | False (desc) | None (unsorted)
        self.sort_state: dict[str, bool | None] = {}

        # ── Config & GUI ──────────────────────────────────────────────────
        self.config = self._load_config()
        self._setup_gui()
        self._load_file()

    # ── Config ────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _initial_dir(self) -> str:
        folder = self.config.get("data_folder", "")
        if folder and os.path.exists(folder):
            return folder
        return os.getcwd()

    # ── GUI setup ─────────────────────────────────────────────────────────

    def _setup_gui(self):
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.left_frame = ttk.Frame(self.paned)
        self.paned.add(self.left_frame, weight=1)

        self.right_frame = ttk.Frame(self.paned)
        self.paned.add(self.right_frame, weight=1)

        self._setup_table_panel()
        self._setup_viz_panel()

    def _setup_table_panel(self):
        ttk.Label(self.left_frame, text="Movement Proxy – Astrocyte bPAC+ Rows",
                  font=("Arial", 12, "bold")).pack(pady=(0, 6))

        tree_frame = ttk.Frame(self.left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(tree_frame)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll.config(command=self.tree.yview)
        h_scroll.config(command=self.tree.xview)

        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self.tree.bind("<Button-1>", self._on_treeview_click)

    def _setup_viz_panel(self):
        ttk.Label(self.right_frame, text="Visualization",
                  font=("Arial", 12, "bold")).pack(pady=(0, 6))

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._show_message("Select a file to begin.")

    # ── File loading & filtering ──────────────────────────────────────────

    def _load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select bPAC_metrics_collect.pkl file",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=self._initial_dir(),
        )

        if not file_path:
            self.root.quit()
            return

        try:
            self.file_path = file_path
            raw: pd.DataFrame = pd.read_pickle(file_path)

            # Validate required source columns
            required = ["dir", "Celltype", "bPAC pos", "ex-type"]
            missing = [c for c in required if c not in raw.columns]
            if missing:
                messagebox.showerror("Invalid File",
                                     f"Missing required columns: {missing}")
                self.root.quit()
                return

            # Filter: Astrocytes with bPAC pos 1 or 2
            mask = (raw["Celltype"] == CELLTYPE_FILTER) & \
                   (raw["bPAC pos"].isin(BPAC_POS_VALUES))
            filtered = raw[mask].copy().reset_index(drop=True)

            if filtered.empty:
                messagebox.showwarning("No Data",
                                       "No Astrocyte rows with bPAC pos = 1 or 2 found.")
                self.root.quit()
                return

            # Build display dataframe
            self.source_df = self._build_display_df(filtered)
            self.display_df = self.source_df.copy()

            self._configure_treeview()
            self._populate_treeview()

            # Select first row
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self._on_row_select(None)

        except Exception as exc:
            messagebox.showerror("Error Loading File", f"Failed to load file:\n{exc}")
            self.root.quit()

    def _build_display_df(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the five-column dataframe shown in the treeview.
        All values are strings ready for display.
        """
        rows = []
        for _, row in filtered.iterrows():
            dir_path = str(row["dir"])
            chanA = derive_chanA_path(dir_path)
            folder_exists = os.path.isdir(chanA)
            fneu_state = check_fneu(chanA)

            s2p_display = chanA if folder_exists else "suite2p folder does not exist yet"
            fneu_display = (CHECKBOX_SYMBOLS[True]  if fneu_state is True  else
                            CHECKBOX_SYMBOLS[False] if fneu_state is False else
                            "—")  # folder absent

            rows.append({
                "AC dir":   dir_path,
                "ex-type":  str(row.get("ex-type", "")),
                "bPAC pos": str(int(row["bPAC pos"])),
                "s2p ChanA": s2p_display,
                "Fneu":     fneu_display,
                # Store raw paths for visualization lookup
                "_dir_raw":   dir_path,
                "_chanA_raw": chanA,
                "_fneu_raw":  fneu_state,
            })

        return pd.DataFrame(rows)

    # ── Treeview setup & population ───────────────────────────────────────

    def _configure_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree["columns"] = DISPLAY_COLUMNS
        self.tree["show"] = "headings"

        col_widths = {
            "AC dir":    340,
            "ex-type":   110,
            "bPAC pos":   70,
            "s2p ChanA": 340,
            "Fneu":       55,
        }

        for col in DISPLAY_COLUMNS:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=col_widths.get(col, 120),
                             minwidth=50, anchor="center")
            self.sort_state[col] = None

    def _populate_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx, row in self.display_df.iterrows():
            values = [row[c] for c in DISPLAY_COLUMNS]
            self.tree.insert("", "end", iid=str(idx), values=values, tags=(str(idx),))

    # ── Sorting ───────────────────────────────────────────────────────────

    def _on_treeview_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != "heading":
            return

        col_id = self.tree.identify_column(event.x)
        col_idx = int(col_id.replace("#", "")) - 1
        if 0 <= col_idx < len(DISPLAY_COLUMNS):
            self._sort_by_column(DISPLAY_COLUMNS[col_idx])

    def _sort_by_column(self, col: str):
        current = self.sort_state.get(col)

        # Cycle: None → desc → asc → desc …
        ascending = True if current is False else False
        self.sort_state[col] = ascending

        # Reset other columns' headers
        for c in DISPLAY_COLUMNS:
            label = c
            if c == col:
                label = f"{c} {'▲' if ascending else '▼'}"
            self.tree.heading(c, text=label)
            if c != col:
                self.sort_state[c] = None

        self.display_df = self.display_df.sort_values(
            col, ascending=ascending, na_position="last",
            key=lambda s: s.str.lower() if s.dtype == object else s,
        ).reset_index(drop=True)

        self._populate_treeview()

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    # ── Row selection → visualization ─────────────────────────────────────

    def _on_row_select(self, _event):
        selection = self.tree.selection()
        if not selection:
            return

        iid = selection[0]
        idx = int(iid)
        row = self.display_df.iloc[idx]

        self._update_visualization(row)

    def _update_visualization(self, row: pd.Series):
        """
        Render the right-hand panel for the selected row.

        This is the extension point for future visualization work.
        Currently shows a summary of the row's key paths and status.
        """
        dir_path  = row["_dir_raw"]
        chanA     = row["_chanA_raw"]
        fneu      = row["_fneu_raw"]
        ex_type   = row["ex-type"]
        bpac_pos  = row["bPAC pos"]

        folder_ok = os.path.isdir(chanA)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")

        status_color = "#2ecc71" if folder_ok else "#e74c3c"
        fneu_text    = ("Fneu.npy  ✓" if fneu is True  else
                        "Fneu.npy  ✗" if fneu is False else
                        "Fneu.npy  —")

        summary = (
            f"AC dir:\n  {dir_path}\n\n"
            f"s2p ChanA:\n  {chanA}\n\n"
            f"Folder exists:  {'Yes' if folder_ok else 'No'}\n"
            f"{fneu_text}\n\n"
            f"ex-type: {ex_type}    bPAC pos: {bpac_pos}"
        )

        ax.text(
            0.05, 0.95, summary,
            ha="left", va="top",
            fontsize=10,
            fontfamily="monospace",
            transform=ax.transAxes,
            wrap=True,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#f0f0f0",
                edgecolor=status_color,
                linewidth=2,
            ),
        )

        self.canvas.draw()

    # ── Utility display helpers ────────────────────────────────────────────

    def _show_message(self, msg: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _show_error(self, msg: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, f"ERROR:\n\n{msg}", ha="center", va="center",
                fontsize=11, color="red", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightgray", alpha=0.8))
        ax.axis("off")
        self.canvas.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    MovementProxyGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
movement_proxy.py

Interactive GUI for inspecting suite2p ChanA data availability
for Astrocyte rows flagged as bPAC positive (bPAC pos = 1 or 2).

Layout:
  Left  : filtered dataframe table (sortable)
  Right : ┌── "Astrocytes" ──────────┬── "Neuropil Inspection" ──────────────────────┐
          │  bPAC_dFF_eval_ChanB.png │  [Raster (Fneu)] │ [TIF stack viewer]         │
          │  (static PNG)            │  [Co-Activity  ] │                            │
          │                          │  [══════ shared frame slider ══════════════]  │
          └──────────────────────────┴────────────────────────────────────────────────┘

TIF loading runs in a background daemon thread.  The raster and co-activity
plots are rendered immediately (Fneu.npy is small); the TIF panel shows live
progress while the stack is converted to uint8.
"""

import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELLTYPE_FILTER  = "Astrocytes"
BPAC_POS_VALUES  = [1, 2]
CHECKBOX_SYMBOLS = {True: "☑", False: "☐"}
DISPLAY_COLUMNS  = ["AC dir", "ex-type", "bPAC pos", "s2p ChanA", "Fneu"]
SLIDER_DEBOUNCE_MS = 200


# ---------------------------------------------------------------------------
# Pure path-helper functions
# ---------------------------------------------------------------------------

def derive_chanA_path(dir_path: str) -> str:
    """…/SUPPORT_ChanB  →  …/SUPPORT_ChanA/derippled/suite2p/plane0"""
    clean = dir_path.rstrip("/\\")
    parent = os.path.dirname(clean)
    return os.path.join(parent, "SUPPORT_ChanA", "derippled", "suite2p", "plane0")


def derive_tif_path(chanA_path: str) -> str:
    """Two levels up from plane0: …/SUPPORT_ChanA/derippled/derippled_stk.tif"""
    derippled_dir = os.path.dirname(os.path.dirname(chanA_path))  # suite2p → derippled
    return os.path.join(derippled_dir, "derippled_stack.tif")


def derive_bpac_png_path(dir_path: str) -> str:
    """dir_path/derippled/bPAC_dFF_evaluation_ChanB.png"""
    return os.path.join(dir_path, "derippled", "bPAC_dFF_evaluation_ChanB.png")


def check_fneu(chanA_path: str):
    """True = Fneu.npy present, False = folder exists but file missing, None = no folder."""
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
        self.root.geometry("1800x900")

        # ── Dataframe state ───────────────────────────────────────────────
        self.source_df: pd.DataFrame | None = None
        self.display_df: pd.DataFrame | None = None
        self.file_path: str | None = None
        self.sort_state: dict = {}

        # ── Neuropil data state ───────────────────────────────────────────
        self._fneu_norm: np.ndarray | None = None    # (n_rois, n_frames) float32
        self._coactivity: np.ndarray | None = None   # (n_frames,) float32
        self._n_frames_fneu: int | None = None
        self._tif_frames: np.ndarray | None = None   # (n_frames, h, w) uint8

        # ── Threading / debounce ──────────────────────────────────────────
        self._tif_load_cancel = threading.Event()
        self._debounce_id: str | None = None

        # ── Matplotlib artist refs for fast in-place slider updates ───────
        self._raster_vline = None
        self._coact_vline  = None
        self._tif_im       = None
        self._loading_text = None
        self._ax_coact     = None   # kept for right-click y-limit dialog

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
        # Outer split: table (left) | visualization area (right)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)

        self.viz_outer = ttk.Frame(self.main_paned)
        self.main_paned.add(self.viz_outer, weight=3)

        self._setup_table_panel()

        # Inner split inside visualization area: Astrocytes | Neuropil
        self.viz_paned = ttk.PanedWindow(self.viz_outer, orient=tk.HORIZONTAL)
        self.viz_paned.pack(fill=tk.BOTH, expand=True)

        self.astro_frame    = ttk.Frame(self.viz_paned)
        self.neuropil_frame = ttk.Frame(self.viz_paned)
        self.viz_paned.add(self.astro_frame,    weight=1)
        self.viz_paned.add(self.neuropil_frame, weight=2)

        self._setup_astro_panel()
        self._setup_neuropil_panel()

    def _setup_table_panel(self):
        ttk.Label(self.left_frame,
                  text="Movement Proxy – Astrocyte bPAC+ Rows",
                  font=("Arial", 11, "bold")).pack(pady=(0, 6))

        tree_frame = ttk.Frame(self.left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(tree_frame)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree = ttk.Treeview(tree_frame,
                                  yscrollcommand=v_scroll.set,
                                  xscrollcommand=h_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.tree.yview)
        h_scroll.config(command=self.tree.xview)

        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self.tree.bind("<Button-1>",         self._on_treeview_click)

    def _setup_astro_panel(self):
        ttk.Label(self.astro_frame, text="Astrocytes",
                  font=("Arial", 11, "bold")).pack(pady=(0, 4))

        self.fig_astro   = Figure(figsize=(5, 6), dpi=100)
        self.canvas_astro = FigureCanvasTkAgg(self.fig_astro, self.astro_frame)
        self.canvas_astro.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._astro_message("Select a row to view the bPAC figure.")

    def _setup_neuropil_panel(self):
        ttk.Label(self.neuropil_frame, text="Neuropil Inspection",
                  font=("Arial", 11, "bold")).pack(pady=(0, 4))

        self.fig_neuropil    = Figure(figsize=(10, 6), dpi=100)
        self.canvas_neuropil = FigureCanvasTkAgg(self.fig_neuropil, self.neuropil_frame)
        self.canvas_neuropil.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right-click on co-activity axes → y-limit dialog
        self.canvas_neuropil.mpl_connect(
            "button_press_event", self._on_neuropil_click)

        # Slider row
        slider_frame = ttk.Frame(self.neuropil_frame)
        slider_frame.pack(fill=tk.X, padx=8, pady=(2, 4))

        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT, padx=(0, 4))

        self.frame_label = ttk.Label(slider_frame, text="– / –", width=12)
        self.frame_label.pack(side=tk.RIGHT)

        self.slider = tk.Scale(
            slider_frame,
            orient=tk.HORIZONTAL,
            from_=0, to=100,
            resolution=1,
            showvalue=False,
            state="disabled",
            command=self._on_slider_change,
        )
        self.slider.pack(fill=tk.X, expand=True, side=tk.LEFT)

        self._neuropil_message("Select a row to begin.")

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

            required = ["dir", "Celltype", "bPAC pos", "ex-type"]
            missing  = [c for c in required if c not in raw.columns]
            if missing:
                messagebox.showerror("Invalid File",
                                     f"Missing required columns: {missing}")
                self.root.quit()
                return

            mask = ((raw["Celltype"] == CELLTYPE_FILTER) &
                    (raw["bPAC pos"].isin(BPAC_POS_VALUES)))
            filtered = raw[mask].copy().reset_index(drop=True)

            if filtered.empty:
                messagebox.showwarning(
                    "No Data",
                    "No Astrocyte rows with bPAC pos = 1 or 2 found.")
                self.root.quit()
                return

            self.source_df  = self._build_display_df(filtered)
            self.display_df = self.source_df.copy()
            self._configure_treeview()
            self._populate_treeview()

            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self._on_row_select(None)

        except Exception as exc:
            messagebox.showerror("Error Loading File",
                                 f"Failed to load file:\n{exc}")
            self.root.quit()

    def _build_display_df(self, filtered: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in filtered.iterrows():
            dir_path = str(row["dir"])
            chanA    = derive_chanA_path(dir_path)
            folder_ok   = os.path.isdir(chanA)
            fneu_state  = check_fneu(chanA)

            s2p_display  = chanA if folder_ok else "suite2p folder does not exist yet"
            fneu_display = (CHECKBOX_SYMBOLS[True]  if fneu_state is True  else
                            CHECKBOX_SYMBOLS[False] if fneu_state is False else
                            "—")

            rows.append({
                "AC dir":    dir_path,
                "ex-type":   str(row.get("ex-type", "")),
                "bPAC pos":  str(int(row["bPAC pos"])),
                "s2p ChanA": s2p_display,
                "Fneu":      fneu_display,
                # Raw values for internal use (not shown in treeview)
                "_dir_raw":   dir_path,
                "_chanA_raw": chanA,
                "_fneu_raw":  fneu_state,
            })
        return pd.DataFrame(rows)

    # ── Treeview ──────────────────────────────────────────────────────────

    def _configure_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree["columns"] = DISPLAY_COLUMNS
        self.tree["show"]    = "headings"

        col_widths = {
            "AC dir": 300, "ex-type": 100,
            "bPAC pos": 65, "s2p ChanA": 300, "Fneu": 50,
        }
        for col in DISPLAY_COLUMNS:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=col_widths.get(col, 100),
                             minwidth=40, anchor="center")
            self.sort_state[col] = None

    def _populate_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, row in self.display_df.iterrows():
            values = [row[c] for c in DISPLAY_COLUMNS]
            self.tree.insert("", "end", iid=str(idx), values=values,
                             tags=(str(idx),))

    def _on_treeview_click(self, event):
        if self.tree.identify_region(event.x, event.y) != "heading":
            return
        col_id  = self.tree.identify_column(event.x)
        col_idx = int(col_id.replace("#", "")) - 1
        if 0 <= col_idx < len(DISPLAY_COLUMNS):
            self._sort_by_column(DISPLAY_COLUMNS[col_idx])

    def _sort_by_column(self, col: str):
        ascending = True if self.sort_state.get(col) is False else False
        self.sort_state[col] = ascending

        for c in DISPLAY_COLUMNS:
            label = f"{c} {'▲' if ascending else '▼'}" if c == col else c
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

    # ── Row selection ─────────────────────────────────────────────────────

    def _on_row_select(self, _event):
        selection = self.tree.selection()
        if not selection:
            return

        idx      = int(selection[0])
        row      = self.display_df.iloc[idx]
        dir_path = row["_dir_raw"]
        chanA    = row["_chanA_raw"]

        # Cancel any in-progress TIF load immediately (silent discard)
        self._cancel_tif_load()

        # Invalidate old artist refs
        self._raster_vline = None
        self._coact_vline  = None
        self._tif_im       = None
        self._loading_text = None
        self._ax_coact     = None
        self._tif_frames   = None
        self.slider.config(state="disabled")
        self.frame_label.config(text="– / –")

        # 1. Astrocytes panel — fast synchronous PNG load
        self._load_astro_png(dir_path)

        # 2. Fneu — fast synchronous numpy load
        if not self._load_fneu(chanA):
            return

        # 3. TIF — slow background load
        tif_path = derive_tif_path(chanA)
        if not os.path.isfile(tif_path):
            self._neuropil_error(f"TIF stack not found:\n{tif_path}")
            return

        self._start_tif_load(tif_path)

    # ── Astrocytes panel ──────────────────────────────────────────────────

    def _load_astro_png(self, dir_path: str):
        png_path = derive_bpac_png_path(dir_path)
        self.fig_astro.clear()
        ax = self.fig_astro.add_subplot(111)
        ax.axis("off")

        if os.path.isfile(png_path):
            try:
                img = Image.open(png_path)
                ax.imshow(img)
                ax.set_title(os.path.basename(png_path), fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading PNG:\n{e}",
                        ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"PNG not found:\n{png_path}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        self.fig_astro.tight_layout()
        self.canvas_astro.draw()

    def _astro_message(self, msg: str):
        self.fig_astro.clear()
        ax = self.fig_astro.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.axis("off")
        self.canvas_astro.draw()

    # ── Fneu loading ──────────────────────────────────────────────────────

    def _load_fneu(self, chanA_path: str) -> bool:
        fneu_path = os.path.join(chanA_path, "Fneu.npy")

        if not os.path.isfile(fneu_path):
            self._neuropil_error(f"Fneu.npy not found:\n{fneu_path}")
            return False

        try:
            fneu = np.load(fneu_path).astype(np.float32)  # (n_rois, n_frames)
            if fneu.ndim != 2:
                self._neuropil_error(
                    f"Fneu.npy has unexpected shape: {fneu.shape}\n"
                    "Expected a 2-D array (n_rois × n_frames).")
                return False

            # Per-row min-max normalization: each ROI scaled to [0, 1]
            row_min = fneu.min(axis=1, keepdims=True)
            row_max = fneu.max(axis=1, keepdims=True)
            denom   = np.where(row_max > row_min, row_max - row_min, 1.0)
            fneu_norm = np.clip((fneu - row_min) / denom, 0.0, 1.0)

            self._fneu_norm      = fneu_norm
            self._coactivity     = fneu_norm.mean(axis=0)
            self._n_frames_fneu  = fneu_norm.shape[1]
            return True

        except Exception as e:
            self._neuropil_error(f"Error loading Fneu.npy:\n{e}")
            return False

    # ── TIF background loading ────────────────────────────────────────────

    def _cancel_tif_load(self):
        """Signal the running thread to stop; hand out a fresh Event for the next load."""
        self._tif_load_cancel.set()
        self._tif_load_cancel = threading.Event()

    def _start_tif_load(self, tif_path: str):
        """Render raster immediately, then kick off the TIF loader thread."""
        self._init_neuropil_raster_only()   # draws raster + loading placeholder
        cancel = self._tif_load_cancel
        threading.Thread(
            target=self._tif_load_worker,
            args=(tif_path, cancel),
            daemon=True,
        ).start()

    def _tif_load_worker(self, tif_path: str, cancel: threading.Event):
        try:
            if not HAS_TIFFFILE:
                if not cancel.is_set():
                    self.root.after(0, lambda: self._neuropil_error(
                        "tifffile is not installed.\n"
                        "Run:  pip install tifffile"))
                return

            with tifffile.TiffFile(tif_path) as tf:
                n_frames = len(tf.pages)
                if n_frames == 0 or cancel.is_set():
                    return

                # Pass 1 – sample ~50 frames to estimate global intensity range
                sample_idx  = np.linspace(0, n_frames - 1,
                                          min(50, n_frames), dtype=int)
                sample_data = np.stack(
                    [tf.pages[int(i)].asarray() for i in sample_idx]
                ).astype(np.float32)
                vmin  = float(np.percentile(sample_data, 1))
                vmax  = float(np.percentile(sample_data, 99))
                scale = (vmax - vmin) if vmax > vmin else 1.0

                if cancel.is_set():
                    return

                # Pass 2 – load every frame, normalise to uint8
                first = tf.pages[0].asarray()
                h, w  = first.shape[:2]
                frames = np.empty((n_frames, h, w), dtype=np.uint8)

                for i, page in enumerate(tf.pages):
                    if cancel.is_set():
                        return
                    frame = page.asarray().astype(np.float32)
                    frame = np.clip((frame - vmin) / scale, 0.0, 1.0)
                    frames[i] = (frame * 255).astype(np.uint8)

                    if i % 100 == 0:
                        fi, n = i, n_frames      # capture for lambda
                        if not cancel.is_set():
                            self.root.after(
                                0, lambda fi=fi, n=n:
                                    self._update_tif_progress(fi, n))

                if cancel.is_set():
                    return

                self.root.after(0, lambda: self._on_tif_loaded(frames))

        except Exception as exc:
            if not cancel.is_set():
                msg = str(exc)
                self.root.after(0, lambda: self._neuropil_error(
                    f"Error loading TIF:\n{msg}"))

    def _update_tif_progress(self, frame_i: int, n_frames: int):
        """Update the loading placeholder text; called in the main thread."""
        if self._loading_text is not None:
            pct = int(frame_i / n_frames * 100)
            self._loading_text.set_text(
                f"Loading TIF…\n{frame_i} / {n_frames} frames\n({pct} %)")
            self.canvas_neuropil.draw_idle()

    def _on_tif_loaded(self, frames: np.ndarray):
        """Called in the main thread once all frames have been loaded."""
        n_tif  = frames.shape[0]
        n_fneu = self._n_frames_fneu

        if n_fneu is not None and n_tif != n_fneu:
            self._neuropil_error(
                f"Frame count mismatch:\n"
                f"  Fneu.npy  →  {n_fneu} frames\n"
                f"  TIF stack →  {n_tif} frames\n\n"
                "Cannot display Neuropil Inspection.")
            return

        self._tif_frames = frames
        self.slider.config(state="normal", from_=0, to=n_tif - 1)
        self.slider.set(0)
        self.frame_label.config(text=f"0 / {n_tif - 1}")
        self._init_neuropil_full(initial_frame=0)

    # ── Neuropil figure rendering ──────────────────────────────────────────

    def _make_neuropil_gridspec(self):
        """Return a fresh GridSpec for the neuropil figure."""
        return GridSpec(
            2, 2,
            height_ratios=[4, 1],
            width_ratios=[1, 1],
            hspace=0.35,
            wspace=0.30,
            figure=self.fig_neuropil,
        )

    def _init_neuropil_raster_only(self):
        """
        Draw raster + co-activity immediately after a row is selected.
        The right column shows a 'Loading TIF…' placeholder that is updated
        in-place as the background thread progresses.
        """
        if self._fneu_norm is None:
            return

        n_frames = self._n_frames_fneu
        self.fig_neuropil.clear()
        gs = self._make_neuropil_gridspec()

        ax_raster = self.fig_neuropil.add_subplot(gs[0, 0])
        ax_coact  = self.fig_neuropil.add_subplot(gs[1, 0])
        ax_right  = self.fig_neuropil.add_subplot(gs[:, 1])

        self._draw_raster(ax_raster, initial_frame=0)
        self._draw_coactivity(ax_coact, n_frames, initial_frame=0)

        # Loading placeholder (text object kept for in-place updates)
        ax_right.axis("off")
        self._loading_text = ax_right.text(
            0.5, 0.5, "Loading TIF…\n",
            ha="center", va="center",
            fontsize=11, transform=ax_right.transAxes, color="gray",
        )

        self.fig_neuropil.tight_layout()
        self.canvas_neuropil.draw()

    def _init_neuropil_full(self, initial_frame: int = 0):
        """
        Build the complete 3-axes neuropil figure once the TIF is loaded.
        Replaces the loading placeholder with the actual stack viewer.
        """
        if self._fneu_norm is None or self._tif_frames is None:
            return

        n_frames = self._tif_frames.shape[0]
        self.fig_neuropil.clear()
        gs = self._make_neuropil_gridspec()

        ax_raster = self.fig_neuropil.add_subplot(gs[0, 0])
        ax_coact  = self.fig_neuropil.add_subplot(gs[1, 0])
        ax_tif    = self.fig_neuropil.add_subplot(gs[:, 1])

        self._draw_raster(ax_raster, initial_frame)
        self._draw_coactivity(ax_coact, n_frames, initial_frame)

        # TIF viewer
        self._tif_im = ax_tif.imshow(
            self._tif_frames[initial_frame],
            cmap="gray", vmin=0, vmax=255, aspect="equal",
        )
        ax_tif.axis("off")
        ax_tif.set_title("ChanA Stack", fontsize=9)

        self._loading_text = None
        self.fig_neuropil.tight_layout()
        self.canvas_neuropil.draw()

    def _draw_raster(self, ax, initial_frame: int):
        """Render the normalised Fneu raster and store the red vline ref."""
        ax.imshow(
            self._fneu_norm,
            aspect="auto",
            cmap="hot",
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_ylabel("ROI index", fontsize=8)
        ax.set_xlabel("Frame",     fontsize=8)
        ax.set_title("Neuropil Raster", fontsize=9)
        ax.tick_params(labelsize=7)
        self._raster_vline = ax.axvline(
            x=initial_frame, color="red", linewidth=1.2, zorder=5)

    def _draw_coactivity(self, ax, n_frames: int, initial_frame: int,
                         ylim: tuple | None = None):
        """Render the co-activity trace and store the axes + vline refs."""
        ax.plot(self._coactivity, color="steelblue", linewidth=0.8)
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylabel("Mean", fontsize=7)
        ax.set_xlabel("Frame", fontsize=7)
        ax.set_title("Co-Activity  (right-click to set y-limits)", fontsize=9)
        ax.tick_params(labelsize=6)
        if ylim is not None:
            ax.set_ylim(ylim)
        self._coact_vline = ax.axvline(
            x=initial_frame, color="red", linewidth=1.2, zorder=5)
        self._ax_coact = ax

    # ── Co-activity right-click y-limit dialog ────────────────────────────

    def _on_neuropil_click(self, event):
        """Matplotlib button-press callback — opens y-limit dialog on right-click."""
        if event.button != 3:
            return
        if self._ax_coact is None or event.inaxes is not self._ax_coact:
            return
        self._show_ylim_dialog()

    def _show_ylim_dialog(self):
        """Small popup to set y-axis limits on the co-activity plot."""
        if self._ax_coact is None:
            return

        current_ymin, current_ymax = self._ax_coact.get_ylim()

        dialog = tk.Toplevel(self.root)
        dialog.title("Co-Activity Y-limits")
        dialog.geometry("260x160")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        pad = {"padx": 10, "pady": 4}

        ttk.Label(dialog, text="Set Y-axis limits for Co-Activity",
                  font=("Arial", 9, "bold")).pack(**pad)

        fields_frame = ttk.Frame(dialog)
        fields_frame.pack(**pad)

        ttk.Label(fields_frame, text="Y-min:").grid(row=0, column=0, sticky="e", padx=4)
        entry_ymin = ttk.Entry(fields_frame, width=12)
        entry_ymin.insert(0, f"{current_ymin:.6g}")
        entry_ymin.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(fields_frame, text="Y-max:").grid(row=1, column=0, sticky="e", padx=4)
        entry_ymax = ttk.Entry(fields_frame, width=12)
        entry_ymax.insert(0, f"{current_ymax:.6g}")
        entry_ymax.grid(row=1, column=1, padx=4, pady=2)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)

        def on_apply():
            try:
                ymin = float(entry_ymin.get())
                ymax = float(entry_ymax.get())
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "Y-min and Y-max must be numbers.",
                                     parent=dialog)
                return
            if ymin >= ymax:
                messagebox.showerror("Invalid input",
                                     "Y-min must be less than Y-max.",
                                     parent=dialog)
                return
            self._ax_coact.set_ylim(ymin, ymax)
            self.canvas_neuropil.draw_idle()
            dialog.destroy()

        def on_reset():
            if self._coactivity is not None:
                data_min = float(self._coactivity.min())
                data_max = float(self._coactivity.max())
                margin   = (data_max - data_min) * 0.05
                entry_ymin.delete(0, tk.END)
                entry_ymin.insert(0, f"{data_min - margin:.6g}")
                entry_ymax.delete(0, tk.END)
                entry_ymax.insert(0, f"{data_max + margin:.6g}")

        ttk.Button(btn_frame, text="Apply",  command=on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset",  command=on_reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        entry_ymin.focus()
        dialog.bind("<Return>", lambda _: on_apply())

    # ── Slider ────────────────────────────────────────────────────────────

    def _on_slider_change(self, value: str):
        if self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(
            SLIDER_DEBOUNCE_MS, self._apply_frame, int(float(value)))

    def _apply_frame(self, frame_idx: int):
        """Fast in-place update: move vlines and swap TIF frame data."""
        self._debounce_id = None

        if (self._tif_frames   is None or
                self._raster_vline is None or
                self._coact_vline  is None or
                self._tif_im       is None):
            return

        n = self._tif_frames.shape[0]
        frame_idx = max(0, min(frame_idx, n - 1))

        self._raster_vline.set_xdata([frame_idx, frame_idx])
        self._coact_vline.set_xdata([frame_idx, frame_idx])
        self._tif_im.set_data(self._tif_frames[frame_idx])
        self.frame_label.config(text=f"{frame_idx} / {n - 1}")
        self.canvas_neuropil.draw_idle()

    # ── Neuropil panel message / error helpers ────────────────────────────

    def _neuropil_message(self, msg: str):
        self.fig_neuropil.clear()
        ax = self.fig_neuropil.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.axis("off")
        self.canvas_neuropil.draw()

    def _neuropil_error(self, msg: str):
        self.fig_neuropil.clear()
        ax = self.fig_neuropil.add_subplot(111)
        ax.text(
            0.5, 0.5, f"ERROR\n\n{msg}",
            ha="center", va="center",
            fontsize=10, color="red",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fff0f0",
                      edgecolor="#cc0000",
                      linewidth=1.5),
        )
        ax.axis("off")
        self.canvas_neuropil.draw()


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

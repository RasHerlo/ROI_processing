#!/usr/bin/env python3
"""
movement_proxy.py

Interactive GUI for inspecting suite2p ChanA data availability
for Astrocyte rows flagged as bPAC positive (bPAC pos = 1 or 2).

Saves/loads a companion Mov_Proxy.pkl that stores, per row:
  fneu_thr  – threshold applied to baseline-corrected co-activity
  fneu_bin  – binary vector (int8), 1 where corrected >= threshold

Layout:
  Left  : filtered dataframe table (sortable)
  Right : ┌── "Astrocytes" ──────────┬── "Neuropil Inspection" ─────────────────────┐
          │  bPAC_dFF_eval_ChanB.png │  [Raster (Fneu)]          │ [TIF viewer]     │
          │  (static PNG)            │  [Co-Activity + baseline] │                  │
          │                          │  [Corrected + threshold]  │                  │
          │                          │  [════════ frame slider ════════════════════] │
          └──────────────────────────┴──────────────────────────────────────────────┘
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

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELLTYPE_FILTER  = "Astrocytes"
BPAC_POS_VALUES  = [1, 2]
CHECKBOX_SYMBOLS = {True: "☑", False: "☐"}
DISPLAY_COLUMNS  = [
    "AC dir", "ex-type", "bPAC pos",
    "s2p ChanA", "Fneu", "Fneu Thr", "Fneu Bin", "Fneu Corr",
]
SLIDER_DEBOUNCE_MS = 200
PROXY_FILENAME     = "Mov_Proxy"

# Stimulation (LED/bPAC) artifact windows per supported trace length.
# Each entry is (stim_start, stim_end_exclusive), matching pickle_file_expansion.py.
# The NaN-masked frames are stim_start … stim_end_exclusive - 1 (7 frames each).
STIM_RANGES: dict[int, tuple[int, int]] = {
    1520: (726,  733),   # frames 726–732
    2890: (1381, 1388),  # frames 1381–1387
}

# AP trigger timepoints (frame indices) for LED+APs experiments only.
# Two datasets are kept here; flip AP_ACTIVE_SET to switch between them.
AP_TIMEPOINTS: dict[str, list[int]] = {
    "APs_calculated": [757,  913, 1070, 1226, 1383, 1539, 1696, 1852],
    "APs_inspected":  [566,  726,  884, 1046, 1389, 1542, 1704, 1870],
}
AP_ACTIVE_SET = "APs_inspected"


# ---------------------------------------------------------------------------
# Pure path-helper functions
# ---------------------------------------------------------------------------

def derive_chanA_path(dir_path: str) -> str:
    """…/SUPPORT_ChanB  →  …/SUPPORT_ChanA/derippled/suite2p/plane0"""
    clean  = dir_path.rstrip("/\\")
    parent = os.path.dirname(clean)
    return os.path.join(parent, "SUPPORT_ChanA", "derippled", "suite2p", "plane0")


def derive_tif_path(chanA_path: str) -> str:
    """Two levels up from plane0: …/SUPPORT_ChanA/derippled/derippled_stack.tif"""
    derippled_dir = os.path.dirname(os.path.dirname(chanA_path))
    return os.path.join(derippled_dir, "derippled_stack.tif")


def derive_bpac_png_path(dir_path: str) -> str:
    return os.path.join(dir_path, "derippled", "bPAC_dFF_evaluation_ChanB.png")


def check_fneu(chanA_path: str):
    """True = present, False = folder ok but file missing, None = no folder."""
    if not os.path.isdir(chanA_path):
        return None
    return os.path.isfile(os.path.join(chanA_path, "Fneu.npy"))


# ---------------------------------------------------------------------------
# Stimulation timing helpers
# ---------------------------------------------------------------------------

def get_stim_range(n_frames: int) -> tuple[int, int] | None:
    """Return (stim_start, stim_end_exclusive) for known trace lengths, else None."""
    return STIM_RANGES.get(n_frames)


# ---------------------------------------------------------------------------
# Signal-processing helpers (module-level, no GUI dependency)
# ---------------------------------------------------------------------------

def fit_exponential_baseline(trace: np.ndarray,
                              window: int = 150,
                              percentile: float = 15.0) -> np.ndarray:
    """
    Estimate the slowly-declining baseline by:
      1. Rolling-window percentile (suppresses positive spikes).
      2. Exponential fit  f(t) = a·exp(−b·t) + c  to those anchor points.
    Falls back to linear interpolation if the fit fails.
    """
    n      = len(trace)
    t_full = np.arange(n, dtype=np.float64)
    step   = max(1, window // 2)

    t_anc, y_anc = [], []
    for start in range(0, n - window + 1, step):
        t_anc.append(start + window // 2)
        y_anc.append(float(np.percentile(trace[start: start + window], percentile)))

    if not t_anc or t_anc[0] > 0:
        t_anc.insert(0, 0)
        y_anc.insert(0, float(np.percentile(trace[:window], percentile)))
    if t_anc[-1] < n - 1:
        t_anc.append(n - 1)
        y_anc.append(float(np.percentile(trace[max(0, n - window):], percentile)))

    t_anc = np.array(t_anc, dtype=np.float64)
    y_anc = np.array(y_anc, dtype=np.float64)

    if HAS_SCIPY:
        def _exp(t, a, b, c):
            return a * np.exp(-b * t) + c
        try:
            popt, _ = curve_fit(_exp, t_anc, y_anc,
                                p0=[y_anc[0] - y_anc[-1], 1.0 / max(n, 1), y_anc[-1]],
                                maxfev=10_000)
            return _exp(t_full, *popt).astype(np.float32)
        except Exception:
            pass

    return np.interp(t_full, t_anc, y_anc).astype(np.float32)


def compute_proxy_for_row(chanA_path: str,
                           thr: float = 0.0) -> tuple:
    """
    Load Fneu.npy, normalise, compute baseline-corrected co-activity,
    apply threshold.  Returns (fneu_thr, fneu_bin, fneu_corr) or (None, None, None).
    fneu_bin  – int8 array: 1 where corrected >= thr, else 0.
    fneu_corr – float32 array of the baseline-corrected co-activity trace.
    """
    fneu_path = os.path.join(chanA_path, "Fneu.npy")
    if not os.path.isfile(fneu_path):
        return None, None, None
    try:
        fneu = np.load(fneu_path).astype(np.float32)
        if fneu.ndim != 2:
            return None, None, None

        row_min   = fneu.min(axis=1, keepdims=True)
        row_max   = fneu.max(axis=1, keepdims=True)
        denom     = np.where(row_max > row_min, row_max - row_min, 1.0)
        fneu_norm = np.clip((fneu - row_min) / denom, 0.0, 1.0)

        coactivity = fneu_norm.mean(axis=0)
        win        = int(np.clip(len(coactivity) * 0.05, 50, 300))
        baseline   = fit_exponential_baseline(coactivity, window=win, percentile=15.0)
        corrected  = coactivity - baseline

        fneu_bin = (corrected >= thr).astype(np.int8)
        return float(thr), fneu_bin, corrected
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Display-string helpers
# ---------------------------------------------------------------------------

def _fmt_thr(thr) -> str:
    return f"{thr:.4f}" if thr is not None else "–"


def _fmt_bin(binary_vec) -> str:
    if binary_vec is None:
        return "–"
    return f"{100.0 * binary_vec.mean():.1f} %"


def _fmt_corr(corr) -> str:
    if corr is None:
        return "–"
    return f"{len(corr)} pts"


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class MovementProxyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Movement Proxy")
        self.root.geometry("1800x900")

        # ── Dataframe state ───────────────────────────────────────────────
        self.source_df:  pd.DataFrame | None = None
        self.display_df: pd.DataFrame | None = None
        self.file_path:  str | None = None
        self.sort_state: dict = {}

        # ── Proxy (Mov_Proxy.pkl) state ───────────────────────────────────
        self._proxy_df:   pd.DataFrame | None = None
        self._proxy_path: str | None = None

        # ── Per-row neuropil state (current selection) ────────────────────
        self._fneu_norm:      np.ndarray | None = None  # (n_rois, n_frames) float32
        self._coactivity:     np.ndarray | None = None  # (n_frames,) float32
        self._coact_baseline: np.ndarray | None = None  # (n_frames,) float32
        self._corrected_data: np.ndarray | None = None  # (n_frames,) float32
        self._n_frames_fneu:  int | None = None
        self._tif_frames:     np.ndarray | None = None  # (n_frames, h, w) uint8
        self._current_row_dir: str | None = None
        self._current_ex_type: str = ""
        self._current_thr:    float = 0.0

        # ── Persistent y-limit cache (survives row changes) ───────────────
        # None = auto-scale; set to a tuple to lock limits across rows.
        self._ylim_coact:     tuple | None = None
        self._ylim_corrected: tuple | None = None

        # ── AP-trigger set selection (survives row changes) ───────────────
        self._ap_active_set = tk.StringVar(master=self.root, value=AP_ACTIVE_SET)

        # ── Threading / debounce ──────────────────────────────────────────
        self._tif_load_cancel = threading.Event()
        self._debounce_id:     str | None = None

        # ── Matplotlib artist refs ────────────────────────────────────────
        self._raster_vline    = None
        self._coact_vline     = None
        self._corrected_vline = None
        self._threshold_hline = None   # horizontal dashed purple line
        self._tif_im          = None
        self._loading_text    = None
        self._ax_coact        = None
        self._ax_corrected    = None

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
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)

        self.viz_outer = ttk.Frame(self.main_paned)
        self.main_paned.add(self.viz_outer, weight=3)

        self._setup_table_panel()

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
        self.fig_astro    = Figure(figsize=(5, 6), dpi=100)
        self.canvas_astro = FigureCanvasTkAgg(self.fig_astro, self.astro_frame)
        self.canvas_astro.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._astro_message("Select a row to view the bPAC figure.")

    def _setup_neuropil_panel(self):
        ttk.Label(self.neuropil_frame, text="Neuropil Inspection",
                  font=("Arial", 11, "bold")).pack(pady=(0, 4))

        self.fig_neuropil    = Figure(figsize=(10, 6), dpi=100)
        self.canvas_neuropil = FigureCanvasTkAgg(self.fig_neuropil, self.neuropil_frame)
        self.canvas_neuropil.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_neuropil.mpl_connect(
            "button_press_event", self._on_neuropil_click)

        ap_frame = ttk.Frame(self.neuropil_frame)
        ap_frame.pack(fill=tk.X, padx=8, pady=(2, 0))
        ttk.Label(ap_frame, text="AP triggers:", font=("Arial", 8)).pack(
            side=tk.LEFT, padx=(0, 6))
        for key in AP_TIMEPOINTS:
            ttk.Radiobutton(
                ap_frame,
                text=key.replace("APs_", "").capitalize(),
                variable=self._ap_active_set,
                value=key,
                command=self._on_ap_toggle,
            ).pack(side=tk.LEFT, padx=4)

        slider_frame = ttk.Frame(self.neuropil_frame)
        slider_frame.pack(fill=tk.X, padx=8, pady=(2, 4))
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT, padx=(0, 4))
        self.frame_label = ttk.Label(slider_frame, text="– / –", width=12)
        self.frame_label.pack(side=tk.RIGHT)
        self.slider = tk.Scale(
            slider_frame, orient=tk.HORIZONTAL,
            from_=0, to=100, resolution=1, showvalue=False,
            state="disabled", command=self._on_slider_change,
        )
        self.slider.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self._neuropil_message("Select a row to begin.")

    # ── File loading & proxy workflow ─────────────────────────────────────

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

            mask     = ((raw["Celltype"] == CELLTYPE_FILTER) &
                        (raw["bPAC pos"].isin(BPAC_POS_VALUES)))
            filtered = raw[mask].copy().reset_index(drop=True)

            if filtered.empty:
                messagebox.showwarning(
                    "No Data",
                    "No Astrocyte rows with bPAC pos = 1 or 2 found.")
                self.root.quit()
                return

            self._setup_proxy_workflow(filtered)

        except Exception as exc:
            messagebox.showerror("Error Loading File",
                                 f"Failed to load file:\n{exc}")
            self.root.quit()

    # ── Proxy setup ───────────────────────────────────────────────────────

    def _setup_proxy_workflow(self, filtered: pd.DataFrame):
        """Ask user whether to create a new proxy file or load an existing one."""
        choice = self._ask_proxy_choice()
        if choice == "new":
            self._init_proxy_new(filtered)
        elif choice == "load":
            self._init_proxy_load(filtered)
        else:
            self.root.quit()

    def _ask_proxy_choice(self) -> str:
        """Modal dialog — returns 'new', 'load', or '' (cancelled)."""
        result = tk.StringVar(value="")
        dialog = tk.Toplevel(self.root)
        dialog.title("Movement Proxy – Setup")
        dialog.geometry("400x160")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog,
                  text="How would you like to proceed?",
                  font=("Arial", 10, "bold")).pack(pady=(16, 6))
        ttk.Label(dialog,
                  text="Create a new Mov_Proxy file from the loaded dataset,\n"
                       "or continue editing an existing one.",
                  font=("Arial", 9), justify="center").pack()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=12)

        def _pick(v):
            result.set(v)
            dialog.destroy()

        ttk.Button(btn_frame, text="Create new",
                   command=lambda: _pick("new"),  width=18).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Load existing",
                   command=lambda: _pick("load"), width=18).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel",
                   command=lambda: _pick(""),     width=10).pack(side=tk.LEFT, padx=6)

        dialog.wait_window()
        return result.get()

    def _init_proxy_new(self, filtered: pd.DataFrame):
        proxy_path = self._resolve_proxy_path(os.path.dirname(self.file_path))
        if proxy_path is None:
            self.root.quit()
            return

        self._proxy_path = proxy_path
        self._proxy_df   = self._precompute_all_proxy(filtered)
        self._save_proxy()
        self._finalize_display(filtered)

    def _init_proxy_load(self, filtered: pd.DataFrame):
        file_path = filedialog.askopenfilename(
            title="Select Mov_Proxy.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.file_path) if self.file_path else self._initial_dir(),
        )
        if not file_path:
            self.root.quit()
            return
        try:
            self._proxy_df   = pd.read_pickle(file_path)
            self._proxy_path = file_path
        except Exception as e:
            messagebox.showerror("Error Loading Proxy",
                                 f"Failed to load Mov_Proxy:\n{e}")
            self.root.quit()
            return

        # Migrate older proxy files that pre-date the fneu_corr column.
        # The column is initialised to None; values are filled as rows are visited.
        if "fneu_corr" not in self._proxy_df.columns:
            self._proxy_df["fneu_corr"] = None

        self._ensure_proxy_coverage(filtered)
        self._finalize_display(filtered)

    def _resolve_proxy_path(self, save_dir: str) -> str | None:
        """
        Return a save path for Mov_Proxy.pkl.
        If Mov_Proxy.pkl already exists, warn the user and offer a new name.
        Returns None if the user cancels.
        """
        default = os.path.join(save_dir, PROXY_FILENAME + ".pkl")
        if not os.path.exists(default):
            return default

        # Suggest an auto-incremented name
        n = 2
        while os.path.exists(os.path.join(save_dir, f"{PROXY_FILENAME}_{n}.pkl")):
            n += 1
        suggested = f"{PROXY_FILENAME}_{n}"

        name_var = tk.StringVar(value=suggested)
        result   = tk.StringVar(value="")

        dialog = tk.Toplevel(self.root)
        dialog.title("File Already Exists")
        dialog.geometry("440x175")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog,
                  text=f"Mov_Proxy.pkl already exists in:\n{save_dir}\n\n"
                       "Enter a name for the new file (without .pkl):",
                  font=("Arial", 9), justify="center").pack(pady=(14, 4))

        entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        entry.pack(pady=4)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        def on_ok():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Invalid Name",
                                     "Please enter a name.", parent=dialog)
                return
            result.set(name)
            dialog.destroy()

        ttk.Button(btn_frame, text="Save",   command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        entry.focus()
        entry.bind("<Return>", lambda _: on_ok())
        dialog.wait_window()

        name = result.get()
        if not name:
            return None
        fname = name if name.endswith(".pkl") else name + ".pkl"
        return os.path.join(save_dir, fname)

    def _precompute_all_proxy(self, filtered: pd.DataFrame) -> pd.DataFrame:
        """Silently compute proxy data for every row (no TIF loading)."""
        n    = len(filtered)
        rows = []

        prog_win = tk.Toplevel(self.root)
        prog_win.title("Preparing Movement Proxy data…")
        prog_win.geometry("360x105")
        prog_win.resizable(False, False)
        prog_win.transient(self.root)
        prog_win.grab_set()

        lbl  = ttk.Label(prog_win, text="Computing baseline corrections…")
        lbl.pack(pady=(14, 4))
        prog = ttk.Progressbar(prog_win, length=320, maximum=n, mode="determinate")
        prog.pack(pady=4)

        for i, (_, row) in enumerate(filtered.iterrows()):
            lbl.config(text=f"Row {i + 1} / {n}  ({str(row['dir'])[-40:]})")
            prog["value"] = i
            prog_win.update()

            dir_path = str(row["dir"])
            chanA    = derive_chanA_path(dir_path)
            fneu_thr, fneu_bin, fneu_corr = compute_proxy_for_row(chanA, thr=0.0)
            rows.append({"dir": dir_path,
                         "fneu_thr": fneu_thr,
                         "fneu_bin": fneu_bin,
                         "fneu_corr": fneu_corr})

        prog_win.destroy()
        return pd.DataFrame(rows)

    def _ensure_proxy_coverage(self, filtered: pd.DataFrame):
        """Add proxy entries for any rows missing from a loaded proxy_df."""
        existing = set(self._proxy_df["dir"].tolist())
        new_rows = []
        for _, row in filtered.iterrows():
            dir_path = str(row["dir"])
            if dir_path not in existing:
                chanA = derive_chanA_path(dir_path)
                fneu_thr, fneu_bin, fneu_corr = compute_proxy_for_row(chanA, thr=0.0)
                new_rows.append({"dir": dir_path,
                                 "fneu_thr": fneu_thr,
                                 "fneu_bin": fneu_bin,
                                 "fneu_corr": fneu_corr})
        if new_rows:
            self._proxy_df = pd.concat(
                [self._proxy_df, pd.DataFrame(new_rows)],
                ignore_index=True)

    def _proxy_for_dir(self, dir_path: str) -> tuple:
        """Return (fneu_thr, fneu_bin, fneu_corr) from proxy_df, or (None, None, None).
        fneu_corr may be None for proxy files created before this column was added."""
        if self._proxy_df is None:
            return None, None, None
        mask = self._proxy_df["dir"] == dir_path
        if not mask.any():
            return None, None, None
        row = self._proxy_df[mask].iloc[0]
        fneu_corr = row["fneu_corr"] if "fneu_corr" in self._proxy_df.columns else None
        return row["fneu_thr"], row["fneu_bin"], fneu_corr

    def _save_proxy(self):
        if self._proxy_df is not None and self._proxy_path:
            try:
                self._proxy_df.to_pickle(self._proxy_path)
            except Exception as e:
                print(f"Warning: could not save proxy file: {e}")

    # ── Display init ──────────────────────────────────────────────────────

    def _finalize_display(self, filtered: pd.DataFrame):
        """Build display dataframe, configure treeview, select first row."""
        has_corr = (self._proxy_df is not None and
                    "fneu_corr" in self._proxy_df.columns)
        proxy_lookup = {
            row["dir"]: (row["fneu_thr"], row["fneu_bin"],
                         row["fneu_corr"] if has_corr else None)
            for _, row in self._proxy_df.iterrows()
        } if self._proxy_df is not None else {}

        self.source_df  = self._build_display_df(filtered, proxy_lookup)
        self.display_df = self.source_df.copy()
        self._configure_treeview()
        self._populate_treeview()

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._on_row_select(None)

    def _build_display_df(self, filtered: pd.DataFrame,
                          proxy_lookup: dict) -> pd.DataFrame:
        rows = []
        for _, row in filtered.iterrows():
            dir_path  = str(row["dir"])
            chanA     = derive_chanA_path(dir_path)
            folder_ok = os.path.isdir(chanA)
            fneu_st   = check_fneu(chanA)

            fneu_thr, fneu_bin, fneu_corr = proxy_lookup.get(
                dir_path, (None, None, None))

            rows.append({
                "AC dir":    dir_path,
                "ex-type":   str(row.get("ex-type", "")),
                "bPAC pos":  str(int(row["bPAC pos"])),
                "s2p ChanA": chanA if folder_ok else "suite2p folder does not exist yet",
                "Fneu":      (CHECKBOX_SYMBOLS[True]  if fneu_st is True  else
                              CHECKBOX_SYMBOLS[False] if fneu_st is False else "—"),
                "Fneu Thr":  _fmt_thr(fneu_thr),
                "Fneu Bin":  _fmt_bin(fneu_bin),
                "Fneu Corr": _fmt_corr(fneu_corr),
                "_dir_raw":   dir_path,
                "_chanA_raw": chanA,
                "_fneu_raw":  fneu_st,
            })
        return pd.DataFrame(rows)

    # ── Treeview ──────────────────────────────────────────────────────────

    def _configure_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = DISPLAY_COLUMNS
        self.tree["show"]    = "headings"

        col_widths = {
            "AC dir": 255, "ex-type": 88, "bPAC pos": 58,
            "s2p ChanA": 255, "Fneu": 44,
            "Fneu Thr": 72, "Fneu Bin": 72, "Fneu Corr": 72,
        }
        for col in DISPLAY_COLUMNS:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=col_widths.get(col, 80),
                             minwidth=40, anchor="center")
            self.sort_state[col] = None

    def _populate_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, row in self.display_df.iterrows():
            self.tree.insert("", "end", iid=str(idx),
                             values=[row[c] for c in DISPLAY_COLUMNS],
                             tags=(str(idx),))

    def _update_treeview_row(self, dir_path: str):
        """Refresh Fneu Thr, Fneu Bin and Fneu Corr cells for the row matching dir_path."""
        fneu_thr, fneu_bin, fneu_corr = self._proxy_for_dir(dir_path)
        thr_str  = _fmt_thr(fneu_thr)
        bin_str  = _fmt_bin(fneu_bin)
        corr_str = _fmt_corr(fneu_corr)

        df_mask = self.display_df["_dir_raw"] == dir_path
        if not df_mask.any():
            return
        idx = self.display_df[df_mask].index[0]

        self.display_df.loc[df_mask, "Fneu Thr"]  = thr_str
        self.display_df.loc[df_mask, "Fneu Bin"]  = bin_str
        self.display_df.loc[df_mask, "Fneu Corr"] = corr_str
        src_mask = self.source_df["_dir_raw"] == dir_path
        self.source_df.loc[src_mask, "Fneu Thr"]  = thr_str
        self.source_df.loc[src_mask, "Fneu Bin"]  = bin_str
        self.source_df.loc[src_mask, "Fneu Corr"] = corr_str

        iid = str(idx)
        if self.tree.exists(iid):
            self.tree.item(iid, values=[self.display_df.loc[idx, c]
                                        for c in DISPLAY_COLUMNS])

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
            self.tree.heading(c, text=f"{c} {'▲' if ascending else '▼'}"
                              if c == col else c)
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

        idx       = int(selection[0])
        row       = self.display_df.iloc[idx]
        dir_path  = row["_dir_raw"]
        chanA     = row["_chanA_raw"]

        self._cancel_tif_load()

        # Invalidate all per-row state
        self._raster_vline    = None
        self._coact_vline     = None
        self._corrected_vline = None
        self._threshold_hline = None
        self._tif_im          = None
        self._loading_text    = None
        self._ax_coact        = None
        self._ax_corrected    = None
        self._tif_frames      = None
        self._coact_baseline  = None
        self._corrected_data  = None
        self._current_row_dir  = dir_path
        self._current_ex_type  = str(row.get("ex-type", ""))

        self.slider.config(state="disabled")
        self.frame_label.config(text="– / –")

        # Load saved threshold for this row
        fneu_thr, _, _ = self._proxy_for_dir(dir_path)
        self._current_thr = float(fneu_thr) if fneu_thr is not None else 0.0

        # 1. Astrocytes panel (synchronous)
        self._load_astro_png(dir_path)

        # 2. Fneu (synchronous numpy)
        if not self._load_fneu(chanA):
            return

        # 3. TIF (background thread)
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
            fneu = np.load(fneu_path).astype(np.float32)
            if fneu.ndim != 2:
                self._neuropil_error(
                    f"Fneu.npy unexpected shape: {fneu.shape}\n"
                    "Expected (n_rois × n_frames).")
                return False

            row_min   = fneu.min(axis=1, keepdims=True)
            row_max   = fneu.max(axis=1, keepdims=True)
            denom     = np.where(row_max > row_min, row_max - row_min, 1.0)
            fneu_norm = np.clip((fneu - row_min) / denom, 0.0, 1.0)

            self._fneu_norm     = fneu_norm
            self._coactivity    = fneu_norm.mean(axis=0)
            self._n_frames_fneu = fneu_norm.shape[1]

            win = int(np.clip(self._n_frames_fneu * 0.05, 50, 300))
            self._coact_baseline = fit_exponential_baseline(
                self._coactivity, window=win, percentile=15.0)
            self._corrected_data = (self._coactivity - self._coact_baseline)

            # Back-fill fneu_corr for proxy files that pre-date this column.
            if (self._proxy_df is not None and
                    self._current_row_dir is not None and
                    "fneu_corr" in self._proxy_df.columns):
                mask = self._proxy_df["dir"] == self._current_row_dir
                if mask.any():
                    idx = self._proxy_df[mask].index[0]
                    if self._proxy_df.at[idx, "fneu_corr"] is None:
                        self._proxy_df.at[idx, "fneu_corr"] = self._corrected_data
                        self._save_proxy()
                        self._update_treeview_row(self._current_row_dir)

            return True
        except Exception as e:
            self._neuropil_error(f"Error loading Fneu.npy:\n{e}")
            return False

    # ── TIF background loading ────────────────────────────────────────────

    def _cancel_tif_load(self):
        self._tif_load_cancel.set()
        self._tif_load_cancel = threading.Event()

    def _start_tif_load(self, tif_path: str):
        self._init_neuropil_raster_only()
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
                        "tifffile not installed.\nRun: pip install tifffile"))
                return

            with tifffile.TiffFile(tif_path) as tf:
                n_frames = len(tf.pages)
                if n_frames == 0 or cancel.is_set():
                    return

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

                first  = tf.pages[0].asarray()
                h, w   = first.shape[:2]
                frames = np.empty((n_frames, h, w), dtype=np.uint8)

                for i, page in enumerate(tf.pages):
                    if cancel.is_set():
                        return
                    frame = page.asarray().astype(np.float32)
                    frame = np.clip((frame - vmin) / scale, 0.0, 1.0)
                    frames[i] = (frame * 255).astype(np.uint8)
                    if i % 100 == 0 and not cancel.is_set():
                        fi, n = i, n_frames
                        self.root.after(0, lambda fi=fi, n=n:
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
        if self._loading_text is not None:
            pct = int(frame_i / n_frames * 100)
            self._loading_text.set_text(
                f"Loading TIF…\n{frame_i} / {n_frames} frames\n({pct} %)")
            self.canvas_neuropil.draw_idle()

    def _on_tif_loaded(self, frames: np.ndarray):
        n_tif  = frames.shape[0]
        n_fneu = self._n_frames_fneu
        if n_fneu is not None and n_tif != n_fneu:
            self._neuropil_error(
                f"Frame count mismatch:\n"
                f"  Fneu.npy  →  {n_fneu} frames\n"
                f"  TIF stack →  {n_tif} frames")
            return
        self._tif_frames = frames
        self.slider.config(state="normal", from_=0, to=n_tif - 1)
        self.slider.set(0)
        self.frame_label.config(text=f"0 / {n_tif - 1}")
        self._init_neuropil_full(initial_frame=0)

    # ── Neuropil figure rendering ─────────────────────────────────────────

    def _make_neuropil_gridspec(self):
        return GridSpec(3, 2,
                        height_ratios=[3, 1, 1],
                        width_ratios=[1, 1],
                        hspace=0.45, wspace=0.30,
                        figure=self.fig_neuropil)

    def _init_neuropil_raster_only(self):
        if self._fneu_norm is None:
            return
        n_frames = self._n_frames_fneu
        self.fig_neuropil.clear()
        gs = self._make_neuropil_gridspec()

        ax_raster    = self.fig_neuropil.add_subplot(gs[0, 0])
        ax_coact     = self.fig_neuropil.add_subplot(gs[1, 0])
        ax_corrected = self.fig_neuropil.add_subplot(gs[2, 0])
        ax_right     = self.fig_neuropil.add_subplot(gs[:, 1])

        self._draw_raster(ax_raster, initial_frame=0)
        self._draw_coactivity(ax_coact, n_frames, initial_frame=0,
                              ylim=self._ylim_coact)
        self._draw_corrected(ax_corrected, n_frames,
                             initial_frame=0, threshold=self._current_thr,
                             ylim=self._ylim_corrected)

        ax_right.axis("off")
        self._loading_text = ax_right.text(
            0.5, 0.5, "Loading TIF…\n",
            ha="center", va="center",
            fontsize=11, transform=ax_right.transAxes, color="gray")

        self.fig_neuropil.tight_layout()
        self.canvas_neuropil.draw()
        if self._ax_coact is not None:
            self._ylim_coact = self._ax_coact.get_ylim()
        if self._ax_corrected is not None:
            self._ylim_corrected = self._ax_corrected.get_ylim()

    def _init_neuropil_full(self, initial_frame: int = 0):
        if self._fneu_norm is None or self._tif_frames is None:
            return
        n_frames = self._tif_frames.shape[0]
        self.fig_neuropil.clear()
        gs = self._make_neuropil_gridspec()

        ax_raster    = self.fig_neuropil.add_subplot(gs[0, 0])
        ax_coact     = self.fig_neuropil.add_subplot(gs[1, 0])
        ax_corrected = self.fig_neuropil.add_subplot(gs[2, 0])
        ax_tif       = self.fig_neuropil.add_subplot(gs[:, 1])

        self._draw_raster(ax_raster, initial_frame)
        self._draw_coactivity(ax_coact, n_frames, initial_frame,
                              ylim=self._ylim_coact)
        self._draw_corrected(ax_corrected, n_frames,
                             initial_frame, threshold=self._current_thr,
                             ylim=self._ylim_corrected)

        self._tif_im = ax_tif.imshow(
            self._tif_frames[initial_frame],
            cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax_tif.axis("off")
        ax_tif.set_title("ChanA Stack", fontsize=9)

        self._loading_text = None
        self.fig_neuropil.tight_layout()
        self.canvas_neuropil.draw()
        if self._ax_coact is not None:
            self._ylim_coact = self._ax_coact.get_ylim()
        if self._ax_corrected is not None:
            self._ylim_corrected = self._ax_corrected.get_ylim()

    def _draw_raster(self, ax, initial_frame: int):
        ax.imshow(self._fneu_norm, aspect="auto", cmap="hot",
                  vmin=0.0, vmax=1.0, interpolation="nearest", origin="upper")
        ax.set_ylabel("ROI index", fontsize=8)
        ax.set_xlabel("Frame",     fontsize=8)
        ax.set_title("Neuropil Raster", fontsize=9)
        ax.tick_params(labelsize=7)
        self._raster_vline = ax.axvline(x=initial_frame, color="red",
                                        linewidth=1.2, zorder=5)

    def _draw_coactivity(self, ax, n_frames: int, initial_frame: int,
                         ylim: tuple | None = None):
        ax.plot(self._coactivity, color="steelblue", linewidth=0.8,
                label="Co-activity", zorder=3)
        if self._coact_baseline is not None:
            ax.plot(self._coact_baseline, color="tomato", linewidth=1.2,
                    linestyle="--", alpha=0.85, label="Exp. baseline", zorder=4)
            ax.legend(fontsize=6, loc="upper right",
                      framealpha=0.6, handlelength=1.5)
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylabel("Mean",  fontsize=7)
        ax.set_xlabel("Frame", fontsize=7)
        ax.set_title("Co-Activity  (right-click → y-limits)", fontsize=9)
        ax.tick_params(labelsize=6)
        if ylim is not None:
            ax.set_ylim(ylim)
        stim = get_stim_range(n_frames)
        if stim is not None:
            ax.axvspan(stim[0], stim[1], color="cyan", alpha=0.25, zorder=2)

        self._coact_vline = ax.axvline(x=initial_frame, color="red",
                                       linewidth=1.2, zorder=5)
        self._ax_coact = ax

    def _draw_corrected(self, ax, n_frames: int, initial_frame: int,
                        threshold: float = 0.0, ylim: tuple | None = None):
        if self._corrected_data is not None:
            corrected = self._corrected_data
        else:
            corrected = (self._coactivity - self._coactivity.mean()
                         if self._coactivity is not None
                         else np.zeros(n_frames))

        ax.plot(corrected, color="darkorange", linewidth=0.8, zorder=3)
        ax.axhline(y=0.0, color="gray", linewidth=0.6,
                   linestyle="--", alpha=0.6, zorder=2)
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylabel("Δ Mean", fontsize=7)
        ax.set_xlabel("Frame",  fontsize=7)
        ax.set_title("Baseline-Corrected  (right-click → options)", fontsize=9)
        ax.tick_params(labelsize=6)
        if ylim is not None:
            ax.set_ylim(ylim)

        stim = get_stim_range(n_frames)
        if stim is not None:
            ax.axvspan(stim[0], stim[1], color="cyan", alpha=0.25, zorder=2)

        if self._current_ex_type == "LED+APs":
            active_set = self._ap_active_set.get()
            ap_frames  = AP_TIMEPOINTS[active_set]
            ap_label   = f"APs ({active_set.replace('APs_', '')})"
            for i, frame in enumerate(ap_frames):
                if 0 <= frame < n_frames:
                    ax.axvline(x=frame, color="black", linewidth=2.0,
                               alpha=0.85, zorder=4,
                               label=ap_label if i == 0 else "_nolegend_")

        self._threshold_hline = ax.axhline(
            y=threshold, color="purple", linewidth=1.2,
            linestyle=":", alpha=0.9, zorder=4,
            label=f"Thr: {threshold:.4f}")
        ax.legend(fontsize=6, loc="upper right",
                  framealpha=0.6, handlelength=1.5)

        self._corrected_vline = ax.axvline(x=initial_frame, color="red",
                                           linewidth=1.2, zorder=5)
        self._ax_corrected = ax

    # ── AP toggle ─────────────────────────────────────────────────────────

    def _on_ap_toggle(self):
        """Called when the user switches AP-trigger set; redraws corrected plot."""
        self._redraw_corrected()

    def _redraw_corrected(self):
        """Clear and redraw the corrected co-activity axes in-place."""
        if self._ax_corrected is None or self._corrected_data is None:
            return
        n_frames = self._n_frames_fneu or 0
        current_frame = (int(float(self.slider.get()))
                         if self._tif_frames is not None else 0)
        self._ax_corrected.cla()
        self._draw_corrected(self._ax_corrected, n_frames, current_frame,
                             threshold=self._current_thr,
                             ylim=self._ylim_corrected)
        self.canvas_neuropil.draw_idle()

    # ── Threshold application ─────────────────────────────────────────────

    def _apply_threshold(self, new_thr: float):
        """Update proxy_df, treeview, threshold line. Auto-saves proxy."""
        if self._corrected_data is None or self._current_row_dir is None:
            return

        fneu_bin = (self._corrected_data >= new_thr).astype(np.int8)
        self._current_thr = new_thr

        # Update proxy_df
        if self._proxy_df is not None:
            mask = self._proxy_df["dir"] == self._current_row_dir
            if mask.any():
                self._proxy_df.loc[mask, "fneu_thr"] = float(new_thr)
                self._proxy_df.loc[mask, "fneu_bin"]  = \
                    self._proxy_df.loc[mask, "fneu_bin"].apply(lambda _: fneu_bin)
            else:
                new_row = pd.DataFrame([{"dir": self._current_row_dir,
                                          "fneu_thr": float(new_thr),
                                          "fneu_bin": fneu_bin,
                                          "fneu_corr": self._corrected_data}])
                self._proxy_df = pd.concat([self._proxy_df, new_row],
                                           ignore_index=True)

        self._save_proxy()
        self._update_treeview_row(self._current_row_dir)

        # Update threshold line in-place (no full redraw)
        if self._threshold_hline is not None:
            self._threshold_hline.set_ydata([new_thr, new_thr])
            self._threshold_hline.set_label(f"Thr: {new_thr:.4f}")
            if self._ax_corrected is not None:
                self._ax_corrected.legend(fontsize=6, loc="upper right",
                                          framealpha=0.6, handlelength=1.5)
            self.canvas_neuropil.draw_idle()

    # ── Slider ────────────────────────────────────────────────────────────

    def _on_slider_change(self, value: str):
        if self._debounce_id:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(
            SLIDER_DEBOUNCE_MS, self._apply_frame, int(float(value)))

    def _apply_frame(self, frame_idx: int):
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
        if self._corrected_vline is not None:
            self._corrected_vline.set_xdata([frame_idx, frame_idx])
        self._tif_im.set_data(self._tif_frames[frame_idx])
        self.frame_label.config(text=f"{frame_idx} / {n - 1}")
        self.canvas_neuropil.draw_idle()

    # ── Right-click dialogs ───────────────────────────────────────────────

    def _on_neuropil_click(self, event):
        if event.button != 3 or event.inaxes is None:
            return
        if event.inaxes is self._ax_coact:
            self._show_ylim_dialog(self._ax_coact, "Co-Activity Y-limits")
        elif event.inaxes is self._ax_corrected:
            self._show_corrected_options_dialog()

    def _show_ylim_dialog(self, target_ax, title: str = "Y-limits"):
        if target_ax is None:
            return
        current_ymin, current_ymax = target_ax.get_ylim()

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("260x160")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        pad = {"padx": 10, "pady": 4}
        ttk.Label(dialog, text=title, font=("Arial", 9, "bold")).pack(**pad)

        fields = ttk.Frame(dialog)
        fields.pack(**pad)
        ttk.Label(fields, text="Y-min:").grid(row=0, column=0, sticky="e", padx=4)
        entry_ymin = ttk.Entry(fields, width=12)
        entry_ymin.insert(0, f"{current_ymin:.6g}")
        entry_ymin.grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(fields, text="Y-max:").grid(row=1, column=0, sticky="e", padx=4)
        entry_ymax = ttk.Entry(fields, width=12)
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
                                     "Values must be numbers.", parent=dialog)
                return
            if ymin >= ymax:
                messagebox.showerror("Invalid input",
                                     "Y-min must be less than Y-max.",
                                     parent=dialog)
                return
            target_ax.set_ylim(ymin, ymax)
            self._ylim_coact = (ymin, ymax)
            self.canvas_neuropil.draw_idle()
            dialog.destroy()

        def on_reset():
            ydata = [v for line in target_ax.lines for v in line.get_ydata()
                     if not isinstance(v, str)]
            if ydata:
                lo, hi = float(np.nanmin(ydata)), float(np.nanmax(ydata))
                m = (hi - lo) * 0.05
                entry_ymin.delete(0, tk.END); entry_ymin.insert(0, f"{lo - m:.6g}")
                entry_ymax.delete(0, tk.END); entry_ymax.insert(0, f"{hi + m:.6g}")

        ttk.Button(btn_frame, text="Apply",  command=on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset",  command=on_reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        entry_ymin.focus()
        dialog.bind("<Return>", lambda _: on_apply())

    def _show_corrected_options_dialog(self):
        """
        Right-click dialog for the baseline-corrected plot.
        Controls y-limits AND the threshold for Fneu Thr / Fneu Bin.
        """
        if self._ax_corrected is None:
            return

        cur_ymin, cur_ymax = self._ax_corrected.get_ylim()
        saved_thr, _, _ = self._proxy_for_dir(self._current_row_dir or "")
        saved_thr    = float(saved_thr) if saved_thr is not None else 0.0

        dialog = tk.Toplevel(self.root)
        dialog.title("Baseline-Corrected – Options")
        dialog.geometry("280x230")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Baseline-Corrected Options",
                  font=("Arial", 9, "bold")).pack(pady=(12, 4))
        ttk.Separator(dialog, orient="horizontal").pack(fill="x", padx=10)

        fields = ttk.Frame(dialog)
        fields.pack(padx=14, pady=6)

        ttk.Label(fields, text="Y-min:").grid(row=0, column=0, sticky="e", padx=4)
        entry_ymin = ttk.Entry(fields, width=14)
        entry_ymin.insert(0, f"{cur_ymin:.6g}")
        entry_ymin.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(fields, text="Y-max:").grid(row=1, column=0, sticky="e", padx=4)
        entry_ymax = ttk.Entry(fields, width=14)
        entry_ymax.insert(0, f"{cur_ymax:.6g}")
        entry_ymax.grid(row=1, column=1, padx=4, pady=2)

        ttk.Separator(fields, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(fields, text="Threshold:").grid(row=3, column=0, sticky="e", padx=4)
        entry_thr = ttk.Entry(fields, width=14)
        entry_thr.insert(0, f"{self._current_thr:.6g}")
        entry_thr.grid(row=3, column=1, padx=4, pady=2)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=8)

        def on_apply():
            try:
                ymin = float(entry_ymin.get())
                ymax = float(entry_ymax.get())
                thr  = float(entry_thr.get())
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "All values must be numbers.", parent=dialog)
                return
            if ymin >= ymax:
                messagebox.showerror("Invalid input",
                                     "Y-min must be less than Y-max.", parent=dialog)
                return
            self._ax_corrected.set_ylim(ymin, ymax)
            self._ylim_corrected = (ymin, ymax)
            self._apply_threshold(thr)
            dialog.destroy()

        def on_restore():
            entry_thr.delete(0, tk.END)
            entry_thr.insert(0, f"{saved_thr:.6g}")

        ttk.Button(btn_frame, text="Apply",
                   command=on_apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Restore thr",
                   command=on_restore).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=4)

        entry_ymin.focus()
        dialog.bind("<Return>", lambda _: on_apply())

    # ── Neuropil message / error helpers ──────────────────────────────────

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
        ax.text(0.5, 0.5, f"ERROR\n\n{msg}",
                ha="center", va="center", fontsize=10, color="red",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#fff0f0", edgecolor="#cc0000", linewidth=1.5))
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

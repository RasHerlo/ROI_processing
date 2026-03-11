# ROI Processing

A comprehensive toolkit for filtering, post-processing, and evaluating calcium imaging ROI (Region of Interest) data from Suite2p, with a focus on bPAC (blue-light activated adenylyl cyclase) photostimulation experiments.

## Overview

This project provides both GUI-based and command-line tools for processing Suite2p outputs, calculating bPAC response metrics, and performing quality control on experimental data across neurons (ChanA) and astrocytes (ChanB).

## Pipeline Architecture

The processing pipeline follows this workflow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                   │
│  Suite2p outputs: selected_traces.pkl, ops.npy, stat.npy, rastermap_model.npy│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: TRACE PROCESSING                                  │
│  selected_traces.pkl  ──►  processed_traces.pkl                             │
│  (pickle_file_expansion.py / batch_process_traces.py)                       │
│  • Normalizes traces (0-1 range)                                            │
│  • Calculates z-scores                                                      │
│  • Computes bPAC_dFF metrics                                                │
│  • Applies NaN masking to stimulation periods                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  STEP 2: INTERACTIVE INSPECTION (Optional)                   │
│  (main_gui.py - PyQt5 Application)                                          │
│  • Trace Inspection: Visualize traces with ROI overlay                      │
│  • Trace Selection: Manual filtering and selection                          │
│  • DataFrame Inspection: View tabular data                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 3: bPAC EVALUATION                                    │
│  processed_traces.pkl  ──►  bPAC_metrics.pkl + visualization figures        │
│  (bPAC_dFF_eval_quick.py)                                                   │
│  • Statistical analysis (normal distribution fitting, p<0.05 limits)        │
│  • Generates 2x2 visualization: histogram, rasterplot, traces, ROI map      │
│  • Collects all metrics into bPAC_metrics_collect.pkl                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 4: QUALITY CONTROL                                    │
│  bPAC_metrics_collect.pkl  ──►  Annotated metrics                           │
│  (bPAC_metrics_collect_eval.py - Tkinter Application)                       │
│  • Interactive table with sortable columns                                  │
│  • Quality checkboxes: Derippling, Segmentation, HQ Exp, bPAC pos           │
│  • Experiment type classification                                           │
│  • Comments and notes per experiment                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Suite2p output files

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ROI_processing.git
cd ROI_processing

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ROI_processing/
├── src/
│   ├── main_gui.py              # Main PyQt5 GUI application
│   └── modules/
│       ├── base_module.py       # Abstract base class for modules
│       ├── pickle_file_expansion.py  # Core trace processing
│       ├── trace_inspection/    # Trace visualization module
│       ├── trace_selection/     # Manual trace selection module
│       └── dataframe_inspection/ # DataFrame viewer module
├── bPAC_dFF_eval_quick.py       # bPAC evaluation and metrics
├── bPAC_metrics_collect_eval.py # Metrics QC GUI
├── batch_process_traces.py      # Batch processing CLI
├── read_pickle.py               # Pickle file inspector utility
├── analyze_pickle.py            # Metrics analysis utility
├── config.json                  # Configuration (data folder path)
├── requirements.txt             # Python dependencies
└── README.md
```

## Data Organization

The toolkit expects data organized in the following structure:

```
DATA/
├── SUPPORT_ChanA/          # Channel A (Neurons)
│   └── derippled/
│       ├── selected_traces.pkl
│       ├── processed_traces.pkl  (generated)
│       ├── bPAC_metrics.pkl      (generated)
│       └── suite2p/
│           └── plane0/
│               ├── ops.npy
│               ├── stat.npy
│               └── rastermap_model.npy
└── SUPPORT_ChanB/          # Channel B (Astrocytes)
    └── derippled/
        └── (same structure)
```

## Usage

### 1. Configure Data Location

Edit `config.json` to point to your data folder:

```json
{"data_folder": "E:/path/to/your/DATA"}
```

Or use the "Locate DATA Folder" button in the main GUI.

### 2. Process Traces (Single File)

Using the GUI:
```bash
cd src
python main_gui.py
```

The GUI will automatically process `selected_traces.pkl` when needed.

### 3. Batch Process Multiple Files

```bash
# Process all selected_traces.pkl files in a directory tree
python batch_process_traces.py "E:/path/to/root/directory"

# Force reprocessing of existing files
python batch_process_traces.py "E:/path/to/root/directory" --overwrite

# Use directory from config.json
python batch_process_traces.py
```

### 4. Run bPAC Evaluation

```bash
# Generate evaluation figures and metrics
python bPAC_dFF_eval_quick.py "E:/path/to/root/directory"
```

This creates:
- `bPAC_dFF_evaluation_ChanA.png` / `bPAC_dFF_evaluation_ChanB.png` (2x2 visualization)
- `bPAC_metrics.pkl` (per-folder metrics)
- `bPAC_metrics_collect.pkl` (aggregated metrics in root directory)

### 5. Quality Control Evaluation

```bash
# Launch the metrics evaluation GUI
python bPAC_metrics_collect_eval.py
```

Features:
- Click column headers to sort data
- Click checkbox columns to toggle filters (show only checked items)
- Click cells to toggle checkbox states (☐ → ⊡ → ☑)
- Click "ex-type" cells to set experiment type
- Click "comments" cells to add notes
- All changes auto-save to the pickle file

## Key Metrics

| Metric | Description |
|--------|-------------|
| `bPAC_dFF` | Ratio of post-stimulation mean to baseline mean |
| `bPAC_dFF_norm` | bPAC_dFF calculated from normalized traces |
| `bPAC_dFF_zScr` | bPAC_dFF calculated from z-scored traces |
| `Tot_Counts` | Total number of ROIs analyzed |
| `Count High Trcs` | ROIs with bPAC_dFF above p<0.05 upper limit |
| `Count Low Trcs` | ROIs with bPAC_dFF below p<0.05 lower limit |
| `AUC high/low` | Mean post-stim intensity for high/low significance traces |

## Utility Scripts

### Inspect Pickle Files

```bash
# General pickle inspector
python read_pickle.py "path/to/file.pkl"

# Analyze bPAC metrics collection
python analyze_pickle.py
```

## GUI Modules

### Main GUI (`main_gui.py`)
- **Channel Selection**: Toggle between ChanA (Neurons) and ChanB (Astrocytes)
- **Trace Inspect**: Visualize traces with ROI overlay on mean image
- **Trace Selection**: Manual selection/deselection of traces for analysis
- **DataFrame Inspection**: View processed data in table format

### Metrics Evaluator (`bPAC_metrics_collect_eval.py`)
- **Quality Control Checkboxes**:
  - Derippling: Quality of derippling preprocessing
  - Segmentation: Quality of ROI segmentation
  - HQ Exp: High-quality experiment flag
  - bPAC pos: bPAC-positive response confirmed
  - TBD: To be determined
  - To fix: Needs attention/reprocessing
- **Experiment Types**: LED only, LED+APs, AC Endfeet, Other

## Supported Trace Lengths

The toolkit supports two standard trace lengths:
- **1520 points**: Stimulation range 726-733, post-stim range 726-929
- **2890 points**: Stimulation range 1381-1388, post-stim range 1381-1585

## Requirements

Core dependencies:
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- PyQt5 >= 5.15.0
- scipy >= 1.7.0
- Pillow (for image handling)

## Development

See [TASKS.md](TASKS.md) for current development tasks and progress tracking.

## License

See the [LICENSE](LICENSE) file for details.

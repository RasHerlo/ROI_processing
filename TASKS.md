# Project Tasks

This document tracks ongoing tasks, to-do items, and project progress.

## Task Status Key
- 🔴 TODO - Not started
- 🟡 IN PROGRESS - Currently being worked on
- 🟢 COMPLETED - Finished
- ⚪ DEFERRED - Postponed for later

## Project Setup
- 🟢 Initialize project repository
- 🟢 Create README.md with project structure
- 🟢 Set up requirements.txt for dependency management
- 🟢 Create basic directory structure (src, tests, docs, etc.)
- 🟢 Set up virtual environment (venv_ROI_proc)
- 🟢 Install dependencies from requirements.txt
- 🔴 Set up testing framework

## Core Functionality
- 🟢 Create ROI data loading module (`pickle_file_expansion.py`)
- 🟢 Implement basic ROI filtering methods
- 🟢 Develop trace processing utilities (normalization, z-scoring, bPAC_dFF)
- 🟢 Add visualization tools (rasterplots, ROI maps, trace overlays)
- 🟢 Batch processing capabilities (`batch_process_traces.py`)
- 🟢 bPAC evaluation and statistical analysis (`bPAC_dFF_eval_quick.py`)
- 🟢 Metrics collection and aggregation

## GUI Development
- 🟢 Main application window with channel selection
- 🟢 Trace Inspection module with ROI visualization
- 🟢 Trace Selection module for manual filtering
- 🟢 DataFrame Inspection module
- 🟢 bPAC Metrics Evaluator GUI (`bPAC_metrics_collect_eval.py`)
- 🟢 Quality control checkboxes and filters
- 🟢 Comments and experiment type annotations

## Documentation
- 🟢 Write comprehensive README with pipeline overview
- 🟢 Add usage examples and CLI documentation
- 🔴 Write API documentation
- 🔴 Add docstrings to all functions

## Testing
- 🔴 Write unit tests for core functionality
- 🔴 Set up continuous integration

## Future Enhancements
- 🔴 Support for additional trace lengths
- 🔴 Export functionality (CSV, Excel)
- 🔴 Integration with other imaging analysis tools
- 🔴 Automated quality control scoring

---

## Adding New Tasks

When adding new tasks, please use the following format:

```
## [Category]
- [status emoji] [Task description]
```

Example:
```
## Documentation
- 🔴 Update installation instructions
```

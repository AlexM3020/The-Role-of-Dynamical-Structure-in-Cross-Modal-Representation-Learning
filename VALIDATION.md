# Validation record

## Executed locally (2026-11-07)

- Installed the package in an isolated Python 3.12.10 environment and verified that imports resolve to this repository.
- Ran control, metric, workflow, notebook-schema, citation, and checkpoint tests. Final test totals are reported in the completion message.
- Verified actual interruption/resume behavior against an uninterrupted CPU run, including optimizer and minibatch RNG state.
- Ran the control-audit notebook cells in the VS Code notebook kernel.
- Executed every code cell of the other three notebooks sequentially in separate clean namespaces in that working notebook kernel, using CPU smoke configurations. This was necessary because separate notebook kernel selection was unavailable; it tests the cell source and complete workflows, not three separate editor-kernel sessions.
- All three smoke grids completed: 3 sample-efficiency conditions, 6 noise/observability conditions, and 2 pixel/sensor conditions. Each condition independently trains two encoders. Plots, summaries, and export archives were generated.
- Notebook execution environment: Python 3.14.6; NumPy 2.5.4; SciPy 1.18.1; pandas 2.3.3; Matplotlib 3.11.2; PyTorch 2.15.0; CPU; Windows 11.
- Core tests also executed under Python 3.12.10 with the same listed numerical-library versions.
- Compiled the standalone LaTeX source with portable Tectonic 0.17.0. All bibliography keys are checked against citations; all four experiment subsections and the results/discussion/conclusions placeholders are checked automatically.

## What this does not establish

- Smoke profiles are only 40 updates and one world. They are **not research results**. Several smoke runs trigger the low-effective-rank diagnostic; that flag is intentionally retained rather than hidden. The primary hypothesis is not assumed to be supported.
- The pilot/full grids, statistical power, full-budget stability, CUDA execution, and execution inside hosted Kaggle or Colab have not been established by the local smoke run.
- Local generated outputs and checkpoints are excluded from Git. The source and notebook narratives do not contain empirical conclusions. Construction checks illustrate mathematical identities, not new discoveries.
- Notebook editors may normalize cell metadata to standard nbformat IDs and notebook-level language metadata. All notebooks are valid JSON/nbformat and their code cells pass Python parsing checks.

Before interpreting results, follow [the protocol checklist](docs/PROTOCOL.md), run multiple independent worlds, preserve all configurations and diagnostics, and record the exact Git commit and runtime environment.

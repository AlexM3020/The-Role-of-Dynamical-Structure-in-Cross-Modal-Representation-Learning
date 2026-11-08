# Validation record

## Direct GitHub/Colab setup (2026-11-08)

- Recovered all five revised notebook setup cells after an interrupted publication. Each can fetch the shared source directly from GitHub in a fresh Colab or Kaggle runtime; no uploaded source ZIP is required when Internet is available.
- The resolved source revision is printed. Existing checkouts are deliberately reused without reset or automatic updates; changing `REPO_REF` requires a fresh runtime. Local execution uses the current checkout rather than creating another local repository.
- The recovered suite passes 29 tests locally, including two source-bootstrap tests using an isolated simulated cloud filesystem and mocked Git commands. These verify a first source fetch and safe reuse on rerun; they are not a hosted Colab/GPU run.
- The exact publication tree, based on the existing remote commit, passes 28 tests with one intentional manuscript-only skip; lint passes. Imports were verified to resolve to that isolated tree rather than the local editable installation. All five publication notebooks have valid JSON/nbformat, matching cell source, and no execution outputs or local interpreter metadata.
- Publication copies of the notebooks have execution outputs and local interpreter metadata removed. Local notebook contents and previous result artifacts are preserved. The public experiment tree intentionally excludes manuscript files; its manuscript-only test is skipped when those files are absent.
- The remote's existing Colab history and deliberate manuscript-directory removal are preserved. No new dataset download or full research experiment is part of this publication step.

## Revised workflows (2026-11-08)

- The five notebook sources parse as Python/nbformat and contain no references to being part of a paper in their narratives. Configuration cells directly expose the experimental and optimization settings.
- All 27 unit/regression/document tests pass on Python 3.12.10 CPU. Tests cover memory-only defaults, explicit save/resume, loss-weight behavior, aliasing references, balanced repeated-state diagnostics, pairing before world-level aggregation, raw archive parsing with a small explicit fixture, participant split safety and the real-sensor workflow.
- Notebook 1's analytical cells executed in the notebook editor. Its export produced bytes in RAM, not an archive on disk.
- The exact code cells of notebooks 2–5 were executed in isolated namespaces in the working Python 3.14.6 notebook kernel. Their default paths perform no training, download or result-file saving.
- Notebooks 2–4 were additionally exercised with three-update, 32-pair CPU overrides while preserving the objective/architecture/observation combinations, and with disk saving disabled. These check execution only.
- Notebook 5's data/training/analysis/export branches were exercised on an explicitly labeled in-memory sensor fixture; the raw archive parser was tested separately with synthetic HAPT-shaped text members. The fixture is **not HAPT evidence** and no synthetic fallback exists in the actual notebook.
- No real HAPT archive was downloaded and no real HAPT training, full grid, hosted Colab/Kaggle execution, or CUDA run was performed for this revision. These remain unverified until explicitly run by the user.
- Test checkpoints are confined to temporary test directories. The existing experiment output directory was not used for new runs. Temporary notebook validation cells were removed after checks.

The source-only cloud ZIP contains clean notebook copies with custom language/ID metadata and no execution outputs; generating it does not modify the source notebooks or include local results. Notebook editors may independently save displayed outputs inside local notebook files; clear those outputs or disable editor output saving if this is undesired.

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

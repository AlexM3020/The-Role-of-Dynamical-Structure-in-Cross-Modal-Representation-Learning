# The Role of Dynamical Structure in Cross-Modal Representation Learning

This is the research repository for the paper **The Role of Dynamical Structure in Cross-Modal Representation Learning**.

## Research question

> Does temporal locality relative to fixed observation geometry improve finite-sample alignment between independently trained temporal encoders, compared with isospectral scrambled dynamics, when the stationary observation distribution and latent transition entropy are held fixed?

The main comparison uses a lazy random walk on 32 circularly embedded states and a permutation-conjugated version of the same transition matrix. **Observation functions remain fixed.** Thus the conditions have the same uniform stationary distribution, exactly the same instantaneous cross-modal observation law, the same latent one-step conditional entropy, and the same transition spectrum. They differ in which transitions are local relative to the sensors.

This is a test of the **interaction between temporal relationships, observation geometry, and learning bias**, not a comparison of inherently physical versus nonphysical worlds. It does not isolate a universal effect of physics. It also does not test whether a larger spectral gap helps: the main comparison preserves the complete spectrum.

## Five revised notebooks — editable configurations, memory-first execution

Each notebook explains the question, why its comparisons matter, experiments to try, editable settings, and interpretation limits. Original filenames are retained for existing links, but workflows 2–4 have revised purposes. The manuscript is intentionally not changed by this experiment revision.

| Notebook | Revised purpose | Compute |
|---|---|---|
| [01_control_audit.ipynb](notebooks/01_control_audit.ipynb) | Matched-control audit, geometry checks, and observation-aware conditional-mean references under noise/aliasing | CPU |
| [02_sample_efficiency.ipynb](notebooks/02_sample_efficiency.ipynb) | Alignment versus dimensional collapse: baseline, variance, covariance, and covariance-identity objectives | CPU or one GPU |
| [03_noise_and_observability.ipynb](notebooks/03_noise_and_observability.ipynb) | **Now capacity/optimization:** linear and width 32/64/256 encoders, update checkpoints, and a separate data-budget slice | CPU or one GPU |
| [04_pixels_and_sensors.ipynb](notebooks/04_pixels_and_sensors.ipynb) | **Now combined robustness:** numerical noise, aliasing, and tiny CNN image–sensor comparisons | CPU or one GPU |
| [05_real_sensor_validation.ipynb](notebooks/05_real_sensor_validation.ipynb) | Raw UCI HAPT accelerometer–gyroscope validation, participant-disjoint calibration/test, activity/identity shortcut diagnostics | One small GPU helpful |

### How to use the configuration cell

1. Read the experiment checklist in the notebook.
2. Edit world/seed IDs, data budget, architecture, width, embedding dimension, noise, objectives and loss weights, learning rate, batch size, clipping, weight decay, update checkpoints and evaluation settings directly in the configuration cell. There are no hidden profile presets that replace those choices.
3. Inspect the displayed plan and fit count. Set `RUN_TRAINING=True` only when ready, then rerun the configuration and subsequent cells.
4. Keep `SAVE_RESULTS=False` and `OUTPUT_DIR=None` for memory-only execution. These are the defaults in every notebook.
5. Freeze settings before confirmation. Suggested larger grids (25 new worlds, two repeats, 800/3,200 updates) are instructions, not automatically launched jobs.

## Data and models

Workflows 1–4 use NumPy-generated data from [worlds.py](src/dynalign/worlds.py). The images are synthetic, not real-world evidence. Small built-in `torch.nn` encoders start from random initialization, avoiding unknown multimodal pretraining. Workflow 5 explicitly loads [UCI HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions), Reyes-Ortiz et al. (2015), DOI 10.24432/C54G7M, CC BY 4.0: raw 50-Hz accelerometer and gyroscope recordings from 30 participants, not the precomputed feature table.

In synthetic workflows, every training example is an **independently sampled stationary length-two trajectory**. This removes long-trajectory mixing and overlapping-window leakage from the main comparison. Each modality receives a separate draw. The encoders share no state-label targets, optimizer state, or embeddings. They do share the objective family, dynamical law, and experimental design; "independent" does not mean assumption-free.

Only calibration and evaluation use same-state cross-modal pairing. Known synthetic states enter diagnostics, probes, and state-mean summaries, never encoder-training targets. A memoryless encoder cannot distinguish exactly aliased states; the quotient target and observation-aware Bayes-MSE reference clarify what is accessible. Conditional-mean reference scores are finite Monte Carlo estimates, not universal finite-sample metric ceilings.

HAPT windows are non-overlapping within each raw recording. Subjects are split before normalization; only training subjects determine normalization. Calibration fits the orthogonal map, and held-out participants evaluate it. The default 18/6/6 split is an explicit new split, not the original official partition. Neighbor graphs have reflecting endpoints and never cross recordings; scrambling conjugates those **empirical pair kernels** while leaving windows fixed. It does not show that unknown natural dynamics have matched entropy/spectrum. Activity labels are used only after encoder training for fixed probes and grouped diagnostics. Opt-in short recording prefixes are debugging data, not a representative activity benchmark.

## Running locally

Python 3.10+ is supported by the source; the tested local environment is recorded in [VALIDATION.md](VALIDATION.md). Python 3.11 or 3.12 is a convenient choice for cloud compatibility.

```bash
python -m venv .venv
# Activate the environment using your operating system's standard activation command.
python -m pip install -e ".[dev]"
python -m pytest
```

Select that environment as the notebook kernel. Open a notebook and run from top to bottom to inspect its plan; training is disabled until explicitly enabled. Source fingerprint, package versions, Git commit if available, full configuration and device are recorded in the in-memory result bundle. A frozen confirmation run should use immutable source and an independent world set.

## Budget and confirmation

Default settings describe small execution checks: one world/repeat and checkpoints at 40 and 80 updates; they do not train until enabled. Recommended studies are listed inside each notebook. For the primary objective confirmation, 25 worlds × 2 repeats × 2 conditions × 2 objectives × 2 encoders = 400 encoder fits. Capacity and robustness can use smaller declared secondary grids. Checkpoints at 800 and 3,200 are observations along one run, not two independent replications. Measure runtime before expanding; reported elapsed seconds include evaluation.

Keep the original objective as a baseline. Apply any changed regularization equally to both conditions. A covariance-identity penalty is a soft alternative objective, not guaranteed whitening. Increasing rank through measurement noise is not useful factor recovery. Do not select objectives for the largest local-minus-scrambled test gap; use independent development/validation diagnostics, freeze settings, then use fresh worlds.

Use the same optimizer-update budget across conditions. Different data budgets therefore imply different numbers of reuses of each training transition; this estimates data-budget effects at fixed updates, not an asymptotic sample-complexity theorem. Parameter counts and measured training time are recorded. An optional longer-optimization check can be made with explicit copied configurations; never silently tune only the favored condition.

## No automatic saving to the laptop

- **Default:** `SAVE_RESULTS=False`, `OUTPUT_DIR=None`. No result directory, model checkpoint, CSV, PNG, or ZIP is written. Tables, figures and export bytes stay in RAM. No automatic download is triggered.
- **Important editor distinction:** Jupyter/VS Code/Colab may save displayed cell outputs inside the notebook itself. Clear outputs or disable output saving if those must not persist locally; the Python runner does not control editor autosave.
- **Optional persistence:** set both `SAVE_RESULTS=True` and a deliberate `OUTPUT_DIR`. Use a mounted Drive or explicitly selected Kaggle working location for cloud persistence. Local paths are allowed only by this explicit choice; no location is inferred.
- Opt-in mode writes per-encoder checkpoints including optimizer/minibatch state, evaluation snapshots, and incremental content-addressed ZIP exports. All actual loss settings, source, environment and device enter run identity; changed configurations cannot silently reuse the old cache. Legacy experiment archives are never used by the revised runner.
- Final cells prepare ZIP bytes in RAM. `SHOW_DOWNLOAD_LINK=True` displays a link that downloads only after a manual click; clicking can save to the laptop. Runtime failure loses unsaved memory-only progress.
- Existing outputs are not deleted or changed. The original APIs in [experiments.py](src/dynalign/experiments.py) remain for legacy reproducibility but are no longer called by the revised notebooks; those legacy APIs still write results when invoked explicitly.

## Measurements and interpretation

Primary endpoint: held-out **linear CKA**, with a local-minus-scrambled contrast at each prespecified setting. Supporting endpoints include calibration-fitted orthogonal Procrustes, full covariance spectra, effective rank, variance, state-mean versus within-state noise variation, original-coordinate/quotient/transition-specific spectral probes, and next-coordinate conditional-mean probes. Step-zero random-initialization CKA and shuffled-correspondence diagnostics are included. Full configurations, not only regime names, accompany exports.

Orthogonal Procrustes uses calibration means and one scalar normalization per modality, **not arbitrary linear whitening**. Spectral targets are marked undefined when the selected boundary eigengap is zero. Collapsed or invalid runs are retained and flagged; never select only successful runs. The revised synthetic workflow decomposes state-mean and within-state noise variation using known state IDs. Legacy state-centroid CKA/RSA remain available in the original implementation, but are not outputs of the revised runner.

Repetitions are paired first and then averaged within world. Incomplete pairs are counted and prevent survivor-only averaging of that unit. Intervals require at least five valid units; five-unit intervals are preliminary. Real-data intervals resample held-out participants after averaging paired seed differences, conditional on the fixed training/calibration cohort and fitted models. Report valid sample counts, all settings, uncertainty, missing metrics and collapse rates. No workflow declares a hypothesis proved.

## Repository layout

```text
notebooks/        Five revised, memory-first workflows
src/dynalign/    Simulation, models, training, metrics, experiment runners, reporting
tests/           Control invariants, metrics, checkpoint and workflow tests
docs/            Protocol and bibliography provenance
scripts/         Source-only cloud bundle helper
VALIDATION.md    What was actually executed and what remains untested
```

This repository does not establish a general explanation of the Platonic Representation Hypothesis, causal identification of physical laws, or transfer to natural audiovisual data. Those would require additional theory and evidence beyond these controlled experiments.

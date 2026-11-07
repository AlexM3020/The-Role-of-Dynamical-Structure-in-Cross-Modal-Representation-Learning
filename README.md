# The Role of Dynamical Structure in Cross-Modal Representation Learning

This is the research repository for the paper **The Role of Dynamical Structure in Cross-Modal Representation Learning**.

## Research question

> Does temporal locality relative to fixed observation geometry improve finite-sample alignment between independently trained temporal encoders, compared with isospectral scrambled dynamics, when the stationary observation distribution and latent transition entropy are held fixed?

The main comparison uses a lazy random walk on 32 circularly embedded states and a permutation-conjugated version of the same transition matrix. **Observation functions remain fixed.** Thus the conditions have the same uniform stationary distribution, exactly the same instantaneous cross-modal observation law, the same latent one-step conditional entropy, and the same transition spectrum. They differ in which transitions are local relative to the sensors.

This is a test of the **interaction between temporal relationships, observation geometry, and learning bias**, not a comparison of inherently physical versus nonphysical worlds. It does not isolate a universal effect of physics. It also does not test whether a larger spectral gap helps: the main comparison preserves the complete spectrum.

## Four notebooks

Each notebook is self-contained as a workflow, imports the shared implementation, and describes its analysis without referring to a manuscript.

| Notebook | Purpose | Compute |
|---|---|---|
| [01_control_audit.ipynb](notebooks/01_control_audit.ipynb) | Audit matched quantities, distinguish mixing and embedding gaps, verify co-permutation and geometry controls | CPU |
| [02_sample_efficiency.ipynb](notebooks/02_sample_efficiency.ipynb) | Independently train two small sensor encoders; compare local, scrambled, and IID transitions across data budgets | CPU or one GPU |
| [03_noise_and_observability.ipynb](notebooks/03_noise_and_observability.ipynb) | Repeat comparisons with measurement noise and exact antipodal aliasing in one sensor | CPU or one GPU |
| [04_pixels_and_sensors.ipynb](notebooks/04_pixels_and_sensors.ipynb) | Compare a tiny CNN on 16 x 16 Gaussian-marker images with a numerical-sensor MLP | CPU or one GPU |

Open in Colab:

- [Control audit](https://colab.research.google.com/github/AlexM3020/The-Role-of-Dynamical-Structure-in-Cross-Modal-Representation-Learning/blob/main/notebooks/01_control_audit.ipynb)
- [Sample efficiency](https://colab.research.google.com/github/AlexM3020/The-Role-of-Dynamical-Structure-in-Cross-Modal-Representation-Learning/blob/main/notebooks/02_sample_efficiency.ipynb)
- [Noise and observability](https://colab.research.google.com/github/AlexM3020/The-Role-of-Dynamical-Structure-in-Cross-Modal-Representation-Learning/blob/main/notebooks/03_noise_and_observability.ipynb)
- [Pixels and sensors](https://colab.research.google.com/github/AlexM3020/The-Role-of-Dynamical-Structure-in-Cross-Modal-Representation-Learning/blob/main/notebooks/04_pixels_and_sensors.ipynb)

For Kaggle, import a notebook from GitHub or upload it, enable Internet for the setup cell, and optionally select a GPU. Each notebook works independently; run the control audit first to understand the assumptions. No account keys, dataset licenses to accept, or pretrained weights are required.

## Data and models

All data are generated locally with NumPy from documented equations in [worlds.py](src/dynalign/worlds.py). Importable generators provide transition matrices, nonlinear sensors, and small rendered images. No external dataset is downloaded. The images are synthetic, not evidence of real-world external validity. Small built-in `torch.nn` MLPs and CNNs start from random initialization; avoiding pretrained models also avoids unknown multimodal supervision in their pretraining.

Every training example is an **independently sampled stationary length-two trajectory**. This deliberately removes long-trajectory mixing and overlapping-window leakage from the main comparison. Each modality receives a separate draw of those trajectories. The two encoders share neither training examples, labels, optimizer state, nor embeddings. They do share the objective family, dynamical law, and experimental design; "independent" does not mean assumption-free.

Only calibration and evaluation contain same-state cross-modal pairs. Known states are used for diagnostics, probes, and state-mean summaries, never as encoder-training targets. A single-frame encoder cannot distinguish exactly aliased states. The observability notebook reports an accessible quotient target as well as the unrecoverable full-coordinate target; it does not pretend that dynamics recover information absent from the encoder input.

## Running locally

Python 3.10+ is supported by the source; the tested local environment is recorded in [VALIDATION.md](VALIDATION.md). Python 3.11 or 3.12 is a convenient choice for cloud compatibility.

```bash
python -m venv .venv
# Activate the environment using your operating system's standard activation command.
python -m pip install -e ".[dev]"
python -m pytest
```

Select that environment as the notebook kernel. Open any notebook and run from top to bottom. Cloud setup cells install from a repository checkout; change `REPO_REF` from `main` to an immutable commit before collecting final results. The actual checkout commit, source fingerprint, package versions, configuration, and device are saved alongside each run.

## Profiles and budget

- **smoke** (default): one world, one initialization pair, 40 updates. Execution and pipeline checks only; no confidence intervals.
- **pilot**: three worlds, one initialization pair, 300 updates. Exploratory diagnostics, not confirmatory inference.
- **full**: five worlds, two independent initialization/data repeats per world, 800 updates; data budgets 512, 2,048, and 8,192 transitions, except robustness at 2,048. Five worlds is a starting point, not a power guarantee.

The sample-efficiency full grid trains 180 encoders; robustness trains 120; pixel validation trains 120. Runs are sequential on one device. Measure a representative run before starting a full grid. Small dense models can be faster on CPU than on a remote GPU. "Full" means the configured grid, not publication readiness or adequate statistical power. Repeat more independent worlds if uncertainty warrants it, using a fixed analysis plan rather than stopping when an effect looks favorable.

Use the same optimizer-update budget across conditions. Different data budgets therefore imply different numbers of reuses of each training transition; this estimates data-budget effects at fixed updates, not an asymptotic sample-complexity theorem. Parameter counts and measured training time are recorded. An optional longer-optimization check can be made with explicit copied configurations; never silently tune only the favored condition.

## Checkpoints, persistence, and output

- Set `OUTPUT_DIR` in the configuration cell. Colab local storage is ephemeral: use a mounted Drive location if persistence across VM termination is required. The notebooks do not automatically request Drive access.
- On Kaggle the default output path is under `/kaggle/working`; save a notebook version to retain outputs, and reattach previous outputs when resuming in a new session.
- Each run saves its configuration, checkpoint for each encoder (including optimizer and minibatch RNG state), embeddings, observation tables, transition matrix, and metrics. Checkpoints are written every 100 updates and at completion.
- A finished matching run is reused. Cache keys include source, specification, dependency versions, and device. An interrupted matching run resumes from its last checkpoint. Only load checkpoints produced by your own trusted execution.
- The output directory contains an incrementally updated `summary.csv`; each notebook also saves paired contrasts and plots. A final cell creates a downloadable archive. Copy that archive to durable storage before ending a free cloud session.
- Outputs, checkpoints, and environments are excluded from Git. No private source conversation or PDF is included in this repository.

## Measurements and interpretation

Primary endpoint: held-out **linear CKA**, with a local-minus-scrambled contrast at each prespecified data budget. Supporting endpoints: orthogonal Procrustes error fitted on a separate calibration set, state-centroid RSA, effective covariance rank, variance, fixed-coordinate and observable-quotient linear probes, transition-specific spectral probes, and next-coordinate conditional-mean probes. Also report random-initialization CKA and a shuffled-correspondence diagnostic.

Orthogonal Procrustes uses calibration means and one scalar normalization per modality, **not arbitrary linear whitening**. Spectral targets are marked undefined when the selected boundary eigengap is zero. Collapsed or invalid runs are retained and flagged; never select only successful runs. State-centroid metrics average measurement noise using known state IDs, so they are complementary oracle-assisted diagnostics, not substitutes for the sample-level endpoint.

Initialization repeats are averaged within world before world-level paired bootstrap intervals. Intervals are not returned with fewer than five valid worlds. Individual observations and initialization repeats are not treated as independent experimental worlds. Bootstrap intervals with only five worlds are preliminary and can be unstable. Report every prespecified condition, uncertainty, missing metrics, and collapse rates; there is no automated "hypothesis proved" message. Pairwise label shuffling is a diagnostic, not a formal randomization-test p-value.


## Repository layout

```text

notebooks/        Four cloud-portable workflows
src/dynalign/    Simulation, models, training, metrics, experiment runners, reporting
tests/           Control invariants, metrics, checkpoint and workflow tests
docs/            Protocol and bibliography provenance
VALIDATION.md    What was actually executed and what remains untested
```

This repository does not establish a general explanation of the Platonic Representation Hypothesis, causal identification of physical laws, or transfer to natural audiovisual data. Those would require additional theory and evidence beyond these controlled experiments.

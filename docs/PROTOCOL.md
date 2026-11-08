# Frozen-run checklist and interpretation boundaries

## Revised five-workflow plan (2026-11-08)

The current notebooks use the memory-first engine in `src/dynalign/revised.py` and the real-sensor workflow in `src/dynalign/hapt.py`. The original engine is retained only for old-result reproducibility. The manuscript is unchanged and will need a separate methods update when the final experiment choices are frozen.

1. **Controls and attainable information:** retain exact construction/geometry checks, and estimate known-simulator conditional means for coordinate, quotient and transition-specific spectral targets. These are privileged Bayes-MSE references estimated by Monte Carlo, not training labels or guaranteed finite-sample metric ceilings.
2. **Alignment versus collapse:** original 25/25/1 weights; stronger variance; stronger off-diagonal covariance; and an explicit covariance-identity penalty. The latter is a soft alternative loss, not hard whitening. Apply each regime equally to both conditions. Develop on separate worlds, freeze settings without maximizing favorable test contrasts, then use 25 new worlds and two repeats for the primary baseline/candidate comparison.
3. **Capacity/optimization:** linear and width 32/64/256 encoders at fixed output dimension, with checkpoints along the same training trajectory. Separate the capacity slice from the width-64 data-budget slice. Suggested checkpoints are 800/3,200 and budgets 512/2,048/8,192 after benchmarking. These are finite-budget comparisons, not a sample-complexity theorem.
4. **Observation robustness:** clean/noisy/antipodally aliased numerical sensors and a tiny CNN image–sensor setting. Compare within each sensor setting, baseline versus a previously frozen candidate. An identical reused clean run is not independent replication. Ten fresh worlds and two repeats are a proposed secondary grid, not a power guarantee.
5. **Raw HAPT:** independently trained accelerometer/gyroscope encoders; non-overlapping raw windows; participant-disjoint train/calibration/test; train-only normalization. Permute a symmetric reflecting neighbor graph within each recording, with a common graph construction across modalities and independent pair draws. This preserves the empirical pair kernel spectrum, not unknown physical transition statistics. Activity labels enter only frozen probes and grouped diagnostics.

The new runner records complete `FitSettings`, `EvalSettings`, and per-run specifications. It reports full covariance eigenvalues, effective rank, state-mean/noise variation, and collapse rather than automatically filtering failed representations. Rank restoration caused by measurement noise is not useful factor recovery. State-mean variance is corrected for finite-repeat noise; the untruncated corrected estimate is retained.

### Storage and budget gates

Every notebook starts with `RUN_TRAINING=False` (except the analytical audit), `SAVE_RESULTS=False`, and no output directory. HAPT download/loading is separately disabled. All analysis and export preparation happen in RAM. Only an explicit output directory with saving enabled permits disk artifacts. A browser link downloads only when manually clicked. Notebook-editor autosave of displayed cell outputs is a separate behavior that the runner cannot disable.

The default small configuration is an execution-check plan, not a meaningful scientific sample. Inspect the displayed fit count and resource budget before enabling training. Do not silently run all recommended expanded grids. Existing results are never deleted or relabeled as confirmation.

Repetitions are paired before averaging within a world or test participant. Missing paired scores are counted; a unit with a missing repeat does not contribute a survivor-only mean. For HAPT, participant-bootstrap intervals are conditional on fixed training/calibration cohorts and fitted models, and the default six test participants provide limited precision. Check label coverage: a debugging prefix may omit activity classes and yield undefined probes.

### Provenance of previous results

The supplied feedback refers to full-run archives from commits different from the initial local checkout. Those numerical claims have not been independently reconciled by this revision. Preserve those archives and verify source/configuration/environment before pooling them with new runs. In particular, identical clean runs shared by two analyses are not separate evidence.

## Primary comparison

Local versus permutation-conjugated dynamics **with sensor maps fixed**. This changes locality relative to the observations, not the abstract transition graph. Exact matches: uniform stationary distribution, instantaneous cross-modal observation law, latent transition entropy, self-transition rate, complete spectrum, and both mixing and embedding gaps. Temporal observation laws, sensor roughness, and network approximation difficulty are intentionally not matched.

IID is a diagnostic negative control, not an entropy-matched alternative. Co-permuting transitions and sensors is a pure coordinate-change check, not the main treatment. The design is not a test of a spectral-gap effect: its principal conditions have identical spectra.

## Before a confirmatory run

1. Complete the control-audit notebook and record the commit.
2. Freeze budgets, loss weights, preprocessing, dimension, architectures, noise, random seeds, and the primary CKA contrast. Use a separate exploratory directory for pilot adjustments.
3. Decide the number of independent worlds using pilot uncertainty and available compute; the bundled five-world full profile is a starting point, not a power calculation.
4. Time one representative run. Specify a resource cap before expanding the grid.
5. Set a durable output directory and test checkpoint resume. Never reuse an unrelated checkpoint.
6. Preserve all run records and missing metrics. Do not select successful seeds or stop when an effect becomes favorable.
7. Record deviations from the frozen plan before interpreting the final comparison.

## Sampling and seeds

Every pair is a fresh stationary length-two trajectory. There are no overlapping windows or long trajectories to mix. The two modalities train on independent pair draws, independent noise, initializations, minibatch random generators, and optimizers. The world index controls the sensor functions and scrambling permutation; the repeat index controls data and initialization. Conditions within a world/repeat use common seed choices for paired variance reduction, but different modality streams.

Calibration contains eight observations per state; test contains sixteen, with independent Gaussian noise. States intentionally recur across splits because the question concerns learning a representation on a fixed finite support, not unseen-state extrapolation. Balanced state evaluation is not the same as independently sampled states; inference is over experimental worlds, not per-frame standard errors.

## Metrics

- Primary: sample-level CKA, local minus scrambled at each fixed data budget.
- Supporting: calibration-fitted orthogonal Procrustes error; state-centroid CKA/RSA; effective covariance rank and variance; fixed ridge probes for original coordinates, doubled-angle quotient, transition-specific spectral coordinates, and next-coordinate conditional mean.
- Diagnostic baselines: pretraining-initialization CKA and shuffled evaluation correspondence CKA.
- A zero spectral boundary gap means the selected oracle subspace is not unique. Do not interpret its arbitrary basis as a preferred latent target.
- A probe accesses the simulated states only after encoder training. It measures readout accessibility, not causal identification. Original-coordinate recovery is biased toward the geometry used to define locality; pair it with the transition-specific target.
- Aliased sensors cannot reveal antipodal identity to a memoryless encoder. A quotient target is accessible, but recovery of that quotient is not full-state recovery.

## Analysis

Average initialization/data repeats within each independent world; bootstrap paired world-level differences. The implementation reports no interval below five valid worlds. Five-world bootstrap intervals may be unstable. Missing values and collapse flags must be reported separately from finite-score summaries. A collapsed representation with a finite CKA is still shown, never silently excluded. If metrics become undefined, state how many valid worlds remain and do not claim a population conclusion from survivors alone.

A positive effect supports only the specific dynamics--observation--learner interaction. A null result can reflect a genuine lack of advantage, a ceiling from high initial similarity, limited optimization, or insufficient precision; inspect the prespecified diagnostics rather than asserting whichever explanation is convenient. Any longer-training or changed-architecture follow-up is a separate declared analysis.

## Compute and publication boundaries

Smoke runs establish execution only. The supplied generators are deliberately small and synthetic. No real-world, unseen-dynamics, conservation-law discovery, or universal PRH claim follows from these notebooks. There are no empirical numbers in the manuscript placeholders. Do not insert smoke outputs as research findings.

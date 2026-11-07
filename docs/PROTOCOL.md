# Frozen-run checklist and interpretation boundaries

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

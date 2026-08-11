# Cross Dataset Sleep Staging

Systematic benchmark of unsupervised domain adaptation (DANN, ADAST) for cross-dataset sleep stage classification from single-channel EEG.

> **Status:** work in progress — undergraduate thesis (TCC), UTFPR Campo Mourão.

## Motivation

Deep learning models for automatic sleep staging achieve expert-level performance when trained and evaluated on the same dataset. However, performance degrades substantially when a model trained on one cohort is applied to recordings from a different source, a phenomenon known as *domain shift*, driven by differences in acquisition hardware, electrode montage, sampling rate, and patient population. This gap is a key barrier to clinical adoption.

This project investigates whether unsupervised domain adaptation (UDA) can mitigate that degradation, and at what computational cost.

## Research question

To what extent do unsupervised domain adaptation techniques reduce the cross-dataset performance degradation of EEG-based sleep staging models, and what is the trade-off between generalization gain and computational cost?

## Approach

- **Baseline model:** TinySleepNet (single-channel EEG)
- **UDA methods:** DANN, ADAST (compared against a no-adaptation baseline)
- **Evaluation:** subject-wise splits, multiple cross-dataset pairs in both directions
- **Metrics:** macro-F1 (primary), per-stage F1, Cohen's κ, confusion matrices, and computational cost (parameters, training time, inference latency)

## Datasets

This repository does **not** include any data. The datasets used are publicly available and subject to their own licenses and terms of use — please obtain them from the original sources.

| Dataset | Source | Labeling |
|---|---|---|
| Sleep-EDF Expanded | PhysioNet | R&K (converted to AASM) |
| ISRUC-Sleep | University of Coimbra | AASM |
| HMC | PhysioNet | AASM |

## Repository structure
* src/: source code
* notebooks/: exploratory analysis
* configs/: experiment configuration
* data/: datasets (not versioned)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

- [x] Repository setup
- [ ] Data loading and preprocessing pipeline
- [ ] Baseline reproduction (within-dataset)
- [ ] Cross-dataset evaluation without adaptation
- [ ] UDA implementation and benchmark

## License

Code is released under the MIT License. Datasets are **not** covered by this license and remain subject to the terms of their respective providers.

## Author

Gabriel Farias Marcondes — Computer Science, UTFPR Campo Mourão

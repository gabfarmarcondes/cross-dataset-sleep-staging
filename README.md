# Cross-Dataset Sleep Staging

Systematic benchmark of unsupervised domain adaptation for cross-dataset sleep stage classification from single-channel EEG.

> **Status:** work in progress — undergraduate thesis (TCC), UTFPR Campo Mourão.

## Motivation

Deep learning models for automatic sleep staging achieve High-level performance when trained and evaluated on the same dataset. However, performance degrades substantially when a model trained on one cohort is applied to recordings from a different source, a phenomenon known as *domain shift*, driven by differences in acquisition hardware, electrode montage, sampling rate, and patient population. This gap is a key barrier to clinical adoption.

This project investigates whether unsupervised domain adaptation (UDA) can mitigate that degradation, and at what computational cost.

## Research question

To what extent do unsupervised domain adaptation techniques reduce the cross-dataset performance degradation of EEG-based sleep staging models, and what is the trade-off between generalization gain and computational cost?

## Approach

- **Baseline model:** TinySleepNet (single-channel EEG), reimplemented in PyTorch from the [official code](https://github.com/akaraspt/tinysleepnet)
- **UDA methods:** DANN (feature alignment) and a joint feature–label alignment method (to be defined), compared against a source-only baseline
- **Evaluation:**
  - within-dataset (reference) and cross-dataset: 3 datasets → 6 source → target scenarios
  - subject-wise splits; the target test subjects are fixed and never used during adaptation
  - model selection on source validation data only (no target labels)
  - 3 random seeds per configuration (mean ± std)
- **Ablation:** band-pass filtering (0.3–35 Hz) vs. no additional filtering
- **Metrics:** macro-F1 (primary), per-stage F1, accuracy, Cohen's κ, confusion matrices, and computational cost (parameters, training time, inference time, memory)

## Datasets

This repository does **not** include any data. The datasets used are publicly available and subject to their own licenses and terms of use — please obtain them from the original sources.

| Dataset | Source | Channel | Sampling rate | Labeling |
|---|---|---|---|---|
| Sleep-EDF Expanded (SC) | PhysioNet | Fpz-Cz | 100 Hz | R&K (converted to AASM) |
| ISRUC-Sleep, Subgroup I | University of Coimbra | C3-A2 / C3-M2 | 200 Hz | AASM |
| HMC | PhysioNet | C3-M2 | 256 Hz | AASM |

All signals are resampled to 100 Hz and split into 30-s epochs. For Sleep-EDF, wake is trimmed to 30 min before and after the sleep period, as in DeepSleepNet and TinySleepNet.

### References

- Kemp, B. et al. (2000). Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering*, 47(9), 1185–1194.
- Khalighi, S., Sousa, T., Santos, J. M., & Nunes, U. (2016). ISRUC-Sleep: a comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*, 124, 180–192.
- Alvarez-Estevez, D., & Rijsman, R. (2022). Haaglanden Medisch Centrum sleep staging database (version 1.1). PhysioNet. https://doi.org/10.13026/t79q-fr32
- Alvarez-Estevez, D., & Rijsman, R. M. (2021). Inter-database validation of a deep learning approach for automatic sleep scoring. *PLoS ONE*, 16(8), e0256111.
- Pollard, T. et al. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*, 1(8), 792–795. https://doi.org/10.1038/s44360-026-00096-z
- Supratak, A., & Guo, Y. (2020). TinySleepNet: an efficient deep learning model for sleep stage scoring based on raw single-channel EEG. *EMBC 2020*, 641–644.

### Subjects

The prototype uses a fixed set of subjects, frozen in [`configs/datasets.yaml`](configs/datasets.yaml) together with the rule used to select them:

| Dataset | Subjects | Recordings |
|---|---|---|
| Sleep-EDF Expanded (SC) | 00–19 ("Sleep-EDF-20") | 39 (subject 13 has one night) |
| ISRUC-Sleep, Subgroup I | 1–20 | 20 |
| HMC | SN001–SN021 (SN014 is not in the release) | 20 |

All code reads the subject lists from this file (`src/read_data.py`); splits are made by subject, never by recording.

## Repository structure

- `src/`: source code
- `configs/`: dataset and experiment configuration (`paths.yaml` is local and not versioned)
- `scripts/`: utility scripts (e.g. dataset checks)
- `notebooks/`: exploratory analysis

Data is never stored in the repository; see [Data setup](#data-setup).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data setup

Dataset locations are machine-specific, so they are kept out of version control in `configs/paths.yaml`:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then set `data_root` to the folder that contains each dataset's top folder (e.g. `<data_root>/ISRUC-SLEEP Dataset/`). Folder names and file patterns inside each dataset come from `configs/datasets.yaml`. To check that the local data matches the config:

```bash
python scripts/check_datasets.py
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

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
- **Metrics:** macro-F1 (primary), per-stage F1, accuracy, Cohen's κ, confusion matrices, and computational cost (parameters, training time, inference time)

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
> If you do not have an NVIDIA GPU, see the requirements.txt

## Data setup

Dataset locations are machine-specific, so they are kept out of version control in `configs/paths.yaml`:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

Then set `data_root` to the folder that contains each dataset's top folder (e.g. `<data_root>/ISRUC-SLEEP Dataset/`). Folder names and file patterns inside each dataset come from `configs/datasets.yaml`. To check that the local data matches the config:

```bash
python scripts/check_datasets.py
```

## Reproducing the thesis results

All commands run from the repository root, inside the virtual environment, with `configs/paths.yaml` set (see [Data setup](#data-setup)). Trained models, evaluations, tables and figures are written to `outputs/`, which is not versioned.

### 1. Preprocessing

```bash
python -m src.data.sleep_edf          # raw files -> one .npz per recording, in processed_root
python -m src.data.isruc
python -m src.data.hmc
python scripts/check_datasets.py
```

The subject splits are versioned in `configs/splits/`. They were generated once with `python -m src.data.splits`, which never overwrites an existing file.

The numbers of the data characterization quoted in the thesis come from the logs of the loaders above (epochs per stage), from `python -m scripts.check_preprocessing all` (share of power below 0.3 Hz and above 35 Hz, with the filter off and on) and from `python -m scripts.check_clipping <dataset>` (share of samples at the physical limit of the channel; it reads the raw files).

### 2. Training

Each command trains its runs one after the other, three seeds (0, 1, 2) and, within a dataset, five folds by default. Finished runs are skipped, so an interrupted command is resumed by running it again; `--dry-run` lists the runs and their status. To train the seeds in parallel, start one process per seed with `--seeds 0`, `--seeds 1` and `--seeds 2`.

| Experiment | Runs |
|---|---|
| A: CNN and CNN + LSTM within Sleep-EDF | `python -m src.training.run_many --dataset sleep_edf --scenario in_domain --modes epoch sequence` |
| A: CNN + LSTM without data augmentation | `python -m src.training.run_many --dataset sleep_edf --scenario in_domain --no-augment` |
| B: reference within ISRUC and HMC | `python -m src.training.run_many --dataset isruc --scenario in_domain`, and the same with `--dataset hmc` |
| B: source models, tested on the other two datasets | `python -m src.training.run_many --dataset sleep_edf --scenario source`, and the same with `--dataset isruc` and `--dataset hmc` |
| C: without the band-pass filter | `python -m src.training.run_many --dataset sleep_edf --scenario in_domain --no-filter` and `python -m src.training.run_many --dataset sleep_edf --scenario source --no-filter` |

The reference within Sleep-EDF in experiment B reuses the CNN + LSTM runs of experiment A, and the filtered runs of experiment C are those of experiments A and B.

### 3. Tables and figures

```bash
python -m src.evaluation.export
```

This regenerates every table and figure of the Experiments chapter: it runs `src.evaluation.report`, `src.evaluation.paired`, `src.evaluation.confusion_figure` and `src.data.figures` (runs not yet evaluated are evaluated first; nothing is trained). It also writes the environment table of the machine it runs on and `outputs/results/thesis_manifest.md`, which marks any missing file. `--dry-run` prints the commands without running them.

| Thesis label | File | Formatted by hand in the thesis |
|---|---|---|
| `fig:distribuicao-estagios` | `outputs/figures/class_distribution.png` | — |
| `fig:epocas-exemplo` | `outputs/figures/example_epochs.png` | — |
| `fig:psd-estagios` | `outputs/figures/psd_per_stage.png` | — |
| `tab:ambiente` | `outputs/results/environment.tex` | — |
| `tab:exp-a` | `outputs/results/experiment_a_metrics.tex` | row of the TinySleepNet paper (Supratak and Guo, 2020) added by hand |
| `tab:pares` | `outputs/results/experiment_a_pares.tex` | — |
| `tab:exp-a-estagios` | `outputs/results/experiment_a_f1_stages.tex` | row of the TinySleepNet paper added by hand |
| `tab:exp-b` | `outputs/results/experiment_b_metrics.tex` | macro-F1 only; the reference of each pair is the '(dentro)' row of its target, and the difference is the pair minus that reference |
| `tab:exp-b-estagios` | `outputs/results/experiment_b_f1_stages.tex` | — |
| `fig:confusao-b` | `outputs/figures/experiment_b_confusion.png` | — |
| `tab:exp-c-pares` | `outputs/results/experiment_c_dentro.tex` | — |
| `tab:exp-c-entre` | `outputs/results/experiment_c_entre.tex` | headers grouped by pair by hand |
| `tab:exp-c-estagios` | `outputs/results/experiment_c_f1_stages.tex` | only the pooled and cross rows; filter in its own column |

## Status

- [x] Repository setup
- [x] Data loading and preprocessing pipeline
- [x] Baseline reproduction (within-dataset)
- [x] Cross-dataset evaluation without adaptation
- [x] Filtering ablation
- [ ] UDA implementation and benchmark

## License

Code is released under the MIT License. Datasets are **not** covered by this license and remain subject to the terms of their respective providers.

## Author

Gabriel Farias Marcondes — Computer Science, UTFPR Campo Mourão

"""Unit tests for the experiment runner (issues 14, 15 and 17). Run from the repository root:

    python -m pytest tests/
"""
import json

from src.training.run_many import effective_preprocessing, planned_runs, status

TRAINING = {"mode": "sequence", "max_epochs": 200, "signal_shift": 0.1, "sequence_skip": 4}


def test_experiment_a_plan_is_seed_by_seed():
    runs = planned_runs("sleep_edf", "in_domain", ["epoch", "sequence"], [0, 1, 2],
                        [0, 1, 2, 3, 4], True, "official", TRAINING)
    names = [name for name, _ in runs]
    assert len(names) == len(set(names)) == 2 * 5 * 3
    assert names[0] == "sleep_edf_fold0_epoch_seed0" and names[9] == "sleep_edf_fold4_sequence_seed0"
    assert names[10] == "sleep_edf_fold0_epoch_seed1"
    assert runs[0][1] == ["--dataset", "sleep_edf", "--mode", "epoch", "--seed", "0",
                          "--batch-norm", "official", "--fold", "0"]


def test_source_plan_and_non_default_names():
    runs = planned_runs("hmc", "source", ["sequence"], [0, 1], [0, 1, 2, 3, 4], False, "standard", TRAINING)
    assert [name for name, _ in runs] == ["hmc_source_sequence_seed0_nofilter_bnstandard",
                                          "hmc_source_sequence_seed1_nofilter_bnstandard"]
    assert "--source" in runs[0][1] and "--no-augment" not in runs[0][1]
    assert runs[0][1][-1] == "--no-filter"  # the filter is switched off on the command line


def test_runs_without_filter_record_it_in_their_settings():
    preprocessing = {"filter": True, "band": [0.3, 35.0], "filter_order": 4, "clip": 20.0}
    assert effective_preprocessing(preprocessing, no_filter=True) == {**preprocessing, "filter": False}
    assert effective_preprocessing(preprocessing, no_filter=False) == preprocessing
    assert preprocessing["filter"]  # the experiment.yaml block itself is not changed
    runs = planned_runs("sleep_edf", "source", ["sequence"], [0], [0], True, "official", TRAINING)
    assert "--no-filter" not in runs[0][1]


def test_runs_without_augmentation_have_their_own_names():
    runs = planned_runs("sleep_edf", "in_domain", ["epoch", "sequence"], [0], [0], True, "official",
                        TRAINING, no_augment=True)
    assert [name for name, _ in runs] == ["sleep_edf_fold0_epoch_seed0_noaug",
                                          "sleep_edf_fold0_sequence_seed0_noaug"]
    assert all(args[-1] == "--no-augment" for _, args in runs)


def test_status_of_finished_interrupted_and_new_runs(tmp_path):
    expected = {"training": {"mode": "sequence", "max_epochs": 200}, "preprocessing": {"filter": True}}

    def write(name, **info):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_text(json.dumps(info), encoding="utf-8")

    write("done", result={}, training={"mode": "epoch", "max_epochs": 200}, preprocessing={"filter": True})
    write("quick", result={}, training={"mode": "sequence", "max_epochs": 2}, preprocessing={"filter": True})
    write("half", run="half")
    write("noaug", result={}, training={"mode": "sequence", "max_epochs": 200, "signal_shift": 0.0},
          preprocessing={"filter": True})
    assert status("done", expected, tmp_path) == "done"  # the mode may differ: it is part of the name
    assert status("quick", expected, tmp_path) == "different settings"
    assert status("noaug", expected, tmp_path) == "different settings"  # augmentation is a setting
    assert status("half", expected, tmp_path) == "interrupted"
    assert status("never", expected, tmp_path) == "new"

"""Train every run of an experiment, one after the other, and pick up where it stopped
(issue 14). From the repository root:

    python -m src.training.run_many --dataset sleep_edf --scenario in_domain \\
        --modes epoch sequence --seeds 0 1 2            # experiment A: 2 x 5 folds x 3 seeds
    python -m src.training.run_many --dataset sleep_edf --scenario source --seeds 0 1 2
    python -m src.training.run_many ... --no-augment    # the same runs without data augmentation
    python -m src.training.run_many ... --no-filter     # the same runs without the band-pass filter
    python -m src.training.run_many ... --dry-run       # only list what would run

Each run is one call to src.training.train in a fresh process (memory is freed between
runs). A run whose run.json already has a result is skipped; a run that was interrupted
(folder without a result) is trained again from the start with --overwrite. A finished run
whose settings differ from configs/experiment.yaml (e.g. a quick test with --max-epochs) is
never reused nor overwritten: the script stops and asks to move it away. If a run fails,
the script stops and says which one, so it can be started again with the same command.

The order is seed by seed, so that complete sets of folds are ready early.
"""
import argparse
import json
import subprocess
import sys
import time

import yaml

from src.data.dataset import without_filter
from src.models.tinysleepnet import BATCH_NORMS
from src.training.augment import is_augmented, without_augmentation
from src.training.loop import MODES
from src.training.train import EXPERIMENT_FILE, RUNS_DIR, run_name, same_training


def effective_training(training, no_augment):
    """The training block the runs will record in run.json (as src.training.train sets it)."""
    return without_augmentation(training) if no_augment else training


def effective_preprocessing(preprocessing, no_filter):
    """The preprocessing block the runs will record in run.json (as src.training.train sets it)."""
    return without_filter(preprocessing) if no_filter else preprocessing


def planned_runs(dataset, scenario, modes, seeds, folds, filtered, batch_norm, training,
                 no_augment=False):
    """(name, arguments) of every run, seed by seed. training: the experiment.yaml block;
    filtered: whether the runs use the band-pass filter (False adds --no-filter)."""
    training = effective_training(training, no_augment)
    runs = []
    for seed in seeds:
        for mode in modes:
            parts = [(f, False) for f in folds] if scenario == "in_domain" else [(None, True)]
            for fold, source in parts:
                name = run_name(dataset, mode, seed, fold, source, None, filtered, batch_norm,
                                is_augmented(training, mode))
                args = ["--dataset", dataset, "--mode", mode, "--seed", str(seed),
                        "--batch-norm", batch_norm]
                args += ["--fold", str(fold)] if fold is not None else ["--source"]
                args += ["--no-augment"] if no_augment else []
                args += [] if filtered else ["--no-filter"]
                runs.append((name, args))
    return runs


def status(name, expected, runs_dir=RUNS_DIR):
    """'done', 'different settings', 'interrupted' (folder without a result) or 'new'.

    expected: the training and preprocessing blocks the run must have been trained with.
    """
    run_json = runs_dir / name / "run.json"
    if run_json.exists():
        info = json.loads(run_json.read_text(encoding="utf-8"))
        if "result" in info:
            same = (same_training(info["training"], expected["training"])
                    and info["preprocessing"] == expected["preprocessing"])
            return "done" if same else "different settings"
    return "interrupted" if (runs_dir / name).exists() else "new"


def main():
    with open(EXPERIMENT_FILE, encoding="utf-8") as f:
        experiment = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Train all runs of an experiment, resumable.")
    parser.add_argument("--dataset", required=True, choices=["sleep_edf", "isruc", "hmc"])
    parser.add_argument("--scenario", required=True, choices=["in_domain", "source"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=[experiment["training"]["mode"]])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-norm", choices=BATCH_NORMS,
                        default=experiment.get("model", {}).get("batch_norm", "official"))
    parser.add_argument("--no-augment", action="store_true", help="train without data augmentation")
    parser.add_argument("--no-filter", action="store_true", help="train without the band-pass filter")
    parser.add_argument("--dry-run", action="store_true", help="only list the runs and their status")
    args = parser.parse_args()

    preprocessing = effective_preprocessing(experiment["preprocessing"], args.no_filter)
    runs = planned_runs(args.dataset, args.scenario, args.modes, args.seeds, args.folds,
                        preprocessing["filter"], args.batch_norm,
                        experiment["training"], args.no_augment)
    expected = {"training": effective_training(experiment["training"], args.no_augment),
                "preprocessing": preprocessing}
    states = {name: status(name, expected) for name, _ in runs}
    todo = [(name, a) for name, a in runs if states[name] != "done"]
    print(f"{len(runs)} runs: {len(runs) - len(todo)} done, {len(todo)} to train")
    for i, (name, run_args) in enumerate(runs, 1):
        print(f"  [{i:2d}/{len(runs)}] {name:<45} {states[name]}")
    if args.dry_run or not todo:
        return
    different = [name for name, _ in runs if states[name] == "different settings"]
    if different:
        sys.exit(f"\n{len(different)} finished runs were trained with other settings than "
                 f"configs/experiment.yaml (e.g. a quick test with --max-epochs, or a run from before "
                 f"issue 15, without the augmentation settings). Move them out of {RUNS_DIR} "
                 f"(they are kept, not overwritten) and run again: {', '.join(different)}")

    for i, (name, run_args) in enumerate(todo, 1):
        overwrite = ["--overwrite"] if states[name] == "interrupted" else []
        print(f"\n=== [{i}/{len(todo)}] {name} ===", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "src.training.train", *run_args, *overwrite])
        if result.returncode != 0:
            sys.exit(f"\n{name} failed (exit code {result.returncode}). Fix the cause and run the "
                     f"same command again: finished runs are skipped.")
        print(f"=== {name} finished in {(time.perf_counter() - start) / 60:.1f} min ===", flush=True)
    print("\nall runs finished")


if __name__ == "__main__":
    main()

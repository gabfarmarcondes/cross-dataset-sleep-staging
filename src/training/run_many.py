"""Train every run of an experiment, one after the other, and pick up where it stopped
(issue 14). From the repository root:

    python -m src.training.run_many --dataset sleep_edf --scenario in_domain \\
        --modes epoch sequence --seeds 0 1 2            # experiment A: 2 x 5 folds x 3 seeds
    python -m src.training.run_many --dataset sleep_edf --scenario source --seeds 0 1 2
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

from src.models.tinysleepnet import BATCH_NORMS
from src.training.loop import MODES
from src.training.train import EXPERIMENT_FILE, RUNS_DIR, run_name


def planned_runs(dataset, scenario, modes, seeds, folds, filtered, batch_norm):
    """(name, arguments) of every run, seed by seed."""
    runs = []
    for seed in seeds:
        for mode in modes:
            parts = [(f, False) for f in folds] if scenario == "in_domain" else [(None, True)]
            for fold, source in parts:
                name = run_name(dataset, mode, seed, fold, source, None, filtered, batch_norm)
                args = ["--dataset", dataset, "--mode", mode, "--seed", str(seed),
                        "--batch-norm", batch_norm]
                args += ["--fold", str(fold)] if fold is not None else ["--source"]
                runs.append((name, args))
    return runs


def without_mode(training):
    return {k: v for k, v in training.items() if k != "mode"}


def status(name, expected, runs_dir=RUNS_DIR):
    """'done', 'different settings', 'interrupted' (folder without a result) or 'new'.

    expected: the training and preprocessing blocks the run must have been trained with.
    """
    run_json = runs_dir / name / "run.json"
    if run_json.exists():
        info = json.loads(run_json.read_text(encoding="utf-8"))
        if "result" in info:
            same = (without_mode(info["training"]) == without_mode(expected["training"])
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
    parser.add_argument("--dry-run", action="store_true", help="only list the runs and their status")
    args = parser.parse_args()

    runs = planned_runs(args.dataset, args.scenario, args.modes, args.seeds, args.folds,
                        experiment["preprocessing"]["filter"], args.batch_norm)
    expected = {"training": experiment["training"], "preprocessing": experiment["preprocessing"]}
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
                 f"configs/experiment.yaml (e.g. --max-epochs). Move them out of {RUNS_DIR} "
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
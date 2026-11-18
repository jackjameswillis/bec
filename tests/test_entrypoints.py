"""Smoke tests for the experiment command-line entry points.

Every argparse-based runner and plotter must at least import and print its
help. Tiny one-iteration ES/GES/MGA runs then exercise the shared runner
plumbing end to end, including the history ordering invariant: the last
history entry must describe the same mode net as the final val metrics.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

HELP_SCRIPTS = [
    "plot_convergence.py",
    "plot_cr05_collapse.py",
    "plot_long_ste.py",
    "plot_mga_sweep.py",
    "plot_paper_sweep.py",
    "plot_theta_expectations.py",
    "plot_theta_neurons.py",
    "run_es.py",
    "run_ges.py",
    "run_mga.py",
    "run_mga_sweep.py",
    "run_paper_sweep.py",
    "run_ste.py",
]

SMALL_COMMON = [
    "--hidden", "4",
    "--layers", "1",
    "--population", "4",
    "--iterations", "1",
    "--fitness-batch", "8",
    "--chunk", "8",
    "--eval-every", "0",
    "--seed", "7",
]


def env():
    out = os.environ.copy()
    out["TF_USE_LEGACY_KERAS"] = "1"
    out["PYTHONUNBUFFERED"] = "1"
    return out


def run(args, timeout=300):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_help():
    for name in HELP_SCRIPTS:
        path = EXPERIMENTS / name
        assert path.exists(), f"missing entry point: {path}"
        proc = run([str(path), "--help"], timeout=180)
        assert proc.returncode == 0, (
            f"{name} --help failed (rc {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        print(f"help OK: {name}")


def check_search(script, extra, method, tmp):
    out = tmp / f"{method}.json"
    proc = run([str(EXPERIMENTS / script), *SMALL_COMMON, *extra,
                "--out", str(out)], timeout=300)
    assert proc.returncode == 0, (
        f"{script} failed (rc {proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    rec = json.loads(out.read_text())
    assert rec["method"] == method, rec["method"]
    assert rec["config"]["iterations"] == 1
    assert len(rec["history"]) == 1, len(rec["history"])
    last = rec["history"][-1]
    assert last["step"] == 1
    assert last["mode_val_loss"] == rec["val"]["loss"], (
        "history is not logged after the final update")
    assert last["mode_val_acc"] == rec["val"]["accuracy"], (
        "history is not logged after the final update")
    assert {"loss", "accuracy"} <= set(rec["test"])
    print(f"tiny {method} run OK")


def check_mga(tmp):
    script = EXPERIMENTS / "run_mga.py"
    out = tmp / "mga.json"
    args = [
        str(script),
        "--hidden", "4", "--layers", "1", "--population", "2",
        "--deme", "2", "--iterations", "1", "--fitness-batch", "8",
        "--chunk", "8", "--eval-every", "0", "--seed", "7",
        "--out", str(out),
    ]
    proc = run(args, timeout=300)
    assert proc.returncode == 0, (
        f"run_mga.py failed (rc {proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    rec = json.loads(out.read_text())
    assert rec["method"] == "mga"
    assert len(rec["history"]) == 1
    assert {"loss", "accuracy"} <= set(rec["test"])
    print("tiny mga run OK")


def main():
    check_help()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        check_search("run_es.py",
                     ["--table-rows", "0", "--sigma", "0"], "es", tmp)
        check_search("run_ges.py",
                     ["--table-rows", "0", "--sigma", "1"], "ges", tmp)
        check_mga(tmp)
    print("entry points OK")


if __name__ == "__main__":
    main()

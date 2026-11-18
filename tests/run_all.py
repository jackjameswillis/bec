"""Run every plain-script test under tests/ and summarise the results.

Each test is a standalone Python script with a ``main()`` entry point. Tests
run sequentially in fresh subprocesses so TF state and GPU memory are released
between scripts.

Usage:
  /path/to/python tests/run_all.py [--pattern 'test_*.py'] [--timeout S]
      [--fail-fast] [--list]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent


def discover(pattern):
    return sorted(TESTS_DIR.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py",
                        help="glob for test scripts (default: test_*.py)")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="per-test timeout in seconds (0 = no limit)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop after the first failing test")
    parser.add_argument("--list", action="store_true",
                        help="print the discovered tests and exit")
    args = parser.parse_args()

    tests = discover(args.pattern)
    if args.list:
        for path in tests:
            print(path.name)
        return 0 if tests else 1
    if not tests:
        print(f"no tests matched {args.pattern!r} under {TESTS_DIR}")
        return 1

    failures = []
    started_all = time.monotonic()
    total = len(tests)
    for i, path in enumerate(tests, start=1):
        print(f"\n[{i}/{total}] {path.name}", flush=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT,
                timeout=args.timeout if args.timeout > 0 else None,
            )
            elapsed = time.monotonic() - started
            if proc.returncode == 0:
                print(f"PASS {path.name} ({elapsed:.1f}s)", flush=True)
                continue
            reason = f"rc={proc.returncode}"
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            reason = f"timeout after {args.timeout:g}s"
        print(f"FAIL {path.name} ({reason}, {elapsed:.1f}s)", flush=True)
        failures.append((path.name, reason))
        if args.fail_fast:
            break

    elapsed_all = time.monotonic() - started_all
    print(f"\n{'=' * 60}")
    if failures:
        print(f"{len(failures)} of {total} test scripts failed:")
        for name, reason in failures:
            print(f"  FAIL {name} ({reason})")
        print(f"total wall time {elapsed_all:.1f}s")
        return 1
    print(f"all {total} test scripts passed in {elapsed_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

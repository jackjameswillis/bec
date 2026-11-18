"""Sweep resume-cache config validation tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._resume import config_mismatches, format_mismatches


def main():
    expected = {
        "population": 100,
        "lr": 0.1,
        "shaping": "rank",
        "antithetic": False,
        "bias_lr": None,
    }

    assert config_mismatches(dict(expected), expected) == []
    assert config_mismatches({**expected, "lr": 0.1 + 1e-12}, expected) == []

    stale = config_mismatches(
        {**expected, "population": 200, "lr": 0.2, "shaping": "none"},
        expected,
    )
    assert [m[0] for m in stale] == ["population", "lr", "shaping"]
    assert stale[0][1:] == (200, 100)
    text = format_mismatches(stale)
    assert "population" in text and "cached=200" in text

    assert config_mismatches(
        {**expected, "antithetic": 0}, expected)[0][0] == "antithetic"
    assert config_mismatches({}, expected)[0][1] == "<missing>"
    assert config_mismatches(None, expected)[0][0] == "<config>"
    print("resume config validation OK")


if __name__ == "__main__":
    main()

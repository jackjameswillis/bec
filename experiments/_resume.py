"""Validation helpers for sweep ``--resume`` caches."""

import math


def _equal(cached, expected, rel_tol):
    if isinstance(cached, bool) or isinstance(expected, bool):
        return (isinstance(cached, bool) and isinstance(expected, bool)
                and cached == expected)
    if isinstance(cached, float) or isinstance(expected, float):
        try:
            return math.isclose(float(cached), float(expected),
                                rel_tol=rel_tol, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return cached == expected


def config_mismatches(cached, expected, rel_tol=1e-9):
    """Return ``(field, cached_value, requested_value)`` recipe mismatches."""
    if not isinstance(cached, dict):
        return [("<config>", cached, expected)]
    mismatches = []
    for key, want in expected.items():
        if key not in cached:
            mismatches.append((key, "<missing>", want))
        elif not _equal(cached[key], want, rel_tol):
            mismatches.append((key, cached[key], want))
    return mismatches


def format_mismatches(mismatches):
    return "\n".join(
        f"    {key}: cached={got!r} requested={want!r}"
        for key, got, want in mismatches
    )

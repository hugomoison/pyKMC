"""Regression: style=constant rates are bit-identical to the legacy Eyring path.

This is the key safety guarantee for the HTST work — enabling the new
``compute_rate`` dispatcher must not change any existing constant-style run.
"""

from typing import Any

from pykmc.rate_constant import compute_rate, compute_rate_Eyring


def test_constant_rate_identical_to_legacy(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """compute_rate (constant style, no nu0) equals compute_rate_Eyring exactly."""
    cfg = config_Ni_4000at_monovacancy_sia  # style=constant from input.in
    for d_e in (0.1, 0.5, 1.2):
        assert compute_rate(d_e, cfg) == compute_rate_Eyring(d_e, cfg)

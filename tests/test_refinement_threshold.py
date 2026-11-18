"""Tests for the rate threshold under which reference events are not refined."""

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from pykmc import constants
from pykmc.refinement import Refinement

K0, T = 10.0, 500.0


def _rate(dE: float, prefactor: float = K0) -> float:
    return prefactor * math.exp(-dE / (constants.KB * T))


def _refinement(refine_thr: float) -> Refinement:
    config = SimpleNamespace(
        control=SimpleNamespace(refine_thr=refine_thr),
        rateconstant=SimpleNamespace(k0=K0, T=T),
    )
    return Refinement(config, None, None, None, None, None)  # type: ignore[arg-type]


def _energy_rule(table: pd.DataFrame, k_cut: float) -> float:
    """Energy above which events were not refined before, for a constant prefactor."""
    under = table[table["k"] < k_cut]
    if len(under) == 0:
        return 0.1
    return under.sort_values("k").iloc[-1]["energy_barrier"] + 0.1


@pytest.mark.parametrize("refine_thr", [0.5, 0.9999, 1e-12])
def test_rate_threshold_is_the_energy_rule_for_a_constant_prefactor(
    refine_thr: float,
) -> None:
    barriers = [0.05, 0.3, 0.42, 0.55, 0.9, 1.4]
    table = pd.DataFrame({"energy_barrier": barriers, "k": map(_rate, barriers)})
    supposed_ktot = float(table["k"].sum())

    k_thr = _refinement(refine_thr).get_rate_thr_refine(table, supposed_ktot)

    e_thr = _energy_rule(table, supposed_ktot * refine_thr)
    assert k_thr == pytest.approx(_rate(e_thr), rel=1e-12)
    for dE in (e_thr - 0.01, e_thr + 0.01, *barriers):
        if abs(dE - e_thr) > 1e-9:
            assert (_rate(dE) < k_thr) == (dE > e_thr)


def test_rate_threshold_follows_the_rate_with_a_prefactor_per_event() -> None:
    """The higher barrier has the higher rate: only the slower event is under the cut."""
    table = pd.DataFrame(
        {
            "energy_barrier": [0.50, 0.55],
            "k": [_rate(0.50, prefactor=0.1), _rate(0.55, prefactor=1000.0)],
        }
    )
    supposed_ktot = float(table["k"].sum())

    k_thr = _refinement(0.5).get_rate_thr_refine(table, supposed_ktot)

    margin = math.exp(-0.1 / (constants.KB * T))
    assert k_thr == table.at[0, "k"] * margin
    assert not (table["k"] < k_thr).any()

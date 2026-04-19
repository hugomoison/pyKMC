from __future__ import annotations

import pandas as pd
import pytest

amsel = pytest.importorskip("amsel")

from pykmc.basins import AmselFPTASelector, StatesConnectivity


def _connectivity(df: pd.DataFrame) -> StatesConnectivity:
    table = StatesConnectivity()
    table.df = df
    return table


def test_adaptive_selector_uses_mean_clock_on_rank1(monkeypatch):
    draws = iter([0.25])
    monkeypatch.setattr("numpy.random.random", lambda: next(draws))

    selector = AmselFPTASelector(clock_mode="adaptive", rank_tol=1.0e-6)
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0],
                "state_connexion": [10],
                "k_forward": [42.0],
            }
        )
    )

    result = selector.select_from_connectivity(table)
    assert result.is_ok()
    assert selector.last_clock_mode == "mean"
    assert selector.last_reduced_kinetics is not None
    assert selector.last_reduced_kinetics.one_rate_clock_is_plausible(1.0e-6)
    assert result.ok_value().t_exit == pytest.approx(1.0 / 42.0)
    assert result.ok_value().exit_state == 10


def test_adaptive_selector_uses_sampled_clock_on_rankk(monkeypatch):
    draws = iter([0.5, 0.0])
    monkeypatch.setattr("numpy.random.random", lambda: next(draws))

    selector = AmselFPTASelector(clock_mode="adaptive", rank_tol=1.0e-6)
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 1, 0, 1],
                "state_connexion": [1, 0, 10, 20],
                "k_forward": [1.0e-4, 1.0e-4, 1.0, 1.0e-4],
            }
        )
    )

    result = selector.select_from_connectivity(table)
    assert result.is_ok()
    assert selector.last_clock_mode == "sampled"
    assert selector.last_reduced_kinetics is not None
    assert selector.last_reduced_kinetics.slow_subspace_rank == 2
    assert selector.last_reduced_kinetics.rank1_invalidity > 0.0
    assert result.ok_value().t_exit > 0.0
    assert result.ok_value().exit_state in (10, 20)

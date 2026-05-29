"""Active events inherit the reference event's nu0, combined with the refined dE."""

import math
from typing import Any

import numpy as np

from pykmc.event_table import ActiveEventTable
from pykmc.rate_constant import compute_rate
from pykmc.result import EventRefinementOutput


def _refinement_output(nu0: float | None) -> EventRefinementOutput:
    return EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=np.zeros((1, 3)),
        E_saddle=0.0,
        min2_positions=np.zeros((1, 3)),
        dE_forward=0.5,
        num_reference_event=0,
        refined="T",
        nu0=nu0,
    )


def test_active_event_k_uses_inherited_nu0(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """build_event_series computes k from the refined dE and the inherited nu0."""
    config = config_Ni_4000at_monovacancy_sia
    config.rateconstant.style = "htst"
    table = ActiveEventTable(config)
    series = table.build_event_series(_refinement_output(nu0=5.0e12))
    assert math.isclose(series["k"], compute_rate(0.5, config, nu0=5.0e12))


def test_active_event_k_falls_back_without_nu0(
    config_Ni_4000at_monovacancy_sia: Any,
) -> None:
    """With nu0 None (constant style or htst fallback), k uses k0."""
    config = config_Ni_4000at_monovacancy_sia  # style=constant from input.in
    table = ActiveEventTable(config)
    series = table.build_event_series(_refinement_output(nu0=None))
    assert math.isclose(series["k"], compute_rate(0.5, config, nu0=None))

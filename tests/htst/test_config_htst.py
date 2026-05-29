"""Tests for RateConstantConfig HTST extension."""

import pytest
from pydantic import ValidationError

from pykmc.config import RateConstantConfig


def test_htst_style_accepts_fields() -> None:
    """HTST style with all new fields is accepted."""
    c = RateConstantConfig(
        style="htst", free_radius=6.0, fd_step=0.01, nu0_min_THz=1.0, nu0_max_THz=100.0
    )
    assert c.style == "htst"
    assert c.free_radius == 6.0


def test_nu0_window_must_be_ordered() -> None:
    """nu0_min_THz >= nu0_max_THz raises ValidationError."""
    with pytest.raises(ValidationError):
        RateConstantConfig(style="htst", nu0_min_THz=100.0, nu0_max_THz=1.0)


def test_constant_style_unchanged() -> None:
    """Existing constant style still parses correctly."""
    c = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    assert c.style == "constant"

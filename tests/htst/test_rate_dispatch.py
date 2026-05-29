"""Tests for the compute_rate dispatcher (constant vs htst prefactor)."""

import math
from types import SimpleNamespace

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import compute_rate


def _cfg(style: str, **kw: float) -> SimpleNamespace:
    return SimpleNamespace(rateconstant=RateConstantConfig(style=style, **kw))


def test_constant_style_uses_k0() -> None:
    """Constant style multiplies k0 by the Eyring exponential."""
    cfg = _cfg("constant", k0=10.0, T=300.0)
    p = PhysicalConstants()
    assert math.isclose(compute_rate(0.5, cfg), 10.0 * math.exp(-0.5 / (p.kb * 300.0)))


def test_htst_uses_nu0_when_present() -> None:
    """HTST style uses the supplied finite nu0 as the prefactor."""
    cfg = _cfg("htst", k0=10.0, T=300.0)
    p = PhysicalConstants()
    assert math.isclose(
        compute_rate(0.5, cfg, nu0=5e12), 5e12 * math.exp(-0.5 / (p.kb * 300.0))
    )


def test_htst_falls_back_to_k0_when_nu0_none() -> None:
    """HTST style falls back to k0 when nu0 is None."""
    cfg = _cfg("htst", k0=10.0, T=300.0)
    p = PhysicalConstants()
    assert math.isclose(
        compute_rate(0.5, cfg, nu0=None), 10.0 * math.exp(-0.5 / (p.kb * 300.0))
    )


def test_htst_falls_back_when_nu0_not_finite() -> None:
    """HTST style falls back to k0 when nu0 is non-finite (inf/nan)."""
    cfg = _cfg("htst", k0=10.0, T=300.0)
    p = PhysicalConstants()
    assert math.isclose(
        compute_rate(0.5, cfg, nu0=float("inf")),
        10.0 * math.exp(-0.5 / (p.kb * 300.0)),
    )

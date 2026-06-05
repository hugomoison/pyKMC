"""Pluggable prefactor solvers + RateConstant (pykmc.rate_constant)."""

import math

from pykmc.config import PhysicalConstants
from pykmc.rate_constant import (
    ConstantPrefactorSolver,
    HtstPrefactorSolver,
    RateConstant,
    RpaPrefactorSolver,
)

KB = PhysicalConstants().kb


def test_constant_solver_ignores_nu0() -> None:
    """ConstantPrefactorSolver always returns k0."""
    solver = ConstantPrefactorSolver(10.0)
    assert solver.prefactor(nu0=5.0e12) == 10.0
    assert solver.prefactor() == 10.0


def test_htst_solver_uses_nu0_else_k0() -> None:
    """HTST returns nu0 when finite, otherwise falls back to k0."""
    solver = HtstPrefactorSolver(10.0)
    assert solver.prefactor(nu0=5.0e12) == 5.0e12
    assert solver.prefactor(nu0=None) == 10.0
    assert solver.prefactor(nu0=float("nan")) == 10.0


def test_rpa_solver_is_bare_vineyard_for_now() -> None:
    """RPA is HTST until kappa lands (subclass, same prefactor)."""
    assert isinstance(RpaPrefactorSolver(10.0), HtstPrefactorSolver)
    assert RpaPrefactorSolver(10.0).prefactor(nu0=7.0e12) == 7.0e12


def test_rate_constant_compute() -> None:
    """RateConstant.compute = prefactor * exp(-dE / (kb T))."""
    rc = RateConstant(HtstPrefactorSolver(10.0), 100.0)
    assert math.isclose(rc.compute(0.5, nu0=5.0e12), 5.0e12 * math.exp(-0.5 / (KB * 100.0)))
    assert math.isclose(rc.compute(0.5, nu0=None), 10.0 * math.exp(-0.5 / (KB * 100.0)))

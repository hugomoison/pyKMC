"""Per-reference-event Vineyard ν₀ orchestration (engine-agnostic).

Given a forces callable and the min1/saddle/min2 geometry of an event, computes
the forward and backward Vineyard prefactors ν₀ (Hz) from frozen-boundary
partial Hessians. Never raises: any failure (bad saddle spectrum, out-of-bounds
ν₀, engine error) yields None prefactors plus a diagnostic ``reason`` so the
caller can fall back to the constant k0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from .free_region import select_free_indices
from .hessian import mass_weighted_partial_hessian
from .vineyard import vineyard_prefactor


@dataclass
class EventPrefactors:
    """Result of a per-event ν₀ computation (both directions, Hz)."""

    nu0_forward: Optional[float]
    nu0_backward: Optional[float]
    n_free: int
    n_neg_saddle: int
    ok_forward: bool
    ok_backward: bool
    reason: str


def _masses_for(types: list[str]) -> np.ndarray:
    """Return per-atom masses (amu) looked up from ASE atomic data."""
    return np.array([atomic_masses[atomic_numbers[t]] for t in types], dtype=float)


def compute_event_prefactors(
    forces_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    min1: np.ndarray,
    saddle: np.ndarray,
    min2: np.ndarray,
    types: list[str],
    central_index: int,
    free_radius: float,
    fd_step: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    nu0_min_hz: float,
    nu0_max_hz: float,
    require_one_negative_mode: bool,
    *,
    hessian_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> EventPrefactors:
    """Compute forward/backward ν₀ (Hz) for one reference event. Never raises.

    Parameters
    ----------
    forces_fn : Callable
        Maps full (N, 3) positions to full (N, 3) forces (engine-bound by caller).
    min1, saddle, min2 : np.ndarray
        (N, 3) positions of the initial minimum, saddle, and final minimum.
    types : list[str]
        Per-atom chemical symbols; masses are looked up via ASE.
    central_index : int
        Index of the moving atom (center of the free region).
    free_radius : float
        Radius (Angstrom) defining the free (movable) atoms.
    fd_step : float
        Finite-difference step (Angstrom).
    cell, pbc : np.ndarray
        Simulation cell (orthogonal) and per-axis periodicity.
    nu0_min_hz, nu0_max_hz : float
        Acceptance window (Hz); ν₀ outside it is rejected (-> None).
    require_one_negative_mode : bool
        Reserved. ``vineyard_prefactor`` already requires exactly one negative
        saddle mode in v1, so a bad saddle always falls back regardless.
    hessian_fn : Callable, optional
        ``(positions, free_indices) -> mass-weighted Hessian`` [eV/(amu·Å²)].
        When given, supplies the Hessian directly (e.g. LAMMPS
        ``dynamical_matrix``) and ``forces_fn`` may be None; otherwise the
        finite-difference Hessian built from ``forces_fn`` is used.

    Returns
    -------
    EventPrefactors
        nu0_forward / nu0_backward are None on any failure (with ``reason`` set).

    Notes
    -----
    **n_zero_modes=0**: The partial Hessian is built from a frozen-boundary
    cluster — boundary atoms are held fixed during finite differences. With a
    fixed boundary, translating the free cluster costs energy, so there are NO
    translational zero modes to project out. Passing n_zero_modes=0 here is
    therefore correct; do NOT use the default of 3 (which applies to clean
    periodic slabs with true translational invariance).

    """
    masses = _masses_for(types)
    free = select_free_indices(min1, central_index, free_radius, cell, pbc)

    if hessian_fn is None:
        # Default: finite-difference mass-weighted partial Hessian from forces.
        def _fd_hessian(geom: np.ndarray, free_idx: np.ndarray) -> np.ndarray:
            return mass_weighted_partial_hessian(forces_fn, geom, masses, free_idx, fd_step)

        hessian = _fd_hessian
    else:
        hessian = hessian_fn

    def _bounded(nu0: float) -> Optional[float]:
        if not np.isfinite(nu0) or nu0 < nu0_min_hz or nu0 > nu0_max_hz:
            return None
        return nu0

    try:
        h_min1 = hessian(min1, free)
        h_sad = hessian(saddle, free)
        h_min2 = hessian(min2, free)
        # Frozen-boundary partial Hessian -> no translational zero modes -> n_zero_modes=0.
        nu0_f = _bounded(vineyard_prefactor(h_min1, h_sad, n_zero_modes=0))
        nu0_b = _bounded(vineyard_prefactor(h_min2, h_sad, n_zero_modes=0))
        ok_f = nu0_f is not None
        ok_b = nu0_b is not None
        return EventPrefactors(
            nu0_forward=nu0_f,
            nu0_backward=nu0_b,
            n_free=len(free),
            n_neg_saddle=-1,
            ok_forward=ok_f,
            ok_backward=ok_b,
            reason="" if (ok_f and ok_b) else "nu0 out-of-bounds or non-finite",
        )
    except Exception as exc:  # bad saddle spectrum, non-PD min, engine error, ...
        return EventPrefactors(
            nu0_forward=None,
            nu0_backward=None,
            n_free=len(free),
            n_neg_saddle=-1,
            ok_forward=False,
            ok_backward=False,
            reason=f"{type(exc).__name__}: {exc}",
        )

"""Harmonic Transition State Theory (HTST) Vineyard prefactor subsystem.

Pure-NumPy mode analysis is vendored from
``apps/PyKMC_Analysis/Analysis/htst/kappa_rpa.py`` (see module headers for
provenance). Engine binding lives in
``pykmc.enginemanager.lmpi.lammps_operations``.
"""

from .constants import HBAR_EV_S, HBAR_OMEGA_EV, KB_EV_PER_K, ZERO_MODE_TOL_EV2
from .free_region import select_free_indices
from .hessian import mass_weighted_partial_hessian
from .normal_modes import normal_modes_from_hessian
from .prefactor import EventPrefactors, compute_event_prefactors
from .vineyard import vineyard_prefactor

__all__ = [
    "HBAR_OMEGA_EV",
    "HBAR_EV_S",
    "ZERO_MODE_TOL_EV2",
    "KB_EV_PER_K",
    "select_free_indices",
    "mass_weighted_partial_hessian",
    "normal_modes_from_hessian",
    "vineyard_prefactor",
    "EventPrefactors",
    "compute_event_prefactors",
]

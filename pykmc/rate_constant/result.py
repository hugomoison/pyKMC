"""Requests, outcomes and error codes of per-event prefactors."""

from dataclasses import dataclass
from enum import auto
from typing import Optional

import numpy as np

from pykmc._core.result import ErrorCode


@dataclass(frozen=True)
class PrefactorRequest:
    """Geometry of one new event, for a strategy with a prefactor per event.

    Attributes
    ----------
    saddle : np.ndarray
        Saddle positions of the whole system, shape (N, 3).
    minima : list[np.ndarray]
        Positions of the whole system at each minimum: the initial one first, then
        the final one for a reference event that has a backward direction.
    centre : int
        Index of the atom the search or the refinement was centred on.
    reference : bool
        True for a reference event, False for a refined active event.
    names : list[str]
        How each direction (one per minimum) is named in the log.

    """

    saddle: np.ndarray
    minima: list[np.ndarray]
    centre: int
    reference: bool
    names: list[str]


@dataclass(frozen=True)
class PrefactorOutcome:
    """Prefactor of one direction of an event.

    Attributes
    ----------
    value : float | None
        Prefactor (ps^-1), or None: the event then uses the strategy's fallback.
    status : str
        'ok', or the key of the error that left the event without a value.

    """

    value: Optional[float]
    status: str


class PrefactorError(ErrorCode):
    """Reasons an event gets no per-event prefactor and uses the fallback."""

    ENGINE_FAILURE = auto()
    NO_FROZEN_BOUNDARY = auto()
    SADDLE_NOT_FIRST_ORDER = auto()
    UNSTABLE_MINIMUM = auto()
    NONFINITE_PREFACTOR = auto()
    FROZEN_OVERLAP = auto()
    WRAPPED_OPEN_AXIS = auto()

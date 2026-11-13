"""Reasons an event gets no HTST prefactor."""

from enum import auto

from pykmc._core.result import ErrorCode


class PrefactorError(ErrorCode):
    """Reasons an event gets no nu0 and uses `k0`."""

    ENGINE_FAILURE = auto()
    NO_FROZEN_BOUNDARY = auto()
    SADDLE_NOT_FIRST_ORDER = auto()
    UNSTABLE_MINIMUM = auto()
    NONFINITE_PREFACTOR = auto()
    FROZEN_OVERLAP = auto()
    WRAPPED_OPEN_AXIS = auto()

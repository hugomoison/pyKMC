"""Outputs and error codes for the rate constant."""

from dataclasses import dataclass
from enum import auto

from pykmc._core.result import ErrorCode


class RateConstantError(ErrorCode):
    """Failures the rate constant module can report.

    Every error carries a backup in `ErrorInfo.variables["backup"]`: a
    `RateConstantOutput` computed with the strategy's default prefactor `k0`.
    """

    COMPUTATION_FAILED = auto()


@dataclass
class RateConstantOutput:
    """Store the rate constant of one hop.

    Attributes
    ----------
    k : float
        The rate constant (ps^-1).
    prefactor : float
        The pre-exponential factor used to compute `k` (ps^-1).

    """

    k: float
    prefactor: float

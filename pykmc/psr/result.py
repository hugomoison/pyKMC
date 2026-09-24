"""Outputs and error codes for point set registration."""

from dataclasses import dataclass
from enum import auto
from typing import Optional

import numpy as np

from pykmc._core.result import ErrorCode


class PSRError(ErrorCode):
    """Failures the point set registration module can report."""

    NO_MATCH_FOUND = auto()
    MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD = auto()


@dataclass
class PSROutput:
    """Store the result of a point set registration operation.

    ``rotation_matrix``, ``translation_matrix`` and ``permutation_matrix`` default
    to ``None``: a strategy used only to check whether a match exists (e.g. event
    deduplication) may skip computing the alignment and report just the score.

    Attributes
    ----------
    matching_score : float
        Score representing the quality of the match.
    rotation_matrix : Optional[np.ndarray]
        Rotation matrix used to align two patterns.
    translation_matrix : Optional[np.ndarray]
        Translation vector applied for alignment.
    permutation_matrix : Optional[np.ndarray]
        Mapping of atom indices from reference to current configuration.

    """

    matching_score: float
    rotation_matrix: Optional[np.ndarray] = None
    translation_matrix: Optional[np.ndarray] = None
    permutation_matrix: Optional[np.ndarray] = None

"""IRA point set registration strategy."""

from typing import Protocol

import ira_mod

from ...result import Err, ErrorInfo, Ok, Result
from ..result import PSROutput, PSRError
from .base import PSRStrategy


class IRAConfig(Protocol):
    """Configuration required by `IRAStrategy`."""

    kmax_factor: float


class IRAStrategy(PSRStrategy):
    """Point set registration using the IRA (Iterative Rotations and Assignments) algorithm."""

    name = "ira"

    def __init__(self, config: IRAConfig) -> None:
        self.config = config

    def match(
        self,
        nat1: int,
        typ1: list,
        coords1,
        nat2: int,
        typ2: list,
        coords2,
        candidate1=None,
        candidate2=None,
    ) -> Result[PSROutput, ErrorInfo]:
        """Run IRA to extract rotation, translation and permutation matrices.

        `candidate1`/`candidate2` are forwarded to IRA as the imposed central
        atoms of each point set (geometric center is used when None).

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The results of the ira psr procedure.
            Transformation's direction convention (coords2 -> coords1).

        """
        ira = ira_mod.IRA()
        try:
            rmat, tr, perm, dh = ira.match(
                nat1,
                typ1,
                coords1,
                nat2,
                typ2,
                coords2,
                self.config.kmax_factor,
                candidate1=candidate1,
                candidate2=candidate2,
            )
            return Ok(
                PSROutput(
                    rotation_matrix=rmat,
                    translation_matrix=tr,
                    permutation_matrix=perm,
                    matching_score=dh,
                )
            )
        except Exception:
            return Err(
                ErrorInfo(
                    type=PSRError.NO_MATCH_FOUND,
                    message="IRA did not find a match",
                )
            )

"""Manages Point Set Registration (shape matching) methods."""

import ira_mod
from ..result import Result, ErrorInfo, Ok, Err
from .result import PSROutput, PSRError


class PointSetRegistration:
    """Perform a point set registration between two point sets, based on the given style.

    Operates on plain point-cloud arrays only. Preparing those arrays from a
    `System`/reference event -- extracting a local neighborhood, colouring atom
    types (including the "grey alloy" species-blind case), unwrapping across
    periodic boundaries, ... -- is the caller's job.

    Parameters
    ----------
    style : str
        Point set registration style to use (e.g. "ira").
    kmax_factor : float
        Passed to the underlying registration algorithm.

    """

    def __init__(self, style: str, kmax_factor: float) -> None:
        self.style = style
        self.kmax_factor = kmax_factor

    def match(
        self, nat1, typ1, coords1, nat2, typ2, coords2
    ) -> Result[PSROutput, ErrorInfo]:
        """Run the point set registration based on the configured style.

        Parameters
        ----------
        nat1 : int
            Number of atoms in the first point set.
        typ1 : list[str]
            Atom types of the first point set.
        coords1 : np.ndarray
            Positions of the first point set.
        nat2 : int
            Number of atoms in the second point set.
        typ2 : list[str]
            Atom types of the second point set.
        coords2 : np.ndarray
            Positions of the second point set.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            Results of the point set registration.

        Raises
        ------
        Exception
            If the style in not known.

        """
        match self.style:
            case "ira":
                return self.ira(nat1, typ1, coords1, nat2, typ2, coords2)
            case _:
                raise Exception("Point set registration style unknown")

    def ira(self, nat1, typ1, coords1, nat2, typ2, coords2) -> Result[PSROutput, ErrorInfo]:
        """Use IRA to extract rotation, translation, permutation matrix to apply on generic event.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The results of the ira psr procedure.

        """
        return simple_ira(nat1, typ1, coords1, nat2, typ2, coords2, self.kmax_factor)


def check_match(
    result_match: Result[PSROutput, ErrorInfo], matching_score: float
) -> Result[PSROutput, ErrorInfo]:
    """Check if a result from the point set registration method is valid and gives a matching score lower than the matching score threshold defined in the configuration.

    Parameters
    ----------
    result_match : Result[PSROutput, ErrorInfo]
        Result of the PSR procedure.
    matching_score : float
        matching score threshold.

    Returns
    -------
    Result[PSROutput, ErrorInfo]
        Result of the check.

    """
    if not result_match.is_ok():
        return result_match  # ErrorInfo no match
    else:
        if result_match.ok_value().matching_score > matching_score:
            return Err(
                ErrorInfo(
                    type=PSRError.MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
                    message="PSR found a match but matching score is above acceptance threshold",
                    details="Hausdorff distance = {}, acceptance threshold = {} ".format(
                        result_match.ok_value().matching_score, matching_score
                    ),
                    variables={
                        "matching_score": result_match.ok_value().matching_score
                    },
                )
            )

        else:
            return result_match  # Ok(PSROutput)


def simple_ira(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor):
    # Run ira to find transformation matrices
    ira = ira_mod.IRA()
    try:
        rmat, tr, perm, dh = ira.match(
            nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor
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

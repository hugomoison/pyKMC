"""Manages Point Set Registration (shape matching) methods."""

from ..result import Err, ErrorInfo, Result
from .result import PSROutput, PSRError
from .strategies import PSRStrategy


class PointSetRegistration:
    """Perform a point set registration between two point sets, using a pluggable strategy.

    Operates on plain point-cloud arrays only. Preparing those arrays from a
    `System`/reference event -- extracting a local neighborhood, colouring atom
    types (including the "grey alloy" species-blind case), unwrapping across
    periodic boundaries, ... -- is the caller's job.

    Parameters
    ----------
    strategy : PSRStrategy
        The point set registration strategy to use.

    """

    def __init__(self, strategy: PSRStrategy) -> None:
        self._strategy = strategy

    def match(
        self, nat1, typ1, coords1, nat2, typ2, coords2
    ) -> Result[PSROutput, ErrorInfo]:
        """Register two point sets using the configured strategy.

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

        """
        return self._strategy.match(nat1, typ1, coords1, nat2, typ2, coords2)

    @classmethod
    def create(cls, style: str, **kwargs) -> "PointSetRegistration":
        """Build a PointSetRegistration wired with the requested strategy.

        Parameters
        ----------
        style : str
            Name of the registered strategy (e.g. "ira").
        **kwargs
            Forwarded to the strategy's constructor (e.g. ``config=config.ira``).

        """
        return cls(strategy=PSRStrategy.create(style, **kwargs))


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

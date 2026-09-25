"""Manages Point Set Registration (shape matching) methods."""

from ..result import Err, ErrorInfo, Result
from .result import PSROutput, PSRError
from .strategies import PSRStrategy


class PointSetRegistration:
    """Perform a point set registration between two point sets, using a pluggable strategy.

    Operates on plain point-cloud arrays only. 

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
        """Try to match two point sets using the configured strategy.

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

    def match_and_check(
        self, nat1, typ1, coords1, nat2, typ2, coords2, matching_score_thr: float
    ) -> Result[PSROutput, ErrorInfo]:
        """Match two point sets and reject the result if its score is above threshold.

        Parameters
        ----------
        nat1, typ1, coords1, nat2, typ2, coords2
            Forwarded to `match`.
        matching_score_thr : float
            Maximum acceptable matching score.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The match, or an error if no match was found or its score is above
            `matching_score_thr`.

        """
        result = self.match(nat1, typ1, coords1, nat2, typ2, coords2)
        if not result.is_ok():
            return result
        if result.ok_value().matching_score > matching_score_thr:
            return Err(
                ErrorInfo(
                    type=PSRError.MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
                    message="PSR found a match but matching score is above acceptance threshold",
                    details="Hausdorff distance = {}, acceptance threshold = {} ".format(
                        result.ok_value().matching_score, matching_score_thr
                    ),
                    variables={"matching_score": result.ok_value().matching_score},
                )
            )
        return result

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

"""Manages Point Set Registration (shape matching) methods."""

import ira_mod
from .result import Result, ErrorInfo, PSROutput, Ok, Err, ErrorType
from .parameters import Parameters
from .system import System, Configuration
import pandas as pd
from .neighbors_list import NeighborsList
from .utils.geometry import unwrap_around


class PointSetRegistration:
    """Perform a point set registration between a reference event and an atomic environment of an atom based on the configuration parameters.

    Parameters
    ----------
    params : Parameters
        The configuration.
    system : System
        The atomic system.
    dfevent : pd.Series
        The reference event.
    neighbors_list : NeighborsList
        The NeighborsList of the System.
    central_atom_index : int
        Index of the central atom in the System for which we want to perfrom the point set registration.

    """

    def __init__(
        self,
        params: Parameters,
        system: System,
        dfevent: pd.Series,
        neighbors_list: NeighborsList,
        central_atom_index: int,
    ) -> None:
        self.system = system
        self.params = params
        self.dfevent = dfevent
        self.neighbors_list = neighbors_list
        self.central_atom_index = central_atom_index
        self.psr_style = self.params.psr.style

    def match(self) -> Result[PSROutput, ErrorInfo]:
        """Run the point set registration based on the style defined in the configuration.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            Results of the point set registration.

        Raises
        ------
        Exception
            If the style in not known.

        """
        match self.psr_style:
            case "ira":
                return self.ira(self.central_atom_index)
            case _:
                raise Exception("Point set registration style unknown")

    def ira(self, central_atom_index: int) -> Result[PSROutput, ErrorInfo]:
        """Use IRA to extract rotation, translation, permutation matrix to apply on generic event.

        Parameters
        ----------
        central_atom_index : int
           index of the system's central atom

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The results of the ira psr procedure.

        """
        initial_configuration = self.dfevent.at["initial_configuration"]

        # atoms around the central atom, exactly rcut (see docstring)
        neighbor_list = self.neighbors_list.get_neighbors(
            "rcut", central_atom_index
        ).copy()
        live_configuration = unwrap_around(
            self.system.configuration[neighbor_list],
            self.system.positions[central_atom_index],
        )

        return simple_ira(
            live_configuration,
            initial_configuration,
            self.params.ira.kmax_factor,
            full=self.params.atomicenvironment.coloring_mode == "full",
            candidate1=neighbor_list.index(central_atom_index),
            candidate2=int(self.dfevent.at["move_atom_idx"]),
        )


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
                    type=ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
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


def simple_ira(
    configuration_1: Configuration,
    configuration_2: Configuration,
    kmax_factor: float,
    full: bool = False,
    candidate1: int | None = None,
    candidate2: int | None = None,
) -> Result[PSROutput, ErrorInfo]:
    """Run IRA between two `Configuration`s; `full` gates real vs. dummy ("X") types.

    `candidate1`/`candidate2`, if given, seed the search with a known
    matching atom pair (e.g. each configuration's central atom).
    """
    nat1 = len(configuration_1)
    nat2 = len(configuration_2)
    typ1 = list(configuration_1.types) if full else nat1 * ["X"]
    typ2 = list(configuration_2.types) if full else nat2 * ["X"]
    candidate_kwargs = {"candidate1": candidate1, "candidate2": candidate2} if candidate1 is not None else {}
    # Run ira to find transformation matrices
    ira = ira_mod.IRA()
    try:
        rmat, tr, perm, dh = ira.match(
            nat1, typ1, configuration_1.positions, nat2, typ2, configuration_2.positions, kmax_factor, **candidate_kwargs
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
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="IRA did not find a match",
            )
        )


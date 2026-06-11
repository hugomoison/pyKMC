"""Manages Point Set Registration (shape matching) methods."""

import ira_mod
import numpy as np
from .result import Result, ErrorInfo, PSROutput, Ok, Err, ErrorType
from .config import Config
from .system import System
import pandas as pd
from .neighbors_list import NeighborsList


class PointSetRegistration:
    """Perform a point set registration between a reference event and an atomic environment of an atom based on the configuration parameters.

    Parameters
    ----------
    config : Config
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
        config: Config,
        system: System,
        dfevent: pd.Series,
        neighbors_list: NeighborsList,
        central_atom_index: int,
    ) -> None:
        self.system = system
        self.config = config
        self.dfevent = dfevent
        self.neighbors_list = neighbors_list
        self.central_atom_index = central_atom_index
        self.psr_style = self.config.psr.style

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
        # Initialize IRA
        ira = ira_mod.IRA()

        # Event informations :
        coords2 = self.dfevent.at["initial_positions"]
        nat2 = len(coords2)

        # atom in the rcutevent around the central atom
        neighbor_list = self.neighbors_list.get_neighbors("rcut", central_atom_index)

        coords1 = self.system.positions[neighbor_list]

        # GREY ALLOY
        typ1 = ["X"] * len(coords1)
        typ2 = typ1

        # typ1 = np.array(self.system.types)[neighbor_list]

        # typ2 = typ1  # If they have same topology id should be always true ?

        # unwrap if close to cell limits :
        alat = self.system.cell[0][0]
        for i in range(len(coords1)):
            if (
                np.linalg.norm(
                    coords1[i][0] - self.system.positions[central_atom_index][0]
                )
                > alat / 2
            ):
                coords1[i][0] = (
                    coords1[i][0]
                    + np.sign(
                        self.system.positions[central_atom_index][0] - coords1[i][0]
                    )
                    * alat
                )
            if (
                np.linalg.norm(
                    coords1[i][1] - self.system.positions[central_atom_index][1]
                )
                > alat / 2
            ):
                coords1[i][1] = (
                    coords1[i][1]
                    + np.sign(
                        self.system.positions[central_atom_index][1] - coords1[i][1]
                    )
                    * alat
                )
            if (
                np.linalg.norm(
                    coords1[i][2] - self.system.positions[central_atom_index][2]
                )
                > alat / 2
            ):
                coords1[i][2] = (
                    coords1[i][2]
                    + np.sign(
                        self.system.positions[central_atom_index][2] - coords1[i][2]
                    )
                    * alat
                )
        nat1 = len(coords1)
        kmax_factor = self.config.ira.kmax_factor

        # Run ira to find transformation matrices
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
                    type=ErrorType.PSR_NO_MATCH_FOUND,
                    message="IRA did not find a match",
                )
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
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="IRA did not find a match",
            )
        )

"""Module to reconstruct an event from saddle positions"""

from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Parameters, Configuration
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
from pykmc.utils.geometry import push_towards, compute_delr_max, wrap_configuration

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method


class Reconstruction:
    def __init__(self, params: Parameters, manager: Manager) -> None:
        self.params = params
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions

    def reconstruct(
        self,
        supposed_min1: Configuration,
        supposed_min2: Configuration,
        saddle: Configuration,
        delr_thr,
        neighbors=None,
    ):
        """From a saddle point, try to reconstruct the event to see if it matches the
        supposed min1 and min2 positions, and that the to minima are connected.

        Since we generaly save only the atomic environment of the central atom
        we can specified neighbors which correspond to the list index of atoms in
        saddle positions that we need to modifie to go toward min pos.

        The reconstruction procede as follow :
        From the saddle positions
        Move the system toward the first minimum (with fraction)
        Minimize and compare minimized positions with supposed min1 positions
        same for min2


        Parameters
        ----------
        supposed_min1 : Configuration
            The hypothesized first minimum's types/positions/cell (typically a
            local, neighbor-indexed cluster sharing `saddle`'s cell).
        supposed_min2 : Configuration
            The hypothesized second minimum's types/positions/cell.
        saddle : Configuration
            The saddle point's types/positions/cell.
        delr_thr : _type_
            _description_
        neighbors : _type_, optional
            _description_, by default None
            typically the neighors list of the in the atomic environment of the atom on which we apply the event
        """
        if neighbors is None:  # len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle))

        # Move toward min1 positions
        push_configuration = saddle.copy()
        push_configuration[neighbors] = push_towards(
            saddle[neighbors],
            supposed_min1,
            fraction=self.params.reconstruction.push_fraction,
        )
        min1_configuration, _ = self.manager.global_minimize_with_results(
            self.params, configuration=push_configuration
        )

        # compaire min1_configuration with system current positions
        delr1 = compute_delr_max(
            supposed_min1,
            wrap_configuration(min1_configuration)[neighbors],
        )  # I guess we need to be carefull here, if atom_modify sort 0 it's ok
        if delr1 > delr_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                    message="did not retreive initial minimum : delr1 = {}".format(
                        delr1
                    ),
                    variables={"delr1": delr1},
                )
            )
        else:
            # positions towards min2 :
            push_configuration = saddle.copy()
            push_configuration[neighbors] = push_towards(
                saddle[neighbors],
                supposed_min2,
                fraction=self.params.reconstruction.push_fraction,
            )
            min2_configuration, min2_etot = self.manager.global_minimize_with_results(
                self.params, configuration=push_configuration
            )

            # Compare min2_configuration with expected final_positions
            delr2 = compute_delr_max(
                supposed_min2,
                wrap_configuration(min2_configuration)[neighbors],
            )
            if delr2 > delr_thr:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message=f"did not retreive expected final minimum : delr2 = {delr2}",
                        variables={"delr2": delr2},
                    )
                )

            else:
                return Ok(
                    ReconstructionOutput(
                        min1_configuration=min1_configuration,
                        saddle_configuration=saddle,
                        min2_configuration=min2_configuration,
                        min2_etot=min2_etot,
                    )
                )

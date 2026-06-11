"""Module to reconstruct an event from saddle positions"""

from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
import copy
from pykmc.utils.geometry import push_towards, compute_delr
import ase.geometry

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method


class Reconstruction:
    def __init__(self, config: Config, manager: Manager, types=None) -> None:
        self.config = config
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions
        self.types = types

    def reconstruct(
        self,
        supposed_min1_positions,
        supposed_min2_positions,
        saddle_positions,
        cell,
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
        supposed_min1_positions : _type_
            _description_
        supposed_min2_positions : _type_
            _description_
        saddle_positions : _type_
            _description_
        central_atom : _type_
            _description_
        cell :
        neighbors : _type_, optional
            _description_, by default None
            typically the neighors list of the in the atomic environment of the atom on which we apply the event
        """

        if neighbors is None:  # len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        # Saddle positions
        tmp_positions = copy.deepcopy(saddle_positions)

        # Move toward min1 positions
        saddle_toward_min1_pos = push_towards(
            saddle_positions[neighbors],
            supposed_min1_positions,
            fraction=self.config.reconstruction.push_fraction,
            cell=cell,
        )
        tmp_positions[neighbors] = saddle_toward_min1_pos
        # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
        min1_pos, _ = self.manager.global_minimize_with_results(
            self.config, positions=tmp_positions, types=self.types
        )
        #        min1_pos, _ = future.result()

        # compaire min1_pos with system current positions
        t1 = ase.geometry.wrap_positions(positions=min1_pos, cell=cell, pbc=True)
        delr1 = compute_delr(
            supposed_min1_positions, t1[neighbors], cell
        )  # I guess we need to be carefull here, if atom_modify sort 0 it's ok
        if delr1 > self.config.psr.matching_score_thr:
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
            saddle_toward_min2_pos = push_towards(
                saddle_positions[neighbors],
                supposed_min2_positions,
                fraction=self.config.reconstruction.push_fraction,
                cell=cell,
            )
            tmp_positions[neighbors] = saddle_toward_min2_pos
            # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
            min2_pos, min2_etot = self.manager.global_minimize_with_results(
                self.config, positions=tmp_positions, types=self.types
            )
            #            min2_pos, _ = future.result()

            # Compare min2pos with expected final_positions
            t2 = ase.geometry.wrap_positions(positions=min2_pos, cell=cell, pbc=True)
            # delr2 = compute_delr(supposed_min2_positions, min2_pos[neighbors], cell)
            delr2 = compute_delr(supposed_min2_positions, t2[neighbors], cell)
            if delr2 > self.config.psr.matching_score_thr:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {}".format(
                            delr2
                        ),
                        variables={"delr2": delr2},
                    )
                )

            else:
                return Ok(
                    ReconstructionOutput(
                        min1_positions=min1_pos,
                        saddle_positions=saddle_positions,
                        min2_positions=min2_pos,
                        min2_etot=min2_etot,
                    )
                )

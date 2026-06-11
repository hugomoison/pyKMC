"""
Old KMC methods, will serve as template when histo will be implemented
"""


def _apply_event_generic(self, idx_atom_apply_event, idx_event_table):
    """ """
    if self.config.control.reconstruction:
        rmat, tr, perm, dh = PointSetRegistration(
            self.config,
            self.system,
            self.reference_table,
            self.neighbors_list,
            idx_event_table,
            idx_atom_apply_event,
        ).match()
        if rmat is None or dh > self.config.psr.matching_score_thr:
            return False
        else:
            current_positions = self.system.positions.copy()
            # initial potential energy
            Eini = self.engine.compute_potential_energy(self.system)
            # go to saddle point
            neighbors = self.neighbors_list.get_neighbors("rcut", idx_atom_apply_event)
            new_positions = np.zeros(
                (
                    len(
                        self.reference_table.table.loc[idx_event_table].at[
                            "saddle_positions"
                        ]
                    ),
                    3,
                )
            )
            for i in range(len(new_positions)):
                new_positions[i] = (
                    np.matmul(
                        rmat,
                        self.reference_table.table.loc[idx_event_table].at[
                            "saddle_positions"
                        ][i],
                    )
                    + tr
                )
            new_positions[:] = new_positions[perm]
            self.system.update_positions(new_positions, atom_idx=neighbors)
            # saddle potential energy
            Esad = self.engine.compute_potential_energy(self.system)
            # check if energy barrier consistent :
            dE = Esad - Eini
            if (
                abs(
                    dE
                    - self.reference_table.table.loc[idx_event_table]["energy_barrier"]
                )
                < 0.5
            ):
                new_positions = np.zeros(
                    (
                        len(
                            self.reference_table.table.loc[idx_event_table].at[
                                "final_positions"
                            ]
                        ),
                        3,
                    )
                )
                for i in range(len(new_positions)):
                    new_positions[i] = (
                        np.matmul(
                            rmat,
                            self.reference_table.table.loc[idx_event_table].at[
                                "final_positions"
                            ][i],
                        )
                        + tr
                    )
                new_positions[:] = new_positions[perm]
                self.system.update_positions(new_positions, atom_idx=neighbors)
                return True
            else:
                # back to current positions
                self.system.update_positions(current_positions)
                return False

    else:
        # neigbors of central atoms :
        neighbors = self.neighbors_list.get_neighbors("rcut", idx_atom_apply_event)
        final_positions = self.reference_table.table.loc[idx_event_table].at[
            "final_positions"
        ]
        # updat positions :
        self.system.update_positions(final_positions, atom_idx=neighbors)


def _select_event_generic(self):
    """ """
    # Find all possible event
    if self.config.control.reconstruction:
        l_env = list(set(self.atomic_environment.atomic_environment_list))
        if l_env == ["crystal"]:
            self._close()
        l_reference_table = [
            i
            for i in range(len(self.reference_table.table))
            if self.reference_table.table.loc[i].at["event_id"] in l_env
        ]
    else:  # all events in reference events are possible
        l_reference_table = [i for i in range(len(self.reference_table.table))]
    # Get constant rate of possible events
    l_k = np.array(
        [
            self.reference_table.table.loc[l_reference_table[i]].at["k"]
            for i in range(len(l_reference_table))
        ]
    )
    # Apply algorithm select event :
    idx_selected_event, delta_t = rejection_free(l_k)
    return l_reference_table[idx_selected_event], delta_t


def _select_central_atom_idx(self, idx_event_table):
    """ """
    if self.config.control.reconstruction:
        id_hash = self.reference_table.table.loc[idx_event_table].at["event_id"]
        possible = [
            i
            for i, e in enumerate(self.atomic_environment.atomic_environment_list)
            if e == id_hash
        ]
        return random.choice(possible)
    else:
        return self.reference_table.table.loc[idx_event_table].at["atom_index"]

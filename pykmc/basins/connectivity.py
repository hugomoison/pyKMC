import pandas as pd

# TODO: See if separated StatesConnectivity and BasinsStateConnectivity is really usefull.
# TODO: See if self.gaph is needed, if only used for analysis we can remove it.


class StatesConnectivity:
    """
    Store connectivity DateFrame describing transitions between visited system states.

    This class stores a pandas DataFrame that records transitions discovered
    during basin exploration.
    Each row represents a directed transition from a `state` to a `state_connexion`,
    associated with a given generic event, symmetry, and transition rates.

    The class provides utilities to:
    - Append new transitions
    - Retrieve transitions leading to a given state
    - Convert data to tuple format
    - Reorder state indices (e.g., to keep transient states first)
    - Save, merge, and clear the table

    Attributes
    ----------
    df : pd.DataFrame
        Internal table containing transition information with columns:
        ['state', 'state_connexion', 'event_connexion', 'central_atom', 'sym',
         'transient', 'dE_forward', 'k_forward', 'dE_backward', 'k_backward'].
    graph : optional
        Placeholder for future graph-based operations (e.g., visualization, path finding).
    """

    def __init__(self) -> None:
        self.df = pd.DataFrame(
            columns=[
                "state",
                "state_connexion",
                "event_connexion",
                "central_atom",
                "sym",
                "transient",
                "dE_forward",
                "k_forward",
                "dE_backward",
                "k_backward",
            ]
        )
        self.graph = None  # Placeholder for later if needed to generate a connectivity graph from the dataframe, for path finding and vizualization

    def add_connectivity(
        self,
        state,
        state_connexion,
        event_connexion,
        central_atom,
        sym,
        transient,
        dE_forward,
        k_forward,
        dE_backward,
        k_backward,
    ) -> None:
        """
        Add a new connectivity entry to the internal DataFrame.

        Parameters
        ----------
        state : int
            Source state index.
        state_connexion : int
            Target state index reached after applying the event.
        event_connexion : int
            Index of the event in the reference event table.
        central_atom : int
            Atom on which the event is applied.
        sym : int
            Symmetry index of the event.
        transient : bool
            Whether the state_connexion is transient (True) or absorbing (False).
        dE_forward : float
            Forward energy barrier.
        k_forward : float
            Forward transition rate.
        dE_backward : float
            Backward energy barrier.
        k_backward : float
            Backward transition rate.
        """
        new_row = pd.DataFrame(
            [
                {
                    "state": state,
                    "state_connexion": state_connexion,
                    "event_connexion": event_connexion,
                    "central_atom": central_atom,
                    "sym": sym,
                    "transient": transient,
                    "dE_forward": dE_forward,
                    "k_forward": k_forward,
                    "dE_backward": dE_backward,
                    "k_backward": k_backward,
                }
            ]
        )
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_transition_to_state(
        self, target_state: int, as_tuples: bool = True, return_all: bool = False
    ) -> tuple | list[tuple] | pd.DataFrame:
        """Return the transition(s) leading to the specified target state.

        This method filters the internal connectivity dataframe to find all transitions
        that reach `target_state`. If `return_all` is False, only the transition
        starting from the smallest `state` is returned.

        Parameters
        ----------
        target_state : int
            Identifier of the target state.
        as_tuples : bool, optional (default=True)
            If True, return the transitions as a list of tuples
            (state, event_connexion, central_atom, sym).
            If False, return a sub pandas DataFrame of the connectivity DataFrame
        return_all : bool, optional (default=False)
            If True, return all possible transitions to `target_state`.
            If False, return only the first transition starting from the smallest `state`.

        Returns
        -------
        tuple or list[tuple] or pd.DataFrame
            - If `as_tuples` is True:
                - If `return_all` is False: returns a list containing a single tuple
                  (state, event_connexion, central_atom, sym) corresponding to the
                  transition from the smallest state.
                - If `return_all` is True: returns a list of tuples for all matching transitions.
            - If `as_tuples` is False:
                - Returns a pandas DataFrame containing the matching row(s).
        """

        # Find sub dataframe with state_connexion == target_state
        sub_df = self.df[self.df["state_connexion"] == target_state]

        if not return_all:
            # Return the first transition with lower from state
            min_state = sub_df["state"].min()
            sub_df = sub_df[sub_df["state"] == min_state].iloc[[0]]  # only first row
            return self.to_tuples(sub_df)[0] if as_tuples else sub_df
        else:
            return self.to_tuples(sub_df) if as_tuples else sub_df

    def get_table(self) -> pd.DataFrame:
        """
        Return the full connectivity table.

        Returns
        -------
        pd.DataFrame
        """
        return self.df

    def to_tuples(self, df: pd.DataFrame) -> list[tuple]:
        """Convert a DataFrame to a list of tuples.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The DataFrame to convert. If None, uses the full internal table.

        Returns
        -------
        list[tuple]
            Each tuple contains (state, event_connexion, central_atom, sym).
        """
        return list(
            df[
                ["state", "event_connexion", "central_atom", "sym", "transient"]
            ].itertuples(index=False, name=None)
        )

    def reorder_states_index(self) -> dict[int, int]:
        """
        Reassign state indices so that transient states come first and numbering is compact and continuous.

        Returns
        -------
        dict[int, int]
            Mapping from old state index to new state index.
        """

        # All states :
        unique_states = sorted(set(self.df["state"]) | set(self.df["state_connexion"]))

        # find transient states (all states in the state row)
        current_transient_states = sorted(self.df["state"].unique())

        # find absorbing states (all other)
        current_absorbing_states = list(
            set(unique_states).difference(current_transient_states)
        )

        # Create mapping
        mapping = {}
        new_idx = 0

        for idx in current_transient_states:
            mapping[idx] = new_idx
            new_idx += 1

        for idx in current_absorbing_states:
            mapping[idx] = new_idx
            new_idx += 1

        # Apply mapping
        self.df["state"] = self.df["state"].map(mapping)
        self.df["state_connexion"] = self.df["state_connexion"].map(mapping)

        return mapping

    def save(self, outfile: str = "basin_connectivity.pickle") -> None:
        """
        Save the connectivity DataFrame to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            Output filename. Default is 'basin_connectivity.pickle'.
        """
        self.df.to_pickle(outfile)

    def clear(self):
        """Reset the connectivity table to an empty DataFrame."""
        self.df = pd.DataFrame(
            columns=[
                "state",
                "state_connexion",
                "event_connexion",
                "central_atom",
                "sym",
                "transient",
                "dE_forward",
                "k_forward",
                "dE_backward",
                "k_backward",
            ]
        )


class BasinStatesConnectivity(StatesConnectivity):
    """Connectivity table with basin exploration specific operations"""

    def change_state_index(self, current_index: int, new_index: int):
        """Update all occurrences of a state index in the connectivity DataFrame.

        This method is used during basin exploration when a state that needs to be
        explored is found to be identical to a previously explored state. All rows
        in the connectivity DataFrame where `state` or `state_connexion` equals `current_index` are updated
        to `new_index`.

        Parameters
        ----------
        current_index : int
            The state index to be replaced.
        new_index : int
            The new state index to assign.
        """
        self.df.loc[self.df["state"] == current_index, "state"] = new_index
        self.df.loc[self.df["state_connexion"] == current_index, "state_connexion"] = (
            new_index
        )

    def change_state_to_absorbing(self, state_connexion):
        """
        Mark a state as absorbing in the connectivity table.

        Sets the `transient` flag to `False` for all transitions whose
        destination (`state_connexion`) equals the given index.
        This is mainly used when a state has unknown local atomic environments and should not be explored further.

        Parameters
        ----------
        state_connexion : int
            Index of the state to mark as absorbing.
        """
        self.df.loc[self.df["state_connexion"] == state_connexion, "transient"] = False

    def merge(self, states_connectivity: "StatesConnectivity"):
        """Merge another StatesConnectivity's DataFrame into this connectivity table.

        This method appends the DataFrame from `states_connectivity` to the internal
        connectivity table (`self.df`) of the current object. It is typically used
        during basin exploration when a new state has been explored.

        Parameters
        ----------
        states_connectivity : StatesConnectivity
            Another StatesConnectivity instance whose connectivity DataFrame (`df`)
            will be appended to this object's DataFrame.

        Notes
        -----
        - The index of the resulting DataFrame is reset.
        - This operation does not remove duplicates; ensure uniqueness if required.
        """
        if self.df.empty:
            self.df = states_connectivity.df.copy()
        else:
            self.df = pd.concat([self.df, states_connectivity.df], ignore_index=True)

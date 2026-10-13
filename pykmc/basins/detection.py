from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


class Detector(ABC):
    """Abstract base class for basin detection algorithms"""

    @abstractmethod
    def detect(self) -> bool:
        """Detect if current configuration is in a basin"""
        pass


class DetectorThreshold(Detector):
    def detect(
        self,
        pds_selected_active_event: pd.Series,
        df_reference_table: pd.DataFrame,
        energy_threshold: float,
        is_refined: Optional[bool] = False,
    ):
        """Check if the current configuration is in a basin.

        Returns True if the active event's barrier is below `energy_threshold`
        and if a corresponding backward event in the reference table also
        has a barrier below this threshold.

        Parameters
        ----------
        pds_selected_active_event : pd.Series
            A pandas Series of the selected active event.
        df_reference_table : pd.DataFrame
            A pandas DataFrame with all generic events.
        energy_threshold : float
            Energy threshold to considere the system in a basin.
        is_refined : Optional[bool]
            Whether `pds_selected_active_event` is an active-table row needing
            its own generic reference row looked up (via `num_reference_event`),
            or is already a reference-table row itself.
        """

        dE_forward = pds_selected_active_event["dE_forward"]

        if dE_forward >= energy_threshold:
            # not in a basin
            return False

        else:
            # Need to check if a backward reaction with low energy barrier exists.

            if is_refined:
                # case where we need to find the generic event from the active one
                idx_reference_event = pds_selected_active_event["num_reference_event"]

                # generic event of the active one
                # pds_generic_event_forward = df_reference_table.iloc[idx_reference_event]
                pds_generic_event_forward = df_reference_table[
                    df_reference_table["idx_ref"] == idx_reference_event
                ].iloc[0]  # is a pd.Serie

            else:
                pds_generic_event_forward = pds_selected_active_event

            # idx_backward already links this row to its real reverse event,
            # resolved once at cataloguing time (see ShapeTable.resolve_sid()) --
            # no need to re-derive a candidate via a coarse id_initial/id_final
            # hash filter plus a geometric re-confirmation. That re-derivation
            # used to compare final_configuration (built to share
            # initial_configuration's atom ordering, for displacement/
            # symmetry/reconstruction purposes -- see _build_event_series)
            # against candidate rows' own initial_configuration (built from
            # their own, generally differently-sized, natural neighbor list)
            # -- a comparison that can fail outright on atom-count mismatch
            # even for an already-known, correctly-catalogued backward event.
            backward_row_match = df_reference_table[
                df_reference_table["idx_ref"]
                == int(pds_generic_event_forward["idx_backward"])
            ]

            # Should always have one (reversibility)
            if backward_row_match.empty:
                raise ValueError(
                    "Basin detection: No backward event for the selected active event."
                )
            backward_row = backward_row_match.iloc[0]

            # Every catalogued pathway sharing the backward row's own decorated
            # shape is a candidate reverse reaction, not just this one row --
            # take the lowest barrier among them, exactly as before.
            df_backward_events = df_reference_table[
                (df_reference_table["id_initial"] == backward_row["id_initial"])
                & (df_reference_table["sid_initial"] == backward_row["sid_initial"])
            ]

            # Check if at least one backward event has a low energy barrier
            dE_backward = df_backward_events["dE_forward"].min()

            return dE_backward < energy_threshold

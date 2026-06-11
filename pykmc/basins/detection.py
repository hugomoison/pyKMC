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
        idx_reference_event : Optional[int]
            Index of the generic event in `df_reference_table` of the `pds_selected_active_event`.
        """

        dE_forward = pds_selected_active_event["energy_barrier"]

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

            # all possible generic backward events
            df_backward_events = df_reference_table[
                df_reference_table["event_id"] == pds_generic_event_forward["id_final"]
            ]

            # Should always have one (reversibility)
            if df_backward_events.empty:
                raise ValueError(
                    "Basin detection: No backward event for the selected active event."
                )

            # Check if at least one backward event has a low energy barrier
            dE_backward = df_backward_events["energy_barrier"].min()

            return dE_backward < energy_threshold

"""FPTA basin selector backed by the amsel Rust kernel.

Mirrors :class:`pykmc.basins.selection.FPTASelector` but delegates the
numerical work to `amsel <https://pypi.org/project/amsel>`_ -- a pure
Rust implementation of the corrected Ferasat 2020 two-step FPTA with
zero-copy DLPack tensor I/O and free-threaded Python wheels.

The API intentionally matches ``FPTASelector`` so that code using
:meth:`select_from_connectivity` can swap the class out with no other
changes. Behaviour is mathematically equivalent for well-conditioned
problems (condition number below ~1e11 in the current amsel v0.1
envelope); the Rust kernel is typically an order of magnitude faster
than the scipy path on superbasins of size 10 and above, and scales
linearly under free-threaded Python because the amsel kernel runs
without holding the GIL.

If ``amsel`` is not importable or rejects the problem (for example on
ill-conditioned matrices outside its f64 envelope) the selector
returns an ``ErrorInfo`` and the caller can fall back to
:class:`FPTASelector`.

References
----------
[1] Puchala, Falk, Garikipati, J. Chem. Phys. 132, 134104 (2010) --
    the original FPTA formulation.
[2] Ferasat et al., J. Chem. Phys. 153, 074109 (2020) -- the two-step
    correction amsel implements.
[3] `amsel <https://github.com/lode-org/amsel>`_ -- the Rust kernel.
"""
from __future__ import annotations

import numpy as np

from pykmc.result import (
    BasinSelectorOutput,
    Err,
    ErrorInfo,
    ErrorType,
    Ok,
    Result,
)

from .connectivity import StatesConnectivity


try:
    import amsel as _amsel

    _AMSEL_AVAILABLE = True
    _AMSEL_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only when amsel missing
    _amsel = None
    _AMSEL_AVAILABLE = False
    _AMSEL_IMPORT_ERROR = exc


class AmselFPTASelector:
    """FPTA selector with the Rust ``amsel`` kernel as its backend."""

    def __init__(self) -> None:
        # Retained for parity with FPTASelector; amsel builds its own
        # matrices internally and does not expose them.
        self.last_t_exit: float | None = None
        self.last_weights: np.ndarray | None = None

    def select_from_connectivity(
        self, connectivity_table: StatesConnectivity
    ) -> Result[BasinSelectorOutput, ErrorInfo]:
        """Find an exit time and exit state for the given basin.

        Parameters
        ----------
        connectivity_table
            :class:`StatesConnectivity` describing the superbasin's
            transient-to-transient and transient-to-absorbing edges.

        Returns
        -------
        Result[BasinSelectorOutput, ErrorInfo]
            Ok(BasinSelectorOutput) on success; Err on amsel absence,
            amsel kernel rejection, or degenerate input.
        """
        if not _AMSEL_AVAILABLE:
            return Err(
                ErrorInfo(
                    error_type=ErrorType.BasinExitTimeSolverError,
                    message=(
                        "amsel is not installed "
                        f"({_AMSEL_IMPORT_ERROR!r}); fall back to FPTASelector "
                        "or install amsel>=0.1"
                    ),
                )
            )

        transient, absorbing, rates = self._extract_graph(connectivity_table)
        if not transient:
            return Err(
                ErrorInfo(
                    error_type=ErrorType.BasinExitTimeSolverError,
                    message="no transient states in connectivity table",
                )
            )
        if not absorbing:
            return Err(
                ErrorInfo(
                    error_type=ErrorType.BasinExitTimeSolverError,
                    message="no absorbing states in connectivity table",
                )
            )

        # Entry is state 0 by pyKMC convention (see FPTASelector).
        entry = transient[0]

        # First random draw: r in (0, 1) for the FPTA bisection target.
        r1 = float(np.random.random())

        try:
            t_exit, weights = _amsel.fpta(
                transient=transient,
                absorbing=absorbing,
                rates=rates,
                entry=entry,
                r=r1,
            )
        except _amsel.AmselError as exc:
            return Err(
                ErrorInfo(
                    error_type=ErrorType.BasinExitTimeSolverError,
                    message=f"amsel kernel rejected problem: {exc}",
                )
            )

        # Second random draw: sample the exit state from the
        # (normalised) weights returned by amsel. Mirrors
        # FPTASelector.select_absorbing_state's np.searchsorted path.
        weights_arr = np.asarray(weights, dtype=np.float64)
        cumul = np.cumsum(weights_arr)
        r2 = float(np.random.random())
        idx = int(np.searchsorted(cumul, r2))
        idx = min(idx, len(absorbing) - 1)
        exit_state = absorbing[idx]

        self.last_t_exit = float(t_exit)
        self.last_weights = weights_arr

        return Ok(BasinSelectorOutput(t_exit=float(t_exit), exit_state=int(exit_state)))

    @staticmethod
    def _extract_graph(
        connectivity_table: StatesConnectivity,
    ) -> tuple[list[int], list[int], list[tuple[int, int, float]]]:
        """Flatten the StatesConnectivity DataFrame into amsel input form.

        Transient states are the unique values in the ``state`` column;
        absorbing states are the values appearing in ``state_connexion``
        but not in ``state``. Each DataFrame row contributes one
        ``(from, to, rate)`` edge using ``k_forward`` as the rate. The
        ordering of ``transient`` and ``absorbing`` lists is stable so
        repeated calls produce deterministic amsel indexing.
        """
        df = connectivity_table.df
        transient_set = set(int(s) for s in df["state"].to_numpy())
        connexion_set = set(int(s) for s in df["state_connexion"].to_numpy())
        absorbing_set = connexion_set - transient_set
        transient = sorted(transient_set)
        absorbing = sorted(absorbing_set)
        rates: list[tuple[int, int, float]] = []
        for _, row in df.iterrows():
            rates.append(
                (
                    int(row["state"]),
                    int(row["state_connexion"]),
                    float(row["k_forward"]),
                )
            )
        return transient, absorbing, rates

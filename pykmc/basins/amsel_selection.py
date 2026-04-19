"""Amsel-backed basin selector with optional adaptive clocking.

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
    """Basin selector with the Rust ``amsel`` kernel as its backend.

    Parameters
    ----------
    clock_mode
        ``"sampled"`` keeps the exact sampled FPTA clock, ``"mean"``
        forces the scalar MRM clock, and ``"adaptive"`` uses
        ``reduced_kinetics`` to choose between them.
    rank_tol
        Tolerance passed to ``ReducedKineticsResult.one_rate_clock_is_plausible``.
    """

    def __init__(self, clock_mode: str = "adaptive", rank_tol: float = 1.0e-8) -> None:
        if clock_mode not in {"sampled", "mean", "adaptive"}:
            raise ValueError("clock_mode must be one of 'sampled', 'mean', or 'adaptive'")
        self.last_t_exit: float | None = None
        self.last_weights: np.ndarray | None = None
        self.last_clock_mode: str | None = None
        self.last_reduced_kinetics = None
        self.clock_mode = clock_mode
        self.rank_tol = rank_tol

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
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
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
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="no transient states in connectivity table",
                )
            )
        if not absorbing:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="no absorbing states in connectivity table",
                )
            )

        # Entry is state 0 by pyKMC convention (see FPTASelector).
        entry = transient[0]
        problem = _amsel.AmcProblem(
            transient=transient,
            absorbing=absorbing,
            rates=rates,
        )

        self.last_reduced_kinetics = None
        self.last_clock_mode = None

        try:
            if self.clock_mode == "sampled":
                fpta_res = problem.fpta(entry=entry, r=float(np.random.random()))
                t_exit = float(fpta_res.t_exit)
                weights_arr = np.asarray(fpta_res.weights, dtype=np.float64)
                self.last_clock_mode = "sampled"
            else:
                rk = problem.reduced_kinetics(entry=entry)
                self.last_reduced_kinetics = rk
                use_mean = self.clock_mode == "mean" or rk.one_rate_clock_is_plausible(self.rank_tol)
                if use_mean:
                    mrm_res = problem.mrm(entry=entry)
                    t_exit = float(mrm_res.tau_total)
                    weights_arr = np.asarray(mrm_res.rate_to_absorbing, dtype=np.float64) * t_exit
                    self.last_clock_mode = "mean"
                else:
                    fpta_res = problem.fpta(entry=entry, r=float(np.random.random()))
                    t_exit = float(fpta_res.t_exit)
                    weights_arr = np.asarray(fpta_res.weights, dtype=np.float64)
                    self.last_clock_mode = "sampled"
        except _amsel.AmselError as exc:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message=f"amsel kernel rejected problem: {exc}",
                )
            )

        weight_sum = float(np.sum(weights_arr))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="amsel returned non-positive absorbing weights",
                )
            )
        weights_arr = weights_arr / weight_sum
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

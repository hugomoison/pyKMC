"""HTST rate constant: per-event Vineyard prefactor nu0, k0 for events without one.

The LAMMPS engine extension, `lammps_extension`, is not imported here: it loads
`lammps` and `pypARTn`, and `engine_extensions` imports it only when
`[RateConstant] style = htst`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from pykmc._core.result import Err, ErrorInfo, Ok, Result

from ...result import PrefactorOutcome, PrefactorRequest
from ..base import RateConstantConfigLike, RateConstantStrategy
from .region import (
    build_region,
    frozen_overlap,
    full_hessian_mb,
    hessian_mode,
    moving_atom,
    region_record,
    wrapped_on_open_axis,
)
from ...result import PrefactorError
from .thresholds import AV_BUFFER_WARN_A, NU0_WARN_BAND_THZ, RMOV_WARN_A

if TYPE_CHECKING:
    from pykmc.config import Config
    from pykmc.engine.base import EngineExtension
    from pykmc.log import LogKMC
    from pykmc.manager import Manager
    from pykmc.system import System


class HTSTStrategy(RateConstantStrategy):
    r"""Rate constant with a per-event prefactor `nu0`, and `k0` for events without one.

    $$
    k = \nu_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$

    `nu0` is the harmonic (Vineyard) prefactor from the Hessians at the minimum and
    at the saddle, each computed by `dynamical_matrix` in a serial scratch LAMMPS
    on a session of the engine manager.
    """

    name = "htst"
    per_event_prefactor = True

    def __init__(self, config: RateConstantConfigLike) -> None:
        super().__init__(config)
        self._run: Any = None  # the full Config, set by setup
        self._manager: Any = None
        self._loggers: Any = None
        self._shell_width: Optional[float] = None
        self._record = ""

    @property
    def engine_extensions(self) -> tuple[type[EngineExtension], ...]:
        """The serial-LAMMPS extension that computes the Hessians."""
        from .lammps_extension import HTSTLammpsExtension

        return (HTSTLammpsExtension,)

    @property
    def settings_record(self) -> str:
        """Hessian region settings, as stored with each nu0 (set by `setup`)."""
        return self._record

    def setup(
        self, config: Config, manager: Manager, system: System, loggers: LogKMC
    ) -> None:
        """Check that HTST prefactors can be computed and set the Hessian region.

        A run that could never compute a prefactor stops here, with the reason in
        the log.

        Parameters
        ----------
        config : Config
            The full simulation configuration.
        manager : Manager
            The engine session pool: every session checks its own LAMMPS.
        system : System
            The simulated system.
        loggers : LogKMC
            The simulation loggers.

        Raises
        ------
        RuntimeError
            If HTST prefactors can never be computed in this run.

        """
        self._run, self._manager, self._loggers = config, manager, loggers
        rc = config.rateconstant
        mode = hessian_mode(config)
        species = sorted(set(system.types))
        n_atoms = len(system.types)
        n_sessions = config.control.n_sessions
        shell = None
        try:
            if not system.cell.orthorhombic:
                raise RuntimeError(
                    "HTST: style = htst needs an orthogonal cell; this cell has "
                    "off-diagonal terms. Use an orthogonal cell or set [RateConstant] "
                    "style = constant."
                )
            if mode == "full" and sum(bool(p) for p in system.pbc) < 2:
                pbc = " ".join("T" if p else "F" for p in system.pbc)
                raise RuntimeError(
                    "HTST: free_radius = full needs at least two periodic axes "
                    f"(pbc = {pbc}): with fewer, rotations are zero modes that are "
                    "not removed. Use a numeric free_radius or set [RateConstant] "
                    "style = constant."
                )
            if mode == "full" and full_hessian_mb(n_atoms) > rc.max_full_hessian_mb:
                raise RuntimeError(
                    f"HTST: free_radius = full needs {full_hessian_mb(n_atoms):.0f} MB "
                    f"per Hessian for {n_atoms} atoms (one 3N x 3N matrix of doubles), "
                    f"above max_full_hessian_mb = {rc.max_full_hessian_mb:g}. Up to "
                    f"n_sessions = {n_sessions} Hessians are built at once, each "
                    "needing about twice that in memory and a scratch file of that "
                    "size in hessian_dir. Raise max_full_hessian_mb or use a numeric "
                    "free_radius."
                )
            # Every session checks its own LAMMPS and its access to hessian_dir.
            manager.broadcast("htst_capability_check", config=config, species=species)
            info = manager.htst_capability_check(
                config=config, species=species
            ).result()
            if mode == "reduced":
                if rc.shell_width is not None:
                    shell = rc.shell_width
                elif info["force_cutoff"] is None or info["force_cutoff"] <= 0:
                    raise RuntimeError(
                        "HTST: the force cutoff could not be read from LAMMPS "
                        "('Communication cutoff' missing or not positive in info "
                        "communication); set [RateConstant] shell_width."
                    )
                else:
                    shell = 2.0 * info["force_cutoff"]
        except Exception as exc:
            # run.py aborts before the exception is printed: log the reason.
            loggers.error("log", str(exc))
            raise
        self._shell_width = shell
        self._record = region_record(config, shell)

        loggers.info(
            "log",
            f"HTST: LAMMPS version {info['lammps_version']}; Hessian scratch files "
            f"go to {info['hessian_dir']}; fd_step = {rc.fd_step:g} A; region record "
            f"'{self._record}'.",
        )
        if mode == "av":
            rmov, ract = config.activevolume.rmov, config.activevolume.ract
            loggers.info(
                "log",
                "HTST: Hessian region: active volume; atoms within rmov = "
                f"{rmov:g} A of the search or refinement centre are displaced, the "
                "rmov-ract buffer is kept frozen, atoms beyond ract = "
                f"{ract:g} A are left out; free_radius, shell_width and "
                "max_full_hessian_mb are ignored.",
            )
            if rmov < RMOV_WARN_A:
                loggers.warning(
                    "log",
                    f"HTST: rmov = {rmov:g} A is below 8 A: with active_volume = True "
                    "the Hessian region is the active volume, and nu0 is not "
                    "converged at this radius (Ni vacancy hop, against the all-atom "
                    "value: about 50 % low at rmov = 4 A, about 2 % at rmov = 8 A). "
                    "See the HTST page of the documentation.",
                )
            if ract - rmov < AV_BUFFER_WARN_A:
                loggers.warning(
                    "log",
                    "HTST: the active-volume buffer ract - rmov = "
                    f"{ract - rmov:g} A is thinner than 4 A: the frozen buffer may "
                    "not shield the displaced atoms (Ni vacancy hop: crop error up "
                    "to 2 % at 2-3 A, up to 39 % at 1 A).",
                )
        elif mode == "reduced":
            source = (
                "[RateConstant] shell_width"
                if rc.shell_width is not None
                else f"2 x force cutoff {info['force_cutoff']:g} A read from LAMMPS"
            )
            loggers.info(
                "log",
                "HTST: Hessian region: atoms within free_radius = "
                f"{rc.free_radius:g} A of the moving atom's saddle position are "
                f"displaced, a frozen shell of {shell:g} A ({source}) is kept, atoms "
                "beyond are left out.",
            )
        else:
            loggers.info(
                "log",
                f"HTST: Hessian region: the whole system ({n_atoms} atoms, "
                f"{full_hessian_mb(n_atoms):.0f} MB per Hessian; up to n_sessions = "
                f"{n_sessions} Hessians at once, each needing about twice that in "
                "memory and a scratch file of that size in hessian_dir).",
            )
        if config.control.basin:
            loggers.warning(
                "log",
                "HTST: basin = True: inside a basin every rate (internal hops and "
                f"exits) uses the constant prefactor k0 = {rc.k0:g} ps^-1; steps "
                "outside basins use nu0.",
            )
        if config.atomicenvironment.atom_coloring_mode == "grey" and len(species) > 1:
            loggers.warning(
                "log",
                f"HTST: atom_coloring_mode = grey with {len(species)} species: a "
                "reference nu0 is computed with the species of the event it was "
                "found on and is reused by unrefined events on other species; "
                "refined events get their own nu0.",
            )

    def prefactor(self, value: Optional[float] = None) -> float:
        """Return the event's `nu0`, or `[RateConstant] k0` when it has none.

        Parameters
        ----------
        value : float, optional
            The event's `nu0` (THz, i.e. ps^-1). `None` or a non-finite value means
            the event has none, and `k0` is used instead.

        Returns
        -------
        float
            `nu0` or `k0` (ps^-1).

        """
        if value is None or not math.isfinite(value):
            return self.config.k0
        return value

    def compute_prefactors(
        self,
        requests: list[PrefactorRequest],
        system: System,
        frozen_ids: np.ndarray,
    ) -> list[list[PrefactorOutcome]]:
        """Compute the nu0 of each direction of new events.

        Each event is one Hessian job on a free session, for all of its minima;
        every job is submitted before any result is awaited. A direction without a
        nu0 uses `k0`, and a warning says why.

        Parameters
        ----------
        requests : list[PrefactorRequest]
            The events.
        system : System
            The simulated system, at the minimum the events start from.
        frozen_ids : np.ndarray
            Indices of the frozen atoms.

        Returns
        -------
        list[list[PrefactorOutcome]]
            For each request, one outcome per minimum: nu0 (THz) and 'ok', or None
            and the key of the error.

        Notes
        -----
        It never raises: a job that cannot be built, sent or collected gets an
        error outcome for each of its minima.

        """
        outcomes = [
            [
                self._outcome(name, result)
                for name, result in zip(request.names, results, strict=True)
            ]
            for request, results in zip(
                requests, self._jobs(requests, system, frozen_ids), strict=True
            )
        ]
        for reference, what in (
            (True, "new reference directions"),
            (False, "refined active events"),
        ):
            done = [
                outcome
                for request, per_event in zip(requests, outcomes, strict=True)
                if request.reference == reference
                for outcome in per_event
            ]
            if done:
                n_ok = sum(outcome.status == "ok" for outcome in done)
                self._loggers.info(
                    "log", f"HTST: nu0 computed for {n_ok} of {len(done)} {what}"
                )
        return outcomes

    def _jobs(
        self,
        requests: list[PrefactorRequest],
        system: System,
        frozen_ids: np.ndarray,
    ) -> list[list[Result[float, ErrorInfo]]]:
        """Run the Hessian jobs of a list of events and return their results.

        Returns
        -------
        list[list[Result[float, ErrorInfo]]]
            For each event, one result per minimum: `Ok` with nu0 (THz) or `Err`
            with the reason there is none.

        """

        def failure(code: PrefactorError, message: str, n: int) -> list:
            return [Err(ErrorInfo(type=code, message=message))] * n

        def well_formed(outcome: object) -> bool:
            if isinstance(outcome, Ok):
                return isinstance(outcome.ok_value(), (float, np.floating))
            return isinstance(outcome, Err) and isinstance(
                outcome.err_value(), ErrorInfo
            )

        try:
            run = self._run
            rc = run.rateconstant
            mode = hessian_mode(run)
            cell = np.asarray(system.cell)
            pbc = tuple(bool(p) for p in system.pbc)
            types = np.asarray(system.types)
            species = sorted(set(system.types))
        except Exception as exc:
            message = f"region build failed: {type(exc).__name__}: {exc}"
            return [
                failure(PrefactorError.ENGINE_FAILURE, message, len(request.minima))
                for request in requests
            ]

        # Submit every job before waiting for any result.
        submitted = []
        for request in requests:
            saddle, minima = request.saddle, request.minima
            centre, reference = request.centre, request.reference
            n = len(minima)
            try:
                if mode == "av":
                    # the active volume the search or the refinement used
                    region = build_region(
                        "av",
                        system.positions,
                        cell,
                        pbc,
                        center_index=centre,
                        r_displaced=run.activevolume.rmov,
                        r_present=run.activevolume.ract,
                    )
                elif mode == "reduced":
                    region = build_region(
                        "reduced",
                        saddle,
                        cell,
                        pbc,
                        center_index=moving_atom(minima[0], saddle, cell, pbc),
                        r_displaced=rc.free_radius,
                        r_present=rc.free_radius + self._shell_width,
                    )
                else:
                    region = build_region("full", saddle, cell, pbc)
                wrapped = (
                    reference
                    and mode != "av"
                    and not all(pbc)
                    and wrapped_on_open_axis(
                        region, (saddle, *minima), system.positions, cell, pbc
                    )
                )
                n_frozen = frozen_overlap(region, frozen_ids)
            except Exception as exc:
                submitted.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"region build failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )
                continue
            if wrapped:
                submitted.append(
                    failure(
                        PrefactorError.WRAPPED_OPEN_AXIS,
                        "recentring wrapped atoms across a non-periodic axis",
                        n,
                    )
                )
                continue
            if n_frozen > 0:
                submitted.append(
                    failure(
                        PrefactorError.FROZEN_OVERLAP,
                        "the frozen region overlapped the HTST calculation space "
                        f"({n_frozen} frozen atoms among the displaced atoms)",
                        n,
                    )
                )
                continue
            present = region.present
            try:
                submitted.append(
                    self._manager.compute_htst_prefactors(
                        config=run,
                        saddle=saddle[present],
                        minima=[minimum[present] for minimum in minima],
                        types=types[present].tolist(),
                        species=species,
                        cell=cell,
                        pbc=region.pbc,
                        displaced=region.displaced,
                        n_zero_modes=region.n_zero_modes,
                    )
                )
            except Exception as exc:
                submitted.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"HTST job failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )

        results = []
        for request, job in zip(requests, submitted, strict=True):
            n = len(request.minima)
            if isinstance(job, list):  # not sent
                results.append(job)
                continue
            try:
                outcomes = job.result()
            except Exception as exc:
                results.append(
                    failure(
                        PrefactorError.ENGINE_FAILURE,
                        f"HTST job failed: {type(exc).__name__}: {exc}",
                        n,
                    )
                )
                continue
            if not (
                isinstance(outcomes, list)
                and len(outcomes) == n
                and all(well_formed(outcome) for outcome in outcomes)
            ):
                outcomes = failure(
                    PrefactorError.ENGINE_FAILURE,
                    "HTST job returned an unexpected result: "
                    f"{type(outcomes).__name__}",
                    n,
                )
            results.append(outcomes)
        return results

    def _outcome(
        self, name: str, outcome: Result[float, ErrorInfo]
    ) -> PrefactorOutcome:
        """Turn the result of a Hessian job into the outcome of one direction.

        Parameters
        ----------
        name : str
            Name of the event direction in the log.
        outcome : Result[float, ErrorInfo]
            `Ok` with nu0 (THz), or `Err` with the reason there is none.

        Returns
        -------
        PrefactorOutcome
            nu0 and 'ok', or None and the error key: the event then uses `k0`, and
            a warning says why.

        """
        if isinstance(outcome, Err):
            error = outcome.err_value()
            self._loggers.warning("log", f"HTST: {name} uses k0: {error.message}")
            return PrefactorOutcome(None, error.type.key)
        nu0 = outcome.ok_value()
        if not (np.isfinite(nu0) and nu0 > 0):
            self._loggers.warning(
                "log",
                f"HTST: {name} uses k0: nu0 is not a finite positive number "
                f"(value {float(nu0)!r} THz)",
            )
            return PrefactorOutcome(None, PrefactorError.NONFINITE_PREFACTOR.key)
        low, high = NU0_WARN_BAND_THZ
        if not low <= nu0 <= high:
            self._loggers.warning(
                "log",
                f"HTST: {name}: nu0 = {nu0:.4g} THz is outside 0.1-1000 THz; "
                "used as computed",
            )
        return PrefactorOutcome(float(nu0), "ok")

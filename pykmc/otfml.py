"""On-the-fly ML potential retraining controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Callable
from .result import ErrorType

if TYPE_CHECKING:
    from .kmc import KMC


# Fixed dump directory — LAMMPS sessions always write here; kmtp-otf moves
# the files out after each cycle, so this dir is empty at the start of each run.
OTFML_DUMP_DIR = Path("extrapolative_dumps")
OTF_THERMO_LOG_PREFIX = "lammps.log"

OTFML_TOL_FLAG = "grade_over_tol"
OTFML_MAX_FLAG = "grade_over_max"
OTFML_LATCH = "grade_trigger"


def ensure_otf_dirs() -> None:
    """Create the OTF dump directory."""
    OTFML_DUMP_DIR.mkdir(parents=True, exist_ok=True)


def session_dump_path(session_id: int) -> Path:
    """Return the dump path for a session."""
    ensure_otf_dirs()
    return OTFML_DUMP_DIR / f"extrapolating_dump.{session_id}.lammps"


def otf_thermo_path(engine) -> Path:
    """Return the dedicated OTF thermo log path for one engine."""
    return Path(f"{OTF_THERMO_LOG_PREFIX}.{engine.engine_id}")


def read_otf_thermo(engine):
    """Read the last thermo block using LAMMPS's official log parser."""
    log_path = otf_thermo_path(engine)
    if not log_path.exists() or not log_path.is_file():
        return None

    from lammps.formats import LogFile

    try:
        runs = LogFile(str(log_path)).runs
    except Exception:
        raise RuntimeError(
            f"Failed to parse LAMMPS log file at {log_path} for OTFML diagnostics."
        )

    if not runs:
        raise RuntimeError(f"No runs found in LAMMPS log file at {log_path}")

    return runs[-1]


@dataclass(frozen=True)
class OTFExtrapolationFlags:
    """Latched extrapolation state for one completed operation."""

    extrapolated: bool = False
    extreme_extrapolated: bool = False


class OTFMLController:
    """Coordinate OTF retraining around the existing KMC workflow."""

    def __init__(self, kmc: KMC) -> None:
        self.kmc = kmc
        self.params = kmc.params.otfml
        self.enabled = bool(kmc.params.control.otfml and self.params)
        self._consecutive_failed_retrain_exit = 0
        if self.enabled:
            ensure_otf_dirs()

    def is_enabled_for_phase(self, phase: str) -> bool:
        """Return whether OTF handling is enabled for a phase."""
        return self.enabled and phase in self.params.enabled_phases

    def retry_extrapolating(self, phase: str, obj) -> None:
        """Retry extrapolating tasks for a phase until stable."""
        self._retry_until_stable(
            phase,
            lambda: self._collect_extrapolation_retry_ids(obj.results),
            obj.retry,
        )

    def _retry_until_stable(self, phase: str, collect_fn, retry_fn) -> None:
        if not self.is_enabled_for_phase(phase):
            return
        cycle = 0
        while True:
            retry_task_ids = collect_fn()
            if not retry_task_ids:
                return
            self._log(
                "log",
                "\t :=> OTFML retry cycle {} for {} jobs in phase '{}'.".format(
                    cycle + 1, len(retry_task_ids), phase
                ),
            )
            self._retrain_and_reload()
            self.kmc._minimize_system_once(configuration=self.kmc.system.configuration)
            retry_fn(retry_task_ids)
            cycle += 1

    def retry_extrapolating_minimization(
        self, minimize_once: Callable[[], None]
    ) -> None:
        """Retry minimization until no further extrapolation is detected."""
        if not self.is_enabled_for_phase("minimize"):
            minimize_once()
            return

        flags = self._coerce_flags(minimize_once())
        while True:
            if not flags.extrapolated:
                return
            self._log(
                "log",
                "\t :=> OTFML detected minimization extrapolation{}.".format(
                    " above gamma_max" if flags.extreme_extrapolated else ""
                ),
            )
            self._retrain_and_reload()
            flags = self._coerce_flags(minimize_once())

    def _collect_extrapolation_retry_ids(self, results) -> list:
        return [
            task_id
            for task_id, result in enumerate(results)
            if result is not None
            and not result.is_ok()
            and result.err_value().type
            in {ErrorType.EXTRAPOLATION, ErrorType.EXTREME_EXTRAPOLATION}
        ]

    def _build_retrain_command(self) -> str:
        """Assemble the full retrain command from params fields."""
        c = self.params
        dumps_glob = str(OTFML_DUMP_DIR / "extrapolating_dump.*.lammps")
        parts = [c.retrain_command]
        parts.append(f"--potential {c.potential_file}")
        parts.append(f"--training_set {c.training_set_file}")
        parts.append(f"--gamma_tolerance {c.gamma_tolerance}")
        parts.append(f"--gamma_max {c.gamma_max}")
        if c.launcher:
            parts.append(f"--launcher {c.launcher}")
        if c.batch_args:
            parts.append(f'--batch-args="{c.batch_args}"')
        if c.runner_args:
            parts.append(f'--runner-args="{c.runner_args}"')
        if c.sequential_eval:
            parts.append("--sequential-eval")
        if c.extra_args:
            parts.append(c.extra_args)
        parts.append(f"--extrapolative_dumps {dumps_glob}")
        return " ".join(parts)

    def _retrain_and_reload(self) -> None:
        full_command = self._build_retrain_command()
        self._log("log", "\t :=> OTFML retraining command: {}".format(full_command))
        # does nothing for now
        # clean_env = {k: v for k, v in os.environ.items() if not any(k.startswith(p) for p in self._MPI_PREFIXES)}

        with self.kmc.manager.sleeping_workers():
            result = subprocess.run(full_command, shell=True)
        if result.returncode == 67:
            self._consecutive_failed_retrain_exit += 1
            self._log(
                "log",
                f"\t :=> Retraining exited with code 67 ({self._consecutive_failed_retrain_exit}/5 consecutive)",
            )
            if self._consecutive_failed_retrain_exit > 5:
                self._log(
                    "log",
                    "Retraining returned exit code 67 more than 5 times in a row; aborting.",
                )
                self.kmc._close()
        else:
            self._consecutive_failed_retrain_exit = 0

        was_global = self.kmc.manager.using_global
        self.kmc.manager.setup_otf_cycle(self.kmc.params)
        self.kmc.manager.use_local()
        self.kmc.manager.set_all_positions(self.kmc.system.positions)
        if was_global:
            self.kmc.manager.use_global()

    _MPI_PREFIXES = (
        "OMPI_0000",
        "PMI_0000",
        "I_MPI_0000",
        "MPI_0000",
        "HYDRA_0000",
        "MPIEXEC_0000",
    )

    def _coerce_flags(self, value) -> OTFExtrapolationFlags:
        if isinstance(value, OTFExtrapolationFlags):
            return value
        return OTFExtrapolationFlags()

    def _log(self, logger_name: str, message: str) -> None:
        if getattr(self.kmc, "loggers", None) is not None:
            self.kmc.loggers.info(logger_name, message)


class OTFMLStreamCheckpoint:
    """Batches phase completions behind a full-drain checkpoint for `OTFMLController.retry_extrapolating`.

    Retraining reloads the potential and repositions every session directly
    (bypassing the job queue entirely), so it may only ever run with nothing
    else outstanding -- a caller streaming work through a bounded pool needs
    to periodically stop admitting new work and let everything in flight
    drain before calling `run()`. This class owns exactly that policy (when
    to defer a result, when to stop admitting, when a drained batch is ready)
    so a streaming caller only ever talks to this narrow interface instead
    of reimplementing OTF-ML's own batching rules itself.

    A no-op wrapper when `controller` isn't enabled for `phase`: every
    method still does the right (nothing) thing, so a caller never needs its
    own conditional to skip this machinery when OTF-ML is inactive.
    """

    def __init__(self, controller: OTFMLController, phase: str, batch_size: int) -> None:
        self.controller = controller
        self.phase = phase
        self.batch_size = batch_size
        self.active = controller.is_enabled_for_phase(phase)
        self._pending: list[tuple] = []

    def should_defer(self, result) -> bool:
        """Whether `result` should be held back rather than processed immediately."""
        if not self.active:
            return False
        return not result.is_ok() and result.err_value().type in {ErrorType.EXTRAPOLATION, ErrorType.EXTREME_EXTRAPOLATION}

    def track(self, task, payload) -> None:
        """Remember `task`'s completion, with caller-owned `payload`, as pending this checkpoint."""
        if self.active:
            self._pending.append((task, payload))

    def admits_more(self) -> bool:
        """Whether dispatch should keep admitting new work, or pause to let a due checkpoint drain."""
        return not self.active or len(self._pending) < self.batch_size

    def due(self) -> bool:
        """Whether there's a pending batch ready to run, once nothing is left in flight."""
        return bool(self._pending)

    def run(self, event_search) -> list[tuple]:
        """Retrain/retry, then return (task, payload) for every pending task whose outcome changed."""
        retry_ids = set(self.controller._collect_extrapolation_retry_ids(event_search.results))
        self.controller.retry_extrapolating(self.phase, event_search)
        changed = [(task, payload) for task, payload in self._pending if task.task_id in retry_ids]
        self._pending.clear()
        return changed

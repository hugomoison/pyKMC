"""Module implementing the Refinement class that deals with the event refinement procedure."""

from dataclasses import dataclass, field

from .result import (
    Result,
    EventRefinementOutput,
    ErrorInfo,
    ErrorType,
    Err,
    Ok,
    PSROutput,
    RefinementCandidate,
    RefinementTask,
)
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
from .parameters import Parameters
from .system import System, Configuration
from .neighbors_list import NeighborsList
from .log import LogKMC
from .enginemanager.lmpi.pool import Manager
import numpy as np
import concurrent.futures


@dataclass
class PreparedRefinementTask:
    """Prepared refinement task ready for submission or immediate completion."""

    task: RefinementTask
    min2_configuration: Configuration | None
    reference_energy_barrier: float
    neighbors: np.ndarray
    immediate_result: Result[EventRefinementOutput, ErrorInfo] | None = None
    submit_kwargs: dict = field(default_factory=dict, repr=False)


class Refinement:
    """Perfrom event refinements and deal with results.

    Parameters
    ----------
    params : Parameters
        The configuration of the simulation.
    loggers : LogKMC
        The logger of the KMC simulation.
    system : System
        The atomic system.
    neighbors_list : NeighborsList
        The neighbors lists of the system.
    manager : Manager
        The engine manager to use for the refinement.

    """

    def __init__(
        self,
        params: Parameters,
        loggers: LogKMC,
        system: System,
        neighbors_list: NeighborsList,
        manager: Manager,
    ) -> None:
        self.params = params
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.manager = manager
        self.results = None
        self.tasks = []
        self._psr_cache: dict[tuple[int, int], Result[PSROutput, ErrorInfo]] = {}

    def execute(self, candidates: list[RefinementCandidate]) -> None:
        """Execute event refinements for each already-selected candidate.

        It stores the results of the event refinements in self.results.

        Parameters
        ----------
        candidates : list[RefinementCandidate]
            Candidates selected by `KMC.build_refinement_candidates()`,
            each already carrying its verify-vs-trust decision.

        """
        tasks = self.build_tasks(candidates)
        self.loggers.info("log", "\t :=> Refining {} events".format(len(tasks)))
        self.tasks = tasks
        self.results = [None] * len(tasks)
        for task_id, result in self._run_tasks(tasks).items():
            self.results[task_id] = result

    def build_tasks(self, candidates: list[RefinementCandidate]) -> list[RefinementTask]:
        """Assign stable task ids to already-selected refinement candidates."""
        return [
            RefinementTask(
                task_id=task_id,
                central_atom_index=candidate.central_atom_index,
                num_reference_event=int(candidate.dfevent["idx_ref"]),
                symmetry_index=candidate.symmetry_index,
                dfevent=candidate.dfevent,
                verify=candidate.verify,
            )
            for task_id, candidate in enumerate(candidates)
        ]

    def retry(self, retry_task_ids: list[int]) -> None:
        """Rerun only the requested refinement jobs."""
        if not retry_task_ids:
            return
        retry_tasks = [self.tasks[task_id] for task_id in retry_task_ids]
        for task_id, result in self._run_tasks(retry_tasks).items():
            self.results[task_id] = result

    def _run_tasks(
        self,
        tasks: list[RefinementTask],
    ) -> dict[int, Result[EventRefinementOutput, ErrorInfo]]:
        if not tasks:
            return {}

        future_to_prepared = {}
        for task in tasks:
            prepared = self._prepare_task(task)
            future = self._submit_task(prepared)
            future_to_prepared[future] = prepared

        run_results = {}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_prepared)):
            prepared = future_to_prepared[future]
            try:
                res = future.result()
            except Exception as exc:
                self.loggers.error(
                    "log",
                    f"\n\t task {prepared.task.task_id:5d} | atom {prepared.task.central_atom_index:6d} | {'RAISE':<5} type={type(exc).__name__}",
                )
                raise
            self.loggers.progress_bar("progress", i + 1, len(tasks))
            run_results[prepared.task.task_id] = self._finalize_result(res, prepared)
        return run_results

    def _prepare_task(
        self,
        task: RefinementTask,
    ) -> PreparedRefinementTask:
        at_idx = task.central_atom_index
        num_reference_event = task.num_reference_event
        symmetry_index = task.symmetry_index
        dfevent = task.dfevent
        matching_score_thr = self.params.psr.matching_score_thr + 0.25*self.params.psr.matching_score_thr

        neighbors = self.neighbors_list.get_neighbors("rcut", at_idx).copy()

        # PSR/IRA doesn't depend on symmetry_index, so every symmetry-variant
        # sibling task of the same (atom, row) pair reuses the same match.
        psr_key = (at_idx, num_reference_event)
        if psr_key not in self._psr_cache:
            result_psr = PointSetRegistration(
                self.params, self.system, dfevent, self.neighbors_list, at_idx
            ).match()
            self._psr_cache[psr_key] = check_match(
                result_psr, matching_score_thr
            )
        result_psr = self._psr_cache[psr_key]
        if not result_psr.is_ok():
            return PreparedRefinementTask(
                task=task,
                min2_configuration=None,
                reference_energy_barrier=dfevent["dE_forward"],
                neighbors=neighbors,
                immediate_result=result_psr,
            )

        output_psr = result_psr.ok_value()
        initial_configuration = dfevent.at["initial_configuration"]
        displacement_saddle = dfevent.at["saddle_configuration"] - initial_configuration
        displacement_final = dfevent.at["final_configuration"] - initial_configuration
        sym_matrix = dfevent.at["sym_matrix"][symmetry_index]
        perm_matrix = dfevent.at["sym_perm"][symmetry_index]

        # Symmetries act on the reaction's displacement field (saddle/final
        # relative to initial), which is pivot-independent -- not on absolute
        # positions -- so wrapping is explicitly skipped here (a displacement
        # vector isn't a position to wrap into the cell).
        new_displacement_saddle = geometry.transform_positions(
            displacement_saddle, sym_matrix, 0, perm_matrix, wrap=False,
        )
        new_displacement_final = geometry.transform_positions(
            displacement_final, sym_matrix, 0, perm_matrix, wrap=False,
        )

        saddle_configuration = initial_configuration + new_displacement_saddle
        final_configuration = initial_configuration + new_displacement_final

        new_saddle_configuration = geometry.transform_positions(
            saddle_configuration,
            output_psr.rotation_matrix,
            output_psr.translation_matrix,
            output_psr.permutation_matrix,
        )
        new_final_configuration = geometry.transform_positions(
            final_configuration,
            output_psr.rotation_matrix,
            output_psr.translation_matrix,
            output_psr.permutation_matrix,
        )
        min2_configuration = new_final_configuration.with_types(
            np.asarray(self.system.types)[neighbors]
        )

        saved_configuration = self.system.configuration.copy()
        self.system.update_positions(new_saddle_configuration, atom_idx=neighbors)
        if not task.verify:
            immediate_result = Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle=self.system.configuration.copy(),
                    E_saddle=dfevent["dE_forward"],
                    num_reference_event=num_reference_event,
                    symmetry_index=symmetry_index,
                    refined="F",
                )
            )
            self.system.update_positions(saved_configuration)
            return PreparedRefinementTask(
                task=task,
                min2_configuration=min2_configuration,
                reference_energy_barrier=dfevent["dE_forward"],
                neighbors=neighbors,
                immediate_result=immediate_result,
            )

        submit_kwargs = {
            "params": self.params,
            "central_atom": at_idx,
            "configuration": saved_configuration,
            "saddle_idx": neighbors.copy(),
            "saddle_positions": self.system.positions.copy()[neighbors.copy()],
            "num_reference_event": num_reference_event,
            "symmetry_index": symmetry_index,
        }
        self.system.update_positions(saved_configuration)
        return PreparedRefinementTask(
            task=task,
            min2_configuration=min2_configuration,
            reference_energy_barrier=dfevent["dE_forward"],
            neighbors=neighbors,
            submit_kwargs=submit_kwargs,
        )

    def _submit_task(self, prepared: PreparedRefinementTask):
        if prepared.immediate_result is not None:
            future = concurrent.futures.Future()
            future.set_result(prepared.immediate_result)
            return future
        return self.manager.partn_refine(**prepared.submit_kwargs)

    def _finalize_result(
        self,
        res,
        prepared: PreparedRefinementTask,
    ):
        task = prepared.task
        if res.is_ok():
            res.ok_value().min2 = prepared.min2_configuration
            res.ok_value().num_reference_event = task.num_reference_event
            res.ok_value().symmetry_index = task.symmetry_index
            res.ok_value().neighbors = prepared.neighbors
            res.ok_value().saddle = res.ok_value().saddle[prepared.neighbors]
            res.ok_value().dE_forward = res.ok_value().E_saddle
            return self.check_refinement_energy(
                res,
                abs(res.ok_value().dE_forward - prepared.reference_energy_barrier),
                self.params.eventsearch.refined_energy_thr,
                task.num_reference_event,
            )

        err = res.err_value()
        if not isinstance(err.variables, dict):
            err.variables = {}
        err.variables["n_ref_event"] = task.num_reference_event
        err.variables.setdefault("num_reference_event", task.num_reference_event)
        err.variables.setdefault("symmetry_index", task.symmetry_index)
        return res

    def check_refinement_energy(
        self,
        result_refine: Result[EventRefinementOutput, ErrorInfo],
        energy_mismatch: float,
        refined_energy_thr: float,
        num_reference_event: int,
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Check if the energy barrier of the refinement correspond the one of the reference event.

        Parameters
        ----------
        result_refine : Result[EventRefinementOutput, ErrorInfo]
            Results of the refinement procedure.
        energy_mismatch : float
            Difference between the reference event energy barrier and the refine one.
        refined_energy_thr : float
            maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy
        num_reference_event : int
            Reference event this refinement task was attempting, recorded on failure for diagnostics.

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            list of results of the procedure.

        """
        if energy_mismatch > refined_energy_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
                    message="refinement energy barrier does not match reference one",
                    variables={
                        "n_ref_event": num_reference_event,
                        "num_reference_event": num_reference_event,
                    },
                )
            )
        else:
            return result_refine

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return successful results.

        Returns
        -------
        list[EventRefinementOutput]
            list of EventRefinementOutpout dataclass with refine event's informations.

        """
        return [e.ok_value() for e in self.results if e is not None and e.is_ok()]

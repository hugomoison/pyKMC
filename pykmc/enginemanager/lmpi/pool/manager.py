"""Class for managing a pool of LAMMPS instances."""

from ..sessions import MpiApiSession
from dataclasses import dataclass
from concurrent.futures import Future
from contextlib import contextmanager
import queue
import threading

# TODO : commented print should be log depending of the verbosity but need to thing of how we modify log before (also loggers are
# initiated in kmc, after the initialization of manager ...))


@dataclass
class Job:
    operation_name: str
    params: dict
    future: Future


class Manager:
    """A class to manage a pool of Lammps sessions."""

    def __init__(
        self, sessions: list[MpiApiSession], global_session: MpiApiSession = None
    ) -> None:
        """
        Initialize the LammpsPoolManager with a specified number of sessions.
        """
        self.sessions = sessions
        self.global_session = global_session
        self.using_global = True
        self.job_queue: queue.Queue[Job] = queue.Queue()
        # Thread that dispatch job to workers
        # self.dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True)
        # self.dispatcher_thread.start()

        # pool de workers
        self.workers = []
        for session in self.sessions:
            t = threading.Thread(target=self._worker_loop, args=(session,), daemon=True)
            t.start()
            self.workers.append(t)

    def broadcast_command(self, cmd: str):
        """
        Send the same command to all sessions and wait for all to finish.
        """
        # print("[PoolManager] Broadcasting command:", cmd)
        for session in self.sessions:
            session.command(cmd)

    def initialize_sessions(self, params, system):
        """
        Initialize engines with the same system and params
        """
        print("[Manager] use local")
        self.use_local()
        print("[Manager] Initializing all Lammps engines")
        for session in self.sessions:
            session.initialize_parameters()
            session.initialize_system(system.configuration, params)
            session.initialize_potential(params)
        print("[Manager] use global")
        self.use_global()
        print("[Manager] Initializing global Lammps engines")
        if self.global_session is not None:
            self.global_initialize_parameters()
            self.global_initialize_system(system.configuration, params)
            self.global_initialize_potential(params)

    def use_local(self):
        """
        Have engines switch from global pool to local pools
        """
        if self.using_global:
            self.global_session.use_local()
            self.using_global = False

    def use_global(self):
        """
        Have engines switch from local pools to global pool
        """
        if not self.using_global:
            for session in self.sessions:
                session.use_global()
            self.using_global = True

    def _worker_loop(self, session: MpiApiSession):
        """Boucle infinie tournant dans un thread dédié à 'session'."""
        while True:
            job = self.job_queue.get()

            if job is None:
                break

            try:
                method = getattr(session, job.operation_name)
                if job.params is None:
                    result = method()
                else:
                    result = method(**job.params)

                job.future.set_result(result)

            except Exception as e:
                job.future.set_exception(e)
            finally:
                self.job_queue.task_done()

    # def _dispatcher(self) :
    #    while True :
    #        job = self.job_queue.get() #block until a job is get
    #        while True :
    #            session = self._get_available_engine()
    #            if session is not None :
    #                #print(f"[PoolManager] Found available session: {session.session_id}")
    #                threading.Thread(target=self._run_job, args=(session, job), daemon=True).start()
    #                threading.Event().wait(0.1) # Wait a bit to allow the job to be processed
    #                break #job is submited
    #            else :
    #                threading.Event().wait(0.1)

    # def _get_available_engine(self) :
    #    """Check if worker is available, if yes return worker, if not, return None"""
    #    for session in self.sessions :
    #        if session._is_busy == False :
    #            return session
    #    return None

    # def _run_job(self, session, job: Job) :
    #    try :
    #        #find method session having job.method_name
    #        method = getattr(session, job.operation_name)
    #        #print(f"[PoolManager] Running job: {job.operation_name}  on session: {session.session_id}")
    #        if job.params is None :
    #            result = method()
    #        else :
    #            result = method(**job.params)
    #        job.future.set_result(result)
    #    except Exception as e :
    #        job.future.set_exception(e)

    def _active_sessions(self) -> list[MpiApiSession]:
        if self.using_global:
            return [self.global_session] if self.global_session is not None else []
        return list(self.sessions)

    @contextmanager
    def sleeping_workers(self):
        """Put the active workers in their sleep loop for the duration of the scope."""
        sessions = self._active_sessions()
        slept_sessions = []
        try:
            for session in sessions:
                session.sleep()
                slept_sessions.append(session)
            yield
        finally:
            for session in reversed(slept_sessions):
                session.wake()

    def set_all_positions(self, positions):
        # print("[Manager] Setting positions to all sessions.")
        for session in self.sessions:
            session.set_positions(positions=positions)

    def setup_otf_cycle(self, params) -> None:
        """Reload potentials in all sessions. Ends in global mode, ready for minimize."""
        self.use_local()
        for session in self.sessions:
            session.setup_otf_cycle(params)
        if self.global_session is not None:
            self.use_global()
            self.global_session.setup_otf_cycle(params)

    def submit_job(self, method_name: str, params: dict = None) -> Future:
        future = Future()
        job = Job(method_name, params, future)
        # print(f"[PoolManager] Submitting job: {job.operation_name}") #with params: {job.params}")
        self.job_queue.put(job)
        return future

    # API

    def minimize(self, params):
        future = self.submit_job("minimize", {"params": params})
        return future

    def minimize_with_results(self, params, configuration):
        future = self.submit_job(
            "minimize_with_results",
            {"params": params, "configuration": configuration},
        )
        return future

    def get_potential_energy(self, positions=None):
        future = self.submit_job("get_potential_energy", {"positions": positions})
        return future

    def get_total_energy(self, positions=None):
        future = self.submit_job("get_total_energy", {"positions": positions})
        return future

    def partn_search(
        self, params, central_atom: list[int], configuration
    ) -> list[Future]:
        futures = []
        for atom in central_atom:
            f = self.submit_job(
                "partn_search",
                {
                    "params": params,
                    "central_atom_idx": atom,
                    "configuration": configuration,
                },
            )
            futures.append(f)
        return futures

    def partn_refine(
        self,
        params,
        central_atom: int,
        configuration,
        saddle_idx=None,
        saddle_positions=None,
        minimize_outter_atoms: bool = True,
        num_reference_event: int | None = None,
        symmetry_index: int | None = None,
    ) -> Future:
        future = self.submit_job(
            "partn_refine",
            {
                "params": params,
                "central_atom_idx": central_atom,
                "configuration": configuration,
                "saddle_idx": saddle_idx,
                "saddle_positions": saddle_positions,
                "minimize_outter_atoms": minimize_outter_atoms,
                "num_reference_event": num_reference_event,
                "symmetry_index": symmetry_index,
            },
        )
        return future

    def close_all(self):
        """
        Close all sessions and their underlying engines.
        """
        # print("[PoolManager] Closing all sessions.")
        # global_session shares its master rank with self.sessions[0] (see ManagerFactory),
        # so closing it separately races with that session's close over the same rank.
        # use_local() is required here: each session's close message is only scoped to
        # that session's own chunk if the engines are already in local mode when they
        # receive it -- in global mode the same message broadcasts to every chunk at
        # once, so the first session's close kills every engine and every later
        # session's close then waits forever for a reply from an already-dead rank.
        self.use_local()
        self._stop_workers()
        for session in self.sessions:
            session.close(wait_status=True)

    def _stop_workers(self):
        """Discard queued jobs and stop worker threads before touching sessions directly.

        Otherwise a worker thread can still be mid-flight on a session (or pick up a
        leftover queued job) while close_all() closes that same session, either racing
        it over the shared MPI channel or blocking forever sending to a rank that has
        already exited its engine loop.
        """
        while True:
            try:
                job = self.job_queue.get_nowait()
            except queue.Empty:
                break
            job.future.set_exception(RuntimeError("Manager is closing; job discarded."))
            self.job_queue.task_done()

        for _ in self.workers:
            self.job_queue.put(None)
        for worker in self.workers:
            worker.join()

    def __getattr__(self, name: str):
        """Check if method start with global_, if yes, then return global_session.method"""
        if name.startswith("global_"):
            method_name = name[7:]  # remove prefixe 'global_'
            if not self.global_session:
                raise RuntimeError("Global session is not available")

            def global_method(*args, **kwargs):
                method = getattr(self.global_session, method_name)
                return method(*args, **kwargs)

            return global_method

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

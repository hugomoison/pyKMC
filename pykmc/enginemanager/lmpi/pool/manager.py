"""Class for managing a pool of LAMMPS instances."""
from ..sessions import MpiApiSession
from dataclasses import dataclass
from concurrent.futures import Future
import queue
import threading

#TODO : commented print should be log depending of the verbosity but need to thing of how we modify log before (also loggers are 
#initiated in kmc, after the initialization of manager ...))

@dataclass 
class Job: 
    operation_name: str 
    params: dict
    future: Future


class Manager:
    """A class to manage a pool of Lammps sessions."""

    def __init__(self, sessions: list[MpiApiSession], global_session: MpiApiSession = None) -> None:
        """
        Initialize the LammpsPoolManager with a specified number of sessions.
        """
        self.sessions = sessions
        self.global_session = global_session
        self.using_global = True
        self.job_queue: queue.Queue[Job] = queue.Queue()
        #Thread that dispatch job to workers
        #self.dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True)
        #self.dispatcher_thread.start()

        #pool de workers
        self.workers = []
        for session in self.sessions:
            t = threading.Thread(target=self._worker_loop, args=(session,), daemon=True)
            t.start()
            self.workers.append(t)

    def broadcast_command(self, cmd: str):
        """
        Send the same command to all sessions and wait for all to finish.
        """
        #print("[PoolManager] Broadcasting command:", cmd)
        for session in self.sessions:
            session.command(cmd)

    def initialize_sessions(self, config, system) :
        """
        Initialize engines with the same system and config
        """
        from ....config import pbc_to_lammps_boundary
        if config.lammps.boundary is not None:
            boundary = config.lammps.boundary
        else:
            boundary = pbc_to_lammps_boundary(system.pbc) if system.pbc is not None else "p p p"
        print("[Manager] use local")
        self.use_local()
        print("[Manager] Initializing all Lammps engines")
        for session in self.sessions :
            session.initialize_parameters(boundary=boundary)
            session.initialize_system(system)
            session.initialize_potential(config)
        print("[Manager] use global")
        self.use_global()
        print("[Manager] Initializing global Lammps engines")
        if self.global_session is not None :
            self.global_initialize_parameters(boundary=boundary)
            self.global_initialize_system(system)
            self.global_initialize_potential(config)


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
            for session in self.sessions :
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

    #def _dispatcher(self) :
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


    #def _get_available_engine(self) :
    #    """Check if worker is available, if yes return worker, if not, return None"""
    #    for session in self.sessions :
    #        if session._is_busy == False :
    #            return session
    #    return None

    #def _run_job(self, session, job: Job) :
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

    def reinitialize_all_sessions(self, config, system):
        """Reinitialize all LAMMPS sessions with a new system after atom removal."""
        self.use_local()
        for session in self.sessions:
            session.reinitialize_system(config, system)
        self.use_global()
        if self.global_session is not None:
            self.global_session.reinitialize_system(config, system)

    def set_all_positions(self, positions) :
        #print("[Manager] Setting positions to all sessions.")
        for session in self.sessions : 
            session.set_positions(positions=positions)

    def submit_job(self, method_name: str, params: dict = None) -> Future:

        future = Future()
        job = Job(method_name, params, future)
        #print(f"[PoolManager] Submitting job: {job.operation_name}") #with params: {job.params}")
        self.job_queue.put(job)
        return future

    # API

    def minimize(self, config ) : 
        future = self.submit_job("minimize", {"config" : config})
        return future
    
    def minimize_with_results(self, config, positions=None, types=None) :
        future = self.submit_job("minimize_with_results", {"config": config, "positions": positions, "types": types})
        return future
    
    def get_potential_energy(self, positions=None) :
        future = self.submit_job("get_potential_energy", {"positions": positions})
        return future

    def get_total_energy(self, positions=None) :
        future = self.submit_job("get_total_energy", {"positions": positions})
        return future

    def partn_search(self, config, central_atom: list[int], positions=None, cell=None, types=None) -> list[Future] :
        futures = []
        for atom in central_atom :
            f = self.submit_job("partn_search", {"config": config, "central_atom_idx": atom, "positions": positions, "cell":cell, "types":types})
            futures.append(f)
        return futures

    def partn_refine(self, config, central_atom: int, positions=None, cell=None, types=None, saddle_idx=None, saddle_positions=None) -> list[Future] :
        future = self.submit_job("partn_refine", {"config": config, "central_atom_idx": central_atom, "positions": positions, "cell":cell, "types":types, "saddle_idx":saddle_idx, "saddle_positions":saddle_positions})
        return future

    def close_all(self):
        """
        Close all sessions and their underlying engines.
        """
        #print("[PoolManager] Closing all sessions.")
        if self.global_session is not None :
            self.global_session.close(wait_status=False)
        for session in self.sessions:
            session.close(wait_status=True)


    def __getattr__(self, name:str) :
        """Check if method start with global_, if yes, then return global_session.method"""
        if name.startswith('global_'):
            method_name = name[7:]  # remove prefixe 'global_'
            if not self.global_session:
                raise RuntimeError("Global session is not available")

            def global_method(*args, **kwargs):
                method = getattr(self.global_session, method_name)
                return method(*args, **kwargs)

            return global_method

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")




from .session import Session
from concurrent.futures import Future
from dataclasses import dataclass, field 
import queue
import threading 

@dataclass 
class Job: 
    """Unit of work dispatched to a thread worker.

    Parameters
    ----------
    op_name : str
        Operation name in the Worker registry.
    kwargs : dict
        Keyword arguments forwarded to the operation.
    future : Future
        Resolved with the result once the job completes.
    use_global : bool
        If the job is run on global mode.
        set automatically by submit() and submit_global()
    """
    op_name: str 
    kwargs: dict = field(default_factory=dict)
    future: Future = field(default_factory=Future)
    use_global: bool = False

class Manager: 

    def __init__(self, local_sessions: list[Session], global_session: Session|None = None) -> None : 

        self.local_sessions = local_sessions 
        self.global_session = global_session

        self._local_queue: queue.Queue[Job] = queue.Queue()
        self._global_queue: queue.Queue[Job] = queue.Queue()
        self._local_threads = []
        self._global_thread = None

        self.using_global = False

    def start(self) -> None: 
        for session in self.local_sessions : 
            t = threading.Thread(target=self._worker_loop, args=(session, self._local_queue), daemon=True)
            t.start()
            self._local_threads.append(t)

        if self.global_session is not None: 
            self._global_thread = threading.Thread(target=self._worker_loop, args = (self.global_session, self._global_queue), daemon = True )
            self._global_thread.start()

    def close(self) -> None : 
        """Stop all thread worker and close sessions""" 
        for _ in self._local_threads : 
            self._local_queue.put(None)
        for t in self._local_threads : 
            t.join() 
        
        if self._global_thread is not None : 
            self._global_queue.put(None)
            self._global_thread.join() 
        
        for session in self.local_sessions : 
            session.close()
        
        if self.global_session is not None : 
            self.global_session.close()
            
    def broadcast(self, op_name:str, **kwargs) -> None : 
        """Send the same op to all sessions sequantially.i

        Usefull when initializing and mode switching."""

        for session in self.local_sessions : 
            session.call(op_name, **kwargs)

    def _use_local(self) -> None:
        """Switch all workers to local mode."""
        self._global_queue.join() #wait for global job to finish
        self.global_session.use_local()
        self.using_global = False

    def _use_global(self) -> None:
        """Wait for local queue to drain, then switch all workers to global."""
        self._local_queue.join()   # wait for all local jobs to finish
        for session in self.local_sessions :
            session.use_global()
        self.using_global = True

    def _worker_loop(self, session: Session, job_queue: queue.Queue)-> None : 
        """Pull jobs from the queue and execute via the session."""
        while True : 
            job = job_queue.get() 
            if job is None: # sentinel -stop -> read when want to close
                break 
            try : 
                result = session.call(job.op_name, **job.kwargs) 
                job.future.set_result(result)
            except Exception as e : 
                job.future.set_exception(e)
            finally : 
                job_queue.task_done()

    

    def submit(self, op_name: str, **kwargs) -> Future: 
        """Submit a job to the local worker pool.

        Parameters
        ----------
        op_name : str
            Operation name in the Worker registry.
        **kwargs
            Forwarded to the operation.

        Returns
        -------
        Future
            Resolved when the job completes.
        """ 
        print(f"submit called, using_global={self.using_global}")
        if self.using_global : 
            self._use_local()
        job = Job(op_name=op_name, kwargs=kwargs)
        self._local_queue.put(job)
        return job.future 
    
    def submit_global(self, op_name: str, **kwargs) -> Future:
        """Submit a job to the global worker.

        Parameters
        ----------
        op_name : str
        **kwargs

        Returns
        -------
        Future

        Raises
        ------
        RuntimeError
            If no global session was configured.
        """
        if self.global_session is None:
            raise RuntimeError("No global session configured.")
        if not self.using_global : 
            self._use_global()
        job = Job(op_name=op_name, kwargs=kwargs, use_global=True)
        self._global_queue.put(job)
        return job.future

    def __getattr__(self, name: str) : 
        """Auto-generate submit/submit_global wrappers.

        mgr.minimize(positions=pos)        → submit("minimize", positions=pos)
        mgr.global_minimize(positions=pos) → submit_global("minimize", positions=pos)
        """
        if name.startswith("global_"):
            op = name[len("global_"):]
            return lambda **kw: self.submit_global(op, **kw)
        return lambda **kw: self.submit(name, **kw)



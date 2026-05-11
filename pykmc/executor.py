from mpi4py import MPI
import inspect

class Executor() :
    """
    MPI executor used to run a *parallelizable* function simultaneously on all MPI ranks.

    This executor is instantiated on *all* ranks by the factory,
    but it is intended to be called **only from rank 0**.

    Design philosophy
    -----------------
    - The executor is responsible only for broadcasting the command
      (function + parameters) to all MPI ranks.
    - All ranks receive and execute the function.
    - The *function itself* is responsible for its parallel logic
      (scatter, local compute, gather, reduction, etc.) and must return
      a meaningful result on rank 0.
    - The executor *does not* perform scatter or gather: it simply returns
      whatever the function returns locally.
    - The parallelized function should have a comm parameter

    Usage
    -----
    On rank 0 :
    executor = Executor(comm)
    result = executor.run(fonction, arg1, arg2)
    executor.shutdown()

    On rank > 0 :
    executor = Executor(comm)
    executor.worker_loop()

    Result behavior
    ---------------
    - On rank 0: returns `result_local` → the final result of the parallel computation.
    - On rank > 0: returns `None`.

    This makes usage simple:
        result = executor.run(graph, neighbors, env)
        # `result` contains the final output only on rank 0
    """

    def __init__(self, comm:MPI.Comm) -> None:
        self.comm = comm
        self.rank = comm.Get_rank()

    def run(self, func, *args, **kwargs)  :

        if self.rank != 0 :
            raise RuntimeError("run() should be called from rank 0")
        
        if 'comm' not in inspect.signature(func).parameters:
            raise TypeError(f"{func.__name__}() must have a `comm` parameter")

        #add comm to kwargs
        kwargs['comm'] = self.comm

        # Broadcast command
        cmd = (func, args, kwargs)
        cmd = self.comm.bcast(cmd, root=0)
        func, args, kwargs = cmd


        #rank 0
        result_local = func(*args, **kwargs) # result rank 0

        return result_local

    def worker_loop(self) -> None:

        if self.rank == 0:
            raise RuntimeError("worker_loop() should not be called from rank 0.")

        while True:
            #receive command
            cmd = self.comm.bcast(None, root=0)

            if isinstance(cmd, str) and cmd == 'shutdown':
                break

            #excute command
            func, args, kwargs = cmd
            _ = func(*args, **kwargs)

    def close(self) :
        if self.rank != 0 :
            raise RuntimeError("close() should be call from rank 0.")

        self.comm.bcast("shutdown", root=0)
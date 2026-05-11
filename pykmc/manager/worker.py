from __future__ import annotations
from typing import Callable 
from mpi4py import MPI 
import inspect
from pykmc import Engine

def build_registry(obj: object) -> dict[str, Callable]:
    """
    Scan an object and return a dict of its public callable methods.

    Used by the Worker to discover available operations on an engine,
    including any extensions registered via ``engine.register()``.

    Parameters
    ----------
    obj : object
        Any instance — typically an Engine with optional extensions.

    Returns
    -------
    dict[str, Callable]
        Mapping of method name → bound method for all public callables.
    """
    return {
        name: method
        for name, method in inspect.getmembers(obj, predicate=callable)
        if not name.startswith("_")
    }


class Worker : 

    def __init__(self, local_engine: Engine, local_comm: "MPI.COMM", engine_id:int, global_engine: Engine|None = None, global_comm: "MPI.COMM"|None = None, world_comm: "MPI.COMM"|None = None)-> None : 
        """MPI worker. Runs on all ranks in the engine communicator.

        Rank 0 reads incoming messages from ``MPI.COMM_WORLD``, broadcasts
        them to all ranks in the active communicator (local or global), and
        dispatches to the engine registry. Other ranks only participate in
        the collective broadcast and execute the dispatched operation.

        The worker supports two modes:
        - **global mode** (default): uses ``global_engine`` and ``global_comm``.
        - **local mode**: uses ``local_engine`` and ``local_comm``.

        Mode is switched via ``use_local`` / ``use_global`` builtins, which
        can be sent as messages from the session.

        Extensions registered on the engine (via ``engine.register()``) are
        automatically included in the registry — no extra wiring needed.

        Parameters
        ----------
        local_engine : Engine
            Started engine instance using ``local_comm``.
        local_comm : MPI.Comm
            Sub-communicator for local mode.
        global_engine : Engine, optional
            Started engine instance using ``global_comm``.
            If None, worker starts in local mode.
        global_comm : MPI.Comm, optional
            Sub-communicator for global mode.
        world_comm : MPI.Comm, optional
        Parent communicator from which ``local_comm`` and ``global_comm``
        were derived (typically via ``Split()``). Used for point-to-point
        messaging with the session (send/recv). Defaults to
        ``MPI.COMM_WORLD``. Pass a ``Dup()`` to avoid tag conflicts if
        necessary.

        Raises
        ------
        ValueError
            If ``local_engine`` and ``global_engine`` expose different operations.
            Both engines must have the same extensions registered.
        """ 

        self.local_engine = local_engine
        self.local_comm = local_comm 
        self.global_engine = global_engine
        self.global_comm = global_comm
        self.local_rank = local_comm.Get_rank()
        self.engine_id = engine_id
        self._is_alive = False

        self.world_comm = world_comm or MPI.COMM_WORLD

        self._builtins_op = {"use_local" : self.use_local, 
                             "use_global" : self.use_global, 
                             "close" : self.close}
        self.local_registry = build_registry(local_engine)


        if self.global_comm is not None : 
            self.global_rank = self.global_comm.Get_rank()
            self.global_registry = build_registry(global_engine)
            self._check_registry()
            self.use_global() #start using global mode
        else : 
            self.use_local()
        

    def _check_registry(self) : 
        """Raise if local and global engines expose different operations.

        Both engines must have the same extensions registered to ensure
        consistent behaviour regardless of which mode is active.
        """
        local_ops = set(self.local_registry)
        global_ops = set(self.global_registry)
        if local_ops != global_ops : 
            diff = local_ops.symmetric_difference(global_ops)
            raise ValueError(f"local_engine and global_engine have different operations: {diff}."
                             f"Register the same extension on both engines.")

    def use_local(self) -> None : 
        """Switch active communicator and rank to local mode."""
        self.global_mode = False 
        self.comm = self.local_comm 
        self.rank = self.local_comm.Get_rank()
        self.registry = self.local_registry

    def use_global(self) -> None : 
        """Switch active communicator and rank to global mode."""
        if self.global_comm is None : 
            return #Worker is not using global mode
        self.global_mode = True 
        self.comm = self.global_comm 
        self.rank = self.global_comm.Get_rank()
        self.registry = self.global_registry

    def start(self) -> None : 
        """Enter the message loop. Blocks until ``close`` is dispatched."""
        self._is_alive = True
        self._loop()

    def close(self) -> None : 
        """Stop the message loop and close both engines."""
        self._is_alive = False 
        self.local_engine.close() 
        if self.global_engine is not None : 
            self.global_engine.close()

    def _loop(self) -> None : 
        """Main loop. All ranks run this until ``_is_alive`` is False.

        Rank 0 reads from ``world_comm`` and broadcasts to all ranks
        in the active communicator. All ranks then execute the operation
        collectively.
        """

        while self._is_alive: 

            if self.rank == 0 : 
                msg = self._read_messages()
                if msg is None : 
                    continue 
            else : 
                msg = None 

            #Broadcast to all rank in self.comm 
            msg = self.comm.bcast(msg, root=0)
            if msg is None : 
                continue 

            result = self._dispatch(msg)

            #return message 
            if self.rank == 0 and result !="_no_status" :
                self.world_comm.send({"type" : "status", "value": {"alive": self._is_alive, "has_result": result is not None}}, dest = 0, tag = 0)
                if result is not None :
                    self.world_comm.send({"type" : "result", "value": result}, dest = 0, tag = 1)
            
            if not self._is_alive : 
                break
    
    def _read_messages(self) -> None : 
        """Read one message from ``MPI.COMM_WORLD`` on rank 0.

        Returns
        -------
        dict or None
            The received message, or None if no message is available.
        """
        if self.world_comm.probe(source = MPI.ANY_SOURCE, tag = 2) : 
            return self.world_comm.recv(source = MPI.ANY_SOURCE, tag = 2)
        return None
    
    def _dispatch(self, msg: dict) -> None : 
        """Dispatch a message to the appropriate handler.

        Builtins (``use_local``, ``use_global``, ``close``) take priority
        over registry operations.

        Parameters
        ----------
        msg : dict
            Message with ``"type"`` (operation name) and optional ``"value"``
            (kwargs dict or scalar).

        Returns
        -------
        Any
            Result of the operation, or None for void operations.
        """

        op_type = msg.get("type") 

        #Check if class methods
        if op_type in self._builtins_op : 
            self._builtins_op[op_type]() 
            return "_no_status" #special case

        #Find operation in registry 
        handler = self.registry.get(op_type) 
        if handler is None : 
            raise ValueError(f"Unknown operation '{op_type}'."
                             f"Available: {list(self._builtins_op)+list(self.registry)}")

        value = msg.get("value")
        #get parameters 
        if value is None : 
            kwargs = {} 
        elif isinstance(value, dict) : 
            kwargs = value 
        else : 
            kwargs = {"value" : value}

        #syncrhonise  
        self.comm.barrier() 

        try : 
            return handler(**kwargs) 
        except Exception as e : 
            print(f"[Worker rank {self.rank}] Error in '{op_type}': {e}")
        finally : 
            self.comm.barrier()

    def __repr__(self) -> str : 
        ops = list(self.registry)
        builtins = list(self._builtins_op)
        return(
            f"Worker(\n"
            f" ops      = {ops}, \n"
            f" builtins = {builtins}\n"
            f")"
        )
from __future__ import annotations
from mpi4py import MPI 
from typing import Any

class Session : 

    def __init__(self, engine_master_rank: int, session_id: int = 0, world_comm: "MPI.COMM"|None = None) -> None : 

        self.engine_master_rank = engine_master_rank 
        self.session_id = session_id
        self.world_comm = world_comm or MPI.COMM_WORLD
        self._is_alive = False 
        self._is_busy = False 

        if self.world_comm.Get_rank() != 0 : 
            raise RuntimeError("Session must be used from rank 0.")
        
    def call(self, op_name: str, **kwargs) -> Any : 
        """Send an operation to the worker and optionally retrieve a result."""

        self._is_busy = True 

        try :
            msg = {"type": op_name}
            if kwargs:
                msg["value"] = kwargs
            self.world_comm.send(msg, dest = self.engine_master_rank, tag=2)
            has_result = self._recv_status()
            if has_result :
                return self._recv_result()
        finally : 
            self._is_busy = False 

    def _recv_status(self) -> bool :
        msg = self.world_comm.recv(source = self.engine_master_rank, tag = 0)
        if msg.get("type") != "status":
            raise RuntimeError(
                f"Expected 'status', got '{msg.get('type')}'"
            )
        value = msg.get("value", {})
        self._is_alive = value.get("alive", False)
        return value.get("has_result", False)

    def _recv_result(self) -> Any : 
        msg = self.world_comm.recv(source=self.engine_master_rank, tag = 1)
        if msg.get("type") != "result" : 
            raise RuntimeError(f"Expected 'result' got '{msg.get('type')}'")
        return msg["value"]
    
    def is_alive(self) -> bool : 
        return self._is_alive
    
    def is_busy(self) -> bool : 
        return self._is_busy

    

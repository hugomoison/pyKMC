from __future__ import annotations
import pytest 
from mpi4py import MPI 
from pykmc import Worker, Engine 
import numpy as np

class FakeEngine(Engine) : 
    """ Minimial Enginge with simple testable operations."""

    def __init__(self, comm: "MPI.COMM" = None, **kwargs) : 
        super().__init__()
        self.comm = comm

    def start(self)                          : pass
    def close(self)                          : pass
    def initialize_parameters(self)          : pass
    def initialize_system(self, **kwargs)    : pass
    def initialize_potential(self, **kwargs) : pass
    def get_positions(self)                  : return None
    def set_positions(self, positions)       : pass
    def get_total_energy(self, **kwargs)     : return None
    def get_potential_energy(self, **kwargs) : return None
    def minimize(self, **kwargs)             : pass
    def minimize_with_results(self, **kwargs): return None

    def collective(self) -> float | None:
        """Reduce sum of ranks — result depends on which comm is active."""
        rank  = self.comm.Get_rank()
        local = np.array([float(rank)])
        total = np.zeros(1) if rank == 0 else None
        self.comm.Reduce(local, total, op=MPI.SUM, root=0)
        return float(total[0]) if rank == 0 else None

_BUILTIN_OPS = {"use_local", "use_global", "close"}

def _call(op_name: str, world_comm: MPI.Comm, expect_result: bool = False, **kwargs):
    """Simulate a session call to worker master (rank 1)."""
    msg = {"type": op_name}
    if kwargs:
        msg["value"] = kwargs
    world_comm.send(msg, dest=1, tag=2)
    if op_name in _BUILTIN_OPS:
        return  # builtins return "_no_status" — worker sends nothing
    status = world_comm.recv(source=1, tag=0)
    assert status["type"] == "status"
    if expect_result:
        result = world_comm.recv(source=1, tag=1)
        assert result["type"] == "result"
        return result["value"]


class TestWorker:

    @pytest.fixture(autouse=True)
    def require_mpi(self):   
        if MPI.COMM_WORLD.Get_size() < 1:
            pytest.skip("Requires mpirun -n > 1")

    @pytest.fixture(autouse=True)
    def setup(self):
        world_comm   = MPI.COMM_WORLD
        rank         = world_comm.Get_rank()
        local_color  = 0 if rank == 1           else MPI.UNDEFINED
        global_color = 0 if rank != 0           else MPI.UNDEFINED
        local_comm   = world_comm.Split(color=local_color,  key=rank)
        global_comm  = world_comm.Split(color=global_color, key=rank)
        self.world_comm = world_comm   
        self.rank       = rank

        if rank != 0:
            local_engine  = FakeEngine(comm=local_comm  if local_comm  != MPI.COMM_NULL else global_comm)
            global_engine = FakeEngine(comm=global_comm)
            lc = local_comm if local_comm != MPI.COMM_NULL else global_comm
            Worker(
                local_engine=local_engine,
                local_comm=lc,
                global_engine=global_engine,
                global_comm=global_comm,
                world_comm=world_comm,
            ).start()   # ← bloque ici jusqu'à close

    def test_local_and_global_collective(self):
        if self.rank == 0:
            _call("use_local", self.world_comm)
            r1 = _call("collective", self.world_comm, expect_result=True)
            assert r1 == 0.0

            _call("use_global", self.world_comm)   # ← world_comm obligatoire
            r2 = _call("collective", self.world_comm, expect_result=True)
            expected = float(np.sum(np.arange(0, self.world_comm.Get_size()-1)))
            assert r2 == expected

            _call("close", self.world_comm) 
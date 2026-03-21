from .test_worker import FakeEngine
from pykmc import Worker, Session
import pytest 
import numpy as np 
from mpi4py import MPI 

class TestSession: 

    @pytest.fixture(autouse=True)
    def require_mpi(self):   
        if MPI.COMM_WORLD.Get_size() < 1:
            pytest.skip("Requires mpirun -n > 1")

    @pytest.fixture(autouse=True)
    def setup(self) : 
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
            ).start()
        
        else : 
            self.session = Session(engine_master_rank=1, world_comm=world_comm)
            #self.session.call("start")
        
    def test_local_and_global_collective(self) : 
        if self.rank == 0 : 
            print("here")
            self.session.call("use_local")
            r1 = self.session.call("collective") 
            assert r1 == 0.0 

            print("here")
            self.session.call("use_global")
            r2 = self.session.call("collective") 
            expected = float(np.sum(np.arange(0, self.world_comm.Get_size()-1)))
            assert r2 == expected 

            self.session.call("close")
            


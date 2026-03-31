from pykmc import Manager, Session, Worker
from .test_worker import FakeEngine
import pytest 
from mpi4py import MPI
import numpy as np

class TestManager: 

    @pytest.fixture(autouse=True)
    def require_mpi(self)  : 
        if MPI.COMM_WORLD.Get_size() < 3 : 
            pytest.skip("Require mpirun -n > 2") 
        
    @pytest.fixture(autouse=True)
    def setup(self) : 
        world_comm = MPI.COMM_WORLD 
        self.rank = world_comm.Get_rank()
        self.available = list(range(1,world_comm.Get_size()))
        #2 sessions 
        self.chunks = [arr.tolist() for arr in np.array_split(self.available, 2)]
        local_color  = next((i for i, c in enumerate(self.chunks) if self.rank in c), MPI.UNDEFINED)
        global_color = 0 if self.rank in self.available else MPI.UNDEFINED

        local_comm  = world_comm.Split(color=local_color,  key=self.rank)
        global_comm = world_comm.Split(color=global_color, key=self.rank) 

        if self.rank != 0:
            Worker(
                local_engine=FakeEngine(comm=local_comm  if local_comm  != MPI.COMM_NULL else global_comm),
                local_comm=local_comm   if local_comm  != MPI.COMM_NULL else global_comm,
                global_engine=FakeEngine(comm=global_comm),
                global_comm=global_comm,
            ).start()

        else : 
            #On rank 0 
            self.manager = Manager(
                local_sessions=[
                    Session(engine_master_rank=self.chunks[0][0]),
                    Session(engine_master_rank=self.chunks[1][0]),
                ],
                global_session=Session(engine_master_rank=self.available[0]),
            )
            self.manager.start()


    def test_local_then_global(self):
        """Submit local jobs then a global job — mode switches correctly."""

        local_size    = len(self.chunks[0])
        global_size   = len(self.available)
        local_expected  = float(sum(range(local_size)))
        global_expected = float(sum(range(global_size)))

        if self.rank == 0:
            self.manager._use_local()
            #submit 10 local jobs
            futures = [self.manager.submit("collective") for _ in range(10)]
            results = [f.result() for f in futures]
            assert all(r == local_expected for r in results)

            self.manager._use_global()
            f3 = self.manager.submit_global("collective")
            assert f3.result() == global_expected

            self.manager.close() 
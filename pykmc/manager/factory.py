from mpi4py import MPI 
import numpy as np
from .manager import Manager
from pykmc import Engine, LammpsEngine
from .worker import Worker
from .session import Session
from typing import Any


class ManagerFactory: 

    """ 
    Responsible for splitting ranks and instantiating Sessions, Workers, Engines, 
    and returning a configured Manager
    """

    def __init__(self, engine_style:str, n_workers: int, comm: MPI.Comm, has_global:bool = True, global_size:int = None, engine_config: Any | None = None) -> None : 

        self.engine_style = engine_style
        self.engine_config = engine_config
        self.comm = comm 
        self.n_workers = n_workers 
        self.has_global = has_global 

        self.start_rank = 1 #we don't use rank 0 for workers
        self.size = self.comm.Get_size() 
        self.rank = self.comm.Get_rank() 
        self.global_size = global_size if global_size is not None else  self.size-self.start_rank

        


        if self.size < self.n_workers + self.start_rank : 
            raise ValueError("Not enough MPI ranks to allocates workers")
        
        self.available_ranks = list(range(self.start_rank, self.size))

        self.chunks = self._split_ranks() 

        if self.global_size is not None and self.global_size % len(self.chunks[0]) != 0 : 
            raise ValueError("The global engine must run on a mulitple of local engine size.")

    def _split_ranks(self) -> list[list[int]] : 
        split_arrays = np.array_split(self.available_ranks, self.n_workers) 
        chunks = [arr.tolist() for arr in split_arrays] 

        return chunks 
    
    def launch(self) -> Manager | None : 

        my_color = MPI.UNDEFINED 
        worker_id = None 
        for session_id, chunk in enumerate(self.chunks) : 
            if self.rank in chunk : 
                my_color = session_id+1 
                worker_id = session_id 
                break

        #split communicator 
        worker_comm = self.comm.Split(color=my_color, key=self.rank)

        global_comm = None
        if self.has_global : 
                if self.rank < self.start_rank or self.rank >= self.start_rank + self.global_size :
                    global_comm = self.comm.Split(color=MPI.UNDEFINED, key=self.rank)
                else : 
                    global_comm = self.comm.Split(color=1, key=self.rank)
                if global_comm == MPI.COMM_NULL : 
                    global_comm = None


        if worker_id is not None : #rank is in a chunk 
            worker = self._create_worker(engine_style = self.engine_style, local_comm=worker_comm, engine_id = worker_id, global_comm = global_comm, engine_config = self.engine_config)
            worker.start()

        else : #On rank 0 
            manager = Manager(local_sessions=[Session(engine_master_rank=self.chunks[i][0], world_comm=self.comm,  session_id=i+1) for i in range(self.n_workers)], 
                              global_session=Session(engine_master_rank=self.available_ranks[0], session_id=0, world_comm=self.comm))
            manager.start() 
            return manager
            


    def _create_worker(self, engine_style: str, local_comm: MPI.Comm , engine_id: int, global_comm: MPI.Comm | None, engine_config: Any | None = None) -> Worker : 
        match engine_style:
            case 'lammps':
                if engine_config is None:
                    raise ValueError("Lammps need a config object to be initialized.")
            
                local_engine = LammpsEngine(config=engine_config, comm=local_comm, engine_id=engine_id)
                local_engine.start()
                local_engine.initialize_parameters()

                global_engine = None
                if global_comm is not None : 
                    global_engine = LammpsEngine(config=engine_config, comm=global_comm, engine_id=0)
                    global_engine.start()
                    global_engine.initialize_parameters()

                return Worker(
                    local_engine=local_engine,
                    local_comm=local_comm,
                    engine_id=engine_id,
                    global_engine=global_engine,
                    global_comm=global_comm,
                    world_comm=self.comm
                )
            case _ : 
                raise ValueError("Unknown engine style")
            

        
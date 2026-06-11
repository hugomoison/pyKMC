from mpi4py import MPI
from ...messenger import MpiMessenger, QueueMessenger
from ...lmpi.pool import Manager
from ...lmpi.sessions import MpiApiSession
from ...lmpi.engines import MpiApiEngine
import numpy as np
import threading


class ManagerFactory:
    """
    Responsible for splitting ranks and instantiating Engines, Sessions,
    and returning a configured PoolSessionManager.
    """

    def __init__(self, n_sessions: int, use_rank_0: bool, has_global: bool = True):
        self.world = MPI.COMM_WORLD
        self.world_rank = self.world.Get_rank()
        self.world_size = self.world.Get_size()
        self.n_sessions = n_sessions
        self.use_rank_0 = use_rank_0
        self.has_global = has_global

        if self.use_rank_0:
            self.start_rank = 0
        else:
            self.start_rank = 1

        if self.world_size < n_sessions + self.start_rank:
            raise ValueError("Not enough MPI ranks to allocate sessions")

        self.available_ranks = list(range(self.start_rank, self.world_size))
        self.chunks = self._split_ranks()

    def _split_ranks(self) -> list[list[int]]:
        split_arrays = np.array_split(self.available_ranks, self.n_sessions)
        chunks = [arr.tolist() for arr in split_arrays]

        return chunks

    def launch(self) -> Manager | None:

        my_color = MPI.UNDEFINED
        engine_id = None
        for session_id, chunk in enumerate(self.chunks):
            if self.world_rank in chunk:
                my_color = session_id + 1
                engine_id = session_id
                break

        # Split communicator
        engine_comm = self.world.Split(color=my_color, key=self.world_rank)

        sessions = []

        # messenger for each chunk
        messengers = []
        for session_id, chunk in enumerate(self.chunks):
            if 0 in chunk:
                messengers.append(QueueMessenger())
            else:
                messengers.append(MpiMessenger(comm=self.world))

        # communicator and messenger for global
        if self.world_rank < self.start_rank:
            global_comm = self.world.Split(color=MPI.UNDEFINED, key=self.world_rank)
        else:
            global_comm = self.world.Split(color=1, key=self.world_rank)
        global_messenger = MpiMessenger(comm=self.world)

        if engine_id is not None:  #  rank in a chunk
            messenger = messengers[engine_id]
            engine = MpiApiEngine(
                local_messenger=messenger,
                local_engine_comm=engine_comm,
                local_engine_id=engine_id + 1,
                global_messenger=global_messenger,
                global_engine_comm=global_comm,
                global_engine_id=0,
            )
            engine.start()  # bloque ici

        # --- Session (On rank 0) ---
        if self.world_rank == 0:
            print("rank0")
            for session_id, chunk in enumerate(self.chunks):
                messenger = messengers[session_id]
                print(f"[Factory] Creating session {session_id} for ranks: {chunk}")
                session = MpiApiSession(
                    messenger=messenger, engine_ranks=chunk, session_id=session_id + 1
                )
                sessions.append(session)

            global_session = MpiApiSession(
                messenger=global_messenger,
                engine_ranks=list(range(self.start_rank, self.world_size)),
                session_id=0,
            )
            return Manager(sessions=sessions, global_session=global_session)

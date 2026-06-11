from mpi4py import MPI
import numpy as np
from ...messenger import MpiMessenger
from threading import RLock
from functools import wraps
# TODO more general way to deal with operations
# TODO : commented print should be log depending of the verbosity but need to thing of how we modify log before (also loggers are
# initiated in kmc, after the initialization of manager ...))


def session_locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            self._is_busy = True
            try:
                return method(self, *args, **kwargs)
            finally:
                self._is_busy = False

    return wrapper


class MpiApiSession:
    """A class to manage an MPI API session for LAMMPS.
    This class provides an interface to send messages to the Lammps MPI API engine.
    It should live on the rank 0 of the MPI World communicator.
    It should knows on which ranks the LAMMPS engine is running.
    """

    def __init__(self, messenger: MpiMessenger, engine_ranks, session_id) -> None:
        self.messenger = messenger
        self.engine_ranks = engine_ranks
        self.engine_master_rank = engine_ranks[0]
        self.session_id = session_id
        self._is_alive = False
        self._is_busy = False
        self._lock = RLock()

        if MPI.COMM_WORLD.Get_rank() != 0:
            raise RuntimeError("MpiApiSession must be used from rank 0.")

    def send_message(self, msg: dict, expect_status: bool = True) -> None:
        """
        Send a message to the engine's master rank.
        """
        self.messenger.send(msg, dest=self.engine_master_rank, tag=2)
        # NOTE : If a lot of message are sent, it will slow down a lot, it is ok if it's just at the initialization, but if
        # it became a bottleneck, we will need to implement a more efficient way to get status.
        if expect_status:
            self.receive_status()

    def receive_status(self) -> None:
        """
        Receive the status of the engine.
        """
        msg = self.messenger.recv(source=self.engine_master_rank, tag=0)
        if msg.get("type") == "status":
            value = msg.get("value", {})
            self._is_alive = value.get("alive", False)
            self._is_busy = value.get("busy", False)
        else:
            raise RuntimeError(
                f"Unexpected message type received: {msg}, expected 'status' but got '{msg.get('type')}'"
            )

    # @session_locked
    def command(self, cmd: str) -> None:
        """
        Send a LAMMPS command to the engine.
        """
        # print(f"[Session] Sending command: {cmd}")
        self.send_message({"type": "command", "value": cmd})

    # @session_locked
    def use_local(self) -> None:
        """
        Instruct the engine to use local pool
        """
        # print(f"[Session {self.session_id}] sending 'use local' to rank {self.engine_master_rank}")
        self.send_message({"type": "use_local"})

    def use_global(self) -> None:
        """
        Instruct the engine to use global pool
        """
        # print(f"[Session {self.session_id}] sending 'use global' to rank {self.engine_master_rank}")
        self.send_message({"type": "use_global"})

    # @session_locked
    def close(self, wait_status: bool = False) -> None:
        """
        Instruct the engine to shut down.
        Parameters
        ----------
        wait_status : bool, default False
            If True, wait for the engine to send a status message (for normal sessions).
            If False, just send the close message (for global / long-running engines).
        """
        print(
            f"[Session] Sending close message to engine at rank {self.engine_master_rank}"
        )
        self.send_message({"type": "close"}, expect_status=wait_status)
        self._is_alive = False

    def is_alive(self) -> bool:
        """
        Check if the engine is alive.
        """
        return self._is_alive

    def is_busy(self) -> bool:
        """
        Check if the engine is busy.
        """
        return self._is_busy

    # ACTIONS
    # @session_locked
    def initialize_parameters(self) -> None:
        """
        Initialize LAMMPS engine with default parameters
        """
        # print(f"[Session {self.session_id}] Initializing Lammps parameters")
        self.send_message({"type": "initialize_parameters"})

    # @session_locked
    def initialize_system(self, system) -> None:
        """
        Initialize Lammps system
        """
        # print(f"[Session {self.session_id}] Initializing Lammps System")
        self.send_message({"type": "initialize_system", "value": system})

    # @session_locked
    def initialize_potential(self, config) -> None:
        """
        Initialize Lammps potential
        """
        # print(f"[Session {self.session_id}] Initializing Lammps Potential")
        self.send_message({"type": "initialize_potential", "value": config})

    # @session_locked
    def minimize(self, config, positions=None) -> None:
        """
        Minimize the system
        """
        # print(f"[Session] Minimizing the system")
        self.send_message(
            {"type": "minimize", "value": {"config": config, "positions": positions}}
        )

    # @session_locked
    def get_total_energy(self) -> float:
        """ """
        self._is_busy = True  # Mark the session as busy
        # print(f"[Session] Get total energy")
        try:
            self.send_message({"type": "get_total_energy"})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def get_positions(self) -> np.ndarray[float]:
        self._is_busy = True
        # print(f"[Session] Get Positions")
        try:
            self.send_message({"type": "get_positions"})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def set_positions(self, positions: np.ndarray[float]) -> None:
        self._is_busy = True
        # print(f"[Session] Set new positions")
        try:
            self.send_message({"type": "set_positions", "value": positions})
        finally:
            self._is_busy = False

    # @session_locked
    def minimize_with_results(self, config, positions=None, types=None):
        """Minimize and return the minimized positions and the total energy."""
        self._is_busy = True
        # print(f"[Session n°{self.session_id}] Minimizing and get positions and total energy")
        try:
            self.send_message(
                {
                    "type": "minimize_with_results",
                    "value": {"config": config, "positions": positions, "types": types},
                }
            )
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def get_total_energy(self, positions=None):
        self._is_busy = True
        # print(f"[Session n°{self.session_id}]  get potential energy")
        try:
            self.send_message(
                {"type": "get_total_energy", "value": {"positions": positions}}
            )
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def get_potential_energy(self, positions=None):
        self._is_busy = True
        # print(f"[Session n°{self.session_id}]  get potential energy")
        try:
            self.send_message({"type": "get_potential_energy"})
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def partn_search(
        self, config, central_atom_idx, positions=None, cell=None, types=None
    ):
        self._is_busy = True
        # print(f"[Session] Launching pARTn search")
        try:
            self.send_message(
                {
                    "type": "partn_search",
                    "value": {
                        "config": config,
                        "central_atom_idx": central_atom_idx,
                        "positions": positions,
                        "cell": cell,
                        "types": types,
                    },
                }
            )
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

    # @session_locked
    def partn_refine(
        self,
        config,
        central_atom_idx,
        positions=None,
        cell=None,
        types=None,
        saddle_idx=None,
        saddle_positions=None,
    ):
        self._is_busy = True
        # print(f"[Session] Launching pARTn search")
        try:
            self.send_message(
                {
                    "type": "partn_refine",
                    "value": {
                        "config": config,
                        "central_atom_idx": central_atom_idx,
                        "positions": positions,
                        "cell": cell,
                        "types": types,
                        "saddle_idx": saddle_idx,
                        "saddle_positions": saddle_positions,
                    },
                }
            )
            msg = self.messenger.recv(source=self.engine_master_rank, tag=1)
            if msg.get("type") == "result":
                return msg["value"]
            else:
                raise RuntimeError(f"Unexpected message type: {msg}")
        finally:
            self._is_busy = False

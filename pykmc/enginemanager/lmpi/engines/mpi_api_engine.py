from lammps import lammps
import threading
from mpi4py import MPI
import queue
import traceback
from ..lammps_operations import (
    initialize_parameters,
    initialize_system,
    initialize_potential,
    setup_otf_cycle,
    reset_otf_flags,
    get_otf_flags,
    minimize,
    get_total_energy,
    get_positions,
    set_positions,
    partn_search,
    partn_refine,
    minimize_with_results,
    get_potential_energy,
)
from ...messenger import QueueMessenger, MpiMessenger


class MpiApiEngine:
    """ """

    def __init__(
        self,
        local_messenger,
        local_engine_comm: MPI.Comm,
        local_engine_id: int,
        global_messenger,
        global_engine_comm: MPI.Comm,
        global_engine_id: int,
    ) -> None:
        self.local_messenger = local_messenger
        self.local_engine_comm = local_engine_comm
        if local_engine_comm is not None:
            self.local_rank = (
                local_engine_comm.Get_rank()
            )  # Ranks of the current process in the engine communicator
        else:
            self.local_rank = (
                0  # Ranks of the current process in the engine communicator
            )
        self.local_engine_id = local_engine_id  # Identifier
        self.local_lmp = None  # Placeholder for Lammps instance

        self.global_messenger = global_messenger
        self.global_engine_comm = global_engine_comm
        if global_engine_comm is not None:
            self.global_rank = (
                global_engine_comm.Get_rank()
            )  # Ranks of the current process in the engine communicator
        else:
            self.global_rank = 0
        self.global_engine_id = global_engine_id  # Identifier
        self.global_lmp = None  # Placeholder for Lammps instance

        self.use_global()  # Start with active properties mapped to global Lammps

        self._is_alive = False
        self._is_busy = False
        self._last_error = None
        self.message_reader_thread = None
        self.message_queue = queue.Queue()  # Queue to hold messages from the session

        # Dispatch map of possible lammps operation
        self._operations_map = {
            "use_global": self.use_global,
            "use_local": self.use_local,
            "sleep": self.sleep,
            "wake": self.wake,
            "close": self.close,
            "command": self.command,
            "initialize_parameters": initialize_parameters,
            "initialize_system": initialize_system,
            "initialize_potential": initialize_potential,
            "setup_otf_cycle": setup_otf_cycle,
            "reset_otf_flags": reset_otf_flags,
            "get_otf_flags": get_otf_flags,
            "minimize": minimize,
            "get_total_energy": get_total_energy,
            "get_positions": get_positions,
            "set_positions": set_positions,
            "partn_search": partn_search,
            "partn_refine": partn_refine,
            "minimize_with_results": minimize_with_results,
            "get_potential_energy": get_potential_energy,
        }

    def start(self) -> None:
        """Start Lammps"""
        self.start_engine()
        if self.rank == 0 and isinstance(self.messenger, QueueMessenger):
            # TODO: this only with engine_use_rank_0=True so can probably cut without that option
            t = threading.Thread(target=self.run_engine_loop, daemon=True)
            t.start()
        else:
            self.run_engine_loop()

    def start_engine(self) -> None:
        if self.global_engine_comm is None:
            raise RuntimeError("Missing engine_comm for global LAMMPS")
        if self.local_engine_comm is None:
            raise RuntimeError("Missing engine_comm for local LAMMPS")

        self.global_lmp = lammps(
            comm=self.global_engine_comm,
            cmdargs=[
                "-screen",
                "none",
                "-log",
                "lammps.log." + str(self.global_engine_id),
            ],
        )
        self.local_lmp = lammps(
            comm=self.local_engine_comm,
            cmdargs=[
                "-screen",
                "none",
                "-log",
                "lammps.log." + str(self.local_engine_id),
            ],
        )
        self.lmp = self.global_lmp
        self._is_alive = True

    # RUN ON RANK 0
    def _read_messages(self):
        """Read messages from the LAMMPS session. it is run in on the master rank.
        The message is a dictionnary of the form : {'type': str, 'value', str}"""
        if isinstance(self.messenger, MpiMessenger):
            #  MPI : use blocking calls
            if self.messenger.comm.probe(source=MPI.ANY_SOURCE, tag=2):
                msg = self.messenger.recv(source=MPI.ANY_SOURCE, tag=2)
            else:
                msg = None

        elif isinstance(self.messenger, QueueMessenger):
            # Queue : block until reception
            msg = self.messenger.recv(tag=2)

        else:
            raise RuntimeError("Unsupported messenger type")

        return msg

    def _read_sleep_message(self):
        """Wait for a control message while the engine is sleeping."""
        if isinstance(self.messenger, MpiMessenger):
            return self.messenger.recv(source=MPI.ANY_SOURCE, tag=2)
        if isinstance(self.messenger, QueueMessenger):
            return self.messenger.recv(tag=2)
        raise RuntimeError("Unsupported messenger type")

    # RUN ON ALL RANKS
    def run_engine_loop(self):
        """All ranks run this while lammps is alive, when a message is broadcaster from the master rank, execute the command."""
        while self._is_alive:
            entry_global_mode = (
                self.global_mode
            )  # store for consistent behaviour when switching

            if self.rank == 0:
                msg = self._read_messages()
                if msg is None:
                    continue
            else:
                msg = None

            # broadcast to all ranks in engine_comm
            msg = self.engine_comm.bcast(msg, root=0)
            if msg is None:
                continue

            self._last_error = None
            result = self._handle_message(msg)

            if entry_global_mode and self.global_rank == 0:
                self._send_status(self.global_messenger)
            if (not entry_global_mode) and self.local_rank == 0:
                self._send_status(self.local_messenger)

            # Check if engine was told to shutdown
            if not self._is_alive:
                break

            if result is not None:
                if entry_global_mode:
                    if self.global_rank == 0:
                        self.global_messenger.send(
                            {"type": "result", "value": result}, dest=0, tag=1
                        )
                else:
                    if self.local_rank == 0:
                        self.local_messenger.send(
                            {"type": "result", "value": result}, dest=0, tag=1
                        )

    def _send_status(self, messenger):
        """Send the status of the engine to the session."""
        status_msg = {
            "type": "status",
            "value": {
                "alive": self._is_alive,
                "busy": self._is_busy,
                "error": self._last_error,
            },
        }
        messenger.send(status_msg, dest=0, tag=0)

    def _handle_message(self, msg: dict) -> None:
        """Handle incoming messages from the session."""

        msg_type = msg.get("type")

        operation_handler = self._operations_map.get(msg_type)
        if operation_handler is None:
            raise ValueError(f"Unknown message type: {msg_type}")

        else:
            # Call method of function (and so check if self needs to be provided or not)
            value = msg.get("value", None)
            entry_engine_comm = self.engine_comm  # needed for when msg is use_global
            entry_engine_comm.barrier()
            try:
                if value is None:
                    args = ()
                    kwargs = {}
                elif isinstance(value, dict):
                    args = ()
                    kwargs = value
                else:
                    args = (value,)
                    kwargs = {}

                if (
                    hasattr(operation_handler, "__self__")
                    and operation_handler.__self__ is self
                ):
                    # it s a method
                    result = operation_handler(*args, **kwargs)
                else:
                    # it s an external method that takes engine as a parameter
                    result = operation_handler(self, *args, **kwargs)
                return result
            except Exception as e:
                self._last_error = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "operation": msg_type,
                }
                print(f"[Engine Rank {self.rank}] Error in handler {msg_type}: {e}")
            finally:
                entry_engine_comm.barrier()

    def command(self, cmd: str) -> None:
        """Send a command to the LAMMPS"""
        if self.lmp is None:
            raise RuntimeError(
                "LAMMPS instance is not initialized. Please start the engine first."
            )
        if not self._is_alive:
            raise RuntimeError("LAMMPS process is not running.")
        if self._is_busy:
            raise RuntimeError("LAMMPS process is already busy.")

        self._is_busy = True
        try:
            self.lmp.command(cmd)
        except Exception as e:
            raise RuntimeError(f"Error executing command '{cmd}': {e}")
        finally:
            self._is_busy = False

    def sleep(self) -> None:
        """Block the worker until a wake or close control message arrives."""
        while self._is_alive:
            if self.rank == 0:
                msg = self._read_sleep_message()
            else:
                msg = None

            msg = self.engine_comm.bcast(msg, root=0)
            msg_type = msg.get("type")

            if msg_type == "wake":
                return
            if msg_type == "close":
                self.close()
                return
            raise ValueError(f"Unknown sleep control message: {msg_type}")

    def wake(self) -> None:
        """Wake is consumed by the sleep loop."""
        return None

    def use_global(self) -> None:
        """Switch to global LAMMPS"""
        self.global_mode = True
        self.messenger = self.global_messenger
        self.engine_comm = self.global_engine_comm
        self.engine_id = self.global_engine_id
        self.rank = self.global_rank
        self.lmp = self.global_lmp

    def use_local(self) -> None:
        """Switch to local LAMMPS"""
        self.global_mode = False
        self.messenger = self.local_messenger
        self.engine_comm = self.local_engine_comm
        self.engine_id = self.local_engine_id
        self.rank = self.local_rank
        self.lmp = self.local_lmp

    def close(self) -> None:
        """Close the LAMMPS engine."""
        print(f"[Engine Rank {self.rank}] Closing LAMMPS engine.")
        if self.local_lmp is not None:
            self.local_lmp.close()
            self.local_lmp = None

        if self.global_lmp is not None:
            self.global_lmp.close()
            self.global_lmp = None

        self._is_alive = False

        if self.message_reader_thread is not None:
            self.message_reader_thread.join(timeout=1)
            self.message_reader_thread = None

    def is_alive(self) -> bool:
        """Check if the LAMMPS engine is alive."""
        return self._is_alive

    def is_busy(self) -> bool:
        """Check if the LAMMPS engine is busy."""
        return self._is_busy

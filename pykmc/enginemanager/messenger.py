""" """

from abc import ABC, abstractmethod
from mpi4py import MPI
import queue


class Messenger(ABC):
    """Abstract communication interface."""

    @abstractmethod
    def send(self, msg, dest, tag=0):
        pass

    @abstractmethod
    def recv(self, source=None, tag=0):
        pass


class MpiMessenger(Messenger):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def send(self, msg, dest, tag=0):
        self.comm.send(msg, dest=dest, tag=tag)

    def recv(self, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
        return self.comm.recv(source=source, tag=tag)


class QueueMessenger(Messenger):
    def __init__(self):
        self.message_queue = queue.Queue()

    def send(self, msg, dest=None, tag=None):
        self.message_queue.put((tag, msg))

    def recv(self, source=None, tag=None):
        while True:
            t, msg = self.message_queue.get()
            if t == tag:
                return msg
            else:
                # put back in queue
                self.message_queue.put((t, msg))

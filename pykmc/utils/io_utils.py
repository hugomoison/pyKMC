import os
import threading
from functools import wraps


class _capture_fd:
    """Suppress all C/Fortran-level writes to a single file descriptor by redirecting to /dev/null."""

    def __init__(self, fd: int):
        self._fd = fd

    def __enter__(self):
        self._saved_fd = os.dup(self._fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, self._fd)
        os.close(devnull)
        self.output = ""
        return self

    def __exit__(self, *_):
        os.dup2(self._saved_fd, self._fd)
        os.close(self._saved_fd)


class capture_output:
    """Suppress C/Fortran-level stdout/stderr as a context manager or decorator."""

    _thread_lock = threading.RLock()

    def __init__(
        self,
        stdout: bool = True,
        stderr: bool = True,
        threaded: bool = False,
    ):
        self._capture_stdout = stdout
        self._capture_stderr = stderr
        self._threaded = threaded

    def __enter__(self):
        self._lock_acquired = False
        if self._threaded:
            type(self)._thread_lock.acquire()
            self._lock_acquired = True
        self._cap_out = _capture_fd(1).__enter__() if self._capture_stdout else None
        self._cap_err = _capture_fd(2).__enter__() if self._capture_stderr else None
        return self

    def __exit__(self, *args):
        try:
            if self._cap_out is not None:
                self._cap_out.__exit__(*args)
            if self._cap_err is not None:
                self._cap_err.__exit__(*args)
            self.stdout = self._cap_out.output if self._cap_out is not None else None
            self.stderr = self._cap_err.output if self._cap_err is not None else None
        finally:
            if self._lock_acquired:
                type(self)._thread_lock.release()

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with type(self)(
                stdout=self._capture_stdout,
                stderr=self._capture_stderr,
                threaded=self._threaded,
            ):
                return func(*args, **kwargs)

        return wrapped

from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.lmpi.pool import ManagerFactory, Manager
from pykmc.enginemanager.messenger import MpiMessenger
from pykmc import System, Config
from mpi4py import MPI 
import pytest
from pytest_lazy_fixtures import lf
import os
import time
import numpy as np


class TestManager: 

    def test_initialize_manager(self)  :
        factory = ManagerFactory(n_sessions=2, use_rank_0=False, has_global=True)
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        # Per-chunk commands require LOCAL mode (engines boot in global mode);
        # issuing them in global mode deadlocks every chunk-master but the global one.
        manager.use_local()
        manager.broadcast_command("units metal")
        manager.broadcast_command("log flush")
        # Global-session commands require GLOBAL mode.
        manager.use_global()
        manager.global_session.command("dimension 3")
        manager.global_session.command("log flush")
        # close_all is only safe in local mode (per-chunk close handshake).
        manager.use_local()
        manager.close_all()


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_minimize_manager(self, system: System, config: Config)  :
        # use_rank_0=True: rank 0 is BOTH orchestrator and an engine (its engine loop
        # runs in a background thread). Exercises that path end-to-end.
        factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=True, has_global=True)
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        # initialize_sessions ends in GLOBAL mode (inits chunks in local, then global).
        manager.initialize_sessions(config, system)
        # Per-chunk minimize needs LOCAL mode.
        manager.use_local()
        f = manager.minimize(config)
        re = f.result()
        # Global minimize needs GLOBAL mode.
        manager.use_global()
        manager.global_minimize(config)
        # Return to local before closing.
        manager.use_local()
        manager.close_all()
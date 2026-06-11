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


class TestLammpsApiMpiEngine:
    # def test_send_commands_engine(self) :
    #    comm = MPI.COMM_WORLD
    #    rank = comm.Get_rank()
    #    size = comm.Get_size()

    #    #test when engine also live on the master session rank or not
    #    start_rank_engine = 1

    #    if size < 2:
    #        raise RuntimeError("This test requires at least 2 MPI ranks.")

    #    engine_ranks = list(range(start_rank_engine, size))

    #    engine_comm = comm.Split(color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank)

    #    # Start the MPI API Engine only on the specified engine ranks
    #    if rank in engine_ranks:
    #        engine = MpiApiEngine(engine_comm, engine_id=0)
    #        engine.start()

    #    # Master rank sends a message to the engine
    #    if rank == 0:
    #        msg = {"type": "command", "value": "units metal"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #        msg = {"type": "command", "value": "log flush"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #        msg = {"type": "close"}
    #        comm.send(msg, dest=engine_ranks[0], tag=1)

    #    time.sleep(4)
    #    # Test if command was sent to lammps :
    #    if rank == 0 :
    #        logfile = os.path.join(os.getcwd(), 'lammps.log.0')
    #        with open(logfile) as f :
    #            log_text = f.read()
    #        assert 'units metal' in log_text

    def test_send_commends_from_session(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.command("units metal")
        session.command("dimension 3")
        session.command("log flush")

        session.close()

        time.sleep(4)
        # Test if command was sent to lammps :
        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as f:
            log_text = f.read()
        assert "units metal" in log_text

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_initialize_session(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.command("log flush")
        session.close()
        time.sleep(1.5)
        # Test if command was sent to lammps :
        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as f:
            log_text = f.read()
        assert "units metal" in log_text
        assert "atom_style atomic" in log_text
        assert "dimension 3" in log_text
        assert "boundary p p p" in log_text
        assert "atom_modify sort 0 0.0" in log_text
        assert "region box" in log_text
        assert "create_box" in log_text

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        session.command("log flush")
        session.close()
        time.sleep(1.5)
        # Test if command was sent to lammps :
        logfile = os.path.join(os.getcwd(), "lammps.log.0")
        with open(logfile) as f:
            log_text = f.read()
        assert "units metal" in log_text
        assert "atom_style atomic" in log_text
        assert "dimension 3" in log_text
        assert "boundary p p p" in log_text
        assert "atom_modify sort 0 0.0" in log_text
        assert "region box" in log_text
        assert "create_box" in log_text

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_get_total_energy(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        e = session.get_total_energy()
        print(e)
        session.command("log flush")
        session.close()
        assert round(e, 3) == round(-1139.1999963495148, 3)

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_get_positions(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        e = session.get_total_energy()
        print(e)
        e = session.get_total_energy()
        print(e)
        positions = session.get_positions()
        print(positions)
        session.command("log flush")
        session.close()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_set_positions(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        positions = session.get_positions()
        positions[0][0], positions[0][1], positions[0][2] = 0.1, 0.2, 0.3
        session.set_positions(positions)
        positions = session.get_positions()
        session.command("log flush")
        session.close()

        assert positions[0][0] == 0.1
        assert positions[0][1] == 0.2
        assert positions[0][2] == 0.3

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_partn_search(self, system: System, config: Config):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # test when engine also live on the master session rank or not
        start_rank_engine = 1
        messenger = MpiMessenger(comm=MPI.COMM_WORLD)

        if size < 2:
            raise RuntimeError("This test requires at least 2 MPI ranks.")

        engine_ranks = list(range(start_rank_engine, size))

        engine_comm = comm.Split(
            color=1 if rank in engine_ranks else MPI.UNDEFINED, key=rank
        )

        # Start the MPI API Engine only on the specified engine ranks
        if rank in engine_ranks:
            engine = MpiApiEngine(
                messenger=messenger, engine_comm=engine_comm, engine_id=0
            )
            engine.start()
            return

        # ------------ SESSION CODE (rank 0) ------------
        session = MpiApiSession(
            messenger=messenger, engine_ranks=engine_ranks, session_id=0
        )
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        result = session.partn_search(config, 0)
        result = session.partn_refine(config, 0)
        if result.is_ok():
            print(result.ok_value())
        else:
            print(result.err_value())
        session.command("log flush")
        session.close()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_initialize_manager(self, system: System, config: Config):
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
        )
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------a
        print("HERERER")
        manager.initialize_sessions(config, system)
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_manager(self, system: System, config: Config):
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        print("done")
        f = manager.minimize(config)
        re = f.result()
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_partn_manager(self, system: System, config: Config):
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        idx = 20 * [0]
        futures = manager.partn_refine(config, idx)
        re = [f.result() for f in futures]
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_with_results_manager(self, system: System, config: Config):
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        futures = manager.minimize_with_results(config)
        positions, total_energy = futures.result()
        print(positions)
        print(total_energy)
        manager.close_all()

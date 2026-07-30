from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc.enginemanager.messenger import MpiMessenger
from pykmc import System, Config
from mpi4py import MPI
import pytest
from pytest_lazy_fixtures import lf
import os


def _launch_direct_session() -> MpiApiSession | None:
    """Launch one dual-mode engine and return its rank-zero session."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise RuntimeError("This test requires at least 2 MPI ranks.")

    engine_ranks = list(range(1, size))
    color = 1 if rank in engine_ranks else MPI.UNDEFINED

    # MpiApiEngine owns distinct local and global LAMMPS instances. Match the
    # current ManagerFactory wiring by giving each instance its own MPI context.
    local_engine_comm = comm.Split(color=color, key=rank)
    global_engine_comm = comm.Split(color=color, key=rank)
    local_messenger = MpiMessenger(comm=comm)
    global_messenger = MpiMessenger(comm=comm)

    if rank in engine_ranks:
        engine = MpiApiEngine(
            local_messenger=local_messenger,
            local_engine_comm=local_engine_comm,
            local_engine_id=1,
            global_messenger=global_messenger,
            global_engine_comm=global_engine_comm,
            global_engine_id=0,
        )
        engine.start()
        return None

    return MpiApiSession(
        messenger=global_messenger,
        engine_ranks=engine_ranks,
        session_id=0,
    )


class TestLammpsApiMpiEngine:
    def test_send_commends_from_session(self):
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
        session.command("units metal")
        session.command("dimension 3")
        session.command("log flush")

        session.close(wait_status=True)

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
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.command("log flush")
        session.close(wait_status=True)
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
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        session.command("log flush")
        session.close(wait_status=True)
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
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        session.minimize(config)
        e = session.get_total_energy()
        print(e)
        session.command("log flush")
        session.close(wait_status=True)
        assert round(e, 3) == round(-1139.1999963495148, 3)

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_get_positions(self, system: System, config: Config):
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
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
        session.close(wait_status=True)

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_set_positions(self, system: System, config: Config):
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
        session.initialize_parameters()
        session.initialize_system(system)
        session.initialize_potential(config)
        positions = session.get_positions()
        positions[0][0], positions[0][1], positions[0][2] = 0.1, 0.2, 0.3
        session.set_positions(positions)
        positions = session.get_positions()
        session.command("log flush")
        session.close(wait_status=True)

        assert positions[0][0] == 0.1
        assert positions[0][1] == 0.2
        assert positions[0][2] == 0.3

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_partn_search(self, system: System, config: Config):
        session = _launch_direct_session()
        if session is None:
            return

        # ------------ SESSION CODE (rank 0) ------------
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
        session.close(wait_status=True)

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_initialize_manager(self, system: System, config: Config):
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=False,
        )
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_manager(self, system: System, config: Config):
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=False,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.use_local()
        f = manager.minimize(config)
        f.result()
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_partn_manager(self, system: System, config: Config):
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=False,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.use_local()
        idx = 20 * [0]
        futures = [manager.partn_refine(config, atom) for atom in idx]
        for future in futures:
            future.result()
        manager.close_all()

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_with_results_manager(self, system: System, config: Config):
        config.control.engine_use_rank_0 = False
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=False,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.use_local()
        futures = manager.minimize_with_results(config)
        positions, total_energy = futures.result()
        print(positions)
        print(total_energy)
        manager.close_all()

import pytest
from pykmc import System, Config, AtomicEnvironment, NeighborsList
import numpy as np
from mpi4py import MPI
from pykmc.basins import (
    StatesConnectivity,
    BasinStatesConnectivity,
    StateData,
    ReferenceEventTable,
)
from copy import deepcopy
import pandas as pd
import logging
from unittest.mock import Mock
import os, sys
import pickle
from functools import wraps
import subprocess
from ase.build import bulk, surface


def _abort_mpi_on_failure(
    report: pytest.TestReport | pytest.CollectReport,
) -> None:
    """Abort an MPI pytest invocation when any rank reports a failure."""
    if not report.failed or not MPI.Is_initialized() or MPI.Is_finalized():
        return

    world = MPI.COMM_WORLD
    size = world.Get_size()
    if size <= 1:
        return

    rank = world.Get_rank()
    phase = getattr(report, "when", "unknown")
    try:
        sys.stderr.write(
            f"\n[pytest-mpi] failure on rank={rank}, size={size}, "
            f"phase={phase}, node={report.nodeid}\n"
            f"{report.longreprtext}\n"
        )
        for title, content in getattr(report, "sections", ()):
            sys.stderr.write(f"\n--- {title} ---\n{content}\n")
        sys.stderr.flush()
    finally:
        try:
            world.Abort(1)
        finally:
            os._exit(1)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Terminate the MPI world after a failed setup, call, or teardown report."""
    _abort_mpi_on_failure(report)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Terminate the MPI world after a failed collection report."""
    _abort_mpi_on_failure(report)


# Systems


@pytest.fixture(scope="session")
def ni_orthorhombic():
    """FCC Ni — orthorhomic."""
    atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True)
    atoms = atoms.repeat([4, 4, 4])
    system = System(
        types=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
        index=np.arange(len(atoms)),
    )
    return system


@pytest.fixture(scope="session")
def ni_triclinic():
    """FCC Ni — triclinic."""
    atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=False)
    atoms = atoms.repeat([4, 4, 4])
    system = System(
        types=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
        index=np.arange(len(atoms)),
    )
    return system


@pytest.fixture(scope="session")
def ni_slab():
    """
    Slab Ni(100) — 4 layers, pbc=[True, True, False], vacuum 10 Å en z.
    """
    atoms = surface("Ni", (1, 0, 0), layers=4, vacuum=5.0)
    atoms = atoms.repeat([4, 4, 1])
    system = System(
        types=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
        index=np.arange(len(atoms)),
    )
    return system


@pytest.fixture(scope="session")
def test_logger():
    """Logger for tests."""
    logger = logging.getLogger("tests")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[\033[1;36m %(levelname)s \033[0m]  %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


@pytest.fixture
def mock_config():
    """Fixture pour créer une configuration mock"""
    config = Mock()
    config.atomicenvironment.rnei = 3.01
    config.atomicenvironment.rcut = 6.5
    config.atomicenvironment.style = "cna/graph"
    config.atomicenvironment.neighbors_add = 0
    config.basin.energy_thr = 0.4
    config.psr.style = "ira"
    config.psr.matching_score_thr = 0.1
    config.ira.kmax_factor = 1.8
    return config


@pytest.fixture
def config_system_single_type():
    config = Config.from_ini_file("./tests/data/input.in")
    return config


@pytest.fixture
def system_binary_fcc() -> System:
    """4x4x4 FCC with alternating Ni/Fe types (L1_0-like ordering)."""
    system = System()

    a = 3.52
    repeat = 4

    basis = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
        * a
    )

    positions = []
    types = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                shift = np.array([i, j, k]) * a
                for b, atom in enumerate(basis):
                    positions.append(atom + shift)
                    # Alternate types: basis sites 0,1 = Ni, 2,3 = Fe
                    types.append("Ni" if b < 2 else "Fe")

    system.positions = np.array(positions)
    system.types = np.array(types)
    system.cell = np.array(
        [[repeat * a, 0.0, 0.0], [0.0, repeat * a, 0.0], [0.0, 0.0, repeat * a]]
    )
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(system.positions))

    return system


@pytest.fixture
def system_single_type_fcc() -> System:
    system = System()

    a = 3.52
    repeat = 4

    basis = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
        * a
    )

    positions = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                shift = np.array([i, j, k]) * a
                for atom in basis:
                    positions.append(atom + shift)

    system.positions = np.array(positions)
    system.types = ["Ni"] * len(system.positions)
    system.cell = np.array(
        [[repeat * a, 0.0, 0.0], [0.0, repeat * a, 0.0], [0.0, 0.0, repeat * a]]
    )
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(system.positions))

    return system


@pytest.fixture
def system_single_type_fcc_vacancy(system_single_type_fcc: System) -> System:
    system = deepcopy(system_single_type_fcc)
    # remove atom
    system.positions = np.delete(system.positions, 0, axis=0)
    system.types = np.delete(system.types, 0, axis=0)
    system.index = np.delete(system.index, 0, axis=0)

    return system


@pytest.fixture
def mock_state_data(system_Ni_4000at_monovacancy_sia):
    """Fixture returning a dummy StateData for testing basin exploration."""
    system = system_Ni_4000at_monovacancy_sia
    nl = NeighborsList(system=system, rnei=3.01, rcut=6.5)
    ae = AtomicEnvironment(
        style="cna/graph",
        neighbors_list=nl.neighbors_list["rnei"],
        environment_list=nl.neighbors_list["rcut"],
    )
    state = StateData(
        system=system_Ni_4000at_monovacancy_sia,
        environment=ae,
        neighbors_list=nl,
        transient=True,
        visited=False,
    )
    return state


@pytest.fixture
def system_Ni_4000at_monovacancy_sia() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(
        data_dir, "initial_config_Ni_fcc_4000at_monovacancy+sia.xyz"
    )
    system = System.create_from_file(filepath)
    return system


@pytest.fixture
def system_Cu() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "initial_config_Cu.xyz")
    system = System.create_from_file(filepath)
    return system


@pytest.fixture
def reference_table_Ni_4000at_monovacancy_sia() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "input.in")
    config = Config.from_ini_file(filepath)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(
        data_dir, "reference_table_Ni_fcc_4000at_monovacancy+sia.pickle"
    )
    config.control.reference_table = filepath
    reference_table = ReferenceEventTable(config)
    return reference_table


@pytest.fixture
def reference_table_Cu_fake() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "input.in")
    config = Config.from_ini_file(filepath)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "reference_table_Cu_fake.pickle")
    config.control.reference_table = filepath
    reference_table = ReferenceEventTable(config)
    return reference_table


@pytest.fixture
def visited_environments_Ni_4000at_monovacancy_sia() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(
        data_dir, "visited_environments_Ni_fcc_4000at_monovacancy+sia.pickle"
    )

    with open(filepath, "rb") as file:
        loaded_set_environments = pickle.load(file)
    return loaded_set_environments


@pytest.fixture
def visited_environments_Cu() -> System:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "visited_environments_Cu.pickle")

    with open(filepath, "rb") as file:
        loaded_set_environments = pickle.load(file)
    return loaded_set_environments


@pytest.fixture
def config_Ni_4000at_monovacancy_sia():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "input.in")
    config = Config.from_ini_file(filepath)
    return config


@pytest.fixture
def config_Cu():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    filepath = os.path.join(data_dir, "input_Cu.in")
    config = Config.from_ini_file(filepath)
    return config


def mpi_test(nproc=2):
    """Décorateur pour lancer un test sous mpirun."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Si déjà lancé sous MPI, exécuter le test directement
            if "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ:
                return func(*args, **kwargs)

            # Sinon relancer ce test avec mpirun
            cmd = [
                "mpirun",
                "-n",
                str(nproc),
                sys.executable,
                "-m",
                "pytest",
                "-v",
                f"{__file__}::{func.__name__}",
            ]
            print(f"\n[pytest-mpi] Launching under MPI: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                pytest.fail(
                    f"MPI test {func.__name__} failed with code {result.returncode}"
                )

        return wrapper

    return decorator

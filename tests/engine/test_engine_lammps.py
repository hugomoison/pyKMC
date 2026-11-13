import pytest

pytest.importorskip("lammps")

from pykmc.engine import Engine, EngineExtension
from .test_engine_contract import EngineContractTests
from dataclasses import dataclass, field


@pytest.fixture(params=["ni_orthorhombic", "ni_triclinic", "ni_slab"])
def system(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def lammps_config_Ni():
    @dataclass
    class LammpsConfig:
        pair_style: str = "lj/cut 6.0"
        pair_coeff: list[str] = field(default_factory=lambda: ["* * 0.52 2.274"])
        min_style: str = "cg"
        minimize: str = "1e-6 1e-8 1000 10000"
        frz_min: str = "1e-4 1e-6 100 1000"
        verbosity: int = 0

    return LammpsConfig()


class TestLammpsEngineSerial(EngineContractTests):
    @pytest.fixture(autouse=True)
    def require_serial(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        self.config = lammps_config_Ni
        self.system = system

    def make_engine(self):
        return Engine.create("lammps", config=self.config, comm=None)

    def make_test_extension(self, engine) -> EngineExtension:
        return _ComputeKineticEnergy(engine=engine)

    def make_conflicting_extension(self, engine) -> EngineExtension:
        return _ConflictingExtension(engine=engine)

    def test_kinetic_energy_returns_float(self):
        """_ComputeKineticEnergy.get_kinetic_energy() returns a float."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        _ComputeKineticEnergy(engine=engine)
        ke = engine.get_kinetic_energy()
        assert isinstance(ke, float)
        engine.close()


@pytest.mark.mpi
class TestLammpsEngineMPI(EngineContractTests):
    """
    Run with: mpirun -n 4 pytest tests/engine/test_engine_lammps.py

    All ranks execute each test collectively.
    Scalar assertions are restricted to rank 0 via is_rank0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        from mpi4py import MPI

        self.config = lammps_config_Ni
        self.system = system
        self.comm = MPI.COMM_WORLD
        yield
        MPI.COMM_WORLD.Barrier()

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N")

    @property
    def is_rank0(self) -> bool:
        from mpi4py import MPI

        return self.comm.Get_rank() == 0

    def make_engine(self) -> Engine:
        return Engine.create("lammps", config=self.config, comm=self.comm)

    def make_test_extension(self, engine) -> EngineExtension:
        return _ComputeKineticEnergy(engine=engine)

    def make_conflicting_extension(self, engine) -> EngineExtension:
        return _ConflictingExtension(engine=engine)

    def test_kinetic_energy_returns_float(self):
        """_ComputeKineticEnergy.get_kinetic_energy() returns a float on rank 0."""
        engine = self.make_engine()
        engine.start()
        self.initialize(engine)
        _ComputeKineticEnergy(engine=engine)
        ke = engine.get_kinetic_energy()
        if self.is_rank0:
            assert isinstance(ke, float)
        engine.close()


class _ComputeKineticEnergy(EngineExtension):
    """Test extension: adds get_kinetic_energy() via a LAMMPS compute."""

    def __init__(self, engine):
        super().__init__(engine)

    def get_kinetic_energy(self) -> float | None:
        compute_id = "test_ke"
        reduce_id = "test_ke_reduce"
        if not self.engine._has_compute(compute_id):
            self.engine.lmp.command(f"compute {compute_id} all ke/atom")
            self.engine.lmp.command(
                f"compute {reduce_id} all reduce sum c_{compute_id}"
            )
        self.engine.lmp.command("run 0 post no")
        result = self.engine.lmp.extract_compute(reduce_id, 0, 0)
        if self.engine._is_rank0:
            return float(result)
        return None


class _ConflictingExtension(EngineExtension):
    """Test extension: same method name as _ComputeKineticEnergy to test conflict detection."""

    def __init__(self, engine):
        super().__init__(engine)

    def get_kinetic_energy(self) -> None:
        pass

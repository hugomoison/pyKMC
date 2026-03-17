import pytest
from pykmc import LammpsEngine, LammpsConfigProtocol
from .test_engine_contract import EngineContractTests
from dataclasses import dataclass

@pytest.fixture(params=["ni_orthorhombic", "ni_triclinic", "ni_slab"])
def system(request):
    return request.getfixturevalue(request.param)

@pytest.fixture(scope="session")
def lammps_config_Ni():
    @dataclass
    class LammpsConfig:
        pair_style: str = "lj/cut 6.0"
        pair_coeff: str = "* * 0.52 2.274"
        min_style:  str = "cg"
        minimize:   str = "1e-6 1e-8 1000 10000"
        verbosity:  int = 0
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
        return LammpsEngine(config=self.config, comm=None)
    
@pytest.mark.mpi
class TestLammpsEngineMPI(EngineContractTests):
    """
    Lancé avec : mpirun -n 4 pytest --with-mpi tests/engine/test_lammps_engine.py

    Tous les ranks exécutent chaque test collectivement.
    Les assertions sont restreintes au rank 0 via is_rank0.
    """

    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        from mpi4py import MPI
        self.config = lammps_config_Ni
        self.system = system
        self.comm = MPI.COMM_WORLD

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N --with-mpi")

    @property
    def is_rank0(self) -> bool:
        from mpi4py import MPI
        return self.comm.Get_rank() == 0

    def make_engine(self) -> LammpsEngine:
        return LammpsEngine(config=self.config, comm=self.comm)
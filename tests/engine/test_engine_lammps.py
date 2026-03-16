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
    def setup(self, lammps_config_Ni, system):
        self.config = lammps_config_Ni
        self.system = system

    def make_engine(self): 
        return LammpsEngine(config=self.config, comm=None)
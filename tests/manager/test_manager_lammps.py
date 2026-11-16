import pytest

pytest.importorskip("lammps")
import numpy as np
from mpi4py import MPI
from pykmc.factory import EngineManagerFactory
from dataclasses import dataclass, field


def _require_8_ranks():
    if MPI.COMM_WORLD.Get_size() != 8:
        pytest.skip("TestManagerLammps must be run with mpirun -n 8.")


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
        verbosity: int = 1

    return LammpsConfig()


class TestManagerLammps:
    @pytest.fixture(autouse=True)
    def setup(self, lammps_config_Ni, system):
        _require_8_ranks()
        MPI.COMM_WORLD.Barrier()
        self.system = system
        self.manager = EngineManagerFactory(
            engine_style="lammps",
            engine_config=lammps_config_Ni,
            n_workers=4,
            comm=MPI.COMM_WORLD,
            group_size=4,
        ).launch()
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.manager.broadcast("start")
            self.manager.broadcast("initialize_parameters")
            self.manager.broadcast(
                "initialize_system",
                types=system.types,
                positions=system.positions,
                cell=system.cell,
                pbc=system.pbc,
            )
            self.manager.broadcast("initialize_potential")
            self.manager.submit_group("start")
            self.manager.submit_group("initialize_parameters")
            self.manager.submit_group(
                "initialize_system",
                types=system.types,
                positions=system.positions,
                cell=system.cell,
                pbc=system.pbc,
            )
            self.manager.submit_group("initialize_potential")
        yield
        if self.manager is not None:
            self.manager.shutdown()
        MPI.COMM_WORLD.Barrier()

    def test_get_total_energy_local(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        result = self.manager.get_total_energy().result()
        assert isinstance(result, float)

    def test_get_total_energy_group(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        result = self.manager.group_get_total_energy()
        assert isinstance(result, float)

    def test_minimize_local(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        rng = np.random.default_rng(seed=0)
        perturbed = self.system.positions.copy() + rng.uniform(
            -0.05, 0.05, size=self.system.positions.shape
        )
        e_before = self.manager.get_potential_energy(positions=perturbed).result()
        _, e_after = self.manager.minimize_with_results(positions=perturbed).result()
        assert e_after < e_before

    def test_minimize_group(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        rng = np.random.default_rng(seed=0)
        perturbed = self.system.positions.copy() + rng.uniform(
            -0.05, 0.05, size=self.system.positions.shape
        )
        e_before = self.manager.group_get_potential_energy(positions=perturbed)
        _, e_after = self.manager.group_minimize_with_results(positions=perturbed)
        assert e_after < e_before

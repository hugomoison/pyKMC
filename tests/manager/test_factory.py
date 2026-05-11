import pytest 
from mpi4py import MPI
from .test_worker import FakeEngine
from unittest.mock import patch
from pykmc import ManagerFactory, LammpsConfigProtocol, LammpsEngine
from dataclasses import dataclass

#@pytest.fixture(params=["ni_orthorhombic", "ni_triclinic", "ni_slab"])
@pytest.fixture(params=["ni_orthorhombic"])
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
        verbosity:  int = 1
    return LammpsConfig()

class TestFactoryManager:

    @pytest.fixture(autouse=True)
    def require_mpi(self):
        if MPI.COMM_WORLD.Get_size() < 3:
            pytest.skip("Require mpirun -n > 2")

    def test_local_then_global(self):
        with patch('pykmc.manager.factory.LammpsEngine', FakeEngine):
            factory = ManagerFactory(
                engine_style='lammps',
                n_workers=4,
                comm=MPI.COMM_WORLD,
                has_global=True,
                engine_config=object(),
                global_size=3
            )
            manager = factory.launch()

            if MPI.COMM_WORLD.Get_rank() == 0:
                manager._use_local()
                local_expected = float(sum(range(len(factory.chunks[0]))))
                print("test_local_then_global, local_expected = ", local_expected)
                global_expected = float(sum(range(factory.global_size)))
                print("test_local_then_global, global_expected = ", global_expected)

                futures = [manager.submit("collective") for _ in range(10)]
                results = [f.result() for f in futures]
                print(results)
                assert all(r == local_expected for r in results)

                manager._use_global()
                f_global = manager.submit_global("collective")
                result = f_global.result()
                print(result)
                assert result == global_expected

                manager.close()

    def test_lammps(self, lammps_config_Ni, system) : 
        config = lammps_config_Ni
        system = system
        factory = ManagerFactory(engine_style='lammps', 
                                 n_workers=2, 
                                 comm=MPI.COMM_WORLD, 
                                 has_global=True, 
                                 engine_config=config, 
                                 )
        print("yes")
        manager = factory.launch() 

        if MPI.COMM_WORLD.Get_rank() == 0 : 
            print(manager.using_global)
            manager._use_local()
            manager.broadcast('initialize_parameters')
            manager.broadcast('initialize_system', types=system.types, positions=system.positions, cell=system.cell, pbc=system.pbc)  
            manager.broadcast('initialize_potential')
            manager._use_global()
            manager.global_initialize_parameters()
            manager.global_initialize_system(type=system.types, positions=system.positions, cell=system.cell, pbc=system.pbc)
            manager.global_initialize_potential()


            manager._use_local()
            # 5 compute energy locaux
            futures = [manager.submit("get_total_energy") for _ in range(5)]
            results = [f.result() for f in futures]
            print(results, flush=True)
            assert all(isinstance(r, float) for r in results)
            assert all(abs(r - results[0]) < 1e-6 for r in results)  # tous identiques

            manager._use_global()

            # 1 compute energy global
            f_global = manager.submit_global("get_total_energy")
            result_global = f_global.result()
            print(result_global, flush=True)
            assert isinstance(result_global, float)
            assert abs(result_global - results[0]) < 1e-6  # cohérent avec local

            manager.close()






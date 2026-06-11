from pykmc import System, Config, NeighborsList, AtomicEnvironment
import pytest 
from pytest_lazy_fixtures import lf
from collections import Counter


expected_results = {
    "system_single_type_fcc_vacancy" : {"different": 2, "noncrystal" : 12}
}

expected_graph_results = {
    "system_single_type_fcc_vacancy": {
        "different": 7,
        "counts": [169, 24, 24, 12, 12, 8, 6],
    }
}

class TestAtomicEnvironment : 

    @pytest.mark.parametrize("system_name, system, config", [("system_single_type_fcc_vacancy", lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))])
    def test_cna(self, system_name:str, system: System, config: Config) :

        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment('cna', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'])

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_results[system_name]

        assert len(hash_count) == expected['different']
        assert expected['noncrystal'] in hash_count.values()

    @pytest.mark.parametrize("system_name, system, config", [("system_single_type_fcc_vacancy", lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))])
    def test_graph(self, system_name:str, system: System, config: Config) :

        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment('graph', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'])

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_graph_results[system_name]

        assert len(hash_count) == expected['different']
        assert sorted(hash_count.values(), reverse=True) == expected["counts"]


    @pytest.mark.parametrize("system_name, system, config", [("system_single_type_fcc_vacancy", lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))])
    def test_cna_graph(self, system_name:str, system: System, config: Config) :

        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment('cna/graph', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'])

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_results[system_name]

        assert len(hash_count) == expected['different']
        assert expected['noncrystal'] in hash_count.values()


class TestAtomColoringMode :

    def test_grey_mode_ignores_types(self, system_binary_fcc, config_system_single_type) :
        """In grey mode, Ni and Fe atoms with same geometry get the same environment ID."""
        config = config_system_single_type
        nl = NeighborsList(system_binary_fcc, config.atomicenvironment.rnei, config.atomicenvironment.rcut)

        # Grey mode: types=None
        ae_grey = AtomicEnvironment('cna/graph', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'], types=None)

        # All atoms in perfect FCC are crystal regardless of type
        assert all(e == "crystal" for e in ae_grey.atomic_environment_list)

    def test_full_mode_graph_distinguishes_types(self, system_binary_fcc, config_system_single_type) :
        """Graph mode with types produces more distinct IDs than without."""
        config = config_system_single_type
        nl = NeighborsList(system_binary_fcc, config.atomicenvironment.rnei, config.atomicenvironment.rcut)

        # Grey mode graph
        ae_grey = AtomicEnvironment('graph', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'], types=None)
        grey_ids = set(ae_grey.atomic_environment_list)

        # Full mode graph
        ae_full = AtomicEnvironment('graph', nl.neighbors_list['rnei'], nl.neighbors_list['rcut'],
                                    types=list(system_binary_fcc.types))
        full_ids = set(ae_full.atomic_environment_list)

        # Full mode should produce more distinct IDs than grey mode
        assert len(full_ids) > len(grey_ids)

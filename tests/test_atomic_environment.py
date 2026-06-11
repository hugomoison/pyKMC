from pykmc import System, Config, NeighborsList, AtomicEnvironment
import pytest
from pytest_lazy_fixtures import lf
from collections import Counter


expected_results = {
    "system_single_type_fcc_vacancy": {"different": 2, "noncrystal": 12}
}


class TestAtomicEnvironment:
    @pytest.mark.parametrize(
        "system_name, system, config",
        [
            (
                "system_single_type_fcc_vacancy",
                lf("system_single_type_fcc_vacancy"),
                lf("config_system_single_type"),
            )
        ],
    )
    def test_cna(self, system_name: str, system: System, config: Config):

        config["AtomicEnvironment"]["style"] = "cna"
        nl = NeighborsList(system, config)
        ae = AtomicEnvironment(
            config, nl.neighbors_list["rnei"], nl.neighbors_list["rcut"]
        )

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_results[system_name]

        assert len(hash_count) == expected["different"]
        assert expected["noncrystal"] in hash_count.values()

    @pytest.mark.parametrize(
        "system_name, system, config",
        [
            (
                "system_single_type_fcc_vacancy",
                lf("system_single_type_fcc_vacancy"),
                lf("config_system_single_type"),
            )
        ],
    )
    def test_graph(self, system_name: str, system: System, config: Config):

        config["AtomicEnvironment"]["style"] = "graph"
        nl = NeighborsList(system, config)
        ae = AtomicEnvironment(
            config, nl.neighbors_list["rnei"], nl.neighbors_list["rcut"]
        )

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_results[system_name]

        assert len(hash_count) == expected["different"]
        assert expected["noncrystal"] in hash_count.values()

    @pytest.mark.parametrize(
        "system_name, system, config",
        [
            (
                "system_single_type_fcc_vacancy",
                lf("system_single_type_fcc_vacancy"),
                lf("config_system_single_type"),
            )
        ],
    )
    def test_cna_graph(self, system_name: str, system: System, config: Config):

        config["AtomicEnvironment"]["style"] = "cna/graph"
        nl = NeighborsList(system, config)
        ae = AtomicEnvironment(
            config, nl.neighbors_list["rnei"], nl.neighbors_list["rcut"]
        )

        hash_count = Counter(ae.atomic_environment_list)

        expected = expected_results[system_name]

        assert len(hash_count) == expected["different"]
        assert expected["noncrystal"] in hash_count.values()

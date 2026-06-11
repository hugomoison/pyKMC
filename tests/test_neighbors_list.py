from pykmc import System, Config, NeighborsList
import pytest
from pytest_lazy_fixtures import lf

expected_neighbors = {
    "system_single_type_fcc": 12,
}


class TestNeighborsList:
    @pytest.mark.parametrize(
        "system_name, system, config",
        [
            (
                "system_single_type_fcc",
                lf("system_single_type_fcc"),
                lf("config_system_single_type"),
            )
        ],
    )
    def test_get_neighbors_list(self, system_name: str, system: System, config: Config):
        nl = NeighborsList(system, config)

        expected_nb_neighbors = expected_neighbors[system_name]

        for i in range(len(system.types)):
            neighbors = nl.get_neighbors("rnei", i)
            assert len(neighbors) == expected_nb_neighbors

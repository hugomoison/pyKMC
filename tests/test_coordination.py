from pykmc import AtomicEnvironment, NeighborsList
import pytest
from pytest_lazy_fixtures import lf
from collections import Counter


class TestCoordination :

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_coordination_bulk_fcc(self, system, config) :
        """All atoms in a perfect FCC bulk should be classified as crystal (12 neighbors)."""
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment('coordination', nl.neighbors_list['rnei'], coordination_threshold=12)

        # All atoms in perfect FCC should be crystal
        assert all(e == "crystal" for e in ae.atomic_environment_list)

    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))])
    def test_coordination_vacancy(self, system, config) :
        """FCC with vacancy: vacancy neighbors should be noncrystal (< 12 neighbors)."""
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
        ae = AtomicEnvironment('coordination', nl.neighbors_list['rnei'], coordination_threshold=12)

        hash_count = Counter(ae.atomic_environment_list)
        # Should have some noncrystal atoms (vacancy neighbors)
        assert "noncrystal" in hash_count
        # 12 atoms around the vacancy lose a neighbor
        assert hash_count["noncrystal"] == 12

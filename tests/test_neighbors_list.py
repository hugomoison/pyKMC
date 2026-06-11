from pykmc import System, Config, NeighborsList
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

expected_neighbors = {
    "system_single_type_fcc": 12, 
}


class TestNeighborsList : 


    @pytest.mark.parametrize("system_name, system, config", [("system_single_type_fcc", lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_get_neighbors_list(self, system_name: str, system: System, config: Config) :
        nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)

        expected_nb_neighbors = expected_neighbors[system_name]

        for i in range(len(system.types)) : 
            neighbors = nl.get_neighbors('rnei', i)
            assert len(neighbors) == expected_nb_neighbors




def _simple_cubic_system(pbc: "np.ndarray") -> System:
    """4x4x4 simple cubic lattice, spacing 1.0, in a 4.0 box."""
    coords = np.array(
        [[x, y, z] for x in range(4) for y in range(4) for z in range(4)],
        dtype=float,
    )
    return System(
        positions=coords,
        types=np.array(["Ni"] * len(coords)),
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=pbc,
        index=np.arange(len(coords)),
    )


def test_mixed_pbc_rcut_includes_central_atom() -> None :
    """Mixed-PBC environments must match the periodic convention.

    The rcut environment includes the central atom itself (event building
    locates the moving atom inside its own environment via np.where), while
    rnei excludes it. Both conventions must hold in the mixed-PBC branch
    exactly as they do in the fully periodic branch.
    """
    nl_mixed = NeighborsList(_simple_cubic_system(np.array([True, True, False])), 1.1, 1.8)
    nl_periodic = NeighborsList(_simple_cubic_system(np.array([True, True, True])), 1.1, 1.8)

    for i in range(64) :
        assert i in nl_mixed.get_neighbors("rcut", i)
        assert i not in nl_mixed.get_neighbors("rnei", i)
        assert i in nl_periodic.get_neighbors("rcut", i)
        assert i not in nl_periodic.get_neighbors("rnei", i)

    # An atom in the slab interior (z=1,2 layers stay fully coordinated when
    # only z is non-periodic) sees the same environment in both systems.
    interior = 4 * 4 * 1 + 4 * 1 + 1  # (x=1, y=1, z=1)
    assert sorted(nl_mixed.get_neighbors("rcut", interior)) == sorted(
        nl_periodic.get_neighbors("rcut", interior)
    )

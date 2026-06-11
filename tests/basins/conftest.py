import pytest
import os
from pykmc.basins import (
    BasinStatesConnectivity,
    StateData,
    StatesConnectivity,
    BasinStatesConnectivity,
)
from pykmc import NeighborsList, AtomicEnvironment
import pandas as pd


@pytest.fixture
def connectivity_table_Cu():
    """Connectivity table for Copper system with 1SIA and 1Vac, with removed SIA translation event"""
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    filepath = os.path.join(data_dir, "basin_connectivity_Cu_fake.pickle")

    connectivity_table = BasinStatesConnectivity()
    connectivity_table.df = pd.read_pickle(filepath)

    return connectivity_table


@pytest.fixture
def mock_state_data_Cu(system_Cu):
    """Fixture returning a dummy StateData for testing basin exploration."""
    system = system_Cu
    nl = NeighborsList(system=system, rnei=2.9, rcut=6.5)
    ae = AtomicEnvironment(
        style="cna/graph",
        neighbors_list=nl.neighbors_list["rnei"],
        environment_list=nl.neighbors_list["rcut"],
    )
    state = StateData(
        system=system_Cu,
        environment=ae,
        neighbors_list=nl,
        transient=True,
        visited=False,
    )
    return state


@pytest.fixture
def fake_connectivity_df():
    """Fake connectivity table shared across StatesConnectivity fixtures."""
    return pd.DataFrame(
        {
            "state": [0, 0, 1, 2],
            "state_connexion": [1, 2, 0, 0],
            "event_connexion": [12, 34, 13, 35],
            "central_atom": [345, 7, 911, 20],
            "sym": [0, 1, 0, 0],
            "transient": [True, False, True, False],
            "dE_forward": [0.2, 0.1, 0.8, 1.1],
            "k_forward": [2, 1, 8, 1],
            "dE_backward": [0.09, 0.13, 0.7, 1.3],
            "k_backward": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def mock_statesconnectivity(fake_connectivity_df):
    sc = StatesConnectivity()
    sc.df = fake_connectivity_df.copy()
    return sc


@pytest.fixture
def mock_basinstatesconnectivity(fake_connectivity_df):
    sc = BasinStatesConnectivity()
    sc.df = fake_connectivity_df.copy()
    return sc

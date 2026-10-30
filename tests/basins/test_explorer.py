import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.basins import BasinGenericEventExplorer
from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant


class TestBasinExplorer:
    def test_generic_event_explorer(
        self, test_logger, mock_config, reference_table_Cu_fake, mock_state_data
    ):

        test_logger.debug(":=> Running basin exploration with generic event.")

        basin_explorer = BasinGenericEventExplorer(mock_config, reference_table_Cu_fake)
        basin_explorer.explore(state=mock_state_data)

        test_logger.debug(
            "connexion_table : \n{}".format(basin_explorer.get_connectivity_table())
        )


def _explore_one_hop(style: str, config: RateConstantConfig) -> pd.Series:
    """Explore a state where one hop and its backward hop are both transient."""
    table = pd.DataFrame(
        {
            "idx_ref": [0, 1],
            "event_id": ["a", "b"],
            "id_final": ["b", "a"],
            "idx_backward": [1, 0],
            "energy_barrier": [0.2, 0.3],
            # stored k from another k0, as in a loaded table
            "k": [111.0, 222.0],
            "sym_matrix": [[np.eye(3)], [np.eye(3)]],
        }
    )
    reference_table = SimpleNamespace(
        table=table,
        rate_constant=RateConstant.create(style, config=config),
        has_id_subset_table=lambda ids: table[table["event_id"].isin(ids)],
    )
    environment = SimpleNamespace(
        atomic_environment_list=["a"], get_atoms_with_id=lambda event_id: [3]
    )
    run_config = SimpleNamespace(
        basin=SimpleNamespace(energy_thr=0.4), rateconstant=config
    )

    explorer = BasinGenericEventExplorer(run_config, reference_table)
    explorer.explore(state=SimpleNamespace(environment=environment))

    connectivity = explorer.get_connectivity_table()
    assert len(connectivity) == 1
    return connectivity.iloc[0]


def test_explorer_reads_the_stored_k_without_a_per_event_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)

    row = _explore_one_hop("constant", config)

    assert row["transient"]
    assert row["k_forward"] == 111.0
    assert row["k_backward"] == 222.0


def test_explorer_uses_k0_with_a_per_event_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    kT = PhysicalConstants.kb * config.T

    row = _explore_one_hop("htst", config)

    assert row["k_forward"] == pytest.approx(10.0 * math.exp(-0.2 / kT), rel=1e-15)
    assert row["k_backward"] == pytest.approx(10.0 * math.exp(-0.3 / kT), rel=1e-15)

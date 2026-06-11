from pykmc.basins import BasinGenericEventExplorer


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

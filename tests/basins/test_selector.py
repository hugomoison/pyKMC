from pykmc.basins import FPTASelector
import pandas as pd


class TestSelector:
    def test_ftpa(self, test_logger, connectivity_table_Cu):

        test_logger.debug("FTPA selector for Copper fake")
        # Get fake connectivity table (Cu 1 sia 1 vac, remove transition sia event)
        connectivity_table = connectivity_table_Cu

        selector = FPTASelector()
        result = selector.select_from_connectivity(connectivity_table)

        test_logger.debug(
            "For connectivity table : \n {}".format(connectivity_table.df)
        )
        test_logger.debug(
            "FTPASelector build Generator matrix : \n {}".format(selector.M_abs)
        )
        test_logger.debug("And reduced matrix : \n {}".format(selector.M_abs_reduced))
        test_logger.debug(
            "Got exit time = {} and exit state = {}".format(
                result.ok_value().t_exit, result.ok_value().exit_state
            )
        )

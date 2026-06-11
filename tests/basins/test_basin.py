from pykmc.basins import BasinsGenericEvents
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory

logger = logging.getLogger("tests")


class TestBasin:
    def test_connectivity_table_construction(
        self,
        test_logger,
        config_Cu,
        reference_table_Cu_fake,
        system_Cu,
        visited_environments_Cu,
    ):

        # Create Manager
        factory = ManagerFactory(
            n_sessions=config_Cu.control.n_sessions, use_rank_0=True
        )
        manager = factory.launch()

        if manager is not None:  # On rank 0
            manager.initialize_sessions(config_Cu, system_Cu)

            self.basin = BasinsGenericEvents(
                config=config_Cu,
                reference_table=reference_table_Cu_fake,
                known_environments=visited_environments_Cu,
                manager=None,
            )
            self.basin.manager = manager

            result = self.basin.execute(system=system_Cu)
            if result.is_ok():
                test_logger.debug("Find Exit State : ")
                test_logger.debug(
                    "Exit time t_exit = {}ps".format(result.ok_value().t_exit)
                )
                test_logger.debug(
                    "Exit state n : {}".format(result.ok_value().exit_state)
                )
            else:
                test_logger.debug("Error: {}".format(result.err_value()))

            manager.close_all()

from pykmc.basins import StatesConnectivity
import logging

logger = logging.getLogger("tests")


class TestStatesConnectivity:
    def test_add_connectivity(self, test_logger):

        state_connectivity = StatesConnectivity()
        test_logger.debug(
            "From States Connectivity DataFrame : \n {}".format(state_connectivity.df)
        )
        test_logger.debug("Add a new connexion between state 0 and 1")
        state_connectivity.add_connectivity(
            state=0,
            state_connexion=1,
            event_connexion=11,
            central_atom=342,
            sym=0,
            transient=True,
            dE_forward=0.1,
            k_backward=9,
            dE_backward=0.3,
            k_forward=6,
        )
        test_logger.debug(
            "New States Connectivity DataFrame : \n {}".format(state_connectivity.df)
        )

    def test_get_transition_to_state(self, test_logger, mock_statesconnectivity):

        test_logger.debug(
            "From States Connectivity DataFrame : \n {}".format(
                mock_statesconnectivity.df
            )
        )
        res = mock_statesconnectivity.get_transition_to_state(
            1, as_tuples=True, return_all=False
        )
        test_logger.debug("Get first transition to state 1 : \n {}".format(res))

    def test_merge_connectivity_table(self, test_logger, mock_basinstatesconnectivity):

        conn1 = mock_basinstatesconnectivity
        conn2 = mock_basinstatesconnectivity

        test_logger.debug("Before merging : df = {} \n".format(conn1.get_table()))
        conn1.merge(conn2)
        test_logger.debug("After merging : df = {}\n".format(conn1.get_table()))

    def test_update_state_index(self, test_logger, mock_basinstatesconnectivity):

        conn1 = mock_basinstatesconnectivity

        test_logger.debug(
            "Before update state index : df = {} \n".format(conn1.get_table())
        )
        test_logger.debug("Changing state 2 to state 1")
        conn1.change_state_index(2, 1)
        test_logger.debug(
            "After update state index : df = {} \n".format(conn1.get_table())
        )

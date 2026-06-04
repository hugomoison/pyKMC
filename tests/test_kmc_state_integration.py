"""Integration tests for the State object inside the KMC object (issue #63).

These exercise the property-delegation wiring without needing the engine/MPI:
the KMC object's ``system`` / ``neighbors_list`` / ``atomic_environment`` must
read through ``self.state``.
"""

from unittest.mock import Mock

from pykmc import System
from pykmc.kmc import KMC
from pykmc.state import State


def test_kmc_properties_delegate_to_state(
    system_single_type_fcc: System, mock_config: Mock
) -> None:
    """Derived properties read through self.state."""
    kmc = KMC(mock_config)
    kmc.state = State(system_single_type_fcc, mock_config)
    assert kmc.system is kmc.state.system
    assert kmc.neighbors_list is kmc.state.neighbors_list
    assert kmc.atomic_environment is kmc.state.atomic_environment


def test_kmc_property_without_state_raises(mock_config: Mock) -> None:
    """Accessing a derived property before the State is set is a clear error."""
    kmc = KMC(mock_config)
    raised = False
    try:
        _ = kmc.system
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when state is None"

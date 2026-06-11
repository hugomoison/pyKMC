import pytest
from pytest_lazy_fixtures import lf
from pykmc import System
import numpy as np


class TestSystem:
    def test_create_from_file_xyz(self, tmp_path):
        file_xyz = """4 
Lattice=\"3.52 0.0 0.0 0.0 3.52 0.0 0.0 0.0 3.52\" pbc=\"T T T\" 
Ni 1.0 0.0 0.0 
Ni 0.0 1.0 0.0 
Ni 0.0 0.0 1.0 
Ni 1.0 1.0 1.0
"""
        # Temporary file
        test_file = tmp_path / "test.xyz"
        test_file.write_text(file_xyz)
        # Initialise ton système
        system = System.create_from_file(test_file)

        np.testing.assert_array_equal(system.types, np.array(["Ni", "Ni", "Ni", "Ni"]))
        positions = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        np.testing.assert_allclose(system.positions, positions)
        cell = np.array([[3.52, 0.0, 0.0], [0.0, 3.52, 0.0], [0.0, 0.0, 3.52]])
        np.testing.assert_allclose(system.cell[:], cell)
        np.testing.assert_array_equal(system.pbc, np.array([True, True, True]))
        np.testing.assert_array_equal(system.index, np.array([0, 1, 2, 3]))

    @pytest.mark.parametrize("system", [lf("system_single_type_fcc")])
    def test_update_all_positions_nowrap(self, system: System):
        new_positions = np.array(
            [[1.0, 1.0, 0.0], [1.5, 3.4, 2.4], [2.3, 0.3, 0.8], [2.6, 1.4, 1.4]]
        )
        system.update_positions(new_positions)
        np.testing.assert_allclose(system.positions, new_positions, rtol=1e-6)

    @pytest.mark.parametrize("system", [lf("system_single_type_fcc")])
    def test_update_all_positions_wrap(self, system: System):
        cell = np.diag(system.cell)
        half_cell = cell / 2.0
        new_positions = np.array(
            [
                [cell[0], cell[1], cell[2]],
                [-half_cell[0], -half_cell[1], -half_cell[2]],
                [
                    cell[0] + half_cell[0],
                    cell[1] + half_cell[1],
                    cell[2] + half_cell[2],
                ],
                [1.0, -1.0, cell[2] + 1.0],
            ]
        )
        system.update_positions(new_positions)
        expected_positions = new_positions % cell
        np.testing.assert_allclose(
            system.positions, expected_positions, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize("system", [lf("system_single_type_fcc")])
    def test_update_index_positions(self, system: System):
        new_positions = np.array([1.5, 3.4, 2.4])
        system.update_positions(new_positions, atom_idx=np.array([1]))
        expected_positions = system.positions
        expected_positions[1] = new_positions
        np.testing.assert_allclose(system.positions, expected_positions, rtol=1e-6)

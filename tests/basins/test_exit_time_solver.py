import numpy as np
import numpy.testing as npt
from pykmc.basins import solve_master_equation, BisectionSolver


class TestSolver:
    def test_solve_master_equation(self, test_logger):

        # Mock reduced matric
        M_abs_reduced = np.array(
            [
                [+0.17, -0.30, -0.20, 0.00],
                [-0.05, 0.48, -0.25, 0.00],
                [-0.10, -0.10, 0.45, 0.00],
                [-0.02, -0.08, -0.00, 0.00],
            ]
        )

        p0 = np.array([1, 0, 0, 0])
        t0 = 1.0 / np.sum(np.diag(M_abs_reduced))

        test_logger.debug("Solve Master Equation for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, False)
        test_logger.debug("found p = \n {}".format(p))

        # computed with gnu octave
        res_expected = np.array([0.869014, 0.041671, 0.070828, 0.018488])
        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)

        test_logger.debug("Solve Master Equation Using Sprectral Decomposition for : ")
        test_logger.debug("M = \n {}".format(M_abs_reduced))
        test_logger.debug("p0 = \n {}".format(p0))
        test_logger.debug("t = {}".format(t0))
        p = solve_master_equation(M_abs_reduced, t0, p0, True)
        test_logger.debug("found p = \n {}".format(p))

        test_logger.debug("Expected p = \n {}".format(res_expected))

        npt.assert_allclose(p, res_expected, rtol=1e-4)

    def test_find_texit(self, test_logger):

        M_abs_reduced = np.array(
            [
                [1.89645002e-02, -9.48225009e-03, -9.48225009e-03, 0.00000000e00],
                [-9.48225009e-03, 1.89645002e-02, -9.48225009e-03, 0.00000000e00],
                [-9.48225009e-03, -9.48225009e-03, 1.89645002e-02, 0.00000000e00],
                [-2.83934789e-10, -2.83934789e-10, -2.83934789e-10, 0.00000000e00],
            ]
        )

        p0 = np.array([1, 0, 0, 0])
        r = 0.9

        solver = BisectionSolver(
            M=M_abs_reduced, p0=p0, r=r, spectral_decomposition=True
        )

        test_logger.debug("Find t_exit for r = {}".format(r))
        test_logger.debug("With M = \n {}".format(M_abs_reduced))
        test_logger.debug("And p0 = \n {}".format(p0))

        res = solver.solve()

        if res.is_ok():
            t_exit = res.ok_value().t_exit
            test_logger.debug("Find t_exit = {}ps".format(t_exit))
        else:
            err = res.err_value()
            test_logger.debug("Err while searching t_exit : {}".format(err))

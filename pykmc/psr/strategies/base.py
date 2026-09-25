"""Base class for point set registration strategies."""

from abc import abstractmethod

from pykmc._core import Registrable

from ...result import ErrorInfo, Result
from ..result import PSROutput


class PSRStrategy(Registrable, root=True):
    """Base class for point set registration algorithms.

    A strategy registers two point sets and reports the rigid transformation
    (rotation, translation, permutation) and matching score found between them.
    Operates on plain point-cloud arrays only.
    """

    @abstractmethod
    def match(
        self, nat1: int, typ1: list, coords1, nat2: int, typ2: list, coords2
    ) -> Result[PSROutput, ErrorInfo]:
        """Register two point sets.

        Parameters
        ----------
        nat1 : int
            Number of atoms in the first point set.
        typ1 : list[str]
            Atom types of the first point set.
        coords1 : np.ndarray
            Positions of the first point set.
        nat2 : int
            Number of atoms in the second point set.
        typ2 : list[str]
            Atom types of the second point set.
        coords2 : np.ndarray
            Positions of the second point set.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The transformation found, or the failure encountered.

        """

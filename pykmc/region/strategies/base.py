"""Base class for geometric region strategies."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from pykmc._core import Registrable


class RegionStrategy(Registrable, root=True):
    """Base class for the geometric shapes of a region.

    A strategy only answers the geometric question: which positions lie inside
    the shape. Selection by type or index, and the ``outside`` inversion, are
    done by the :class:`~pykmc.region.Region` facade.

    A concrete strategy declares its own config `Protocol` and stores it in
    `self.config`.
    """

    @abstractmethod
    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return which positions lie inside the shape.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Cartesian positions.
        cell : np.ndarray, shape (3, 3), optional
            Simulation cell. When given, periodic boundary conditions are
            applied in all three directions. When None, positions are used as
            they are.

        Returns
        -------
        np.ndarray, shape (N,)
            Boolean mask, True for positions inside the shape.

        """

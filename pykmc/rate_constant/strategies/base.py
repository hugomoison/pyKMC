"""Base class for rate constant strategies."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

import numpy as np

from pykmc._core import Registrable

from pykmc._core.constants import PhysicalConstants

if TYPE_CHECKING:
    from ...config import Config
    from ...engine.base import EngineExtension
    from ...manager import Manager
    from ...log import LogKMC
    from ...system import System
    from ..result import PrefactorOutcome, PrefactorRequest


class RateConstantConfigLike(Protocol):
    """Configuration read by every rate constant strategy."""

    k0: float  # ps^-1 (THz)
    T: float  # K


class RateConstantStrategy(Registrable, root=True):
    r"""Base class for rate constant computation methods.

    Every strategy computes the same functional form,
    $$
    k = A \, e^{-\frac{dE}{k_{b}T}}
    $$
    and differs only in the prefactor `A` returned by `prefactor`.

    Parameters
    ----------
    config : RateConstantConfigLike
        The `[RateConstant]` section of the configuration (`k0`, `T`).

    """

    per_event_prefactor: ClassVar[bool] = False
    """True when events carry their own prefactor, computed by `compute_prefactors`.

    The core then keeps full-size saddles through refinement, adds the prefactor
    columns to the event tables, and compares refinement thresholds as rates (the
    rate is no longer monotone in the barrier).
    """

    def __init__(self, config: RateConstantConfigLike) -> None:
        self.config = config

    @property
    def engine_extensions(self) -> tuple[type[EngineExtension], ...]:
        """Engine extensions every session must load for this strategy.

        A property, so that a strategy imports its extensions, and the engine they
        need, only when it is the one configured. Default: none.
        """
        return ()

    def setup(
        self, config: Config, manager: Manager, system: System, loggers: LogKMC
    ) -> None:
        """Check that the strategy can run, once the engines exist.

        Parameters
        ----------
        config : Config
            The full simulation configuration.
        manager : Manager
            The engine session pool.
        system : System
            The simulated system.
        loggers : LogKMC
            The simulation loggers.

        Notes
        -----
        Default: nothing.

        """

    @property
    def settings_record(self) -> str:
        """Settings stored with each prefactor, compared when a table is loaded."""
        return ""

    def compute_prefactors(
        self,
        requests: list[PrefactorRequest],
        system: System,
        frozen_ids: np.ndarray,
    ) -> list[list[PrefactorOutcome]]:
        """Compute the prefactor of each direction of new events.

        Called only when `per_event_prefactor` is True, after `setup`.

        Parameters
        ----------
        requests : list[PrefactorRequest]
            The events.
        system : System
            The simulated system, at the minimum the events start from.
        frozen_ids : np.ndarray
            Indices of the frozen atoms.

        Returns
        -------
        list[list[PrefactorOutcome]]
            For each request, one outcome per minimum.

        Notes
        -----
        It never raises: an event whose prefactor cannot be computed gets an
        outcome without a value.

        """
        raise NotImplementedError(f"{self.name} has no per-event prefactor")

    @abstractmethod
    def prefactor(self, value: Optional[float] = None) -> float:
        """Return the pre-exponential factor of one event.

        Parameters
        ----------
        value : float, optional
            The event's own prefactor (ps^-1). `None` or a non-finite value (a
            missing cell of a float table column reads back as NaN) means the
            event has none. A strategy that does not use per-event prefactors
            ignores it.

        Returns
        -------
        float
            The prefactor (ps^-1).

        """

    def compute_rate(self, dE: float, prefactor: Optional[float] = None) -> float:
        r"""Compute the rate constant of an event.

        It uses the following equation :
        $$
        k = A \, e^{-\frac{dE}{k_{b}T}}
        $$
        where `A` is `prefactor(prefactor)` and `T` is `[RateConstant] T`.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        prefactor : float, optional
            The event's own prefactor (ps^-1), forwarded to `prefactor`.

        Returns
        -------
        float
            The rate constant (ps^-1).

        Notes
        -----
        A missing or non-finite `prefactor` is not an error: the strategy falls
        back to its own prefactor (`[RateConstant] k0` for `constant` and `htst`).

        """
        return self.prefactor(prefactor) * math.exp(
            -dE / (PhysicalConstants.kb * self.config.T)
        )

"""Manages rate constant computation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .result import PrefactorOutcome, PrefactorRequest
from .strategies import RateConstantStrategy

if TYPE_CHECKING:
    from ..config import Config
    from ..engine.base import EngineExtension
    from ..log import LogKMC
    from ..manager import Manager
    from ..system import System


class RateConstant:
    """Compute the rate constant of an event, using a pluggable strategy.

    Parameters
    ----------
    strategy : RateConstantStrategy
        The rate constant strategy to use.

    """

    def __init__(self, strategy: RateConstantStrategy) -> None:
        self._strategy = strategy

    @property
    def engine_extensions(self) -> tuple[type[EngineExtension], ...]:
        """Engine extensions every session must load for the strategy."""
        return self._strategy.engine_extensions

    @property
    def per_event_prefactor(self) -> bool:
        """True when events carry their own prefactor, from `compute_prefactors`."""
        return self._strategy.per_event_prefactor

    @property
    def settings_record(self) -> str:
        """Settings stored with each prefactor, compared when a table is loaded."""
        return self._strategy.settings_record

    def setup(
        self, config: Config, manager: Manager, system: System, loggers: LogKMC
    ) -> None:
        """Let the strategy check that it can run, once the engines exist.

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

        """
        self._strategy.setup(config, manager, system, loggers)

    def compute_prefactors(
        self,
        requests: list[PrefactorRequest],
        system: System,
        frozen_ids: np.ndarray,
    ) -> list[list[PrefactorOutcome]]:
        """Compute the prefactor of each direction of new events.

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

        """
        return self._strategy.compute_prefactors(requests, system, frozen_ids)

    def compute_rate(self, dE: float, prefactor: Optional[float] = None) -> float:
        """Compute the rate constant of an event using the configured strategy.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        prefactor : float, optional
            The event's own prefactor (ps^-1). `None` or a non-finite value means
            the event has none.

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        return self._strategy.compute_rate(dE, prefactor=prefactor)

    @classmethod
    def create(cls, style: str, **kwargs) -> "RateConstant":
        """Build a RateConstant wired with the requested strategy.

        Parameters
        ----------
        style : str
            Name of the registered strategy (e.g. "constant").
        **kwargs
            Forwarded to the strategy's constructor (e.g. ``config=config.rateconstant``).

        """
        return cls(strategy=RateConstantStrategy.create(style, **kwargs))

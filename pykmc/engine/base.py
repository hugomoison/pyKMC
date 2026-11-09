from __future__ import annotations
from abc import abstractmethod
from typing import Any
import numpy as np
from ase.cell import Cell
from pykmc._core import Registrable


class EngineExtension:
    """
    Base class for all engine extensions.

    Centralises registration on the engine and guarantees that
    ``self.engine`` is always defined.

    Parameters
    ----------
    engine : Engine
        The engine instance to attach the extension to.

    Notes
    -----
    Subclasses **must** call ``super().__init__(engine)`` in their own
    ``__init__``. This call registers the extension on the engine (via
    ``engine.register(self)``) and makes its public methods accessible
    through ``engine.<method>``. Omitting this call means the extension
    will never be attached.

    Examples
    --------
    ::

        class MyExtension(EngineExtension):
            def __init__(self, engine: Engine, my_param: float):
                super().__init__(engine)   # ← required
                self.my_param = my_param

            def my_method(self): ...
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        engine.register(self)


class Engine(Registrable, root=True):
    """
    Abstract base class for engines use for the KMC simulation.

    An engine can be used standalone or as a backend in a master-worker
    architecture via the manager module.

    All abtract methods are mandatory in order to perform the simulation.
    """

    def __init__(self):
        self._extensions: dict[str, object] = {}

    def register(self, ext: EngineExtension) -> None:
        """
        Register an extension on this engine.

        The extension's public methods become accessible directly on the
        engine instance via ``__getattr__`` delegation.

        Parameters
        ----------
        ext : EngineExtension
            Extension instance to register.

        Raises
        ------
        ValueError
            If an extension with the same class name is already registered
            (class names must be unique across all extensions, even across
            modules), if any public method of `ext` shadows a native engine
            method, or if any public method of `ext` conflicts with a method
            already provided by a registered extension.
        """

        ext_name = type(ext).__name__
        if ext_name in self._extensions:
            raise ValueError(
                f"An extension named '{ext_name}' is already registered. "
                "Extension class names must be unique across all modules."
            )

        # Discover methods statically from the class to avoid executing @property getters,
        # which would crash if called before the subclass __init__ body has run.
        new_methods = {
            m
            for m in dir(type(ext))
            if not m.startswith("_") and callable(getattr(type(ext), m, None))
        }

        # Check against native engine methods
        native = {
            name
            for cls in type(self).__mro__
            for name in cls.__dict__
            if not name.startswith("_")
        }
        clash_with_native = new_methods & native
        if clash_with_native:
            raise ValueError(
                f"Extension '{ext_name}' shadows native engine methods: "
                + ", ".join(f"'{m}'" for m in sorted(clash_with_native))
            )

        # Check against already registered extensions
        for registered_name, registered_ext in self._extensions.items():
            clash = new_methods & {
                m
                for m in dir(type(registered_ext))
                if not m.startswith("_")
                and callable(getattr(type(registered_ext), m, None))
            }
            if clash:
                raise ValueError(
                    f"Extension '{ext_name}' has conflicting methods with '{registered_name}' :\n"
                    + "\n".join(f"  • {m!r}" for m in sorted(clash))
                )
        self._extensions[ext_name] = ext

    def __dir__(self) -> list[str]:
        names = list(super().__dir__())
        for ext in self._extensions.values():
            names.extend(
                m
                for m in dir(ext)
                if not m.startswith("_") and callable(getattr(ext, m, None))
            )
        return names

    def __getattr__(self, name: str) -> Any:
        # Called only when `name` is not found through normal attribute lookup.
        # Protects against RecursionError if _extensions is not yet set.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        extensions = object.__getattribute__(self, "_extensions")
        for ext in extensions.values():
            attr = getattr(ext, name, None)
            if callable(attr):
                return attr
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @abstractmethod
    def start(self) -> None:
        """Start the engine. Must be called before any operation."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the engine and free resources."""

    @abstractmethod
    def initialize_parameters(self) -> None:
        """Set default simulation parameters so the engine can run operations (e.g. units, pbc, ...)."""

    @abstractmethod
    def initialize_system(
        self,
        types: list[str] | np.ndarray,
        positions: np.ndarray,
        cell: Cell,
        pbc: list[bool] | np.ndarray,
        species: list[str] | None = None,
    ) -> None:
        """Load atomic system into the engine.

        ``species`` is every species of the simulation. Pass it when the
        system is a cropped region that may miss a species, so the engine
        keeps the type numbering of the full system.
        """

    @abstractmethod
    def initialize_potential(self) -> None:
        """Set interatomic potential."""

    @abstractmethod
    def get_positions(self) -> np.ndarray | None:
        """Return current atomic positions, shape (N,3)."""

    @abstractmethod
    def set_positions(self, positions: np.ndarray) -> None:
        """Set atomic position, shape(N,3)."""

    @abstractmethod
    def get_total_energy(
        self, positions: np.ndarray | None = None, recompute: bool = True
    ) -> float | None:
        """Return total energy of the system."""

    @abstractmethod
    def get_potential_energy(
        self, positions: np.ndarray | None = None, recompute: bool = True
    ) -> float | None:
        """Return potential energy of the system."""

    @abstractmethod
    def minimize(self, positions: np.ndarray | None = None) -> None:
        """Run energy minimization."""

    @abstractmethod
    def minimize_with_results(
        self, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, float] | None:
        """Run energy minimization and return (positions, total_energy)."""

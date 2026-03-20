from __future__ import annotations
from abc import ABC, abstractmethod 
import numpy as np 

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


class Engine(ABC) : 
    """
    Abstract base class for engines use for the KMC simulation.

    An engine can be used standalone or as a backend in a master-worker
    architecture via the manager module. In the latter case, concrete
    implementations should decorate each method with @registrable('<name>')
    so that build_registry() can expose them to the manager at runtime.

    All abtract methods are mandatory in order to perform the simulation.
    """

    def __init__(self):
        self._extensions: dict[str, object] = {}

    def register(self, ext: object) -> None:
        """
        Register an extension on this engine.

        The extension's public methods become accessible directly on the
        engine instance via ``__getattr__`` delegation.

        Parameters
        ----------
        ext : object
            Extension instance to register. Should be a subclass of
            :class:`EngineExtension`.

        Raises
        ------
        ValueError
            If an extension with the same class name is already registered,
            or if any public method of `ext` conflicts with a method already
            provided by a registered extension.
        """

        #Name of the extension, ie class name
        ext_name = type(ext).__name__
        if ext_name in self._extensions:
            raise ValueError(f"Extension '{ext_name}' already registered.")

        #Get all pulic method of ext
        new_methods = {
            m for m in dir(ext)
            if not m.startswith("_") and callable(getattr(ext, m))
        }

        #Check if method name already present in extension
        for registered_name, registered_ext in self._extensions.items():
            clash = new_methods & {
                m for m in dir(registered_ext)
                if not m.startswith("_") and callable(getattr(registered_ext, m))
            }
            if clash:
                raise ValueError(
                    f"Extension '{ext_name}' has conflicting methods with '{registered_name}' :\n"
                    + "\n".join(f"  • {m!r}" for m in sorted(clash))
                )
        self._extensions[ext_name] = ext

    def __getattr__(self, name: str):
        # Called only when `name` is not found through normal attribute lookup.
        # Protects against RecursionError if _extensions is not yet set.
        extensions = object.__getattribute__(self, "_extensions")
        for ext in extensions.values():
            if hasattr(ext, name):
                return getattr(ext, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @abstractmethod
    def start(self) -> None : 
        """Start the engine. Must be called before any operation. 

        Decorate with @registrable('start') in concrete implementations
        to expose via the manager registry if manager is used.
        """

    @abstractmethod
    def close(self) -> None : 
        """Shut down the engine and free resources.
        
        Decorate with @registrable('close') in concrete implementations
        to expose via the manager registry if manager is used.
        """

    @abstractmethod 
    def initialize_parameters(self) -> None : 
        """Set default simulation parameters so the engine can run operations (e.g. units, pbc, ...).
        
        Decorate with @registrable('initialize_parameters') in concrete implementations
        to expose via the manager registry if manager is used.
        """
    
    @abstractmethod 
    def initialize_system(self, **kwargs) -> None : 
        """Load atomic system into the engine.
        
        Decorate with @registrable('initialize_system') in concrete implementations
        to expose via the manager registry if manager is used.
        """

    @abstractmethod 
    def initialize_potential(self, **kwargs) -> None : 
        """Set interatomic potential.
        
        Decorate with @registrable('initialize_potential') in concrete implementations
        to expose via the manager registry if manager is used. 
        """ 

    @abstractmethod 
    def get_positions(self) -> np.ndarray|None : 
        """Return current atomic positions, shape (N,3).
        
        Decorate with @registrable('get_positions') in concrete implementations
        to expose via the manager registry if manager is used. 
        """
    
    @abstractmethod 
    def set_positions(self, positions: np.ndarray) -> None : 
        """Set atomic position, shape(N,3).
        
        Decorate with @registrable('set_positions') in concrete implementations
        to expose via the manager registry if manager is used. 
        """

    @abstractmethod 
    def get_total_energy(self, **kwargs) -> float|None : 
        """Return total energy of the system.
        
        Decorate with @registrable('get_total_energy') in concrete implementations
        to expose via the manager registry if manager is used. 
        """

    @abstractmethod
    def get_potential_energy(self, **kwargs) -> float|None : 
        """Return potential energy of the system.
        
        Decorate with @registrable('get_potential_energy') in concrete implementations
        to expose via the manager registry if manager is used. 
        """
    
    @abstractmethod 
    def minimize(self, **kwargs) -> None : 
        """Run energy minimization.

        Decorate with @registrable('minimize') in concrete implementations
        to expose via the manager registry if manager is used.
        """

    @abstractmethod 
    def minimize_with_results(self, **kwargs) -> tuple[np.ndarray, float] : 
        """Run energy minimization and return (positions, total_energy) after the minimization.

        Decorate with @registrable('minimize_with_results') in concrete implementations
        to expose via the manager registry if manager is used. 
        """
from abc import ABC, abstractmethod 
import numpy as np 

class Engine(ABC) : 
    """
    Abstract base class for engines use for the KMC simulation.

    An engine can be used standalone or as a backend in a master-worker
    architecture via the manager module. In the latter case, concrete
    implementations should decorate each method with @registrable('<name>')
    so that build_registry() can expose them to the manager at runtime.

    All abtract methods are mandatory in order to perform the simulation.
    """

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
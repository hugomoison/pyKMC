from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

class NeighborsBackend(ABC):
    """Abstract base class for neighbor list computation backends."""

    @property
    @abstractmethod
    def r_neighbors_cutoff(self) -> float: 
        pass

    @property
    @abstractmethod
    def r_env_cutoff(self) -> float:
        pass
    
    @abstractmethod
    def build(self) -> dict[str, list[list[int]]]:
        """Build neighbor lists for all atoms."""
        pass


class KDTreeBackend(NeighborsBackend) : 
    """ 
    Neighbor list backend using cKDTree. 

    Only support orthonhombic cells and pbc = [True, True, True]

    Raise valueError otherwise
    """

    def __init__(self, positions: np.ndarray, alat: float, r_neighbors_cutoff:float, r_env_cutoff: float) -> None : 
        self.positions = positions 
        self._alat = alat
        self._r_neighbors_cutoff = r_neighbors_cutoff
        self._r_env_cutoff = r_env_cutoff

    @property 
    def r_neighbors_cutoff(self) -> float : 
        return self._r_neighbors_cutoff
    @property 
    def r_env_cutoff(self) -> float : 
        return self._r_env_cutoff
    @property 
    def alat(self)-> float : 
        return self._alat


    def build(self) -> dict[str, list[list[int]]] : 
        """Build and populates the `neighbors_list`."""
        # Construct the kdTree
        box = [self.alat]*3
        tree = cKDTree(self.positions, boxsize=box)

        result = {"neighbors": [], "environments" : []}
        # Find first neighbors and atoms in environments
        for i in range(len(self.positions)):
            neighbors = tree.query_ball_point(self.positions[i], self.r_neighbors_cutoff)
            neighbors.remove(i)  # don't have self as neighbor
            result["neighbors"].append(neighbors)
            neighbors = tree.query_ball_point(self.positions[i], self.r_env_cutoff)
            result["environments"].append(neighbors)
        return result

    def __repr__(self):
        return (f"alat = {self.alat}, "
                f"r_neighbors_cutoff = {self.r_neighbors_cutoff}, "
                f"r_env_cutoff = {self.r_env_cutoff}")
    


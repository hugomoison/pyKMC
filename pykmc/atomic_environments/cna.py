from atomic_environments import BaseAtomicEnvironment
import numpy as np

class CnaAtomicEnvironment(BaseAtomicEnvironment) : 
    """ """

    def __init__(self, r_cna: float, neighbors_list: list | np.ndarray=None, backend: str = "default") -> None:
        self._r_cna = r_cna
        self._neighbors_list = neighbors_list
        self._hash_list = []
        self._backend = backend

    @property
    def r_cna(self) -> float:
        return self._r_cna

    @property
    def hash_list(self) -> list:
        return self._hash_list

    def compute_hash(self) -> None:
        match self._backend:
            case "default":
                self._hash_list = compute_cna_default(self._r_cna, self._neighbors_list)
            case "lammps":
                self._hash_list = self.backend.compute_cna_lammps(self._r_cna, self._neighbors_list)
            case _:
                raise ValueError(f"Unknown backend: '{self._backend}'. Expected 'default' or 'lammps'.")
            

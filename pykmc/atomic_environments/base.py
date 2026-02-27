from abc import ABC, abstractmethod

class BaseAtomicEnvironment(ABC) : 

    @property
    @abstractmethod
    def hash_list(self) -> list:
        pass

    @abstractmethod 
    def compute_hash(self) : 
        pass 

    def get_index_with_hash(self, target_hash: str)-> list : 
        return [i for i, e in enumerate(self.hash_list) if e == target_hash]

    def get_unknown_hash(self, known_hash: set[str])-> list : 
        return list(set(self.hash_list).difference(known_hash))

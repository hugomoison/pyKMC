from __future__ import annotations
from typing import Callable

def registrable(name: str):
    """Decorator that marks a method as exposable to the Manager registry.

    Parameters
    ----------
    name : str
        Key used in the registry.

    Examples
    --------
    >>> class MyEngine(Engine):
    ...     @registrable("minimize")
    ...     def minimize(self, **kwargs): ...
    ...
    ...     @registrable("get_positions")
    ...     def get_positions(self): ...
    """
    def decorator(func: Callable) -> Callable:
        func._registry_name = name
        return func
    return decorator
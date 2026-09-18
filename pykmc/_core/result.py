"""Result handling infrastructure.
 
This module provides a lightweight implementation of a `Result` type, inspired by
Rust/rustedpy, to clearly distinguish between successful and unsuccessful operations.
 
It holds **no** domain logic: output dataclasses (`EventSearchOutput`, `PSROutput`,
...) belong to the modules that produce them.
 
Includes:
- `Ok` / `Err` result wrapper types.
- `Result`, the alias for their union.
- `ErrorInfo`, a structured container describing an error.
- `ErrorType`, the error enumeration .
"""

from typing import Any, Generic, Optional, TypeAlias, TypeVar, Dict
from dataclasses import dataclass
from enum import Enum

# Construction of the Result Type :

TOK = TypeVar("TOK")
TERR = TypeVar("TERR")


class Ok(Generic[TOK]):
    """Wrapper representing a successful computation result.

    Attributes
    ----------
    _value : TOK
        The result of the successful operation.

    """

    _value: TOK

    def __init__(self, value: TOK) -> None:
        self._value = value

    def is_ok(self) -> bool:
        """Return True indicating a successful result."""
        return True

    def ok_value(self) -> TOK:
        """Return the value stored in the successful result."""
        return self._value
    
    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


class Err(Generic[TERR]):
    """Wrapper representing a failed computation result.

    Attributes
    ----------
    _err : TERR
        The error object or message describing the failure.

    """

    _err : TERR

    def __init__(self, err: TERR) -> None:
        self._err = err

    def is_ok(self) -> bool:
        """Return False indicating a failed result."""
        return False

    def err_value(self) -> TERR:
        """Return the error stored in the failed result."""
        return self._err
    
    def __repr__(self) -> str:
        return f"Err({self._err!r})"


Result: TypeAlias = Ok[TOK] | Err[TERR]
"""Alias representing either a successful (`Ok`) or failed (`Err`) result.
 
To unwrap a result, test the class rather than calling `is_ok()`: only
`isinstance` (or `match`) narrows the type in both branches.
 
>>> if isinstance(res, Ok):
...     value = res.ok_value()   # res: Ok[...]
... else:
...     info = res.err_value()   # res: Err[...]
"""


@dataclass
class ErrorInfo:
    """Structured information about an error that occurred during a simulation step.

    Attributes
    ----------
    type : ErrorType
        Type of the error.
    message : str
        Human-readable message describing the error.
    details : Optional[str]
        Optional technical details or context.
    variables : Optional[Dict[str, Any]]
        Optional dictionary of variables related to the error context.

    """

    type: "ErrorType | ErrorCode"
    message: str
    details: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


class ErrorType(Enum):
    """Enumeration of all error types that may occur during the simulation."""

    EVENT_NOT_FOUND = 1
    EVENT_MINIMA_NOT_MATCH_POSITIONS = 2
    EVENT_ENERGY_HIGHER_THAN_THRESHOLD = 11
    EVENT_ENERGY_LOWER_THAN_THRESHOLD = 12
    EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD = 13
    EVENT_ASYMMETRIC = 14
    EVENT_NOT_NEW = 15
    PSR_NO_MATCH_FOUND = 21
    PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD = 22
    REFINEMENT_INVALID_ENERGY_BARRIER = 31
    REFINEMENT_INVALID_MINIMA = 32
    RECONSTRUCTION_INVALID_MIN1 = 41
    RECONSTRUCTION_INVALID_MIN2 = 42
    BASIN_TEXIT_NOT_FOUND = 51

class ErrorCode(Enum):
    """Base class for per-module error enumerations.
 
    Each module declares its own subclass listing the failures it can report::
 
        class PSRError(ErrorCode):
            NO_MATCH_FOUND = auto()
 
    Two subclasses may reuse the same member name without clashing: they are
    distinct classes, so ``PSRError.NOT_FOUND != BasinError.NOT_FOUND``.
    """
 
    @property
    def key(self) -> str:
        """Return a stable identifier for output.
 
        Prefixed with the owning enumeration, so codes sharing a member name
        across modules remain distinguishable::
 
            >>> PSRError.NO_MATCH_FOUND.key
            'PSRError.NO_MATCH_FOUND'
        """
        return f"{type(self).__name__}.{self.name}"
 
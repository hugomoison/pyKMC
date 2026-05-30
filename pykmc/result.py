"""Result handling and data structures.

This module provides a lightweight implementation of a `Result` type, inspired by Rust/rustedpy,
to clearly distinguish between successful and unsuccessful operations. It also defines structured outputs for
various simulation steps such as event search, refinement, and point set registration.

Includes:
- `Ok` / `Err` result wrapper types.
- Enumerated error types for diagnostics.
- Output data containers (`EventSearchOutput`, `PSROutput`, `KMCLoopInfo`, etc.)
"""

from typing import TypeAlias, TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import yaml
import pandas as pd

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


class Err(Generic[TERR]):
    """Wrapper representing a failed computation result.

    Attributes
    ----------
    _err : TERR
        The error object or message describing the failure.

    """

    _err = TERR

    def __init__(self, err: TERR) -> None:
        self._err = err

    def is_ok(self) -> bool:
        """Return False indicating a failed result."""
        return False

    def err_value(self) -> TERR:
        """Return the error stored in the failed result."""
        return self._err


Result: TypeAlias = Ok[TOK] | Err[TERR]
"""Alias representing either a successful (`Ok`) or failed (`Err`) result."""


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

    type: "ErrorType"
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


# Dataclass to store operation outputs


@dataclass
class EventSearchOutput:
    """Store the output of a successful event search operation.

    Attributes
    ----------
    central_atom_index : int
        Index of the atom around which the event was searched.
    min1_positions : np.ndarray
        Atomic positions of the initial state.
    saddle_positions : np.ndarray
        Atomic positions at the saddle point.
    min2_positions : np.ndarray
        Atomic positions of the final state.
    dE_forward : float
        Forward energy barrier (min1 → saddle).
    dE_backward : float
        Backward energy barrier (min2 → saddle).
    move_atom_index : int
        Index of the atom that moved the most during the transition.
    cell : Optional[np.ndarray]
        Simulation cell, if applicable.

    """

    central_atom_index: int
    min1_positions: np.ndarray
    saddle_positions: np.ndarray
    min2_positions: np.ndarray
    dE_forward: float
    dE_backward: float
    move_atom_index: int
    #map: np.ndarray
    cell: Optional[np.ndarray] = None


@dataclass
class EventRefinementOutput:
    """Store the output of a refined transition event.

    Attributes
    ----------
    central_atom_index : int
        Index of the atom around which the event was refined.
    saddle_positions : np.ndarray
        Refined saddle point atomic positions.
    min2_positions : np.ndarray
        Refined atomic positions of the final minimum (if matched)
    E_saddle : float 
        Potential energy at the saddle point.
    dE_forward : Optional[float]
        Refined forward energy barrier (if matched).
    num_reference_event : Optional[int]
        Index of the corresponding reference event (if matched).
    refined: Optional[str] 
        If the event has been refined (T: True, F: False, B: In basin)
    """

    central_atom_index: int
    saddle_positions: np.ndarray
    E_saddle: float
    min2_positions: Optional[np.ndarray] = None
    dE_forward: Optional[float] = None
    num_reference_event: Optional[int] = None
    refined: Optional[str] = None


@dataclass
class PSROutput:
    """Store the result of a point set registration operation.

    Attributes
    ----------
    rotation_matrix : np.ndarray
        Rotation matrix used to align two patterns.
    translation_matrix : np.ndarray
        Translation vector applied for alignment.
    permutation_matrix : np.ndarray
        Mapping of atom indices from reference to current configuration.
    matching_score : float
        Score representing the quality of the match.

    """

    rotation_matrix: np.ndarray
    translation_matrix: np.ndarray
    permutation_matrix: np.ndarray
    matching_score: float

@dataclass 
class ReconstructionOutput : 
    """Store the result of a reconstruction"""
    min1_positions: np.ndarray 
    saddle_positions: np.ndarray 
    min2_positions: np.ndarray
    min2_etot: float

@dataclass 
class BasinSelectorOutput : 
    """"Store the result of the selector"""
    t_exit: float 
    exit_state: int 

@dataclass 
class BasinExitTimeSolverOutput : 
    """Sotre the results of exit time solver"""
    t_exit: float

@dataclass
class BasinOutput : 
    """Store the results of the basin."""
    initial_system_positions: np.ndarray
    central_atom: int
    saddle_positions: np.ndarray 
    final_positions: np.ndarray 
    neighbors: np.ndarray
    energy_barrier: float
    k_tot: float 
    t_exit: float
    exit_state: int
    from_state: int
    num_reference_event: int 

@dataclass
class AtomicEnvironmentInfo:
    """Store informations on atomic environments for one KMC step.

    Attributes
    ----------
    total_atomic_environments_encounter : int
        Total unique atomic environments seen so far.
    n_current_atomic_environments : int
        Number of environments in the current configuration.
    n_new_atomic_environments : int
        Number of new environments discovered in the last step.
    atoms_grouped_by_environment : list[list[int]]
        List of atom index groups sharing identical environments.

    """

    total_atomic_environments_encounter: int = 0
    n_current_atomic_environments: int = 0
    n_new_atomic_environments: int = 0
    atoms_grouped_by_environment: list[list[int]] = field(default_factory=list)


@dataclass
class ReferenceEventSearchInfo:
    """Summary of the outcomes of reference event search attempts.

    Attributes
    ----------
    total_event_searches : int
        Total number of event search attempts performed.
    n_successes : int
        Number of successful event searches.
    n_fails : dict[str, int]
        Dictionary mapping failure reasons (as strings) to the number of occurrences.

    """

    total_event_searches: int
    n_successes: int
    n_fails: dict[str, int]


@dataclass
class ReferenceValidEventsInfo:
    """Summary of valid and invalid events found during reference analysis.

    Attributes
    ----------
    n_valid_events : int
        Number of events considered valid.
    invalid_events : dict[str, int]
        Dictionary mapping invalidity reasons (as strings) to the number of corresponding events.

    """

    n_valid_events: int
    invalid_events: dict[str, int]


@dataclass
class RefinementsInfo:
    """Statistics related to event refinement attempts.

    Attributes
    ----------
    n_attempts : int
        Total number of refinement attempts.
    n_sucesses : int
        Number of successful refinements.
    n_fails : dict[str, int]
        Dictionary mapping refinement failure reasons.

    """

    n_attempts: int
    n_sucesses: int
    n_fails: dict[str, int]

@dataclass
class EventsInfo: 
    """Active events informations."""

    types: list[str] 
    central_atom: list[int]
    initial_topologies: list[str]
    reference_events: list[int]
    dE_forward: list[float]
    dE_backward: list[float]
    dE_asym: list[float]
    k: list[float]
    dra_i: list[float]
    dra_f: list[float]
    refined: list[str]

    def output_msg(self) -> str: 

        df = pd.DataFrame({
                'Types': self.types,
                'Central Atom': self.central_atom,
                'Ref Event': self.reference_events,
                'dE forward': self.dE_forward,
                'dE backward': self.dE_backward,
                'dE asym': self.dE_asym,
                'k': self.k,
                'dra_i': self.dra_i,
                'dra_f': self.dra_f,
                'Refined': self.refined,}).reset_index(drop=True)
        return df.to_string(index=True)        

@dataclass
class KMCLoopInfo:
    """Store summary information for a full KMC step.

    With metadata on atomic environments, valid events, refinement attempts.

    """

    step: int = 0
    atomic_environment_info: AtomicEnvironmentInfo = None
    reference_event_searches_info: ReferenceEventSearchInfo = None
    valid_event_info: ReferenceValidEventsInfo = None
    refinements_info: RefinementsInfo = None

    def output_msg(self) -> str:
        """Return a YAML-formatted summary of the loop info for logging purposes.

        Returns
        -------
        str
            YAML string.

        """
        cleaned = clean_dict(asdict(self))
        return yaml.dump(
            cleaned,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            explicit_start=True,
            Dumper=CustomDumper,
        )


class CustomDumper(yaml.Dumper):
    """YAML dumper class that forces proper indentation and formatting."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Increase indentation level for nested YAML structures.

        Overrides the default behavior to ensure proper formatting of
        nested mappings and sequences in block style.

        Parameters
        ----------
        flow : bool, optional
            Whether to use flow style (default: False).
        indentless : bool, optional
            Whether to omit indentation on the first level (ignored here).

        Returns
        -------
        None

        """
        return super().increase_indent(flow, indentless)


# Custom representer to force inner lists to be in flow style
def represent_list_preserve_flow(
    dumper: yaml.Dumper, data: list
) -> yaml.nodes.SequenceNode:
    """Represent lists in YAML with inline (flow) style if they contain only integers.

    Ensures that short lists (e.g., atom indices) are rendered inline
    for compact and readable YAML output.

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper instance.
    data : list
        The list to represent.

    Returns
    -------
    yaml.nodes.SequenceNode
        YAML node representing the sequence.

    """
    if all(isinstance(i, int|float) for i in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


# custom representer for lists
CustomDumper.add_representer(list, represent_list_preserve_flow)


def clean_dict(d: dict | list) -> dict | list:
    """Recursively remove empty or None fields from a dictionary or list.

    Parameters
    ----------
    d : dict or list
        The input structure to clean.

    Returns
    -------
    dict or list
        Cleaned structure.

    """
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v not in (None, [], {}, "")}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d if v not in (None, [], {}, "")]
    return d

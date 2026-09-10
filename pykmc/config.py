"""Configuration models for PyKMC simulations.

This module defines Pydantic BaseModel classes to structure and validate
all input parameters required for running PyKMC simulations
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, model_validator, ValidationError, field_validator
import configparser
from typing import Any, Literal
from dataclasses import dataclass
import re


@dataclass
class PhysicalConstants:
    """Store physical constants."""

    kb = 8.6173303e-05  # eV.K^-1
    h = 4.135667e-3  # eV.ps


def parse_wall_time_to_seconds(value: Any) -> Optional[float]:
    """Convert wall-time input to seconds.

    Accepted formats:
    - seconds as int/float/string: 300, "300"
    - simple units: "30s", "10m", "2h", "3d"
    - SLURM-like: "HH:MM:SS" or "D-HH:MM:SS"
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip().lower()

        try:
            return float(value)
        except ValueError:
            pass

        unit_match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", value)
        if unit_match:
            number = float(unit_match.group(1))
            unit = unit_match.group(2)
            factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            return number * factors[unit]

        slurm_match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", value)
        if slurm_match:
            days = int(slurm_match.group(1) or 0)
            hours = int(slurm_match.group(2))
            minutes = int(slurm_match.group(3))
            seconds = int(slurm_match.group(4))
            return days * 86400 + hours * 3600 + minutes * 60 + seconds

    raise ValueError(
        f"Invalid wall time value: {value}. "
        "Use seconds, '10m', '2h', '3d', 'HH:MM:SS', or 'D-HH:MM:SS'."
    )


class ControlConfig(BaseModel):
    """Core simulation control parameters."""

    initial_config: str = Field(
        default=...,
        description="File path for the initial atomic structure. This file should be parseable by `ase.io.read()` and contain atom types, positions, simulation cell, and periodic boundary conditions.",
    )

    trajectory_output: Optional[str] = Field(
        default="./trajkmc.xyz",
        description="File path where the simulation trajectory will be saved. The file should be writable by `ase.io.write` using `append=True `",
    )

    reference_table_output: Optional[str] = Field(
        default="./reference_table.pickle",
        description="File path where the reference table will be store in pickle format.",
    )

    visited_environments_output: Optional[str] = Field(
        default="./visited_environments.pickle",
        description="File path where the list of atomic environments that have been explored will be sore in pickle format."
    )

    reference_table: Optional[str] = Field(
        default=None,
        description="Path to a reference table generated from a previous simulation."
    )

    visited_environments: Optional[str] = Field(
        default=None,
        description="Path to a list of visited environment generated from a previous simulation."
    )

    restart_file: Optional[str] = Field(
        default = None,
        description="File with restart informations."
    )

    reconstruction: Optional[bool] = Field(
        default=True, description="If at each KMC step we reconstruct generic events.\n NOT WORKING"
    )

    n_steps: int = Field(
        default=..., description="Total number of simulation steps to run.", gt=0
    )

    wall_time_limit: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Wall-clock time limit. This is real elapsed runtime, not KMC physical time. "
            "Accepted formats: seconds, '10m', '2h', '3d', 'HH:MM:SS', or 'D-HH:MM:SS'. "
            "If set, the simulation stops cleanly when the elapsed wall time from the start "
            "of run() reaches this limit."
        ),
    )

    wall_time_buffer: float = Field(
        default=120.0,
        ge=0,
        description=(
            "Buffer time subtracted from wall_time_limit. "
            "Default is 120 seconds. "
            "Accepted formats: seconds, '10m', '2h', '3d', 'HH:MM:SS', or 'D-HH:MM:SS'."
        ),
    )

    @field_validator("wall_time_limit", "wall_time_buffer", mode="before")
    @classmethod
    def parse_wall_time_fields(cls, value):
        return parse_wall_time_to_seconds(value)

    reference_event_failure_threshold: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Remove a reference event from the catalog after this many "
            "refinement or reconstruction failures. If None, this check is disabled."
        ),
    )

    engine: Literal["lammps"] = Field(
        default=...,
        description="Which E/F Engine to use. Note : Only lammps is implemented.",
    )

    n_sessions: Optional[int] = Field(
        default=1,
        description="Number of Sessions"
    )

    engine_use_rank_0: Optional[bool] = Field(
        default=False,
        description="Deprecated : If use mpi rank 0 or not."
    )

    verbosity: Optional[int] = Field(
        default=1, description="Controls the level of detail in the simulation output."
    )

    refine_thr: Optional[float] = Field(
        default=0.9999,
        description="Event constributing to this percent of ktot are refined."
    )

    basin: Optional[bool] = Field(
        default=False,
        description="Basin mode"
    )

    active_volume: Optional[bool] = Field(
        default=False,
        description="Incorporate AV's into simulations, recommended for large systems"
    )

class AtomicEnvironmentConfig(BaseModel):
    """Atomic environments parameters."""

    style: Literal["cna", "graph", "cna/graph", "diamond/graph"] = Field(
        ...,
        description="Method used to characterize and assign an ID to an atom's local atomic environment",
    )

    rnei: float = Field(
        ...,
        description="Radius cutoff (in Angstrom) for defining the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors.",
    )

    rcut: Optional[float] = Field(
        default=None,
        description="Radius cutoff (in Angstrom) for defining the local atomic environment.",
    )

    neighbors_add: Optional[int] = Field(
        default=0,
        description="When `style` is 'cna/graph', specifies the N-th shell of neighbors whose graph IDs should also be computed.",
    )


class EventSearchConfig(BaseModel):
    """Event search parameters."""

    style: Literal["partn"] = Field(..., description="Method used to find events.")
    nsearch: int = Field(
        ...,
        description="Number of event searches to perform per unique atomic environment.",
    )
    emax_event: float = Field(
        default=5.0,
        description="Maximum energy barrier (in eV) for an event to be added to the reference table.",
    )
    emin_event: float = Field(
        default=0.0,
        description="Minimum energy forward and backward barrier (in eV) for an event to be added to the reference table.",
    )
    backward_emin_event: float = Field(
        default=0.0,
        description="To be used with `energy_assymetry`.",
    )
    energy_asymmetry: int = Field(
        default=5,
        description="Prevent highly asymmetric event to be added to the reference table."
                    "The con"
    )
    refined_minimum_delr_thr: float = Field(
        default = 0.1,
        description="Refinement is accepted only if the central atom moves less than this distance between the current position and the refined minimum."
    )
    refined_energy_thr: float = Field(
        default=0.05,
        description="Maximumallowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy.",
    )

    delr_thr: float = Field(
        default=0.5,
        description="delr threshold between one minima and the intial configuration to consider the event valid.",
    )




class PartnConfig(BaseModel):
    """pARTn parameters."""

    #Control
    verbosity: int = Field(default=2, description="pARTn verbosity")

    delr_thr: float = Field(
        default=0.1,
        description="Threshold at which an atom is considered to have moved. This threshold affects the npart parameter in the artn.out output."
    )

    evalf_max: int = Field(
        default = 9999,
        description="to stop an artn search before end when the number of force evaluations by the force engine is greater to nevalf_max."
    )

    #Exploration
    zseed: int = Field(
        default=0,
        description="The value of zseed is used to seed the random number generator. If the value equals 0, a new radom seed gets geenrated. The exact zseed value of each research is written in file zseed.dat, which can be useful for debugging, or re-running exact same pARTn runs."
    )

    #Initial push
    push_mode: Literal["list", "rad"] = Field(
        default="rad",
        description="Determines how the initial atomic displacement (push) is generated around the central atom "
        "of the currently explored environment:\n"
        "- **'list'**: The push is applied *only* to the central atom.\n"
        "- **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) "
        "from the central atom.",
    )

    push_dist_thr: float = Field(
        default=1.0,
        description="If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom "
        "within which all atoms receive an initial displacement.",
    )

    push_step_size: float = Field(
        default=0.4,
        description="Maximum size of a component in the initial displacement vector.",
    )

    ninit: int = Field(
        default=2,
        description="Specify the minimal number of pushes with the initial push vector.",
    )

    #Lanczos
    lanczos_min_size: int = Field(
        default=10,
        description="Enforce Lanczos to always do at least this number of iterations."
    )

    lanczos_max_size: int = Field(
        default=20,
        description="Maximum number of Lanczos iterations."
    )

    lanczos_disp: float = Field(
        default=0.0005,
        description="Scaling factor for displacement during the Lanczos algorithm",
    )

    lanczos_eval_conv_thr: float = Field(
        default=0.001,
        description="Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits."
    )

    #Eigenvector push
    eigval_thr: float = Field(
        default=-0.01,
        description="Threshold for eigenvalue, which determines when to start following the eigenvector"
    )

    eigen_step_size: float = Field(
        default=0.2,
        description="The limit to the maximum size of the displacement with eigenvector.",
    )

    nsmooth: int = Field(
        default=3,
        description="Number of smoothing steps from initial displacement to eigenvector.",
    )

    neigen: int = Field(
        default=1,
        description="Number of pushes along the eignevector before starting a perpendicular relax."
    )

    alpha_mix_cr: float = Field(
        default=0.2,
        description="This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. ",
        ge=0.0,
        le=1.0
    )

    nnewchance: int = Field(
        default=0,
        description="Number of times a research is allowed to cross a convex region (without counting the starting convex region)."
    )

    #Perpendicular relaxation
    nperp: Optional[int] = Field(default=3, description="Control the perpendicular relaxation.")
    nperp_limitation: Optional[list[int]] = Field(
        default=[ 4, 8, 12, 16, -1 ],
        description="Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL."
    )
    #Convergence
    forc_thr: float = Field(
        default=0.001,
        description="The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.",
    )

    convergence_property: Literal["maxval", "norm"] = Field(
        default="maxval",
        description="Specify how to test convergence of the forces. 'maxval': the convergence will be tested by MAXVAL( ABS( force ) ); 'norm' the convergence will be tested by NORM2( force )."
    )

    nevalf_max: int = Field(
        default=500,
        description="Stop an artn search before end when the number of force evaluations by the force engine is greater to nevalf_max"
    )

    #Final push

    push_over: float = Field(
        default=1.0,
        description="Factor that scales the displacement vector used to push the system from the saddle point towards a local energy minimum. "
        "\n"
        "$$ \\text{displacement} = \\text{push_factor} \\times v_0 \\times \\text{eigen_step_size} \\times \\text{push_over} \\times 0.8 $$"
        "\n",
    )

    #Lammps
    dmax: float = Field(
        default=6.0,
        description="dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.",
    )

#################
#Refinement part#
#################

    r_evalf_max: int = Field(
        default = 300,
        description="to stop an artn refinement before end when the number of force evaluations by the force engine is greater to nevalf_max."
    )

    #Max single refinement attempt
    r_max_attempts: int = Field(
        default=5,
        description="When adjusting the saddle energy and positions, in some rare cases partn has trouble finding the saddle point and goes back to the minium."
        "In that case, we do another attempt with a different seed."
    )

    r_delr_sad_thr: float = Field(
        default = 0.4,
        description="When a saddle point is found by pARTn, we compare artn delr_sad to this threshold to check if the system went back to the minimum. If yes, new attempt."
    )

    #Initial_push
    r_push_mode: Literal["list", "rad"] = Field(
        default="list",
        description="Determines how the initial atomic displacement (push) is generated around the central atom "
        "of the currently explored environment:\n"
        "- **'list'**: The push is applied *only* to the central atom.\n"
        "- **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) "
        "from the central atom.",
    )

    r_push_dist_thr: float = Field(
        default=1.0,
        description="If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom "
        "within which all atoms receive an initial displacement.",
    )

    r_push_step_size: float = Field(
        default=0.0001,
        description="Maximum size of a component in the initial displacement vector.",
    )

    r_ninit: int = Field(
        default=0,
        description="Refinement: Specify the minimal number of pushes with the initial push vector.",
    )

    #Lanczos
    r_lanczos_min_size: int = Field(
        default=20,
        description="Refinement: Enforce Lanczos to always do at least this number of iterations."
    )

    r_lanczos_max_size: int = Field(
        default=50,
        description="Refinement: Maximum number of Lanczos iterations."
    )

    r_lanczos_disp: float = Field(
        default=0.0005,
        description="Refinement: Scaling factor for displacement during the Lanczos algorithm",
    )

    r_lanczos_eval_conv_thr: float = Field(
        default=0.001,
        description="Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits."
    )

    #Eigenvector push
    r_eigval_thr: float = Field(
        default=-0.01,
        description="Refinement: threshold for eigenvalue, which determines when to start following the eigenvector"
    )

    r_eigen_step_size: float = Field(
        default=0.005,
        description="Refinement: The limit to the maximum size of the displacement with eigenvector.",
    )

    r_nsmooth: int = Field(
        default=0,
        description="Refinement: Number of smoothing steps from initial displacement to eigenvector.",
    )

    r_neigen: int = Field(
        default=1,
        description="Refinement: Number of pushes along the eignevector before starting a perpendicular relax."
    )

    r_alpha_mix_cr: float = Field(
        default=0.2,
        description="Refinement: This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. ",
        ge=0.0,
        le=1.0
    )

    r_nnewchance: int = Field(
        default=0,
        description="Refinement: Number of times a research is allowed to cross a convex region (without counting the starting convex region)."
    )


    #Perpendicular relaxation
    r_nperp: Optional[int] = Field(default=3, description="Refinement: Control the perpendicular relaxation.")
    r_nperp_limitation: Optional[list[int]] = Field(
        default=[100],
        description="Refinement: Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL."
    )


    #Convergence
    r_forc_thr: float = Field(
        default=0.001,
        description="Refinement: The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.",
    )

    #Lammps
    r_dmax: float = Field(
        default=1.0,
        description="Refinement: dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.",
    )


    #To deal with nperp None if only using nperp_limitation :
    @field_validator("nperp", "r_nperp", mode="before")
    @classmethod
    def parse_optional_int(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() == "none"):
            return None
        return v

    #To deal with list
    @field_validator("nperp_limitation", "r_nperp_limitation", mode="before")
    @classmethod
    def parse_list_of_ints(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() == "none"):
            return None
        if isinstance(v, str):
            v = v.strip("[]")
            try:
                return [int(x.strip()) for x in v.split(",") if x.strip()]
            except ValueError:
                raise ValueError(f"Invalid list of integers: {v}")
        return v

class RateConstantConfig(BaseModel):
    """Rate constant computation parameters."""

    style: Literal["constant"] = Field(
        default=...,
        description="Method used to compute the prefactor of the rate constant. ",
    )
    k0: float = Field(
        default=1.0,
        description="When `style` is set to **'constant'**, this value is used directly as the pre-exponential factor ($k_0$) "
        "\n"
        "$$ k = k_{0} \\exp\\left(-\\frac{\\Delta E}{k_{b}T}\\right) $$"
        "\n",
    )
    T: float = Field(
        default=300,
        description="Temperature (in Kelvin) used for computing rate constants.",
    )


class PSRConfig(BaseModel):
    """Point set registration parameters."""

    style: Literal["ira"] = Field(
        default=...,
        description="Method used for the point set registration (shape matching) between reference events and atomic environment of an atom having the same atomic environement ID of the event. This method is also used to find atomic environment symmetries.",
    )

    matching_score_thr: float = Field(
        default=0.1,
        description="Maximum value of the matching score of the algorithm used.",
    )

class ActiveVolume(BaseModel):
    """ Active Volume Parameters"""

    ract: float = Field(
        default=6.0,
        description="Radius of entire active volume, spherical"
    )

    rmov: float = Field(
        default=4.0,
        description="Radius of movable atoms in active volume, spherical"
    )

    AV_debug: bool = Field(
        default=False,
        description="Debug flag for active volume size checks",
    )

class LammpsConfig(BaseModel):
    """Lammps parameters."""

    pair_style: str = Field(default=..., description="Lammps pair_style command.")
    pair_coeff: str = Field(default=..., description="Lammps pair_coeff command.")
    min_style: Optional[str] = Field(
        default="cg", description="Lammps min_style command."
    )
    minimize: Optional[str] = Field(
        default="1.0e-6 1.0e-8 1000 1000",
        description="Lammps minimize command",
    )


class IraConfig(BaseModel):
    """IRA parameters."""

    kmax_factor: float = Field(
        default=1.8,
        description="Multiplicative factor that needs to be larger than 1.0. Larger value increases the search space of the rotations.",
    )
    sym_thr: float = Field(
        default=0.01,
        description="Threshold in terms of the Hausdorff distance. If an operation returns a distance value beyond sym_thr, then SOFI will not consider that operation as a symmetry operation.",
    )

class ReconstructionConfig(BaseModel):
    """Reconstruction parameters."""

    push_fraction: float = Field(
        default=0.15,
        description="Fraction used to push the system from the saddle point toward each minimum during reconstruction.",
    )

class BasinConfig(BaseModel):
    """Basin parameters"""
    style: Literal["global", "global/reconstruction"] = Field(default="global", description="Basin style used.")

    energy_thr: float = Field(
    default = 0.0,
    description="Energy threshold"
    )

class Config(BaseModel):
    """Config for the KMC simulations."""

    control: ControlConfig = Field(
        default_factory=ControlConfig, description="Core simulation control parameters."
    )

    atomicenvironment: AtomicEnvironmentConfig = Field(
        default_factory=AtomicEnvironmentConfig,
        description="Parameters defining the local atomic environments and the method used to define them.",
    )

    eventsearch: EventSearchConfig = Field(
        default_factory=EventSearchConfig,
        description="Parameter controling the event searches.",
    )

    psr: PSRConfig = Field(
        default=PSRConfig,
        description="Parameter controlling the point set registration algorithm.",
    )
    rateconstant: RateConstantConfig = Field(
        default_factory=RateConstantConfig,
        description="Parameters used to compute rate constants.",
    )

    lammps: Optional[LammpsConfig] = Field(
        default=None,
        description="LAMMPS-specific parameters. Required if engine == lammps.",
    )

    partn: Optional[PartnConfig] = Field(
        default=None, description="pARTn parameters controling the event searches"
    )

    ira: Optional[IraConfig] = Field(default=None, description="IRA parameters.")

    basin: Optional[BasinConfig] = Field(default=None, description="Basin parameters")

    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig, description="Reconstruction parameters")

    activevolume: Optional[ActiveVolume] = Field(default=None, description="Active volume parameters")

    @classmethod
    def from_ini_file(cls, ini_path: str) -> Config:
        """Load and validates simulation configuration from an INI file.

        Parses the INI file, ensuring all mandatory sections (e.g., `control`,
        `atomicenvironment`, etc.) are present. It then validates the parameters
        against the Pydantic `Config` model, providing clear error messages for
        missing or invalid entries.

        Parameters
        ----------
        ini_path : str
            The file path to the INI configuration file.

        Returns
        -------
        Config
            A validated `Config` instance containing all simulation parameters.

        Raises
        ------
        ValueError
            If the specified INI file does not exist or cannot be read.
        ValueError
        If the INI file has parsing errors, a mandatory section is missing,
        or Pydantic validation fails.

        """
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(ini_path)

        config_dict: dict[str, dict[str, Any]] = {
            section.lower(): dict(parser.items(section))
            for section in parser.sections()
        }

        # Check sections required
        required_sections = [
            "control",
            "atomicenvironment",
            "eventsearch",
            "rateconstant",
            "psr",
        ]
        for sec in required_sections:
            if sec not in config_dict:
                raise ValueError(f"Section [{sec}] mandatory in the INI file.")

        try:
            return cls.model_validate(config_dict)
        except ValidationError as e:
            user_msg = format_pydantic_errors(e)
            raise ValueError(
                f"Error while reading configuration file :\n{user_msg}"
            ) from None




    @model_validator(mode="after")
    def validate_dependencies(self) -> Config:
        """Validate conditional dependencies between configuration sections or fields.

        Ensures that when a specific configuration option is chosen (e.g., a certain
        'engine' or 'style'), its dependent sections or parameters are also provided.

        Validation rules are defined as a dictionary:
            - Key: A tuple `(condition_field_path, condition_value)`
                - `condition_field_path` (str): Dot-separated path to the field that triggers the dependency (e.g., "control.engine").
                - `condition_value` (Any): The specific value of the `condition_field` that activates the dependency.
            - Value: A list of `required_field_paths` (str)
                - `required_field_paths`: Dot-separated paths to the fields or sections that become mandatory (e.g., "lammps").

        Returns
        -------
        Self
            The validated `Config` instance.

        Raises
        ------
        ValueError
            If a required section or parameter is missing based on the validation rules.

        """
        validation_rules = {
            ("control.engine", "lammps"): ["lammps"],
            ("eventsearch.style", "partn"): ["partn"],
            ("psr.style", "ira"): ["ira"],
            ("control.basin", True) : ["basin"],
            ("control.active_volume", True) : ["activevolume"]
        }

        for (field_path, condition_value), required_fields in validation_rules.items():
            actual_value = get_nested_attr(self, field_path)
            if actual_value is not None:  # check if it required
                if actual_value == condition_value:
                    missing_fields = [
                        f for f in required_fields if get_nested_attr(self, f) is None
                    ]
                    if missing_fields:
                        raise ValueError(
                            "The following section or/and parameters are required when {} is {} : {}".format(
                                field_path, condition_value, missing_fields
                            )
                        )
        return self


def get_nested_attr(obj: BaseModel, attr_path: str) -> Optional[Any]:
    """Retrieve a nested attribute from a Pydantic BaseModel.

    Parameters
    ----------
    obj : BaseModel
        The Pydantic BaseModel instance from which to retrieve the attribute.
    attr_path : str
        The dot-separated path to the nested attribute.

    Returns
    -------
    Optional[Any]
        The value of the nested attribute, or None if the attribute
        (or any part of its path) does not exist or is None.

    """
    for part in attr_path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def format_pydantic_errors(e: ValidationError) -> str:
    """Format Pydantic errors to be more readable.

    Parameters
    ----------
    e : ValidationError
        The pydantic error.

    Returns
    -------
    str
        Formatted error message.

    """
    messages = []
    for err in e.errors():
        # Find error informations
        location = ".".join(str(loc_part) for loc_part in err["loc"])
        msg = err.get("msg", "")
        typ = err.get("type", "")
        if "missing" in typ:
            # find section and field in error message
            parts = location.split(".")
            section = parts[0]
            field = parts[1]
            messages.append(f"Section '{section}' : field '{field}' is mandatory.")
        else:
            # brut error
            messages.append(f"{location} : {msg}")
    return "\n".join(messages)

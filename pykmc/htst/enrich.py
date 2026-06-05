r"""Offline HTST enricher: add Vineyard ν₀ to a stored reference table.

Loads a ``reference_table.pickle``, computes the per-event Vineyard prefactor
ν₀ (Hz) by treating each event's stored neighbour subset as a frozen-boundary
cluster, and writes an enriched pickle (with a ``nu0`` column) plus an optional
CSV report. Runs on a serial in-memory LAMMPS engine -- no MPI required.

This is the offline counterpart to the runtime ``style = htst`` path: both
share the engine-agnostic math in :mod:`pykmc.htst.prefactor`. Use it to
inspect / curate per-event ν₀ before committing to a live HTST run.

Example:
-------
::

    python -m pykmc.htst.enrich \\
        --reference-table reference_table.pickle \\
        --potential NiAlH_jea.eam --out reference_table_htst.pickle \\
        --report nu0_report.csv

v1 supports single-element systems (``--elements Ni``); alloy enrichment needs
per-atom species and is deferred.

"""

from __future__ import annotations

import argparse
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .constants import hz_to_thz, thz_to_hz
from .prefactor import compute_event_prefactors

ForcesFn = Callable[[np.ndarray], np.ndarray]
EngineFactory = Callable[[np.ndarray], "tuple[ForcesFn, np.ndarray]"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the offline enricher CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m pykmc.htst.enrich",
        description="Add Vineyard nu0 to a stored pyKMC reference table (offline).",
    )
    parser.add_argument(
        "--reference-table", required=True, help="Input reference_table.pickle path."
    )
    parser.add_argument("--out", required=True, help="Output enriched pickle path.")
    parser.add_argument(
        "--potential", required=True, help="LAMMPS potential file (e.g. NiAlH_jea.eam)."
    )
    parser.add_argument(
        "--pair-style",
        default="eam/alloy",
        help="LAMMPS pair_style (default eam/alloy).",
    )
    parser.add_argument(
        "--elements", default="Ni", help="Space-separated element symbols (type order)."
    )
    parser.add_argument(
        "--free-radius", type=float, default=6.0, help="Free-region radius (Angstrom)."
    )
    parser.add_argument(
        "--fd-step", type=float, default=0.01, help="Finite-difference step (Angstrom)."
    )
    parser.add_argument(
        "--nu0-min-thz", type=float, default=1.0, help="Lower nu0 bound (THz)."
    )
    parser.add_argument(
        "--nu0-max-thz", type=float, default=100.0, help="Upper nu0 bound (THz)."
    )
    parser.add_argument("--report", default=None, help="Optional CSV report path.")
    return parser


def enrich_dataframe(
    df: pd.DataFrame,
    engine_factory: EngineFactory,
    elements: list[str],
    free_radius: float,
    fd_step: float,
    nu0_min_hz: float,
    nu0_max_hz: float,
    require_one_negative_mode: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``nu0`` column (forward ν₀ in Hz).

    Each row's stored ``initial_positions`` / ``saddle_positions`` /
    ``final_positions`` are treated as a frozen-boundary cluster; the forward
    Vineyard prefactor is computed via :func:`pykmc.htst.prefactor`. Rows whose
    ν₀ cannot be computed (no clean saddle, out of bounds, engine error) get
    ``NaN`` -- the same graceful fallback the runtime path logs.

    Parameters
    ----------
    df : pd.DataFrame
        Reference table with position/move-atom columns.
    engine_factory : Callable
        Maps (N, 3) positions to ``(forces_fn, cell)``; ``forces_fn`` returns
        (N, 3) forces for given positions.
    elements : list[str]
        Element symbols; v1 supports a single element (all atoms that species).
    free_radius, fd_step : float
        Hessian free-region radius and finite-difference step (Angstrom).
    nu0_min_hz, nu0_max_hz : float
        Acceptance window (Hz).
    require_one_negative_mode : bool
        Forwarded to the orchestrator (reserved; saddle check always on in v1).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added float ``nu0`` column (Hz; NaN on fallback).

    """
    if len(elements) != 1:
        raise NotImplementedError(
            "Offline enrichment v1 supports a single element; "
            f"got elements={elements}. Alloy enrichment needs per-atom species."
        )
    pbc = np.array([True, True, True])
    nu0_values: list[float] = []
    for _, row in df.iterrows():
        init = np.asarray(row["initial_positions"], dtype=float)
        sad = np.asarray(row["saddle_positions"], dtype=float)
        fin = np.asarray(row["final_positions"], dtype=float)
        n_atoms = init.shape[0]
        forces_fn, cell = engine_factory(sad)
        res = compute_event_prefactors(
            forces_fn=forces_fn,
            min1=init,
            saddle=sad,
            min2=fin,
            types=[elements[0]] * n_atoms,
            central_index=int(row["move_atom_idx"]),
            free_radius=free_radius,
            fd_step=fd_step,
            cell=cell,
            pbc=pbc,
            nu0_min_hz=nu0_min_hz,
            nu0_max_hz=nu0_max_hz,
            require_one_negative_mode=require_one_negative_mode,
        )
        nu0_values.append(res.nu0_forward if res.nu0_forward is not None else np.nan)
    out = df.copy()
    out["nu0"] = nu0_values
    return out


def lammps_forces_factory(
    potential: str, pair_style: str, elements: list[str]
) -> EngineFactory:
    """Build an engine factory that yields a serial-LAMMPS forces callable.

    Each call creates a fresh serial in-memory LAMMPS engine holding the given
    positions as a non-periodic cluster, and returns ``(forces_fn, cell)``.
    Imports of ``lammps`` / engine ops are deferred so the parser and the pure
    :func:`enrich_dataframe` stay importable without a LAMMPS build.
    """
    if len(elements) != 1:
        raise NotImplementedError(
            f"Offline enrichment v1 supports a single element; got elements={elements}."
        )
    from ase.data import atomic_masses, atomic_numbers

    from lammps import lammps

    from ..enginemanager.lmpi import lammps_operations as ops

    mass = float(atomic_masses[atomic_numbers[elements[0]]])
    pair_coeff = f"* * {potential} {' '.join(elements)}"

    class _SerialEngine:
        def __init__(self, lmp: object) -> None:
            self.lmp = lmp
            self.rank = 0

        def command(self, cmd: str) -> None:
            self.lmp.command(cmd)

    def factory(positions: np.ndarray) -> "tuple[ForcesFn, np.ndarray]":
        lo = positions.min(axis=0) - 15.0
        hi = positions.max(axis=0) + 15.0
        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("atom_modify map array")
        lmp.command("boundary f f f")
        bounds = " ".join(
            f"{v:.3f}" for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        )
        lmp.command(f"region box block {bounds} units box")
        lmp.command("create_box 1 box")
        n_atoms = positions.shape[0]
        lmp.create_atoms(
            n_atoms, None, [1] * n_atoms, positions.astype(float).reshape(-1).tolist()
        )
        lmp.command(f"mass 1 {mass}")
        lmp.command(f"pair_style {pair_style}")
        lmp.command(f"pair_coeff {pair_coeff}")
        lmp.command("run 0")
        engine = _SerialEngine(lmp)

        def forces_fn(pos: np.ndarray) -> np.ndarray:
            return ops.get_forces(engine, pos)

        return forces_fn, np.diag(hi - lo)

    return factory


def main(argv: Optional[list[str]] = None) -> None:
    """Run the offline enricher CLI."""
    args = build_parser().parse_args(argv)
    elements = args.elements.split()
    df = pd.read_pickle(args.reference_table)
    factory = lammps_forces_factory(args.potential, args.pair_style, elements)
    out = enrich_dataframe(
        df,
        engine_factory=factory,
        elements=elements,
        free_radius=args.free_radius,
        fd_step=args.fd_step,
        nu0_min_hz=thz_to_hz(args.nu0_min_thz),
        nu0_max_hz=thz_to_hz(args.nu0_max_thz),
    )
    out.to_pickle(args.out)

    nu0 = pd.to_numeric(out["nu0"], errors="coerce")
    n_ok = int(nu0.notna().sum())
    print(f"[htst-enrich] {n_ok}/{len(out)} events got a finite nu0")
    for i in range(len(out)):
        val = nu0.iloc[i]
        thz = f"{hz_to_thz(float(val)):.3f} THz" if pd.notna(val) else "k0 fallback"
        print(f"  event {i:>3}: nu0 = {thz}")
    if args.report:
        report = pd.DataFrame(
            {
                "event": np.arange(len(out)),
                "energy_barrier_eV": pd.to_numeric(
                    out.get("energy_barrier", pd.Series([np.nan] * len(out))),
                    errors="coerce",
                ),
                "nu0_Hz": nu0,
                "nu0_THz": nu0 * 1.0e-12,
            }
        )
        report.to_csv(args.report, index=False)
        print(f"[htst-enrich] wrote report -> {args.report}")


if __name__ == "__main__":
    main()

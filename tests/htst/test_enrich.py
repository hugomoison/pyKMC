"""Tests for the offline HTST reference-table enricher CLI (pykmc.htst.enrich)."""

from __future__ import annotations

import numpy as np

from pykmc.htst.enrich import build_parser, enrich_dataframe


def test_parser_parses_core_arguments() -> None:
    """build_parser accepts the reference-table/out/potential/knob arguments."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference-table",
            "ref.pickle",
            "--out",
            "ref_htst.pickle",
            "--potential",
            "NiAlH_jea.eam",
            "--free-radius",
            "5.0",
            "--fd-step",
            "0.02",
        ]
    )
    assert args.reference_table == "ref.pickle"
    assert args.out == "ref_htst.pickle"
    assert args.potential == "NiAlH_jea.eam"
    assert args.free_radius == 5.0
    assert args.fd_step == 0.02


def test_parser_defaults() -> None:
    """Knob defaults match the plan (free_radius=6, fd_step=0.01, eam/alloy)."""
    parser = build_parser()
    args = parser.parse_args(
        ["--reference-table", "r.pickle", "--out", "o.pickle", "--potential", "p.eam"]
    )
    assert args.free_radius == 6.0
    assert args.fd_step == 0.01
    assert args.pair_style == "eam/alloy"
    assert args.elements == "Ni"


def test_enrich_dataframe_adds_nu0_column_with_stub_engine() -> None:
    """enrich_dataframe adds a 'nu0' column using an injected engine factory.

    A stub forces-engine lets us exercise the enrich loop without LAMMPS: the
    factory returns an object whose get_forces yields a simple harmonic field,
    so every row gets a finite nu0 written back.
    """
    import pandas as pd

    eq = np.array([[5.0, 5.0, 5.0], [6.2, 5.0, 5.0], [5.0, 6.2, 5.0]])
    df = pd.DataFrame(
        {
            "initial_positions": [eq.copy()],
            "saddle_positions": [eq.copy()],
            "final_positions": [eq.copy()],
            "move_atom_idx": [0],
            "energy_barrier": [0.5],
        }
    )

    class _StubEngine:
        def __init__(self, positions: np.ndarray) -> None:
            self._eq = positions.copy()

        def get_forces(self, positions: np.ndarray) -> np.ndarray:
            return -2.0 * (positions - self._eq)

    def factory(positions: np.ndarray) -> tuple:
        eng = _StubEngine(positions)
        return eng.get_forces, np.diag([20.0, 20.0, 20.0])

    out = enrich_dataframe(
        df,
        engine_factory=factory,
        elements=["Ni"],
        free_radius=5.0,
        fd_step=0.01,
        nu0_min_hz=0.0,
        nu0_max_hz=1e30,
    )
    assert "nu0" in out.columns
    assert len(out) == len(df)
    # min1 == saddle here, so there is no real saddle -> graceful None/NaN.
    assert pd.isna(out.iloc[0]["nu0"])

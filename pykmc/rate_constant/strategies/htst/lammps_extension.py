"""LAMMPS engine extension computing HTST prefactors.

Every Hessian is computed with the `dynamical_matrix` command (PHONON package) in
a serial LAMMPS instance of its own, built on the session leader for one event
and closed afterwards. The session engine is never read or changed.
"""

import os
import re
import tempfile
from typing import Any, Optional

import numpy as np
from ase.cell import Cell
from mpi4py import MPI

from pykmc._core.constants import ESKM_METAL_CONVERSION
from pykmc._core.result import Err, ErrorInfo, Ok, Result
from pykmc.engine.base import Engine, EngineExtension
from pykmc.engine.lammps import LammpsEngine

from .modes import Spectrum, minimum_error, saddle_error, spectrum, vineyard_thz
from .region import _minimum_image
from ...result import PrefactorError


def _engine_failure(message: str) -> Err:
    return Err(ErrorInfo(type=PrefactorError.ENGINE_FAILURE, message=message))


def _atoms_lost(got: int, want: int) -> Err:
    return _engine_failure(
        f"scratch box holds {got} of {want} atoms "
        "(atoms outside a non-periodic boundary)"
    )


def _remove(path: Optional[str]) -> None:
    if path is not None:
        try:
            # LAMMPS keeps the file open when dynamical_matrix fails: empty it, so
            # that its disk space is freed even then.
            os.truncate(path, 0)
        except OSError:
            pass
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class HTSTLammpsExtension(EngineExtension):
    """HTST operations of a `LammpsEngine`.

    Parameters
    ----------
    engine : Engine
        The engine to attach the extension to. Only its configuration, rank and
        id are used.

    Notes
    -----
    Both operations run on the leader of the engine's communicator and return
    `None` on its other ranks.

    """

    def __init__(self, engine: Engine) -> None:
        super().__init__(engine)
        self._jobs = 0

    def compute_htst_prefactors(
        self,
        config: Any,
        saddle: np.ndarray,
        minima: list[np.ndarray],
        types: list[str],
        species: list[str],
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
        displaced: np.ndarray,
        n_zero_modes: int,
    ) -> Optional[list[Result[float, ErrorInfo]]]:
        """Compute the Vineyard prefactor from each minimum to the saddle.

        Parameters
        ----------
        config : Config
            The simulation configuration (`rateconstant.fd_step` and
            `rateconstant.hessian_dir` are used).
        saddle : np.ndarray
            Saddle positions of the present atoms, shape (n_present, 3).
        minima : list[np.ndarray]
            Positions of the present atoms at each minimum, same shape as `saddle`.
        types : list[str]
            Chemical symbol of each present atom.
        species : list[str]
            Every species of the simulation, in the order of the LAMMPS types.
        cell : np.ndarray
            Simulation cell, shape (3, 3).
        pbc : tuple[bool, bool, bool]
            Periodic boundary conditions of the scratch box.
        displaced : np.ndarray
            Positions within the present atoms of the atoms the Hessian displaces.
            The other present atoms stay fixed.
        n_zero_modes : int
            3 when the displaced atoms are the whole system, else 0.

        Returns
        -------
        list[Result[float, ErrorInfo]] or None
            One entry per minimum: `Ok` with nu0 (THz), or `Err` with the reason
            there is none. `None` on ranks other than the leader.

        Notes
        -----
        It never raises: a failure of the scratch engine is returned as
        `PrefactorError.ENGINE_FAILURE` for every entry.

        """
        if self.engine.rank != 0:
            return None
        self._jobs += 1
        n = len(minima)
        try:
            n_atoms = len(types)
            if n_zero_modes == 0 and len(displaced) == n_atoms:
                return [
                    Err(
                        ErrorInfo(
                            type=PrefactorError.NO_FROZEN_BOUNDARY,
                            message="every present atom is displaced but the region "
                            "is not the whole system",
                        )
                    )
                ] * n
            scratch = self._scratch()
            try:
                scratch.start()
                scratch.initialize_parameters()
                scratch.initialize_system(
                    types, saddle, Cell(cell), pbc, species=species
                )
                # create_atoms skips the atoms outside a non-periodic boundary
                got = scratch.lmp.get_natoms()
                if got != n_atoms:
                    return [_atoms_lost(got, n_atoms)] * n
                scratch.initialize_potential()

                if len(displaced) == n_atoms:
                    group = "all"
                else:
                    group = "htst_displaced"
                    ids = " ".join(str(i + 1) for i in displaced)
                    scratch.command(f"group {group} id {ids}")
                sqrt_masses = None
                if n_zero_modes == 3:
                    # masses as LAMMPS uses them (the potential may set them)
                    mass = scratch.lmp.numpy.extract_atom("mass")
                    sqrt_masses = np.sqrt(
                        [mass[species.index(atom_type) + 1] for atom_type in types]
                    )[displaced]

                def spectrum_of(positions: np.ndarray, label: str) -> Result:
                    return self._spectrum_of(
                        scratch,
                        config,
                        positions,
                        label,
                        group,
                        len(displaced),
                        n_zero_modes,
                        sqrt_masses,
                        np.asarray(cell),
                        pbc,
                    )

                saddle_spectrum = spectrum_of(saddle, "saddle")
                if isinstance(saddle_spectrum, Err):
                    return [saddle_spectrum] * n
                error = saddle_error(saddle_spectrum.ok_value())
                if error is not None:
                    return [Err(error)] * n

                results: list[Result[float, ErrorInfo]] = []
                for i, minimum in enumerate(minima):
                    minimum_spectrum = spectrum_of(minimum, f"min{i + 1}")
                    if isinstance(minimum_spectrum, Err):
                        failure = minimum_spectrum.err_value().type
                        if failure == PrefactorError.ENGINE_FAILURE:
                            # the scratch engine is unusable for the next minima
                            return results + [minimum_spectrum] * (n - i)
                        results.append(minimum_spectrum)
                        continue
                    error = minimum_error(minimum_spectrum.ok_value())
                    if error is not None:
                        results.append(Err(error))
                        continue
                    results.append(
                        vineyard_thz(
                            minimum_spectrum.ok_value(), saddle_spectrum.ok_value()
                        )
                    )
                return results
            finally:
                scratch.close()
        except Exception as exc:
            return [
                _engine_failure(f"scratch LAMMPS failed: {type(exc).__name__}: {exc}")
            ] * n

    def htst_capability_check(
        self, config: Any, species: list[str]
    ) -> Optional[dict[str, Any]]:
        """Check that this engine can compute Hessians, before the simulation.

        It checks, in this order, that the Hessian scratch directory can be
        written (it is created if missing), that LAMMPS has the PHONON package,
        and that the potential gives a finite `dynamical_matrix` on a box with one
        atom per species. It then reads the force cutoff of the potential.

        Parameters
        ----------
        config : Config
            The simulation configuration (`rateconstant.fd_step` and
            `rateconstant.hessian_dir` are used).
        species : list[str]
            Every species of the simulation, in the order of the LAMMPS types.

        Returns
        -------
        dict or None
            `force_cutoff` (A, `None` if it could not be read), `lammps_version`
            and `hessian_dir` (absolute path). `None` on ranks other than the
            leader.

        Raises
        ------
        RuntimeError
            If one of the checks fails. The message says which and what to do.

        """
        if self.engine.rank != 0:
            return None
        rc = config.rateconstant
        hessian_dir = os.path.abspath(rc.hessian_dir or tempfile.gettempdir())
        prefix = f"pykmc_htst_{os.getpid()}_{self.engine.engine_id}_0_"
        # First: LAMMPS writes nothing, without an error, to a path it cannot open.
        try:
            os.makedirs(hessian_dir, exist_ok=True)
            fd, path = tempfile.mkstemp(
                dir=hessian_dir, prefix=prefix + "probe_", suffix=".bin"
            )
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(b"\0")
            finally:
                os.remove(path)
        except OSError as exc:
            raise RuntimeError(
                f"HTST: hessian_dir '{hessian_dir}' cannot be used: {exc}. "
                "Give a writable directory."
            ) from exc

        probe_path = cutoff_path = None
        scratch = self._scratch()
        try:
            scratch.start()
            if not scratch.lmp.has_package("PHONON"):
                raise RuntimeError(
                    "HTST: this LAMMPS build has no PHONON package (needed for "
                    "dynamical_matrix). Rebuild LAMMPS with -DPKG_PHONON=on or set "
                    "[RateConstant] style = constant."
                )
            fd, probe_path = tempfile.mkstemp(
                dir=hessian_dir, prefix=prefix + "probe_", suffix=".bin"
            )
            os.close(fd)
            # one atom per species, 3 A apart, in a 40 A periodic cube
            positions = np.full((len(species), 3), 3.0)
            positions[:, 0] += 3.0 * np.arange(len(species))
            try:
                scratch.initialize_parameters()
                scratch.initialize_system(
                    species,
                    positions,
                    Cell(40.0 * np.eye(3)),
                    (True, True, True),
                    species=species,
                )
                scratch.initialize_potential()
                scratch.command("run 0 post no")
                scratch.command("group htst_probe id 1")
                scratch.command(
                    f"dynamical_matrix htst_probe eskm {rc.fd_step!r} "
                    f'file "{probe_path}" binary yes'
                )
            except Exception as exc:
                raise RuntimeError(
                    self._bad_potential(
                        f"LAMMPS error in a one-atom-per-species box: {exc}"
                    )
                ) from exc
            values = np.fromfile(probe_path, dtype=np.float64)
            if values.size < 9:
                raise RuntimeError(
                    self._bad_potential(
                        f"the dynamical_matrix probe returned {values.size} values "
                        "(expected 9)"
                    )
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    self._bad_potential(
                        "the dynamical_matrix probe returned non-finite values"
                    )
                )
            fd, cutoff_path = tempfile.mkstemp(
                dir=hessian_dir, prefix=prefix + "cutoff_", suffix=".txt"
            )
            os.close(fd)
            return {
                "force_cutoff": self._force_cutoff(scratch, cutoff_path),
                "lammps_version": int(scratch.lmp.version()),
                "hessian_dir": hessian_dir,
            }
        finally:
            try:
                _remove(probe_path)
                _remove(cutoff_path)
            finally:
                scratch.close()

    def _scratch(self) -> LammpsEngine:
        """Return a serial engine with the potential of this engine, not started."""
        return LammpsEngine(
            config=self.engine.config,
            comm=MPI.COMM_SELF,
            engine_id=f"{self.engine.engine_id}-htst",
        )

    @staticmethod
    def _bad_potential(reason: str) -> str:
        return (
            f"HTST: the potential cannot be used for Hessians: {reason}. Check "
            "[Lammps] pair_style and pair_coeff or set [RateConstant] style = "
            "constant."
        )

    @staticmethod
    def _force_cutoff(scratch: LammpsEngine, path: str) -> Optional[float]:
        """Read the largest force cutoff of the potential, or `None`.

        With a zero neighbor skin, the communication cutoff LAMMPS reports is the
        largest pair cutoff.
        """
        try:
            scratch.command("neighbor 0.0 bin")
            scratch.command("run 0 post no")
            scratch.command(f'info communication out overwrite "{path}"')
            with open(path) as file:
                match = re.search(r"Communication cutoff = (\S+)", file.read())
            return float(match.group(1))
        except Exception:
            return None

    def _dynmat(
        self,
        scratch: LammpsEngine,
        config: Any,
        positions: np.ndarray,
        label: str,
        group: str,
        n_displaced: int,
    ) -> Result[np.ndarray, ErrorInfo]:
        """Compute the mass-weighted Hessian of `group` at `positions`.

        Returns it in eV / (amu A^2), shape (3 n_displaced, 3 n_displaced), rows
        in ascending atom order. The scratch file is removed in every case.
        """
        n_atoms = len(positions)
        scratch.set_positions(positions)
        rc = config.rateconstant
        fd, path = tempfile.mkstemp(
            dir=os.path.abspath(rc.hessian_dir or tempfile.gettempdir()),
            prefix=(
                f"pykmc_htst_{os.getpid()}_{self.engine.engine_id}_{self._jobs}_"
                f"{label}_"
            ),
            suffix=".bin",
        )
        os.close(fd)
        try:
            scratch.command(
                f'dynamical_matrix {group} eskm {rc.fd_step!r} file "{path}" binary yes'
            )
            # The command drops the atoms outside a non-periodic boundary and
            # lowers nlocal only; the file keeps its size if a fixed atom is lost.
            got = scratch.lmp.extract_setting("nlocal")
            if got != n_atoms:
                return _atoms_lost(got, n_atoms)
            h = np.fromfile(path, dtype=np.float64)
        finally:
            _remove(path)
        size = 3 * n_displaced
        if h.size != size * size:
            return _engine_failure(
                f"Hessian file holds {h.size} values, expected {size * size}"
            )
        h = h.reshape(size, size)
        h /= ESKM_METAL_CONVERSION
        h += h.T
        h *= 0.5
        return Ok(h)

    def _spectrum_of(
        self,
        scratch: LammpsEngine,
        config: Any,
        positions: np.ndarray,
        label: str,
        group: str,
        n_displaced: int,
        n_zero_modes: int,
        sqrt_masses: Optional[np.ndarray],
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
    ) -> Result[Spectrum, ErrorInfo]:
        """Compute the mode spectrum of `group` at `positions`."""
        hessian = self._dynmat(scratch, config, positions, label, group, n_displaced)
        if isinstance(hessian, Err):
            return hessian
        body = None
        if n_zero_modes == 3:
            # every atom is displaced: the atoms as one body, around the first one
            body = _minimum_image(positions - positions[0], cell, pbc)
        return spectrum(hessian.ok_value(), n_zero_modes, sqrt_masses, body)

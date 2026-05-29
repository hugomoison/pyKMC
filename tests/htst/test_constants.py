"""Round-trip test: eigenvalue -> hbar*omega [eV] -> linear frequency (Hz/THz)."""

import math

from pykmc.htst.constants import eigval_to_omega_eV, omega_eV_to_hz, hz_to_thz


def test_five_thz_round_trip() -> None:
    """Verify the eigenvalue → THz round-trip at 5 THz."""
    # nu[THz] = 15.634 * sqrt(lambda); choose lambda so nu = 5 THz exactly.
    lmbda = (5.0 / 15.634) ** 2
    nu_thz = hz_to_thz(omega_eV_to_hz(eigval_to_omega_eV(lmbda)))
    assert math.isclose(nu_thz, 5.0, rel_tol=1e-3)

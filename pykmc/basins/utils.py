import numpy as np
from scipy.sparse.linalg import expm
from numpy.linalg import eig, inv


def solve_master_equation(
    M: np.ndarray,
    t: float,
    p0: np.ndarray,
    spectral_decomposition: bool = True,
    Valeig: np.ndarray = None,
    Veceig: np.ndarray = None,
    Veceiginv: np.ndarray = None,
) -> np.ndarray:
    """
    Compute p(t) = exp(-M t) p0.

    If spectral_decomposition is True:
        p(t) = V diag(exp(-lambda_i t)) V^{-1} p0

    Otherwise:
        p(t) = expm(-M t) p0

    Parameters
    ----------
    M : np.ndarray
        Generator matrix of shape (n, n).
    t : float
        Time value at which to compute p(t).
    p0 : np.ndarray
        Initial probability vector of size n.
    spectral_decomposition : bool, default=True
        If True, computes exp(-M t) using the eigen decomposition of M.
        If False, falls back to scipy.linalg.expm.
    Valeig : np.ndarray, optional
        Precomputed eigenvalues of M. Used if spectral_decomposition=True.
    Veceig : np.ndarray, optional
        Precomputed eigenvectors of M. Used if spectral_decomposition=True.
    Veceiginv : np.ndarray, optional
        Precomputed inverse of eigenvectors of M. Used if spectral_decomposition=True.

    Returns
    -------
    np.ndarray
        Probability vector p(t) of size n
    """

    if spectral_decomposition:
        # Check if eigenvalues are provided :
        if Valeig is None or Veceig is None or Veceiginv is None:
            Valeig, Veceig = eig(M)
            Veceiginv = inv(Veceig)
        exp_lambdasxt = np.array([np.exp(-t * val) for val in Valeig])
        p = Veceig @ np.diag(exp_lambdasxt) @ Veceiginv @ p0
    else:
        p = expm(-M * t) @ p0
    return p


def solve_master_equation_last_value(
    M: np.ndarray,
    t: float,
    p0: np.ndarray,
    spectral_decomposition: bool = True,
    Valeig: np.ndarray = None,
    Veceig: np.ndarray = None,
    Veceiginv: np.ndarray = None,
) -> np.ndarray:
    """
    Compute only the absorbing probability p_abs(t),
    i.e. the last component of p(t).

    Optimized to avoid constructing the full vector if spectral
    decomposition is used.

    Parameters
    ----------
    M : np.ndarray
        Generator matrix of shape (n, n).
    t : float
        Time value at which to compute p(t).
    p0 : np.ndarray
        Initial probability vector of size n.
    spectral_decomposition : bool, default=True
        If True, computes exp(-M t) using the eigen decomposition of M.
        If False, falls back to scipy.linalg.expm.
    Valeig : np.ndarray, optional
        Precomputed eigenvalues of M. Used if spectral_decomposition=True.
    Veceig : np.ndarray, optional
        Precomputed eigenvectors of M. Used if spectral_decomposition=True.
    Veceiginv : np.ndarray, optional
        Precomputed inverse of eigenvectors of M. Used if spectral_decomposition=True.

    Returns
    -------
    float
        p_abs(t) = p(t)[-1]
    """
    if spectral_decomposition:
        # Check if eigenvalues are provided :
        if Valeig is None or Veceig is None or Veceiginv is None:
            Valeig, Veceig = eig(M)
            Veceiginv = inv(Veceig)
        exp_lambdasxt = np.array([np.exp(-t * val) for val in Valeig])
        p_abs = np.dot(Veceig[-1], np.diag(exp_lambdasxt) @ Veceiginv @ p0)
    else:
        p = expm(-M * t) @ p0
        p_abs = p[-1]
    return p_abs

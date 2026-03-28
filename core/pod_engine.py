"""
POD (Proper Orthogonal Decomposition) engine via truncated SVD.
"""

import time
import numpy as np
import scipy.linalg


def POD_SVD(Utx):
    """Perform POD decomposition on a snapshot matrix using SVD.

    Parameters
    ----------
    Utx : ndarray
        Snapshot matrix of shape (N, m), where N is the number of time steps
        and m is the number of spatial degrees of freedom.

    Returns
    -------
    U0x : ndarray
        Time-averaged mean field, shape (m,).
    An : ndarray
        POD temporal coefficients (scores), shape (N, N).
    PhiU : ndarray
        POD spatial modes (basis), shape (m, N).
    Ds : ndarray
        Eigenvalues (energy spectrum), shape (N,).
    S : ndarray
        Singular values, shape (N,).
    """
    N = Utx.shape[0]
    m = Utx.shape[1]

    print(f"POD input snapshot matrix shape: {Utx.shape}")

    U0x = np.mean(Utx, axis=0)
    Utx_centered = Utx - U0x * np.ones((N, m))

    print("\nRunning POD via SVD...")
    start_time = time.time()

    U, S, PhiU = scipy.linalg.svd(Utx_centered, full_matrices=False)
    An = U @ np.diag(S)
    Ds = (S ** 2) / N

    elapsed = time.time() - start_time
    print(f"POD decomposition complete. Elapsed: {elapsed:.2f} s")

    return U0x, An, PhiU.T, Ds, S

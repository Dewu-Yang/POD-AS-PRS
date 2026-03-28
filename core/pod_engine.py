
import time
import numpy as np
import scipy.linalg


def POD_SVD(Utx):
   
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

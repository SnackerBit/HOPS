"""
This file implements higher-precision svd using mpmath
TODO: This doesn't work correctly right now ...
"""
from mpmath import mp
import numpy as np

def mp_matrix_to_numpy_array(A):
    return np.array([[np.complex256(x) for x in row] for row in A.tolist()], dtype=np.complex256)

def svd_prec(A, prec=100): 
    """
    Helper function that computes a SVD with higher precision than scipy.
    TODO: Benchmark & Improve to 256 bit precision ...
    """
    mp.prec = prec
    A_mp = mp.matrix(A.tolist())
    U, S, V = mp.svd(A_mp)
    U = mp_matrix_to_numpy_array(U)
    S = mp_matrix_to_numpy_array(S)[:, 0]
    V = mp_matrix_to_numpy_array(V)
    return U, S, V

def qr_prec(A, prec=100): 
    """
    Helper function that computes a QR with higher precision than scipy.
    TODO: Benchmark & Improve to 256 bit precision ...
    """
    mp.prec = prec
    A_mp = mp.matrix(A.tolist())
    Q, R = mp.qr(A_mp)
    Q = mp_matrix_to_numpy_array(Q)
    R = mp_matrix_to_numpy_array(R)
    return Q, R

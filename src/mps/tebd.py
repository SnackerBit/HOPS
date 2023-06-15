"""
This file implements a basic version of the Time Evolving Block Decimation (TEBD) algorithm,
see https://arxiv.org/pdf/1008.3477.pdf, 
"The density-matrix renormalization group in the age of matrix product states"
"""

import numpy as np
from scipy.linalg import expm
from ..mps.mps import split_and_truncate

def calc_U_bonds(model, dt):
    """
    Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(-dt * H)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds

def run_TEBD(psi, U_bonds, N_steps, chi_max, eps):
    """Evolve the state `psi` for `N_steps` time steps with (first order) TEBD.

    The state psi is modified in place."""
    Nbonds = psi.L - 1
    assert len(U_bonds) == Nbonds
    for n in range(N_steps):
        for k in [0, 1]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
    # done

def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = i + 1
    # construct theta matrix
    theta = psi.get_theta_2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    chi_vL, chi_i, chi_j, chi_vR = Utheta.shape
    Utheta = np.reshape(Utheta, (chi_vL*chi_i, chi_j*chi_vR))
    # split and truncate
    U, S, V, _, _ = split_and_truncate(Utheta, chi_max, eps)
    # put back into MPS
    U = np.tensordot(U, np.diag(S), ([1], [0])) # (vL i) [vC]; [vC*] vC -> (vL i) vC
    U = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vR
    psi.Bs[i] = np.transpose(U, (0, 2, 1)) # vL i vR -> vL vR i
    V = np.reshape(V, (V.shape[0], chi_j, chi_vR)) # vL (j, vR) -> vL j vR
    psi.Bs[j] = np.transpose(V, (0, 2, 1)) # vL j vR -> vL vR j

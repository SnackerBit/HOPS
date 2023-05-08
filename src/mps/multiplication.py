"""
This file implements multiplication of an MPS with an MPO
"""

import numpy as np
from ..mps import mps

def multiply(psi, Ws, chi_max, eps, inplace=False, compress=True):
    """
    Multiplies the given MPS with a MPO. This happens in two steps: First, the MPO is
    applied to the MPS site by site. This increases the bond dimension drastically.
    Second, the MPS is compressed again to the maximum bond dimension. Eigenvalues
    smaller than eps are discarded.
    
    Parameters
    ----------
    psi : MPS
        the mps we want to multiply the MPO to
    Ws : list of np.ndarray
        list of psi.L arrays of shape (wL, wR, i*, i)
    chi_max : int
        maximal bond dimension
    eps : float
        lower threshhold for the absolute size of eigenvalues
    inplace : bool
        if this is set to true, the MPO is multiplied to the MPS
        in-place and nothing is returned.
    compress : bool
        wether to compress the resulting MPS after the operation.
        If this is set to false, the resulting MPS will NOT be in
        canonical form.
    
    Returns
    -------
    mps' : MPS
        Ws * mps
    """
    assert(len(Ws)==psi.L)
    Bs = [None]*psi.L
    # apply MPO to MPS
    Bs[0] = np.tensordot(psi.Bs[0], Ws[0][0, :, :, :], ([1], [1])) # vL [i] vR; wR [i*] i -> vL vR wR i
    Bs[0] = np.transpose(Bs[0], (0, 3, 1, 2)) # vL vR wR i -> vL i vR wR
    shape = Bs[0].shape
    Bs[0] = np.reshape(Bs[0], (shape[0], shape[1], shape[2]*shape[3]))
    for i in range(1, psi.L-1):
        Bs[i] = np.tensordot(psi.Bs[i], Ws[i], ([1], [2])) # vL [i] vR; wL wR [i*] i -> vL vR wL wR i
        Bs[i] = np.transpose(Bs[i], (0, 2, 4, 1, 3)) # vL vR wL wR i -> vL wL i vR wR
        shape = Bs[i].shape
        # vL wL i vR wR -> (vL wL) i (vR wR)
        Bs[i] = np.reshape(Bs[i], (shape[0]*shape[1], shape[2], shape[3]*shape[4])) 
    Bs[-1] = np.tensordot(psi.Bs[-1], Ws[-1][:, -1, :, :], ([1], [1])) # vL [i] vR; wL [i*] i -> vL vR wL i
    Bs[-1] = np.transpose(Bs[-1], (0, 2, 3, 1)) # vL vR wL i -> vL wL i vR
    shape = Bs[-1].shape
    Bs[-1] = np.reshape(Bs[-1], (shape[0]*shape[1], shape[2], shape[3]))
    
    if inplace == True:
        psi.Bs = Bs
        psi.canonical = False
        if compress:
            psi.compress(chi_max, eps)
    else:
        result = mps.MPS(Bs, [None]*psi.L)
        result.norm = psi.norm
        result.canonical = False
        if compress:
            result.compress(chi_max, eps)
        return result
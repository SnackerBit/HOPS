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
    error : float
        the truncation error
    """
    assert(len(Ws)==psi.L)
    Bs = [None]*psi.L
    # apply MPO to MPS
    Bs[0] = np.tensordot(psi.Bs[0], Ws[0][0, :, :, :], ([2], [1])) # vL vR [i]; wR [i*] i -> vL vR wR i
    shape = Bs[0].shape
    Bs[0] = np.reshape(Bs[0], (shape[0], shape[1]*shape[2], shape[3]))
    for i in range(1, psi.L-1):
        Bs[i] = np.tensordot(psi.Bs[i], Ws[i], ([2], [2])) # vL vR [i]; wL wR [i*] i -> vL vR wL wR i
        Bs[i] = np.transpose(Bs[i], (0, 2, 1, 3, 4)) # vL vR wL wR i -> vL wL vR wR i
        shape = Bs[i].shape
        # vL wL vR wR i -> (vL wL) (vR wR) i
        Bs[i] = np.reshape(Bs[i], (shape[0]*shape[1], shape[2]*shape[3], shape[4])) 
    Bs[-1] = np.tensordot(psi.Bs[-1], Ws[-1][:, -1, :, :], ([2], [1])) # vL vR [i]; wL [i*] i -> vL vR wL i
    Bs[-1] = np.transpose(Bs[-1], (0, 2, 1, 3)) # vL vR wL i -> vL wL vR i
    shape = Bs[-1].shape
    Bs[-1] = np.reshape(Bs[-1], (shape[0]*shape[1], shape[2], shape[3]))
    
    if inplace == True:
        psi.Bs = Bs
        psi.canonical = False
        if compress:
            return psi.canonicalize(chi_max, eps)
        return 0
    else:
        result = mps.MPS(Bs, [None]*psi.L, use_precise_svd=psi.use_precise_svd)
        result.norm = psi.norm
        result.canonical = False
        error = 0
        if compress:
            error = result.canonicalize(chi_max, eps)
        return result, error

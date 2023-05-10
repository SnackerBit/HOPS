"""
This file implements the addition of two MPS of possibly different bond dimension
and with OBC, see https://arxiv.org/pdf/1008.3477.pdf (section 4.3)
"""

import numpy as np
from ..mps import mps

def add(psi1, psi2, chi_max, eps, factor=1., compress=True):
    """
    Adds two MPSs. This happens in two steps: First, the MPSs are added together
    site by site. This increases the bond dimension.
    Second, the resulting MPS is compressed again to the maximum bond dimension. 
    Eigenvalues smaller than eps are discarded.
    
    Parameters
    ----------
    psi1 : MPS
        the first mps
    psi2 : MPS
        the second mps
    chi_max : int
        maximal bond dimension
    eps : float
        lower threshhold for the absolute size of eigenvalues
    factor : float
        scalar factor that the second mps is multiplied with 
        before adding the two MPS together.
    compress : bool
        wether to compress the resulting MPS after the operation.
        If this is set to false, the resulting MPS will NOT be in
        canonical form.
        
    Returns
    -------
    psi : MPS
        psi1 + factor * psi2
    error : float
        the truncation error
    """
    assert(psi1.L == psi2.L)
    
    Bs = []
    # due to the OBC, the first and last site need to be treated differently!
    B1 = psi1.Bs[0]
    B2 = psi2.Bs[0]
    Bs.append(np.zeros((1, B1.shape[1], B1.shape[2]+B2.shape[2]), dtype=complex))
    Bs[-1][0, :, :B1.shape[2]] = psi1.norm * B1[0, :, :].copy()
    Bs[-1][0, :, B1.shape[2]:] = psi2.norm * factor * B2[0, :, :].copy()
    for i in range(1, psi1.L-1):
        B1 = psi1.Bs[i]
        B2 = psi2.Bs[i]
        Bs.append(np.zeros((B1.shape[0]+B2.shape[0], B1.shape[1], B1.shape[2]+B2.shape[2]), dtype=complex))
        Bs[-1][:B1.shape[0], :, :B1.shape[2]] = B1.copy()
        Bs[-1][B1.shape[0]:, :, B1.shape[2]:] = B2.copy()
    B1 = psi1.Bs[psi1.L-1]
    B2 = psi2.Bs[psi1.L-1]
    Bs.append(np.zeros((B1.shape[0]+B2.shape[0], B1.shape[1], 1), dtype=complex))
    Bs[-1][:B1.shape[0], :, 0] = B1[:, :, 0].copy()
    Bs[-1][B1.shape[0]:, :, 0] = B2[:, :, 0].copy()
    
    # compress the result
    result = mps.MPS(Bs, [None]*psi1.L, use_precise_svd=psi1.use_precise_svd)
    result.canonical = False
    error = 0
    if compress:
        error = result.compress(chi_max, eps)
    return result, error

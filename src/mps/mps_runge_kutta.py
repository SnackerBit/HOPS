"""
This file implements the MPS Runge-Kutta algorithm from https://arxiv.org/abs/1907.12044
"""

import numpy as np
from ..mps import addition
from ..mps import multiplication

def integrate_MPS_RK4(psi, dt, update_mpo, chi_max, eps):
    """
    Performs one RK4 timestep for the differential equation
    
    \frac{d}{dt} |Psi> = update_mpo |Psi>
    
    Parameters
    ----------
    psi : MPS
        the current state of the system in MPS form
    dt : float
        time step size
    update_mpo : list of np.ndarray
        the MPO defining the update equation, see function description.
        One generally can define an effective Hamiltonian H_eff, s.t.
        update_mpo = -1.j*H_eff
    chi_max : int
        maximal bond dimension
    eps : float
        lower threshhold for the absolute size of eigenvalues
    
    Returns
    -------
    psi' : MPS
        the updated MPS
    error : float
        the sum of all truncation errors
    """
    errors = np.empty(5, dtype=float)
    k1, errors[0] = multiplication.multiply(psi, update_mpo, chi_max, eps, inplace=False, compress=True)
    temp, _ = addition.add(psi, k1, chi_max, eps, dt/2, compress=False)
    k2, errors[1] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    temp, _ = addition.add(psi, k2, chi_max, eps, dt/2, compress=False)
    k3, errors[2] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    temp, _ = addition.add(psi, k3, chi_max, eps, dt, compress=False)
    k4, errors[3] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    
    result, _ = addition.add(psi, k1, chi_max, eps, dt/6, compress=False)
    result, _ = addition.add(result, k2, chi_max, eps, dt*2/6, compress=False)
    result, _ = addition.add(result, k3, chi_max, eps, dt*2/6, compress=False)
    result, errors[4] = addition.add(result, k4, chi_max, eps, dt/6, compress=True)
    return result, np.sum(errors)

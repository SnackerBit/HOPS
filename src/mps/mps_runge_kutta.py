"""
This file implements the MPS Runge-Kutta algorithm from https://arxiv.org/abs/1907.12044
"""

import numpy as np
from ..mps import addition
from ..mps import multiplication

def integrate_MPS_RK4_with_memory(psi, memory, t_index, dt, f, chi_max, eps):
    """
    Performs one RK4 timestep, updating both the differential equation
    for the state
    
    \frac{d}{dt} |Psi> = update_mpo |Psi>
    
    and the memory
    
    \frac{d}{dt} memory = f(memory, t)
    
    Parameters
    ----------
    psi : MPS
        the current state of the system in MPS form
    memory : np.array of dtype complex
        the current memory of the system stored in a 1D array
    t_index : int
        current time index, used by the homps class
        to compute the rhs. of the update equation
    dt : float
        time step size
    f : function
        update function, defining the rhs. for both the state
        and the memory differential equation.
    chi_max : int
        maximal bond dimension
    eps : float
        lower threshhold for the absolute size of eigenvalues
    
    Returns
    -------
    psi' : MPS
        the updated MPS
    memory' : np.array of dtype complex
        the updated memory
    error : float
        the sum of all truncation errors
    """
    errors = np.empty(5, dtype=complex)
    
    update_mpo, k1_m = f(psi, memory, t_index)
    k1, errors[0] = multiplication.multiply(psi, update_mpo, chi_max, eps, inplace=False, compress=True)
    
    temp, _ = addition.add(psi, k1, chi_max, eps, dt/2, compress=False)
    update_mpo, k2_m = f(temp, memory + 0.5*dt*k1_m, t_index+1)
    k2, errors[1] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    
    temp, _ = addition.add(psi, k2, chi_max, eps, dt/2, compress=False)
    update_mpo, k3_m = f(temp, memory + 0.5*dt*k2_m, t_index+1)
    k3, errors[2] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    
    temp, _ = addition.add(psi, k3, chi_max, eps, dt, compress=False)
    update_mpo, k4_m = f(temp, memory + dt*k3_m, t_index+2)
    k4, errors[3] = multiplication.multiply(temp, update_mpo, chi_max, eps, inplace=False, compress=True)
    
    result, _ = addition.add(psi, k1, chi_max, eps, dt/6, compress=False)
    result, _ = addition.add(result, k2, chi_max, eps, dt*2/6, compress=False)
    result, _ = addition.add(result, k3, chi_max, eps, dt*2/6, compress=False)
    result, errors[4] = addition.add(result, k4, chi_max, eps, dt/6, compress=True)
    return result, memory + 1/6*(k1_m + 2*k2_m + 2*k3_m + k4_m)*dt, np.sum(errors)

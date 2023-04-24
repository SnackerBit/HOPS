import numpy as np

def debye_spectral_density(w, eta, gamma):
    """
    Computes the debye spectral density
    
    S(w) = \eta * \frac{w\gamma}{w^2 + \gamma^2}
    
    Parameters
    ----------
    w : float or np.ndarray
        value(s) for w
    eta : float
        parameter for the spectral density
    gamma : float
        parameter for the spectral density
        
    Returns
    -------
    float or np.ndarray
        the debye spectral density for the given frequence/frequencies
    """
    return eta * w * gamma / (w**2 + gamma**2)
    
def get_debye_coefficients(N_terms, T, gamma, eta):
    """
    Constructs the expansion coefficients g_j and w_j for the
    exponential expansion of the bath correlation function when using
    the debye spectral density
    
    Parameters
    ----------
    w : float or np.ndarray
        value(s) for w
    eta : float
        parameter for the spectral density
    gamma : float
        parameter for the spectral density
        
    Returns
    -------
    np.ndarray
        array containing the expansion coefficients g_j
    np.ndarray
        array containing the expansion coefficients w_j
    """
    gs = np.empty(N_terms, dtype=complex)
    ws = np.empty(N_terms, dtype=complex)
    
    gs[0] = eta * gamma / 2 * (1. / np.tan(gamma / 2 / T) - 1j)
    ws[0] = gamma
    
    for j in range(1, N_terms):
        gs[j] = - eta * 4*np.pi*T**2*gamma*(j-1) / (gamma**2 - 4*np.pi**2*T**2*(j-1)**2)
        ws[j] = 2*np.pi*(j-1)*T
    return gs, ws
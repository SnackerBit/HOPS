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

def get_debye_coefficients(N_terms, T, gamma, eta, mode='matsubara', use_alternative_expansion=False):
    """
    Constructs the expansion coefficients g_j and w_j for the
    exponential expansion of the bath correlation function when using
    the debye spectral density
    
    Parameters
    ----------
    N_terms : int
        the number of terms that are used in the expansion
    T : float
        temperature
    gamma : float
        parameter for the spectral density
    eta : float
        parameter for the spectral density
    mode : str
        which mode to use for the expansion.
        can be one of 'matsubara' and 'pade'
    use_alternative_expansion : bool
        wether to use the alternative expansion of the
        bath correlation function.
        
    Returns
    -------
    np.ndarray
        array containing the expansion coefficients g_j
    np.ndarray
        array containing the expansion coefficients w_j
    """
    gs = np.empty(N_terms, dtype=complex)
    ws = np.empty(N_terms, dtype=complex)
    
    if mode != 'matsubara' and mode != 'pade':
        print(f"[WARNING]: Unknown mode {mode}, defaulting to 'matsubara'!")
    
    if mode == 'matsubara':
        if use_alternative_expansion:
            gs[0] = -eta * gamma / 2 * (np.tan(gamma / 4 / T) + 1j)
            ws[0] = gamma
            for j in range(1, N_terms):
                gs[j] = - 8*eta*gamma*(2*j - 1)*np.pi / ((gamma / T)**2 - (2*(2*j-1)*np.pi)**2)
                ws[j] = 2*(2*j-1)*np.pi*T
        else:
            gs[0] = eta * gamma / 2 * (1. / np.tan(gamma / 2 / T) - 1j)
            ws[0] = gamma
            for j in range(1, N_terms):
                gs[j] = - eta * 4*np.pi*T**2*gamma*j / (gamma**2 - 4*np.pi**2*T**2*j**2)
                ws[j] = 2*np.pi*j*T
    else:
        if use_alternative_expansion:
            raise NotImplementedError
        else:
            beta = 1/T
            xij, etaj = PSD_Nm1_N(N_terms-1, "bose")
            gs[0] = 1/(gamma*beta) - 1.j/2
            for j in range(N_terms-1):
                gs[0] -= 2*etaj[j]*gamma*beta/(xij[j]**2 - gamma**2*beta**2)
            gs[0] *= eta*gamma
            ws[0] = gamma
            for k in range(N_terms-1):
                gs[k+1] = 2*etaj[k]*eta*gamma*xij[k]/(xij[k]**2-gamma**2*beta**2)
                ws[k+1] = xij[k]/beta
    return gs, ws


# performs the [N-1/N] PDS scheme
def PSD_Nm1_N(N, mode='fermi'):
    # compute bs
    if mode=='fermi':
        b = 2*np.arange(1, 2*N+2) - 1
    else:
        b = 2*np.arange(1, 2*N+2) + 1
    # compute xi
    lam = np.zeros((2*N, 2*N), dtype=float)
    for m in range(2*N):
        if m > 0:
            lam[m, m-1] = 1/np.sqrt(b[m]*b[m-1])
        if m < 2*N-1:
            lam[m, m+1] = 1/np.sqrt(b[m]*b[m+1])
    w, _ = np.linalg.eig(lam)
    xi = 2/np.sort(w)[N:]
    # compute sigma
    lam = lam[1:, 1:]
    sigma = []
    if lam.size > 1:
        w, _ = np.linalg.eig(lam)
        sigma = 2/np.sort(w)[N:]
    # compute eta
    etas = []
    for j in range(N):
        eta = 0.5*N*b[N]
        if (len(sigma) > 0):
            eta *= np.prod(sigma**2-xi[j]**2)
        for k in range(N):
            if k != j:
                eta /= (xi[k]**2-xi[j]**2)
        etas.append(eta)
    return xi, etas

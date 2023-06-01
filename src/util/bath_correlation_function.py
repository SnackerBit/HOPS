import scipy.integrate
import numpy as np

"""
This file implements different methods to (approximately) compute the bath correlation function (BCF)

\alpha(\tau) = \frac{1}{\pi} \int_0^{\infty} dw J(w) (coth(w*\beta/2)cos(w*\tau) - i*sin(\tau)).

with spectral density J(w)

There are three different methods available:
1.) Using the expansion of the BCF in term of exponentials. This is the fastest implementation
2.) Using scipy.integrate.quad to approximate the integral
3.) Using a sum to approximate the integral
"""

def alpha(tau, g, w):
    """
    Implements the simple expansion of the bath correlation function in terms of exponentials
    \alpha(\tau) = \sum_{j=1}^{J} g_j e^{-\omega_j\tau} for \tau >= 0 and
    \alpha(\tau) = \alpha(-\tau)^*                      for \tau < 0.
    
    Parameters
    ----------
    tau : float or np.ndarray
        the time value(s) for which the correlation function is to be evaluated
    g : np.ndarray
        array of values g_j. g_j should be of type complex.
    w : np.ndarray
        array of values \omega_j. \omega_j should be of type complex. array should be of same length as g
   
    Returns
    --------
    np.ndarray:
        array of type complex containing the values of the bath correlation function evaluated at the \tau array
    """
    assert(g.shape == w.shape)
    assert(g.ndim == 1)
    arg1 = np.multiply.outer(np.real(w), np.abs(tau))
    arg2 = np.multiply.outer(np.imag(w), tau)
    return np.sum(g[:, np.newaxis]*np.exp(-arg1 - 1j*arg2), axis=0)

def alpha_quad(tau, J, beta, w_cut, limit=1000):
    """
    Approximately computes the bath correlation function
    
    \alpha(\tau) = \frac{1}{\pi} \int_0^{\infty} dw J(w) (coth(w*\beta/2)cos(w*\tau) - i*sin(\tau))
    
    by using scipy.integrate.quad. We integrate only up to a cutoff-frequency w_cut

    Parameters
    ----------
    tau : float
        the time value for which the correlation function is to be evaluated
    J : function
        function that takes a value w of type float and returns a float. Spectral density.
    beta : float
        the inverse temperature beta = 1/T
    w_cut : float
        the cutoff frequency up to which we integrate
    limit : int
        limit of how many subintervals are used in the integration
    
    Returns
    -------
    float :
        the value of the bath correlation function
    """
    integrand_real = lambda w : J(w) * 1./np.tanh(w*beta/2)*np.cos(w*tau)
    integrand_imag = lambda w : J(w) * (-np.sin(w*tau))
    return 1/np.pi * (scipy.integrate.quad(integrand_real, 0, w_cut, limit=limit)[0] + 1j * scipy.integrate.quad(integrand_imag, 0, w_cut, limit=limit)[0])

def alpha_sum(tau, J, beta, N, dw):
    """
    Approximately computes the bath correlation function
    
    \alpha(\tau) = \frac{1}{\pi} \int_0^{\infty} dw J(w) (coth(w*\beta/2)cos(w*\tau) - i*sin(\tau)).
    
    by approximating the integral with a fixed sum 
    
    Parameters
    ----------
    tau : float
        the time value for which the correlation function is to be evaluated
    J : function
        function that takes a value w of type float and returns a float. Spectral density.
    beta : float
        the inverse temperature beta = 1/T
    N : int
        the number of terms we use
    dw : float
        difference in w for two consecutive terms in the sum
    
    Returns
    -------
    float :
        the value of the bath correlation function
    """
    integrand = lambda w : J(w) * (1./np.tanh(w*beta/2)*np.cos(w*tau) - 1j*np.sin(w*tau))
    result = (np.arange(N, dtype=float) + 0.5)*dw
    result = integrand(result)
    return np.sum(result)/np.pi*dw

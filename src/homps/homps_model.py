import numpy as np
from scipy.linalg import svd

from ..util import operators

class HOMPSModel:
    """
    This implements a Model class that holds the Matrix Product Operator (MPO)
    for HOMPS and can be used for TDVP.
    
    Attributes
    ----------
    N_bath : int
        number of bath modes
    N_trunc : int
        truncation order of each bath mode. This directly controls the dimensions of 
        the W-tensors of the MPO
    H_mpo : list of np.ndarray
        list of W-tensors making up the effective Hamiltonian in MPO form. 
        The list contains N_bath + 1 tensors. All tensors have shape (vL vR i i*)
        
                    i*
                    |
              vL---(W)---vR
                    |
                    i
        
        for the first tensor H_mpo[0], it holds dim(i) = dim(i*) = d, where d is the
        dimension of the physical Hilbert space (eg. d = 2 for spin-1/2 system).
        For all other tensors H_mpo[j], j > 0 it holds dim(i) = dim(i*) = N_trunc
    """
    
    def __init__(self, g, w, h, L, N_trunc, rescale_aux=True, alternative_realization=False, gamma_terminator=0):
        """
        Parameters
        ----------
        g : np.ndarray
            array of expansion coefficients g_j for the exponential expansion
            of the bath correlation function. Should be of length N_bath.
        w : np.ndarray
            array of frequency coefficients w_j for the exponential expansion
            of the bath correlation function. Should be of length N_bath.
        h : np.ndarray
            System Hamiltonian. Should be a square matrix with shape (d, d), where
            d is the dimension of the physical Hilbert space (eg. d=2 for spin-1/2)
        L : np.ndarray
            Coupling operator that couples to the heat bath. Should be of the same
            shape as the Hamiltonian.
        N_trunc : int
            truncation order of each bath mode. This directly controls the dimensions of 
            the W-tensors of the MPO
        N : int
            number of tensors in the MPO. It holds N = N_bath + 1
        rescale_aux : bool
            Wether or not to use the HOMPS equation for rescaled auxillary states,
            which gives better numerical stability
        alternative_realization : bool
            Wether to use the alternative realization of HOPS
        """
        self.N_bath = g.shape[0]
        assert(self.N_bath == w.shape[0])
        self.L = L
        self.N_trunc = N_trunc
        self.N = self.N_bath + 1
        self.alternative_realization = alternative_realization
        # get some useful operators
        sigma_x, sigma_z, eye = operators.generate_physical_operators()
        self.eye = eye
        N, b_dagger, b, eye_aux = operators.generate_auxiallary_operators(N_trunc, rescale_aux)
        # construct H_mpo
        self._H_mpo_template = [None]*(self.N_bath + 1)
        self.H_mpo = [None]*(self.N_bath + 1)
        self._H_mpo_template[0] = np.zeros((4, 4, 2, 2), dtype=complex)
        self._H_mpo_template[0][0, 0, :, :] = -1.j * eye
        if self.alternative_realization:
            self._H_mpo_template[0][0, 1, :, :] = -1.j * L
            self._H_mpo_template[0][0, 2, :, :] = 1.j * np.conj(L).T
            self._H_mpo_template[0][0, 3, :, :] = h - 1.j * gamma_terminator * np.conj(L).T@L
        else:
            self._H_mpo_template[0][0, 1, :, :] = 1.j * L
            self._H_mpo_template[0][0, 2, :, :] = -1.j * np.conj(L).T
            self._H_mpo_template[0][0, 3, :, :] = h
        self.H_mpo[0] = self._H_mpo_template[0].copy()
        for i in range(self.N_bath):
            self._H_mpo_template[i+1] = np.zeros((4, 4, N_trunc, N_trunc), dtype=complex)
            self._H_mpo_template[i+1][0, 0, :, :] = eye_aux
            self._H_mpo_template[i+1][1, 1, :, :] = eye_aux
            self._H_mpo_template[i+1][2, 2, :, :] = eye_aux
            self._H_mpo_template[i+1][3, 3, :, :] = eye_aux
            if rescale_aux:
                self._H_mpo_template[i+1][0, 3, :, :] = w[i]*b_dagger@b
                self._H_mpo_template[i+1][1, 3, :, :] = g[i]/np.sqrt(np.abs(g[i]))*b_dagger
                self._H_mpo_template[i+1][2, 3, :, :] = np.sqrt(np.abs(g[i]))*b
            else:
                self._H_mpo_template[i+1][0, 3, :, :] = w[i]*N
                self._H_mpo_template[i+1][1, 3, :, :] = g[i]*N@b_dagger
                self._H_mpo_template[i+1][2, 3, :, :] = b
            self.H_mpo[i+1] = self._H_mpo_template[i+1].copy()
            
    """
    Updates the MPO (linear HOMPS). Should be called before each time step
    ------------------------------------
    Parameters:
        zt: complex
            the noise z_t at the current time (including memory terms, when doing non-linear HOMPS).
    """
    def update_mpo_linear(self, zt):
        self.H_mpo = [W.copy() for W in self._H_mpo_template]
        if self.alternative_realization:
            self.H_mpo[0][0, 3, :, :] += zt * self.L
        else:
            self.H_mpo[0][0, 3, :, :] += 1.j * np.conj(zt) * self.L
                        
    """
    Updates the MPO (non-linear HOMPS). Should be called before each time step
    ------------------------------------
    Parameters:
        zt: complex
            the noise \tilde{z}_t at the current time.
        memory : complex
            current memory term
        expL : float
            the expectation value of the system operator <L^\dagger> at the current
            time.
    """
    def update_mpo_nonlinear(self, zt, memory, expL):
        self.H_mpo = [W.copy() for W in self._H_mpo_template]
        if self.alternative_realization:
            self.H_mpo[0][0, 3, :, :] += (zt + memory) * self.L
            self.H_mpo[0][0, 2, :, :] -= 1.j * expL * self.eye
        else:
            self.H_mpo[0][0, 3, :, :] += 1.j * (np.conj(zt) + memory) * self.L
            self.H_mpo[0][0, 2, :, :] += 1.j * expL * self.eye
        
    """
    Computes self.update_mpo from self.H_mpo. The update MPO is just -1.j*H_mpo.
    """
    def compute_update_mpo(self):
        N = len(self.H_mpo)
        self.update_mpo = [None] * N
        factor = np.power(-1.j, 1/N)
        for i in range(N):
            self.update_mpo[i] = factor * np.transpose(self.H_mpo[i].copy(), (0, 1, 3, 2)) # wL, wR, i, i* -> wL, wR, i*, i

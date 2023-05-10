import numpy as np

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
    
    def __init__(self, g, w, h, L, N_trunc):
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
        """
        self.N_bath = g.shape[0]
        assert(self.N_bath == w.shape[0])
        self.L = L
        self.N_trunc = N_trunc
        self.N = self.N_bath + 1
        # get some useful operators
        sigma_x, sigma_z, eye = operators.generate_physical_operators()
        self.eye = eye
        N, b_dagger, b, eye_aux = operators.generate_auxiallary_operators(N_trunc)
        # construct H_mpo
        self._H0_template = np.zeros((4, 4, 2, 2), dtype=complex)
        self._H0_template[0, 0, :, :] = -1.j * eye
        self._H0_template[0, 1, :, :] = 1.j * L
        self._H0_template[0, 2, :, :] = -1.j * np.conj(L).T
        self._H0_template[0, 3, :, :] = h
        self.H_mpo = [None]*(self.N_bath + 1)
        self.H_mpo[0] = self._H0_template.copy()
        for i in range(self.N_bath):
            self.H_mpo[i+1] = np.zeros((4, 4, N_trunc, N_trunc), dtype=complex)
            self.H_mpo[i+1][0, 0, :, :] = eye_aux
            self.H_mpo[i+1][1, 1, :, :] = eye_aux
            self.H_mpo[i+1][2, 2, :, :] = eye_aux
            self.H_mpo[i+1][3, 3, :, :] = eye_aux
            self.H_mpo[i+1][0, 3, :, :] = w[i]*N
            self.H_mpo[i+1][1, 3, :, :] = g[i]*N@b_dagger
            self.H_mpo[i+1][2, 3, :, :] = b
            
    """
    Updates the MPO (linear HOMPS). Should be called before each time step
    ------------------------------------
    Parameters:
        zt: complex
            the noise z_t^* at the current time. should already be complex conjugated.
    """
    def update_mpo_linear(self, zt):
        self.H_mpo[0] = self._H0_template.copy()
        self.H_mpo[0][0, 3, :, :] += 1.j * zt * self.L
                        
    """
    Updates the MPO (non-linear HOMPS). Should be called before each time step
    ------------------------------------
    Parameters:
        zt: complex
            the noise \tilde{z}_t^* at the current time. should already include
            memory terms and be complex conjugated.
        expL : float
            the expectation value of the system operator <L^\dagger> at the current
            time.
    """
    def update_mpo_nonlinear(self, zt, expL):
        self.update_mpo_linear(zt)
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

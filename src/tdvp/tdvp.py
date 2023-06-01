import numpy as np
from ..mps import mps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import qr
from ..util.rq import rq
import time

class TDVP_Engine:
    """
    Base class for TDVP1_Engine and TDVP2_Engine that implements basic functions used in
    both algorithms
    """
    
    def __init__(self, psi, model, dt, chi_max, eps):
        """
        Initializes the TDVP_Engine
        """
        self.psi = psi
        self.model = model
        self.dt = dt
        self.chi_max = chi_max
        self.eps = eps
        # initialize left and right environment
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        D = self.model.H_mpo[0].shape[0]
        chi = self.psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype="float")  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype="float")  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 0, -1):
            self.update_RP(i) 
            
    def update_RP(self, i):
        """
        Calculate RP right of site `i-1` from RP right of site `i`.
        """
        j = i - 1
        RP = self.RPs[i]  # vR* wR* vR
        B = self.psi.Bs[i]  # vL vR i
        Bc = B.conj()  # vL* vR* i*
        W = self.model.H_mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, ([1, 0]))  # vL [vR] i; [vR*] wR* vR -> vL i wR* vR
        RP = np.tensordot(RP, W, ([1, 2], [3, 1]))  # vL [i] [wR*] vR; wL [wR] i [i*] -> vL vR wL i
        RP = np.tensordot(RP, Bc, ([1, 3], [1, 2]))  # vL [vR] wL [i]; vL* [vR*] [i*] -> vL wL vL*
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)
    
    def update_LP(self, i):
        """
        Calculate LP left of site `i+1` from LP left of site `i`.
        """
        j = i + 1
        LP = self.LPs[i]  # vL wL vL*
        B = self.psi.Bs[i]  # vL vR i
        Bc = B.conj() # vL* vR* i*
        W = self.model.H_mpo[i] # wL wR i i*
        LP = np.tensordot(LP, B, ([2], [0])) # vL wL* [vL*]; [vL] vR i -> vL wL* vR i
        LP = np.tensordot(W, LP, ([0, 3], [1, 3]))  # [wL] wR i [i*]; vL [wL*] vR [i] -> wR i vL vR
        LP = np.tensordot(Bc, LP, ([0, 2], [2, 1]))  # [vL*] vR* [i*], wR [i] [vL] vR -> vR* wR vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)

    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """                  
        raise NotImplementedError

class TDVP2_Engine(TDVP_Engine):
    """
    Class that implements the 2-site Time Dependent Variational Principle (TDVP) algorithm
    """
    
    def __init__(self, psi, model, dt, chi_max, eps):
        super().__init__(psi, model, dt, chi_max, eps)
        
    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """    
        # sweep from left to right
        for i in range(self.psi.L - 1):
            self.update_bond(i, sweep_right=True)
        # sweep from right to left
        for i in range(self.psi.L - 2, -1, -1):
            self.update_bond(i, sweep_right=False)
                      
    def update_bond(self, i, sweep_right):
        """
        Performs a single bond update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        sweep_right : bool
            wether we are currently in a right or left sweep
        """
        j = i + 1
        # get two-site wavefunction
        theta = self.psi.get_theta_2(i) # vL, i, j, vR
        chi_vL, chi_i, chi_j, chi_vR = theta.shape
        # get effective two-site Hamiltonian
        Heff = compute_Heff_Twosite(self.LPs[i], self.RPs[j], self.model.H_mpo[i], self.model.H_mpo[j])
        theta = np.reshape(theta, [Heff.shape[0]])
        # evolve 2-site wave function forward in time
        theta = evolve(theta, Heff, self.dt/2)
        # split and truncate
        theta = np.reshape(theta, (chi_vL*chi_i, chi_j*chi_vR))
        U, S, V, norm_factor, _ = mps.split_and_truncate(theta, self.chi_max, self.eps)
        self.psi.norm *= norm_factor
        # put back into MPS
        if sweep_right:
            U = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vR -> vL i vR 
            self.psi.Bs[i] = np.transpose(U, (0, 2, 1)) # vL i vR -> vL vR i 
            V = np.tensordot(np.diag(S), V, ([1], [0])) # vC [vC*]; [vC] (j vR) -> vC j vR = vL (j vR)
            V = np.reshape(V, (V.shape[0], chi_j, chi_vR)) # vL (j vR) -> vL j vR
            self.psi.Bs[j] = np.transpose(V, (0, 2, 1)) # vL j vR -> vL vR j
        else:
            U = np.tensordot(U, np.diag(S), ([1], [0])) # (vL i) [vC*]; [vC*] vC -> (vL i) vC
            U = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vC = vL i vR
            self.psi.Bs[i] = np.transpose(U, (0, 2, 1)) # vL i vR -> vL vR i
            V = np.reshape(V, (V.shape[0], chi_j, chi_vR)) # vL (j vR) -> vL j vR
            self.psi.Bs[j] = np.transpose(V, (0, 2, 1)) # vL j vR -> vL vR j
        if sweep_right == True:
            self.update_LP(i)
            if i < self.psi.L - 2:
                # extract single-site wavefunction
                psi = self.psi.Bs[j]
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[j], self.RPs[j], self.model.H_mpo[j])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                self.psi.Bs[j] = psi
        else:
            self.update_RP(j)
            if i > 0:
                # extract single-site wavefunction
                psi = self.psi.Bs[i]
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                self.psi.Bs[i] = psi

class TDVP1_Engine(TDVP_Engine):
    
    def __init__(self, psi, model, dt, chi_max=0, eps=0, mode='qr'):
        super().__init__(psi, model, dt, chi_max, eps)
        self.mode = mode
        
    def sweep(self):
        """
        Performs one sweep left -> right -> left.
        """
        # sweep from left to right
        for i in range(self.psi.L - 1):
            self.update_site(i)
            self.update_bond(i, sweep_right=True)
        # update last site
        self.update_site(self.psi.L - 1)
        # sweep from right to left
        for i in range(self.psi.L - 1, 0, -1):
            self.update_site(i)
            self.update_bond(i, sweep_right=False)
        # update first site
        self.update_site(0)
       
    def update_site(self, i):
        """
        Performs a single site update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        """
        # extract single-site wavefunction
        psi = self.psi.Bs[i]
        psi_shape = psi.shape
        psi = psi.flatten()
        # compute effective one-site Hamiltonian
        Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
        # evolve 1-site wave function forwards in time
        psi = evolve(psi, Heff, self.dt/2)
        psi = np.reshape(psi, psi_shape)
        # put back into MPS
        self.psi.Bs[i] = psi
            
    def update_bond(self, i, sweep_right):
        """
        Performs a single bond update at the given bond
        
        Parameters
        ----------
        i : int
            the bond index
        sweep_right : bool
            wether we are currently in a right or left sweep
        """
        # First we factorize the current site to construct the zero-site tensor
        # and update the environment
        C = None
        B = np.transpose(self.psi.Bs[i], (0, 2, 1)) # vL vR i -> vL i vR
        chi_vL, chi_i, chi_vR = B.shape
        if sweep_right:
            B = np.reshape(B, (chi_vL*chi_i, chi_vR)) # vL i vR -> (vL i) vR
            if self.mode == 'qr':
                Q, R = qr(B)
                B = np.reshape(Q, (chi_vL, chi_i, Q.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(U, (chi_vL, chi_i, U.shape[1])) # (vL i) vC -> vL i vC
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vL i vC -> vL vC i = vL vR i
                C = np.tensordot(np.diag(S), V, ([1], [0])) # vC [vC*]; [vC] vR -> vC vR
            self.update_LP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i+1], self.RPs[i])
        else:
            B = np.reshape(B, (chi_vL, chi_i*chi_vR)) # vL i vR -> vL (i vR)
            if self.mode == 'qr':
                R, Q = rq(B)
                B = np.reshape(Q, (Q.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = R
            else:
                U, S, V, norm, _ = mps.split_and_truncate(B, self.chi_max, self.eps)
                self.psi.norm *= norm
                B = np.reshape(V, (V.shape[0], chi_i, chi_vR)) # vC (i vR) -> vC i vR
                self.psi.Bs[i] = np.transpose(B, (0, 2, 1)) # vC i vR -> vC vR i = vL vR i
                C = np.tensordot(U, np.diag(S), ([1], [0])) # vL [vC]; [vC*] vC -> vL vC
            self.update_RP(i)
            # compute effective zero-site Hamiltonian
            Heff = compute_Heff_zero_site(self.LPs[i], self.RPs[i-1])
        C_shape = C.shape
        # evolve zero-site wave function backwards in time
        C = evolve(C.flatten(), Heff, -self.dt/2)
        C = np.reshape(C, C_shape)
        # put back into MPS
        if sweep_right:
            self.psi.Bs[i+1] = np.tensordot(C, self.psi.Bs[i+1], ([1], [0])) # vC [vR]; [vL] vR i -> vC vR i
        else:
            self.psi.Bs[i-1] = np.tensordot(self.psi.Bs[i-1], C, ([1], [0])) # vL [vR] i; [vL] vC -> vL i vC
            self.psi.Bs[i-1] = np.transpose(self.psi.Bs[i-1], (0, 2, 1)) # vL i vC -> vL vC i

def compute_Heff_Twosite(LP, RP, W1, W2):
    """
    Computes the two-site effective hamiltonian
    |theta'> = H_eff |theta>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
    Wi : np.ndarray, shape wL wC i i*
        the MPO tensor acting on site i
    Wj : np.ndarray, shape wC wR j j*
        the MPO tensor acting on sitr j = i + 1
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL i j vR) (vL* i* j* vR*)'
    """
    chi1, chi2 = LP.shape[0], RP.shape[2]
    d1, d2 = W1.shape[2], W2.shape[2]
    result = np.tensordot(LP, W1, ([1], [0])) # vL [wL*] vL*; [wL] wC i i* -> vL vL* wC i i*
    result = np.tensordot(result, W2, ([2], [0])) # vL vL* [wC] i i*; [wC] wR j j* -> vL vL* i i* wR j j*
    result = np.tensordot(result, RP, ([4], [1])) # vL vL* i i* [wR] j j*; vR* [wR*] vR -> vL vL* i i* j j* vR* vR
    result = np.transpose(result, (0, 2, 4, 7, 1, 3, 5, 6)) # vL vL* i i* j j* vR* vR -> vL i j vR vL* i* j* vR*
    mat_shape = chi1*chi2*d1*d2
    result = np.reshape(result, (mat_shape, mat_shape))
    return result

def compute_Heff_Onesite(LP, RP, W):
    """
    Computes the one-site effective hamiltonian
    |psi'> = H_eff |psi>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
    W : np.ndarray, shape wL wR i i*
        the MPO tensor acting on site i
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL vR i) (vL* vR* i*)'
    """
    result = np.tensordot(LP, W, ([1], [0])) # vL [wL*] vL*; [wL] wR i i* -> vL vL* wR i i*
    result = np.tensordot(result, RP, ([2], [1])) # vL vL* [wR] i i*; vR* [wR*] vR -> vL vL* i i* vR* vR
    result = np.transpose(result, (0, 5, 2, 1, 4, 3)) # vL vL* i i* vR* vR -> vL vR i vL* vR* i*
    result = np.reshape(result, (result.shape[0]*result.shape[1]*result.shape[2], result.shape[3]*result.shape[4]*result.shape[5]))
    return result

def compute_Heff_zero_site(LP, RP):
    """
    Computes the one-site effective hamiltonian
    |C'> = H_eff |C>
    
    Parameters
    ----------
    LP : np.nadarray, shape vL wL* vL*
        the left environment tensor
    RP : np.ndarray, shape vR* wR* vR
        the right environment tensor
        
    Returns
    -------
    H_eff : np.ndarray
        the effective Hamiltonian as a matrix of dimensions
        '(vL vR) (vL* vR*)'
    """
    result = np.tensordot(LP, RP, ([1], [1])) # vL [wL*] vL*; vR* [wR*] vR -> vL vL* vR* vR
    result = np.transpose(result, (0, 3, 1, 2)) # vL vL* vR* vR -> vL vR vL* vR*
    result = np.reshape(result, (result.shape[0]*result.shape[1], result.shape[2]*result.shape[3]))
    return result

def evolve(psi, H, dt, debug=False):
    """
    Evolves the given vector by time dt using
    psi(t+dt) = exp(-i*H*dt) @ psi(t)

    Parameters
    ----------
    psi : np.ndarray
        vector of length N, state vector at time t
    H : np.ndarray
        matrix of shape (N, N), (effective) Hamiltonian
    dt : float
        time step
    """
    return expm_multiply(-1.j * H * dt, psi)

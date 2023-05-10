import numpy as np
from ..mps import mps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import time

class TDVPEngine:
    """
    Implements the Two-site Time Dependent Variational Principle (TDVP) algorithm 
    to compute time evolution of matrix product states (MPS).
    """
    
    """
    Initializes the TDVP Engine

    Parameters
    ----------
    psi : MPS
        The current state
    model : 
        The model, containing a MPO representation of the Hamiltonian as the attribute
        model.H_mpo
    dt : float
        time step with which the system is evolved
    chi_max : int
        maximal bond dimension (everything above that gets truncated)
    eps : float
        minimal singular value (everything below that gets truncated)
    normalize : bool
        controls if the singular values are normalized after truncation
    """
    def __init__(self, psi, model, dt, chi_max, eps, normalize=True):
        assert psi.L == model.N  # ensure compatibility
        self.model = model
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        self.dt = dt
        self.normalize = normalize
        # initialize left and right environment
        D = model.H_mpo[0].shape[0]
        chi = psi.Bs[0].shape[0]
        LP = np.zeros([chi, D, chi], dtype="float")  # vL wL* vL*
        RP = np.zeros([chi, D, chi], dtype="float")  # vR* wR* vR
        LP[:, 0, :] = np.eye(chi)
        RP[:, D - 1, :] = np.eye(chi)
        self.LPs[0] = LP
        self.RPs[-1] = RP
        # initialize necessary RPs
        for i in range(psi.L - 1, 1, -1):
            self.update_RP(i)

    """
    Performs one sweep left -> right -> left.
    """
    def sweep(self):
        # sweep from left to right
        for i in range(self.psi.L - 1):
            self.update_bond(i, sweep_right=True)
        # sweep from right to left
        for i in range(self.psi.L - 2, -1, -1):
            self.update_bond(i, sweep_right=False)
        
    """
    Performs a single bond update at the given bond
    
    Parameters
    ----------
    i : int
        the bond index
    sweep_right : bool
        wether we are currently in a right or left sweep
    """
    def update_bond(self, i, sweep_right=True):
        j = i + 1
        # get two-site wavefunction
        theta = self.psi.get_theta2(i) # vL, i, j, vR
        theta_shape = theta.shape
        # get effective two-site Hamiltonian
        Heff = compute_Heff_Twosite(self.LPs[i], self.RPs[j], self.model.H_mpo[i], self.model.H_mpo[j])
        theta = np.reshape(theta, [Heff.shape[0]])
        # evolve 2-site wave function forward in time
        theta = evolve(theta, Heff, self.dt/2)
        theta = np.reshape(theta, theta_shape)
        # split and truncate
        Ai, Sj, Bj, norm_factor = mps.split_truncate_theta(theta, self.chi_max, self.eps)
        if self.normalize == False:
            self.psi.norm *= norm_factor
        # put back into MPS
        Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
        self.psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC*] vC
        self.psi.Ss[j] = Sj  # vC
        self.psi.Bs[j] = Bj  # vC j vR
        if sweep_right == True:
            self.update_LP(i)
            if i < self.psi.L - 2:
                # extract single-site wavefunction
                psi = self.psi.get_theta1(j)
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[j], self.RPs[j], self.model.H_mpo[j])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                Gj = np.tensordot(np.diag(self.psi.Ss[j]**(-1)), psi, axes=[1, 0])  # vL [vL*], [vL] i vR
                self.psi.Bs[j] = Gj
        else:
            self.update_RP(j)
            if i > 0:
                # extract single-site wavefunction
                psi = self.psi.get_theta1(i)
                psi_shape = psi.shape
                psi = psi.flatten()
                # compute effective one-site Hamiltonian
                Heff = compute_Heff_Onesite(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2)
                psi = np.reshape(psi, psi_shape)
                # put back into MPS
                Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), psi, axes=[1, 0])  # vL [vL*], [vL] i vR
                self.psi.Bs[i] = Gi

    def update_RP(self, i):
        """Calculate RP right of site `i-1` from RP right of site `i`."""
        j = i - 1
        RP = self.RPs[i]  # vR* wR* vR
        B = self.psi.Bs[i]  # vL i vR
        Bc = B.conj()  # vL* i* vR*
        W = self.model.H_mpo[i]  # wL wR i i*
        RP = np.tensordot(B, RP, axes=[2, 0])  # vL i [vR], [vR*] wR* vR
        RP = np.tensordot(RP, W, axes=[[1, 2], [3, 1]])  # vL [i] [wR*] vR, wL [wR] i [i*]
        RP = np.tensordot(RP, Bc, axes=[[1, 3], [2, 1]])  # vL [vR] wL [i], vL* [i*] [vR*]
        self.RPs[j] = RP  # vL wL vL* (== vR* wR* vR on site i-1)

    def update_LP(self, i):
        """Calculate LP left of site `i+1` from LP left of site `i`."""
        j = i + 1
        LP = self.LPs[i]  # vL wL vL*
        B = self.psi.Bs[i]  # vL i vR
        G = np.tensordot(B, np.diag(self.psi.Ss[j]**-1), axes=[2, 0])  # vL i [vR], [vR*] vR
        A = np.tensordot(np.diag(self.psi.Ss[i]), G, axes=[1, 0])  # vL [vL*], [vL] i vR
        Ac = A.conj()  # vL* i* vR*
        W = self.model.H_mpo[i]  # wL wR i i*
        LP = np.tensordot(LP, A, axes=[2, 0])  # vL wL* [vL*], [vL] i vR
        LP = np.tensordot(W, LP, axes=[[0, 3], [1, 2]])  # [wL] wR i [i*], vL [wL*] [i] vR
        LP = np.tensordot(Ac, LP, axes=[[0, 1], [2, 1]])  # [vL*] [i*] vR*, wR [i] [vL] vR
        self.LPs[j] = LP  # vR* wR vR (== vL wL* vL* on site i+1)

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
        '(vL i vR) (vL* i* vR*)'
    """
    result = np.tensordot(LP, W, ([1], [0])) # vL [wL*] vL*; [wL] wR i i* -> vL vL* wR i i*
    result = np.tensordot(result, RP, ([2], [1])) # vL vL* [wR] i i*; vR* [wR*] vR -> vL vL* i i* vR* vR
    result = np.transpose(result, (0, 2, 5, 1, 3, 4)) # vL vL* i i* vR* vR -> vL i vR vL* i* vR*
    mat_shape = LP.shape[0] * W.shape[2] * RP.shape[2]
    result = np.reshape(result, (mat_shape, mat_shape))
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

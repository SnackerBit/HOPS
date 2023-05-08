import numpy as np
from ..mps import mps
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import time
from ..util import krylov

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
    N_krylov : int
        number of iterations for the krylov matrix exponentiation
    normalize : bool
        controls if the singular values are normalized after truncation
    """
    def __init__(self, psi, model, dt, chi_max, eps, N_krylov=5, normalize=True):
        assert psi.L == model.N  # ensure compatibility
        self.model = model
        self.psi = psi
        self.LPs = [None] * psi.L
        self.RPs = [None] * psi.L
        self.chi_max = chi_max
        self.eps = eps
        self.dt = dt
        self.N_krylov = N_krylov
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
        Heff = HeffDouble(self.LPs[i], self.RPs[j], self.model.H_mpo[i], self.model.H_mpo[j])
        theta = np.reshape(theta, [Heff.shape[0]])
        # evolve 2-site wave function forward in time
        theta = evolve(theta, Heff, self.dt/2, self.N_krylov)
        theta = np.reshape(theta, theta_shape)
        # split and truncate
        Ai, Sj, Bj = mps.split_truncate_theta(theta, self.chi_max, self.eps, normalize=self.normalize)
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
                Heff = HeffSingle(self.LPs[j], self.RPs[j], self.model.H_mpo[j])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2, self.N_krylov)
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
                Heff = HeffSingle(self.LPs[i], self.RPs[i], self.model.H_mpo[i])
                # evolve 1-site wave function backwards in time
                psi = evolve(psi, Heff, -self.dt/2, self.N_krylov)
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
        
class HeffDouble:
    """
    Class that can be used to perform the multiplication
    |theta'> = H_eff |theta>
    as a "non-matrix" operation!
    """
    
    def __init__(self, LP, RP, W1, W2):
        """
        Initializes the effective hamiltonian
        
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
        """
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wC i i*
        self.W2 = W2  # wC wR j j*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chi1, d1, d2, chi2)  # vL i j vR
        self.shape = (chi1 * d1 * d2 * chi2, chi1 * d1 * d2 * chi2)
        self.dtype = W1.dtype

    def multiply(self, theta):
        """
        Parameters
        ----------
        theta : np.ndarray
            the two-site wavefunction, must be reshapable into (vL, i, j, vR)
            
        Returns
        -------
        np.ndarray :
            Heff@theta
        """
        x = np.reshape(theta, self.theta_shape)  # vL i j vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i j vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] j vR, [wL] wC i [i*]
        x = np.tensordot(x, self.W2, axes=([3, 1], [0, 3]))  # vL [j] vR [wC] i, [wC] wR j [j*]
        x = np.tensordot(x, self.RP, axes=([1, 3], [0, 1]))  # vL [vR] i [wR] j, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x
    
    def get_as_matrix(self):
        """
        Returns the effective Hamiltonian as a matrix
        """
        result = np.tensordot(self.LP, self.W1, ([1], [0])) # vL [wL*] vL*; [wL] wC i i* -> vL vL* wC i i*
        result = np.tensordot(result, self.W2, ([2], [0])) # vL vL* [wC] i i*; [wC] wR j j* -> vL vL* i i* wR j j*
        result = np.tensordot(result, self.RP, ([4], [1])) # vL vL* i i* [wR] j j*; vR* [wR*] vR -> vL vL* i i* j j* vR* vR
        result = np.transpose(result, (0, 2, 4, 7, 1, 3, 5, 6)) # vL vL* i i* j j* vR* vR -> vL i j vR vL* i* j* vR*
        mat_shape = self.theta_shape[0] * self.theta_shape[1] * self.theta_shape[2] * self.theta_shape[3]
        result = np.reshape(result, (mat_shape, mat_shape))
        return result
        
    
class HeffSingle:
    """
    Class that can be used to perform the multiplication
    |psi'> = H_eff |psi>
    as a "non-matrix" operation!
    """
    
    def __init__(self, LP, RP, W):
        """
        Parameters
        ----------
        LP : np.nadarray, shape vL wL* vL*
            the left environment tensor
        RP : np.ndarray, shape vR* wR* vR
            the right environment tensor
        W : np.ndarray, shape wL wR i i*
            the MPO tensor acting on site i
        """
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W = W  # wL wR i i*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d = W.shape[2]
        self.psi_shape = (chi1, d, chi2)  # vL i vR
        self.shape = (chi1 * d * chi2, chi1 * d * chi2)
        self.dtype = W.dtype

    def multiply(self, psi):
        """
        Parameters
        ----------
        psi : np.ndarray
            the one-site wavefunction, must be reshapable into (vL, i, vR)
            
        Returns
        -------
        np.ndarray :
            Heff@theta
        """
        x = np.reshape(psi, self.psi_shape)  # vL i vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i vR
        x = np.tensordot(x, self.W, axes=([1, 2], [0, 3]))  # vL [wL*] [i] vR, [wL] wR i [i*]
        x = np.tensordot(x, self.RP, axes=([1, 2], [0, 1]))  # vL [vR] [wR] i, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0]) # vL i vR
        return x
    
    def get_as_matrix(self):
        """
        Returns the effective Hamiltonian as a matrix
        """
        result = np.tensordot(self.LP, self.W, ([1], [0])) # vL [wL*] vL*; [wL] wR i i* -> vL vL* wR i i*
        result = np.tensordot(result, self.RP, ([2], [1])) # vL vL* [wR] i i*; vR* [wR*] vR -> vL vL* i i* vR* vR
        result = np.transpose(result, (0, 2, 5, 1, 3, 4)) # vL vL* i i* vR* vR -> vL i vR vL* i* vR*
        mat_shape = self.psi_shape[0] * self.psi_shape[1] * self.psi_shape[2]
        result = np.reshape(result, (mat_shape, mat_shape))
        return result

def evolve(psi, H, dt, N_krylov, debug=False):
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
    N_krylov : int
        iterations for the krylov exponentiation
    """
    #return expm(-1.j * H.get_as_matrix() * dt) @ psi
    #return expm_multiply(-1.j * H.get_as_matrix() * dt, psi)
    return krylov.expm_krylov(H.multiply, psi, -1.j*dt, min(int(psi.size), N_krylov), hermitian=True)
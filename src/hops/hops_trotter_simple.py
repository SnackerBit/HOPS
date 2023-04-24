import numpy as np
import scipy.sparse

from ..util import bath_correlation_function
from ..util import noise_generator
from ..util import operators
from ..util import krylov

class HOPS_Simple_Trotter:
    """
    Class that implements the Hierarchy of Stochastic Pure States (HOPS) as described in
    https://arxiv.org/abs/1402.4647, for the simple bath correlation function
    
    \alpha(\tau) = g e^{-\omega \tau}
    
    (only one bath term). Both the linear and the non-linear HOPS equation are implemented.
    For integration, Trotter decomposition with an effective Hamiltonian is used.
    Instead of computing the matrix exponential at every step, we employ a krylov algorithm
    (Arnoldi iteration) to compute expm(dt*A)*v.
    """
    
    def __init__(self, g, w, h, L, duration, N_steps, N_trunc, linear, use_noise=True, N_krylov=5):
        """
        Initializes the HOPS, computes sparse matrices and sets up the noise generator.
        
        Parameters
        ----------
        g : np.ndarray
            array of values g_j. g_j should be of type complex. 
            Because simple HOPS is used here, the array must be of size 1
        w : np.ndarray
            array of values \omega_j. \omega_j should be of type complex. 
            Because simple HOPS is used here, the array must be of size 1
        h : np.ndarray
            the system hamiltonian. Must be a square matrix of dtype complex
        L : np.ndarray
            the coupling operator. Must be the same shape as the system hamiltonian
        duration : float
            the total time duration of the simulation. Time will start at t=0.
        N_steps : int
            number of integration steps
        N_trunc : int
            truncation order of the HOPS
        linear : bool
            used to swap between linear (True) and non-linear (False) HOPS
        use_noise : bool
            wether to use noise or not. For HOPS as described in the paper,
            noise is essential. Therefore set use_noise=False only for testing.
        N_krylov : int
            the amount of iterations used in krylov matrix exponentiation 
        """
        assert(w.size == 1)
        assert(g.size == 1)
        self.dim = h.shape[0] # dimension of the physical hilbert space
        self.h = h
        self.L = L
        self.eye = scipy.sparse.identity(self.dim)
        self.g = g
        self.w = w
        self.aux_N, self.aux_b_dagger, self.aux_b, self.aux_eye = operators.generate_auxiallary_operators_sparse(N_trunc)
        self.alpha0 = bath_correlation_function.alpha(0, g, w).item()
        self.ts = np.linspace(0, duration, N_steps)
        self.dt = (self.ts[1] - self.ts[0])
        self.N_steps = N_steps
        self.N_krylov = N_krylov
        if use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w) 
            self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(N_steps, alpha, 0, duration)
        self.N_trunc = N_trunc
        self.construct_linear_Heff()
        if use_noise:
            self.construct_noise_Heff()
        if not linear:
            self.construct_nonlinear_Heff()
        self.linear = linear
        self.use_noise = use_noise
        self.zts = None
        
    def construct_linear_Heff(self):
        """
        Helper function that constructs the linear effective Hamiltonian

        H_{eff}^{linear} = H \otimes \mathbb{1} - i\omega \mathbb{1} \otimes \hat{N} + i\alpha(0) L \otimes \hat{b}\hat{N} - i L^\dagger \otimes \hat{b}

        as a sparse matrix
        """
        self.Heff_linear = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_linear += scipy.sparse.kron(self.aux_eye, self.h)
        self.Heff_linear += -1.j * self.w.item() * scipy.sparse.kron(self.aux_N, self.eye)
        self.Heff_linear += 1.j * self.alpha0 * scipy.sparse.kron(self.aux_N@self.aux_b_dagger, self.L)
        self.Heff_linear += -1.j * scipy.sparse.kron(self.aux_b, np.conj(self.L).T)
        
    def construct_noise_Heff(self):
        """
        Helper function that constructs the noise effective Hamiltonian

        H_{eff}^{noise} = i L \otimes \mathbb{1} 

        as a sparse matrix
        """
        self.Heff_noise = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_noise += 1.j * scipy.sparse.kron(self.aux_eye, self.L)
        
    def construct_nonlinear_Heff(self):
        """
        Helper function that constructs the non-linear effective Hamiltonian

        H_{eff}^{non-linear} = i \mathbb{1} \otimes \hat{b}^\dagger 

        as a sparse matrix
        """
        self.Heff_nonlinear = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_nonlinear += 1.j * scipy.sparse.kron(self.aux_b, self.eye)
        
    def compute_update(self, t_index):
        """
        Updates self.Psi (and if self.linear == False also updates self.memory)
        ------------------------------------
        Parameters:
            t_index : int
                current time index. Used as index into 
        """
        H_eff = self.Heff_linear.copy()
        if self.linear:
            if self.use_noise:
                H_eff += np.conj(self.zts[t_index]) * self.Heff_noise
        else:
            # compute the expectation value of self.L^\dagger
            self.expL = (np.conj(self.psi[0:self.dim]).T @ np.conj(self.L).T @ self.psi[0:self.dim]) / (np.conj(self.psi[0:self.dim]).T @ self.psi[0:self.dim])
            # add noise term and non-linear term to H_eff
            if self.use_noise:
                H_eff += (np.conj(self.zts[t_index]) + self.memory).item() * self.Heff_noise
                # update memory
                self.memory = np.exp(-np.conj(self.w)*self.dt) * (self.memory + self.dt*np.conj(self.alpha0)*self.expL)
            H_eff += self.expL * self.Heff_nonlinear
        # update state
        Afunc = lambda v : -1j*H_eff@v
        self.psi = krylov.expm_krylov(Afunc, self.psi, self.dt, self.N_krylov)
        
    def compute_realizations(self, N_samples, psi0=np.array([1, 0], dtype=complex), progressBar=iter, zts_debug=None):
        """
        Computes multiple realizations of the HOPS
        
        Parameters
        ----------
        N_samples : int
            How many realizations you want to compute
        psi0 : np.ndarray
            initial state of the system. array should be of shape (self.dim,) and of dtype complex
        progressBar : class
            optional progressBar to visualize how long the computation will take. usage:
            ```
            from tqdm.notebook import tqdm
            hops.compute_realizations(..., progressBar=tqdm)
            ```
            or
            ```
            from tqdm import tqdm
            hops.compute_realizations(..., progressBar=tqdm)
            ```
        zts_debug : np.ndarray or None
            list of N_steps noise values that will be used as noise instead of generating new noise.
            This can be used for debugging (reproducing the exact same evolution using different HOPS methods)
        
        Returns
        -------
            np.ndarray
                array of shape (N_samples, N_steps, dim) of dtype complex containing the physical state \Psi_t^{(k=0)}
                for discrete times t.
        """
        # save psi vectors in list
        psis = np.empty((N_samples, self.N_steps, self.dim), dtype=complex)
        # main loop
        for n in progressBar(range(N_samples)):
            # setup psi vector
            self.psi = np.zeros(self.dim*self.N_trunc, dtype=complex)
            self.psi[0:self.dim] = psi0.copy()
            # setup noise
            if self.use_noise:
                if zts_debug is None:
                    self.zts = self.generator.sample_process()
                else:
                    self.zts = zts_debug
            psis[n, 0, :] = self.psi[0:self.dim].copy()
            # setup memory
            if not self.linear:
                self.memory = complex(0)
            # main loop
            for i in range(0, self.N_steps-1):
                self.compute_update(i)
                psis[n, i+1, :] = self.psi[0:self.dim]
        return psis
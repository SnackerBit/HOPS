import numpy as np

from . import homps_model
from ..util import noise_generator
from ..util import bath_correlation_function
from ..tdvp import mps
from ..tdvp import tdvp

class HOMPS_TDVP:
    """
    This class implements the Hierarchy Of Matrix Pure States (HOMPS) method to
    simulate systems in contact with a heat bath. This class uses the Time Dependent
    Variational Principle (TDVP) to integreate the HOMPS equations
    """
    
    def __init__(self, g, w, h, L, duration, N_steps, N_trunc, linear, use_noise=True, chimax=10, epsilon=1e-12, N_krylov=5, normalize=False):
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
        duration : float
            the total time duration of the simulation. Time will start at t=0.
        N_steps : int
            number of integration steps
        N_trunc : int
            truncation order of each bath mode. This directly controls the dimensions of 
            the W-tensors of the MPO
        linear : bool
            used to swap between linear (True) and non-linear (False) HOPS
        use_noise : bool
            wether to use noise or not. For HOMPS as described in the paper,
            noise is essential. Therefore set use_noise=False only for testing.
        chimax : int
            maximum virtual bond dimension of the MPS
        epsilon : float
            minimum value of singular values. During the bond update of TDVP
            singular values smaller than epsilon will be discarded
        N_krylov : int
            the number of iterations used in krylov matrix exponentiation
        normalize : bool
            if this is set to True, the states are normalized during TDVP (only in the non-linear mode)
        """
        self.g = g
        self.w = w
        self.ts = np.linspace(0, duration, N_steps)
        self.dt = (self.ts[1] - self.ts[0])
        self.N_steps = N_steps
        self.linear = linear
        self.use_noise = use_noise
        self.dim = h.shape[0]
        self.chimax = chimax
        self.epsilon = epsilon
        self.N_krylov = N_krylov
        # construct model
        self.model = homps_model.HOMPSModel(g, w, h, L, N_trunc)
        # construct noise generator
        if use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w)
            self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(N_steps, alpha, 0, duration)
            
    def compute_realizations(self, N_samples, psi0=np.array([1, 0], dtype=complex), progressBar=iter, zts_debug=None, collect_debug_info=False):
        """
        Computes multiple realizations of the HOMPS
        
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
        collect_debug_info : bool
            If this is set to true, debug information will be collected during the computation.
            After the computation is done, the collected information will be available under
            self.debug_info
            
            
        Returns
        -------
        np.ndarray :
            array of shape (N_samples, N_steps, dim) of dtype complex containing the physical state \Psi_t^{(k=0)}
            for discrete times t.
        """
        # setup vector storing psis
        psis = np.empty((N_samples, self.N_steps, self.dim), dtype=complex)
        if collect_debug_info:
            self.prepare_debug_info(N_samples)
        # main loop
        try:
            for n in progressBar(range(N_samples)):
                # initialize MPS
                psi = mps.init_HOMPS_MPS(psi0, self.model.N_bath, self.model.N_trunc)
                # generate noise
                if self.use_noise:
                    if zts_debug is None:
                        self.zts = self.generator.sample_process()
                    else:
                        self.zts = zts_debug
                # setup TDVP engine
                self.engine = tdvp.TDVPEngine(psi, self.model, self.dt, self.chimax, self.epsilon, normalize=False, N_krylov=self.N_krylov)
                if self.linear:
                    # linear HOPS
                    for i in range(self.N_steps):
                        # update MPO
                        if self.use_noise:
                            self.engine.model.update_mpo_linear(np.conj(self.zts[i]))
                        # update psi
                        self.engine.sweep()
                        # save psi
                        psis[n, i, :] = self.extract_physical_state(self.engine.psi)
                        # collect debug info
                        if collect_debug_info:
                            self.collect_debug_info(n, i)
                else:
                    # non-linear HOPS
                    if self.use_noise:
                        self.memory = np.zeros(self.g.size, dtype=complex)
                    for i in range(self.N_steps):
                        # compute expectation value of coupling operator
                        psi_phys = self.extract_physical_state(self.engine.psi)
                        expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
                        # update MPO
                        if self.use_noise:
                            self.engine.model.update_mpo_nonlinear(np.conj(self.zts[i]) + np.sum(self.memory), expL)
                            # update memory
                            self.update_memory(expL)
                        else:
                            self.engine.model.update_mpo_nonlinear(0, expL)
                        # update psi
                        self.engine.sweep()
                        # save psi
                        psis[n, i, :] = self.extract_physical_state(self.engine.psi)
                        # normalize
                        if self.normalize:
                            norm = np.linalg.norm(psis[n, i, :])
                            N = len(self.engine.psi.Ss)
                            for i in range(len(self.engine.psi.Ss))
                                self.engine.psi
                        # collect debug info
                        if collect_debug_info:
                            self.collect_debug_info(n, i)
            return psis
        except KeyboardInterrupt:
            # If a keyboard interruption occurs, return progress up to this point!
            if n > 0:
                print("detected keyboard interrupt. Returning", n, "realizations!")
                return psis[0:n, :, :]
            else:
                print("detected keyboard interrupt.")
                return None
    
    def extract_physical_state(self, psi):
        """
        Extracts the physical state Psi_t^{(0)} from the wavefunction in MPS form
        
        Parameters
        ----------
        psi : MPS class
            the current wavefunction containing all Psi_t^{(k)}
        
        Returns
        -------
        np.ndarray :
            the current physical state as a vector of shape (self.dim, )
        """
        contr = psi.Bs[-1][:, 0, 0] # vL
        for i in range(self.model.N_bath-1, 0, -1):
            contr = np.tensordot(psi.Bs[i][:, 0, :], contr, ([1], [0])) # vL [vR], [vL] -> vL
        result = np.tensordot(psi.get_theta1(0)[0, :, :], contr, ([1], [0])) # i [vR], [vL] -> i
        return result
    
    def update_memory(self, expL):
        """
        Updates the memory vector that is used in nonlinear HOMPS
        
        Parameters
        ----------
        expL : float
            the expectation value of the system operator <L^\dagger> at the current time.
        """
        self.memory = np.exp(-np.conj(self.w)*self.dt) * (self.memory + self.dt*np.conj(self.g)*expL)
        
    def prepare_debug_info(self, N_samples):
        self.debug_info = {}
        self.debug_info["average_bond_dim"] = np.empty((N_samples, self.N_steps))
        
    def collect_debug_info(self, n, i):
        self.debug_info["average_bond_dim"][n, i] = self.engine.psi.get_average_bond_dim()
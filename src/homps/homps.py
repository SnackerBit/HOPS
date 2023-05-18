import numpy as np

from . import homps_model
from ..util import noise_generator
from ..util import bath_correlation_function
from ..mps import mps
from ..tdvp import tdvp
from ..mps import mps_runge_kutta
from . import alternative_TDVP

class HOMPS_Engine:
    """
    This class implements the Hierarchy Of Matrix Pure States (HOMPS) method to
    simulate systems in contact with a heat bath. This class uses the Time Dependent
    Variational Principle (TDVP) or Runge-Kutta of fourth order (RK4) to 
    integreate the HOMPS equations
    """
    
    def __init__(self, g, w, h, L, duration, N_steps, N_trunc, options={}):
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
        options [optional] : dictionary
            dictionary with more options. In the following, all possible options are explained:
            
            'linear' : bool
                used to swap between linear (True) and non-linear (False) HOPS. Default: False
            'use_noise' : bool
                wether to use noise or not. For HOPS as described in the paper, noise is essential. 
                Therefore set use_noise=False only for testing. Default: True
            'method' : string
                which method to use. Possible methods are 'RK4' (Runge-Kutta), 'TDVP'
                (Time Dependent Variational Principle) and 'TDVP2' (TDVP with double site update). 
                Default: 'RK4'
            'chi_max' : int
                maximum virtual bond dimension of the MPS. Default: 10
            'eps' : float
                minimum value of singular values. During the bond update of TDVP
                singular values smaller than epsilon will be discarded.
                Default value: 1.e-10
            'g_noise' : list of complex
                expansion coefficients for the BCF used for computing noise.
                It can be beneficial to use more terms for noise computation than for coupling
                to heat baths, because the noise computation is relatively cheap. Default: None
            'w_noise' : list of complex
                frequencies for the expansion of the BCF. See above. Default: None
            'use_precise_svd' : bool
                wether to use the slower but more precise SVD from util.svd_prec.
                Default: False
            'optimize_mpo' : bool
                wether to optimize the bond dimension of the MPO hamiltonian by using SVDs
        """
        self.g = g
        self.w = w
        self.N_bath = len(g)
        assert(len(w) == self.N_bath)
        self.N_steps = N_steps
        self.N_trunc = N_trunc
        self.ts = np.linspace(0, duration, N_steps)
        self.dt = (self.ts[1] - self.ts[0])
        self.dim = h.shape[0]
        # parse the options
        self.linear = False
        self.use_noise = True
        self.method = 'RK4'
        self.chi_max = 10
        self.eps = 1.e-10
        self.g_noise = g
        self.w_noise = w
        self.use_precise_svd = False
        self.optimize_mpo = False
        if options is not None:
            if 'linear' in options:
                self.linear = options['linear']
            if 'use_noise' in options:
                self.use_noise = options['use_noise']
            if 'method' in options:
                if options['method'] == 'RK4' or options['method'] == 'TDVP' or options['method'] == 'TDVP2' or options['method'] == 'TDVP_alternative':
                    self.method = options['method']
                else:
                    print(f"Unknown method \'{options['method']}\'. Defaulting to \'RK4\'")
            if 'chi_max' in options:
                self.chi_max = options['chi_max']
            if 'eps' in options:
                self.eps = options['eps']
            if 'g_noise' in options and 'w_noise' in options:
                # g_noise and w_noise need to have the same length
                assert(len(options['g_noise']) == len(options['w_noise']))
                self.g_noise = options['g_noise']
                self.w_noise = options['w_noise']
            else:
                # You need to specify both g_noise and w_noise
                assert('g_noise' not in options and 'w_noise' not in options)
            if 'use_precise_svd' in options:
                self.use_precise_svd = options['use_precise_svd']
            if 'optimize_mpo' in options:
                self.optimize_mpo = options['optimize_mpo']
        # construct model
        self.model = homps_model.HOMPSModel(g, w, h, L, N_trunc)
        # determine dtype
        if self.use_precise_svd:
            self.dtype = np.complex256
        else:
            self.dtype = np.complex128
        # construct noise generator
        if self.use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w)
            if self.g_noise is not None:
                alpha = lambda tau : bath_correlation_function.alpha(tau, self.g_noise, self.w_noise)
            self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(N_steps, alpha, 0, duration)
            
    def compute_realizations(self, N_samples, psi0=np.array([1, 0], dtype=np.complex256), progressBar=iter, zts_debug=None, collect_debug_info=False):
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
        psis = np.empty((N_samples, self.N_steps, self.dim), dtype=self.dtype)
        # setup debug info
        if collect_debug_info:
            self.expL = 0
            self.initialize_debug_info(N_samples)
        # main loop
        try:
            for n in progressBar(range(N_samples)):   
                # setup psi vector
                if self.method == 'TDVP' or self.method == 'TDVP_alternative':
                    self.psi = mps.MPS.init_HOMPS_MPS(psi0, self.N_bath, self.N_trunc, use_precise_svd=self.use_precise_svd, chi_max=self.chi_max)
                else:
                    self.psi = mps.MPS.init_HOMPS_MPS(psi0, self.N_bath, self.N_trunc, use_precise_svd=self.use_precise_svd)
                psis[n, 0, :] = self.extract_physical_state(self.psi)
                # setup noise
                if self.use_noise:
                    if zts_debug is None:
                        self.zts = self.generator.sample_process()
                    else:
                        self.zts = zts_debug
                # setup memory
                if not self.linear:
                    self.memory = np.zeros(self.g_noise.size, dtype=complex)
                # initially compute debug_info
                if collect_debug_info:
                    self.compute_debug_info(n, 0)
                # Compute realization
                if self.method == 'RK4':
                    # Runge-Kutta
                    if self.linear and not self.use_noise:
                        # Initial computation of the update MPO
                        self.model.compute_update_mpo()
                    for i in range(0, self.N_steps-1):
                        self.compute_update_RK4(i)
                        if self.linear == False:
                            self.psi.norm = 1.
                        psis[n, i+1, :] = self.extract_physical_state(self.psi)
                        if collect_debug_info:
                            self.compute_debug_info(n, i+1)
                elif self.method == 'TDVP_alternative':
                    for i in range(0, self.N_steps-1):
                        self.compute_update_TDVP_alternative(i)
                        if self.linear == False:
                            self.psi.norm = 1.
                        psis[n, i+1, :] = self.extract_physical_state(self.psi)
                        if collect_debug_info:
                            self.compute_debug_info(n, i+1)
                else:
                    # TDVP
                    if self.method == 'TDVP2':
                        self.engine = tdvp.TDVP2_Engine(self.psi, self.model, self.dt, self.chi_max, self.eps)
                    else:
                        self.engine = tdvp.TDVP1_Engine(self.psi, self.model, self.dt, self.chi_max, self.eps)
                    for i in range(0, self.N_steps-1):
                        self.compute_update_TDVP(i)
                        if self.linear == False:
                            self.engine.psi.norm = 1.
                        psis[n, i+1, :] = self.extract_physical_state(self.engine.psi)
                        if collect_debug_info:
                            self.compute_debug_info(n, i+1)
        except KeyboardInterrupt:
            # If a keyboard interruption occurs, return progress up to this point!
            if n > 0:
                print("detected keyboard interrupt. Returning", n, "realizations!")
                return psis[0:n, :, :]
            else:
                print("detected keyboard interrupt.")
                return None
        return psis
            
    def compute_update_TDVP(self, i):
        """
        Computes a single TDVP update step
        """
        if self.linear:
            # linear HOPS
            if self.use_noise:
                self.engine.model.update_mpo_linear(np.conj(self.zts[i]))
            # update psi
            self.engine.sweep()
        else:
            # non-linear HOPS
            # compute expectation value of coupling operator
            psi_phys = self.extract_physical_state(self.engine.psi)
            self.expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
            # update MPO
            if self.use_noise:
                self.engine.model.update_mpo_nonlinear(np.conj(self.zts[i]) + np.sum(self.memory), self.expL)
            else:
                self.engine.model.update_mpo_nonlinear(np.sum(self.memory), self.expL)
            # update psi
            self.engine.sweep()
            # update memory
            self.update_memory(self.expL)
            
    def compute_update_TDVP_alternative(self, i):
        if self.linear:
            # linear HOPS
            if self.use_noise:
                self.model.update_mpo_linear(np.conj(self.zts[i]))
            # update psi
            self.psi.Bs = alternative_TDVP.single_sweep_TDVP(self.psi.Bs, self.model.H_mpo, self.dt)
        else:
            # non-linear HOPS
            # compute expectation value of coupling operator
            psi_phys = self.extract_physical_state(self.psi)
            self.expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
            # update MPO
            if self.use_noise:
                self.model.update_mpo_nonlinear(np.conj(self.zts[i]) + np.sum(self.memory), self.expL)
            else:
                self.model.update_mpo_nonlinear(np.sum(self.memory), self.expL)
            # update psi
            self.psi.Bs = alternative_TDVP.single_sweep_TDVP(self.psi.Bs, self.model.H_mpo, self.dt)
            # update memory
            self.update_memory(self.expL)
            
    def compute_update_RK4(self, i):
        """
        Computes a single RK4 update step
        """
        if self.linear:
            # linear HOPS
            # update MPO
            if self.use_noise:
                self.model.update_mpo_linear(np.conj(self.zts[i]))
                if self.optimize_mpo:
                    self.model.optimize_mpo_bonds()
                self.model.compute_update_mpo()
            # update psi
            self.psi, self.error = mps_runge_kutta.integrate_MPS_RK4(self.psi, self.dt, self.model.update_mpo, self.chi_max, self.eps)
        else:
            # compute expectation value of coupling operator
            psi_phys = self.extract_physical_state(self.psi)
            self.expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
            # update MPO
            if self.use_noise:
                self.model.update_mpo_nonlinear(np.conj(self.zts[i]) + np.sum(self.memory), self.expL)
            else:
                self.model.update_mpo_nonlinear(np.sum(self.memory), self.expL)
            if self.optimize_mpo:
                self.model.optimize_mpo_bonds()
            self.model.compute_update_mpo()
            # update psi
            self.psi, self.error = mps_runge_kutta.integrate_MPS_RK4(self.psi, self.dt, self.model.update_mpo, self.chi_max, self.eps)
            # update memory
            self.update_memory(self.expL)
            
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
            contr = np.tensordot(psi.Bs[i][:, :, 0], contr, ([1], [0])) # vL [vR], [vL] -> vL
        result = np.tensordot(psi.Bs[0][0, :, :], contr, ([0], [0])) # [vR] i, [vL] -> i
        return result * psi.norm
    
    def update_memory(self, expL):
        """
        Updates the memory vector that is used in nonlinear HOMPS
        
        Parameters
        ----------
        expL : float
            the expectation value of the system operator <L^\dagger> at the current time.
        """
        # update memory
        self.memory = np.exp(-np.conj(self.w_noise)*self.dt) * (self.memory + self.dt*np.conj(self.g_noise)*expL)
        
    def initialize_debug_info(self, N_samples):
        """
        Initializes the debug_info dictionary
        """
        self.debug_info = {
            'memory' : np.empty((N_samples, self.N_steps, len(self.g_noise)), dtype=complex),
            'expL' : np.empty((N_samples, self.N_steps), dtype=float),
            'average_bond_dim' : np.empty((N_samples, self.N_steps))
        }
        
    def compute_debug_info(self, n, i):
        """
        Computes debug information. Should be called at each time step
        """
        self.debug_info['memory'][n, i, :] = self.memory
        self.debug_info['expL'][n, i] = np.real_if_close(self.expL)
        if self.method == 'TDVP':
            self.debug_info['average_bond_dim'][n, i] = self.engine.psi.get_average_bond_dim()
        else:
            self.debug_info['average_bond_dim'][n, i] = self.psi.get_average_bond_dim()

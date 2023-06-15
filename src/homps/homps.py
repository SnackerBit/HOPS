import numpy as np

from . import homps_model
from ..util import noise_generator
from ..util import bath_correlation_function
from ..mps import mps
from ..tdvp import tdvp
from ..mps import mps_runge_kutta

class HOMPS_Engine:
    """
    This class implements the Hierarchy Of Matrix Pure States (HOMPS) method to
    simulate systems in contact with a heat bath. This class uses the Time Dependent
    Variational Principle (TDVP) or Runge-Kutta of fourth order (RK4) to 
    integreate the HOMPS equations.
    See Phys. Rev. A 105, L030202 (2022), "Non-Markovian stochastic Schrödinger equation: 
    Matrix-product-state approach to the hierarchy of pure states"
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
                which method to use. Possible methods are 'RK4' (Runge-Kutta), 'TDVP' (single-site TDVP)
                and 'TDVP2' (double-site TDVP). Default: 'RK4'
            'chi_max' : int
                maximum virtual bond dimension of the MPS. Default: 10
            'eps' : float
                minimum value of singular values. During the bond update of TDVP
                singular values smaller than epsilon will be discarded.
                Default value: 1.e-10
            'g_memory' : list of complex
                expansion coefficients for the BCF used for computing memory updates.
                It can be beneficial to use more terms for memory computation than for coupling
                to heat baths, because the memory computation is relatively cheap. Default: None
            'w_memory' : list of complex
                frequencies for the expansion of the BCF. See above. Default: None
            'optimize_mpo' : bool
                wether to optimize the bond dimension of the MPO hamiltonian by using SVDs
            'noise_generator' : class instance
                if this option is given, the given noise generator is used instead of
                the standard one. Default: None
            'rescale_aux' : bool
                Wether or not to rescale the auxillary vectors for better numerical stability.
                Default: True
            'alternative_realization' : bool
                Wether to use the alternative HOPS
            'gamma_terminator' : complex
                terminator coefficient for alternative HOPS
        """
        options = dict(options) # create copy
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
        self.g_memory = g
        self.w_memory = w
        self.optimize_mpo = False
        self.generator = None
        self.rescale_aux = True
        self.alternative_realization = False
        self.gamma_terminator = 0
        if options is not None:
            if 'linear' in options:
                self.linear = options['linear']
                del options['linear']
            if 'use_noise' in options:
                self.use_noise = options['use_noise']
                del options['use_noise']
            if 'method' in options:
                if options['method'] == 'RK4' or options['method'] == 'TDVP2':
                    self.method = options['method']
                else:
                    print(f"Unknown method \'{options['method']}\'. Defaulting to \'RK4\'")
                del options['method']
            if 'chi_max' in options:
                self.chi_max = options['chi_max']
                del options['chi_max']
            if 'eps' in options:
                self.eps = options['eps']
                del options['eps']
            if 'g_memory' in options and 'w_memory' in options:
                # g_memory and w_memory need to have the same length
                assert(len(options['g_memory']) == len(options['w_memory']))
                self.g_memory = options['g_memory']
                self.w_memory = options['w_memory']
                del options['g_memory']
                del options['w_memory']
            else:
                # You need to specify both g_memory and w_memory
                assert('g_memory' not in options and 'w_memory' not in options)
            if 'optimize_mpo' in options:
                self.optimize_mpo = options['optimize_mpo']
                del options['optimize_mpo']
            if 'noise_generator' in options:
                self.generator = options['noise_generator']
                del options['noise_generator']
            if 'rescale_aux' in options:
                self.rescale_aux = options['rescale_aux']
                del options['rescale_aux']
            if 'alternative_realization' in options:
                self.alternative_realization = options['alternative_realization']
                del options['alternative_realization']
            if 'gamma_terminator' in options:
                self.gamma_terminator = options['gamma_terminator']
                del options['gamma_terminator']
            for key, item in options.items():
                print("[WARNING]: Unused option", key, ":", item)
        # construct model
        self.model = homps_model.HOMPSModel(g, w, h, L, N_trunc, self.rescale_aux, self.alternative_realization, self.gamma_terminator)
        # construct noise generator
        if self.use_noise:
            if self.generator is None:
                alpha = lambda tau : bath_correlation_function.alpha(tau, g, w)
                if self.g_memory is not None:
                    alpha = lambda tau : bath_correlation_function.alpha(tau, self.g_memory, self.w_memory)
                self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(alpha, 0, duration)
            if self.method == 'RK4':
                self.generator.initialize(2*N_steps)
            else:
                self.generator.initialize(N_steps)
        if self.linear:
            self.memory = 0
            
    def compute_realizations(self, N_samples, start=0, psi0=np.array([1, 0], dtype=complex), data_path=None, progressBar=iter, zts_debug=None, collect_debug_info=False):
        """
        Computes multiple realizations of the HOMPS
        
        Parameters
        ----------
        N_samples : int
            How many realizations you want to compute
        start : int
            the realization we start with. Can be used to continue interrupted runs.
        psi0 : np.ndarray
            initial state of the system. array should be of shape (self.dim,) and of dtype complex
        data_path : str
            if this is set, the realizations are not returned but instead stored at
            the given path, with increasing numbering. Default: None
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
        # setup debug info
        if collect_debug_info:
            self.expL = 0
            self.initialize_debug_info(N_samples)
        # main loop
        try:
            for n in progressBar(range(start, N_samples)):   
                # setup psi vector
                if self.method == 'TDVP':
                    self.psi = mps.MPS.init_HOMPS_MPS(psi0, self.N_bath, self.N_trunc, chi_max=self.chi_max)
                else:
                    self.psi = mps.MPS.init_HOMPS_MPS(psi0, self.N_bath, self.N_trunc)
                psis[n, 0, :] = self.extract_physical_state(self.psi)
                # setup noise
                if self.use_noise:
                    if zts_debug is None:
                        self.zts = self.generator.sample_process()
                    else:
                        self.zts = zts_debug
                # setup memory
                if not self.linear:
                    self.memory = np.zeros(self.g_memory.size, dtype=complex)
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
                        self.compute_update_RK4(2*i)
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
                # save realization
                if data_path is not None:
                    np.save(data_path+str(n), psis[n, :, :])
        except KeyboardInterrupt:
            # If a keyboard interruption occurs, return progress up to this point!
            if n > 0:
                print("detected keyboard interrupt. Returning", n, "realizations!")
                return psis[0:n, :, :]
            else:
                print("detected keyboard interrupt.")
                return None
        if data_path is None:
            return psis
            
    def compute_update_TDVP(self, i):
        """
        Computes a single TDVP update step
        """
        if self.linear:
            # linear HOPS
            if self.use_noise:
                self.engine.model.update_mpo_linear(self.zts[i])
            # update psi
            self.engine.sweep()
        else:
            # non-linear HOPS
            # compute expectation value of coupling operator
            psi_phys = self.extract_physical_state(self.engine.psi)
            self.expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
            # update MPO
            if self.use_noise:
                self.engine.model.update_mpo_nonlinear(self.zts[i], np.sum(self.memory), self.expL)
            else:
                self.engine.model.update_mpo_nonlinear(0, np.sum(self.memory), self.expL)
            # update psi
            self.engine.sweep()
            # update memory
            self.update_memory(self.expL)
            
    def compute_update_RK4(self, i):
        """
        Computes a single RK4 update step
        """
        self.psi, self.memory, self.error = mps_runge_kutta.integrate_MPS_RK4_with_memory(self.psi, self.memory, i, self.dt, self.compute_mpo_and_memory_update, self.chi_max, self.eps)
           
    def compute_mpo_and_memory_update(self, psi, memory, t_index):
        """
        Computes the right hand side of the HOMPS equation as an MPO, and the right hand side
        of the memory update equation. Used for the RK4 integration
        
        Parameters
        ----------
        psi : MPS
            the current state
        memory : list of complex
            the current memory
        t_index : int
            current time index (index into self.zts)
        """
        if self.linear:
            if self.use_noise:
                self.model.update_mpo_linear(self.zts[t_index])
                if self.optimize_mpo:
                    self.model.optimize_mpo_bonds()
                self.model.compute_update_mpo()
            return self.model.update_mpo, 0
        else:
            psi_phys = self.extract_physical_state(psi)
            self.expL = (np.conj(psi_phys).T @ np.conj(self.model.L).T @ psi_phys) / (np.conj(psi_phys).T @ psi_phys)
            # update MPO
            if self.use_noise:
                self.model.update_mpo_nonlinear(self.zts[t_index], np.sum(memory), self.expL)
            else:
                self.model.update_mpo_nonlinear(0, np.sum(memory), self.expL)
            if self.optimize_mpo:
                self.model.optimize_mpo_bonds()
            self.model.compute_update_mpo()
            memory_update = -np.conj(self.w_memory)*memory + np.conj(self.g_memory)*self.expL
            return self.model.update_mpo, memory_update
            
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
        self.memory = np.exp(-np.conj(self.w_memory)*self.dt) * (self.memory + self.dt*np.conj(self.g_memory)*expL)
        
    def initialize_debug_info(self, N_samples):
        """
        Initializes the debug_info dictionary
        """
        self.debug_info = {
            'memory' : np.empty((N_samples, self.N_steps, len(self.g_memory)), dtype=complex),
            'expL' : np.empty((N_samples, self.N_steps), dtype=float),
            'bond_dims' : np.empty((N_samples, self.N_steps, self.N_bath)),
            'full_state' : [[None]*self.N_steps for i in range(N_samples)],
        }
        
    def compute_debug_info(self, n, i):
        """
        Computes debug information. Should be called at each time step
        """
        self.debug_info['memory'][n, i, :] = self.memory
        self.debug_info['expL'][n, i] = np.real_if_close(self.expL)
        if self.method == 'TDVP':
            self.debug_info['bond_dims'][n, i] = self.engine.psi.get_bond_dims()
            self.debug_info['full_state'][n, i] = self.engine.psi.copy()
        else:
            self.debug_info['bond_dims'][n, i] = self.psi.get_bond_dims()
            self.debug_info['full_state'][n][i] = self.psi.copy()


''

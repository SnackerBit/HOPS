import numpy as np
import scipy.sparse

from ..util import bath_correlation_function
from ..util import noise_generator
from ..util import operators
from . import runge_kutta

class HOPS_Engine_Simple:
    """
    Class that implements the Hierarchy of Stochastic Pure States (HOPS) as described in
    Phys. Rev. Lett. 113, 150403 (2014), "Hierarchy of Stochastic Pure States for 
    Open Quantum System Dynamics", for the simple bath correlation function

    \alpha(\tau) = g e^{-\omega \tau}
    
    (only one bath mode). Both the linear and the non-linear HOPS equation are implemented.
    For integration,Runge-Kutta (RK4) is used. To speed up the computation we use scipy sparse matrices.
    """
    def __init__(self, g, w, h, L, duration, N_steps, N_trunc, options={}):
        """
        Initializes the HOPS and sets up the noise generator
        
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
        options [optional] : dictionary
            dictionary with more options. In the following, all possible options are explained:
            
            'linear' : bool
                used to swap between linear (True) and non-linear (False) HOPS. Default: False
            'use_noise' : bool
                wether to use noise or not. For HOPS as described in the paper, noise is essential. 
                Therefore set use_noise=False only for testing. Default: True
        """
        # only implemented for one bath mode
        assert(w.size == 1)
        assert(g.size == 1)
        options = dict(options) # create copy
        self.g = g
        self.w = w
        self.h = h
        self.dim = h.shape[0] # dimension of the physical hilbert space
        self.L = L
        self.N_steps = N_steps
        self.N_trunc = N_trunc
        self.eye = scipy.sparse.identity(self.dim)
        self.zts = None
        # parse the options
        self.linear = False
        self.use_noise = True
        if options is not None:
            if 'linear' in options:
                self.linear = options['linear']
                del options['linear']
            if 'use_noise' in options:
                self.use_noise = options['use_noise']
                del options['use_noise']
            for key, item in options.items():
                print("[WARNING]: Unused option", key, ":", item)
        # construct propagators
        self.ts = np.linspace(0, duration, 2*N_steps)
        self.dt = (self.ts[2] - self.ts[0])
        self.construct_linear_propagator()
        if self.use_noise or not self.linear:
            self.construct_noise_propagator()
        if not self.linear:
            self.construct_nonLinear_propagator()
        # setup noise generator
        if self.use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w) 
            self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(alpha, 0, duration)
            self.generator.initialize(2*N_steps)
                
    def construct_linear_propagator(self):
        """
        Helper function that constructs the linear propagator as a sparse matrix:
        
        (-iH - kw) \Pis^{(k)} + k*alpha(0)*L*\Psi^{(k-1)} - L^*\Psi^{(k+1)}
        
        This is used when self.method=='RK4'.
        """
        self.linear_propagator = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        # terms acting on \Pis^{k}
        for k in range(self.N_trunc):
            Hk = -1j*self.h - k*self.w[0]*self.eye
            self.linear_propagator[self.dim*k:self.dim*(k+1), self.dim*k:self.dim*(k+1)] += Hk
        # term acting on \Psi^{(k-1)}
        for k in range(1, self.N_trunc):
            Hkm1 = k*self.g*self.L
            self.linear_propagator[self.dim*k:self.dim*(k+1), self.dim*(k-1):self.dim*k] += Hkm1
        # term acting on \Psi^{(k+1)}
        for k in range(self.N_trunc-1):
            self.linear_propagator[self.dim*k:self.dim*(k+1), self.dim*(k+1):self.dim*(k+2)] -= np.conj(self.L).T
        # convert to csr for faster computation
        self.linear_propagator = scipy.sparse.csr_matrix(self.linear_propagator)
        
    def construct_noise_propagator(self):
        """
        Helper function that constructs the noise propagator as a sparse matrix
        
        L*z_t^* \Psi^{(k)}
        
        Note that we still need to multiply by z^*_t when updating.
        This is used when self.method=='RK4'.
        """
        self.noise_propagator = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        for k in range(self.N_trunc):
            self.noise_propagator[self.dim*k:self.dim*(k+1), self.dim*k:self.dim*(k+1)] += self.L
        # convert to csr for faster computation
        self.noise_propagator = scipy.sparse.csr_matrix(self.noise_propagator)
        
    def construct_nonLinear_propagator(self):
        """
        Helper function that constructs the non-linear propagator as a sparse matrix
        
        \mathbb{1} \Psi^{(k+1)}
        
        Note that we still need to multiply by <L^*>_t when updating.
        This is used when self.method=='RK4'.
        """
        self.non_linear_propagator = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        for k in range(self.N_trunc-1):
            self.non_linear_propagator[self.dim*k:self.dim*(k+1), self.dim*(k+1):self.dim*(k+2)] += self.eye   
        
    def f_linear(self, t_index, psi):
        """
        Computes right hand side of the linear differential equation:
        
        \frac{d}{dt} \Psi_t^{(k)} = (-iH -kw + L z_t^*) \Psi_t^{(k)} + k \alpha(0) L \Psi_t{(k-1)} - L^* \Psi_t^{k+1}
        
        This is used when self.method=='RK4'.
        
        Parameters
        ----------
        t_index : int
            the current time index. Note that (t_index + 1) refers to the time (self.ts[t_index] + 0.5*self.dt)
        psi : np.ndarray
            the current state of the system
    
        Returns
        -------
        np.ndarray
            array of the same shape as psi, representing the right hand side of the linear differential equation.
        """
        result = self.linear_propagator.dot(psi)
        if self.use_noise:
            result += np.conj(self.zts[t_index])*self.noise_propagator.dot(psi)
        return result
    
    def f_nonlinear(self, t_index, psi, memory):
        """
        Computes right hand side of the linear differential equation:
        
        \frac{d}{dt} \Psi_t^{(k)} = (-iH -kw + L \tilde{z}_t^*) \Psi_t^{(k)} + k \alpha(0) L \Psi_t{(k-1)} - (L^* - <L^*>_t) \Psi_t^{k+1}
        
        with 
        
        \tilde{z}_t^* = z_t^* + z_memory_t^*
        \frac{d}{dt} z_memory_t^* = -w^*z_memory_t^* + g<L^*>_t
        
        This is used when self.method=='RK4'.
        
        Parameters
        ----------
        t_index : int
            the current time index. Note that (t_index + 1) refers to the time (self.ts[t_index] + 0.5*self.dt)
        psi : np.ndarray
            the current state of the system
        memory : complex
            the current memory state of the system
            
        Returns
        -------
        np.ndarray
            array of the same shape as psi, representing the right hand side of the non-linear differential equation.
        complex
            right hand side of the memory differential equation
        """
        # compute <L^*>_t
        self.expL = ((np.conj(psi[0:2])@np.conj(self.L).T@psi[0:2])/(np.conj(psi[0:2])@psi[0:2])).item()
        # compute psi update
        psi_update = self.linear_propagator.dot(psi)
        if self.use_noise:
            psi_update += (np.conj(self.zts[t_index]) + memory) * self.noise_propagator.dot(psi)
        else:
            psi_update += memory * self.noise_propagator.dot(psi)
        psi_update += self.expL*self.non_linear_propagator.dot(psi)
        # compute memory update
        memory_update = -np.conj(self.w.item())*memory + np.conj(self.g)*self.expL
        return psi_update, memory_update
    
    def compute_update(self, t_index):
        """
        Updates self.Psi (and if self.linear == False also updates self.memory),
        using the RK4 method
        ------------------------------------
        Parameters:
            t_index : int
                current time index. Used as index into self.ts
        """
        if self.linear == True:
            self.psi = runge_kutta.integrate_RK4(self.psi, 2*t_index, self.dt, self.f_linear)
        else:
            self.psi, self.memory = runge_kutta.integrate_RK4_with_memory(self.psi, self.memory, 2*t_index, self.dt, self.f_nonlinear)
        
    def compute_realizations(self, N_samples, start=0, psi0=np.array([1, 0], dtype=complex), data_path=None, progressBar=iter, zts_debug=None, compute_debug_info=False):
        """
        Computes multiple realizations of the HOPS
        
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
            list of 2*N_steps (RK4) or N_steps (Trotter) noise values that will be used as noise instead of
            generating new noise. This can be used for debugging 
            (reproducing the exact same evolution using different HOPS methods)
        compute_debug_info : bool
            Wether to compute and return additional debug information, like the memory and expL.
            Default: False
            
        Returns
        -------
        np.ndarray
            array of shape (N_samples, N_steps, dim) of dtype complex containing the physical state \Psi_t^{(k=0)}
            for discrete times t. Is only returned if data_path == None.
        """
        # save psi vectors in list
        psis = np.empty((N_samples, self.N_steps, self.dim), dtype=complex)
        # setup debug info
        if compute_debug_info:
            self.expL = 0
            self.initialize_debug_info(N_samples)
        # main loop
        for n in progressBar(range(start, N_samples)):
            # setup psi vector
            self.psi = np.zeros(self.dim*self.N_trunc, dtype=complex)
            self.psi[0:self.dim] = psi0.copy()
            psis[n, 0, :] = self.psi[0:self.dim].copy()
            # setup noise
            if self.use_noise:
                if zts_debug is None:
                    self.zts = self.generator.sample_process()
                else:
                    self.zts = zts_debug
            # setup memory
            if not self.linear:
                self.memory = complex(0)
            # initially compute debug_info
            if compute_debug_info:
                self.compute_debug_info(n, 0)
            # Compute realization
            for i in range(0, self.N_steps-1):
                self.compute_update(i)
                if not self.linear:
                    # normalize
                    self.psi /= np.linalg.norm(self.psi)
                psis[n, i+1, :] = self.psi[0:self.dim]
                if compute_debug_info:
                    self.compute_debug_info(n, i+1)
            # save realization
            if data_path is not None:
                np.save(data_path+str(n), psis[n, :, :])
        if data_path is None:
            return psis
    
    def initialize_debug_info(self, N_samples):
        """
        Initializes the debug_info dictionary
        """
        self.debug_info = {
            'memory' : np.empty((N_samples, self.N_steps), dtype=complex),
            'expL' : np.empty((N_samples, self.N_steps), dtype=float)
        }
        
    def compute_debug_info(self, n, i):
        """
        Computes debug information. Should be called at each time step
        """
        self.debug_info['memory'][n, i] = self.memory
        self.debug_info['expL'][n, i] = np.real_if_close(self.expL)

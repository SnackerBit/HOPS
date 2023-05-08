import numpy as np
import scipy.sparse

from ..util import bath_correlation_function
from ..util import noise_generator
from ..util import operators
from . import runge_kutta

class HOPS_Engine_Simple:
    """
    Class that implements the Hierarchy of Stochastic Pure States (HOPS) as described in
    https://arxiv.org/abs/1402.4647, for the simple bath correlation function

    \alpha(\tau) = g e^{-\omega \tau}
    
    (only one bath mode). Both the linear and the non-linear HOPS equation are implemented.
    For integration, either Runge-Kutta (RK4) or trotter decomposition with an effective
    Hamiltonian is used. To speed up the computation we use scipy sparse matrices.
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
            'method' : string
                which method to use. Possible methods are 'RK4' (Runge-Kutta) and 'Trotter'
                (Trotter decomposition). Default: 'RK4'
        """
        # only implemented for one bath mode
        assert(w.size == 1)
        assert(g.size == 1)
        self.g = g
        self.w = w
        self.h = h
        self.dim = h.shape[0] # dimension of the physical hilbert space
        self.L = L
        self.N_steps = N_steps
        self.N_trunc = N_trunc
        self.alpha0 = bath_correlation_function.alpha(0, g, w).item()
        self.eye = scipy.sparse.identity(self.dim)
        self.zts = None
        # parse the options
        self.linear = False
        self.use_noise = True
        self.method = 'RK4'
        if options is not None:
            if 'linear' in options:
                self.linear = options['linear']
            if 'use_noise' in options:
                self.use_noise = options['use_noise']
            if 'method' in options:
                if options['method'] == 'RK4' or options['method'] == 'Trotter':
                    self.method = options['method']
                else:
                    print(f"Unknown method \'{options['method']}\'. Defaulting to \'RK4\'")
        # construct propagators/Heffs
        if self.method == 'RK4':
            self.ts = np.linspace(0, duration, 2*N_steps)
            self.dt = (self.ts[2] - self.ts[0])
            self.construct_linear_propagator()
            if self.use_noise:
                self.construct_noise_propagator()
            if not self.linear:
                self.construct_nonLinear_propagator()
        else:
            self.ts = np.linspace(0, duration, N_steps)
            self.dt = (self.ts[1] - self.ts[0])
            self.aux_N, self.aux_b_dagger, self.aux_b, self.aux_eye = operators.generate_auxiallary_operators_sparse(N_trunc)
            self.construct_linear_Heff()
            if self.use_noise:
                self.construct_noise_Heff()
            if not self.linear:
                self.construct_nonlinear_Heff()
        # setup noise generator
        if self.use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w) 
            if self.method == 'RK4':
                self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(2*N_steps, alpha, 0, duration)
            else:
                self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(N_steps, alpha, 0, duration)
                
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
            Hkm1 = k*self.alpha0*self.L
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
            
    def construct_linear_Heff(self):
        """
        Helper function that constructs the linear effective Hamiltonian

        H_{eff}^{linear} = H \otimes \mathbb{1} - i\omega \mathbb{1} \otimes \hat{N} + i\alpha(0) L \otimes \hat{b}\hat{N} - i L^\dagger \otimes \hat{b}

        as a sparse matrix. This is used when self.method=='Trotter'.
        """
        self.Heff_linear = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_linear += scipy.sparse.kron(self.aux_eye, self.h)
        self.Heff_linear -= 1.j * self.w.item() * scipy.sparse.kron(self.aux_N, self.eye)
        self.Heff_linear += 1.j * self.alpha0 * scipy.sparse.kron(self.aux_N@self.aux_b_dagger, self.L)
        self.Heff_linear -= 1.j * scipy.sparse.kron(self.aux_b, np.conj(self.L).T)
        
    def construct_noise_Heff(self):
        """
        Helper function that constructs the noise effective Hamiltonian

        H_{eff}^{noise} = i L \otimes \mathbb{1} 

        as a sparse matrix. This is used when self.method=='Trotter'.
        """
        self.Heff_noise = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_noise += 1.j * scipy.sparse.kron(self.aux_eye, self.L)
        
    def construct_nonlinear_Heff(self):
        """
        Helper function that constructs the non-linear effective Hamiltonian

        H_{eff}^{non-linear} = i \mathbb{1} \otimes \hat{b}^\dagger 

        as a sparse matrix. This is used when self.method=='Trotter'.
        """
        self.Heff_nonlinear = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        self.Heff_nonlinear += 1.j * scipy.sparse.kron(self.aux_b, self.eye)
        
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
        psi_update += self.expL*self.non_linear_propagator.dot(psi)
        # compute memory update
        memory_update = -np.conj(self.w.item())*memory + np.conj(self.alpha0)*self.expL
        return psi_update, memory_update
    
    def compute_update_RK4(self, t_index):
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
        
    def compute_update_Trotter(self, t_index):
        """
        Updates self.Psi (and if self.linear == False also updates self.memory),
        using the Trotter decomposition method
        ------------------------------------
        Parameters:
            t_index : int
                current time index. Used as index into self.ts
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
        self.psi = scipy.sparse.linalg.expm_multiply(-1.j*H_eff*self.dt, self.psi)
        
    def compute_realizations(self, N_samples, psi0=np.array([1, 0], dtype=complex), progressBar=iter, zts_debug=None, compute_debug_info=False):
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
            for discrete times t.
        """
        # save psi vectors in list
        psis = np.empty((N_samples, self.N_steps, self.dim), dtype=complex)
        # setup debug info
        if compute_debug_info:
            self.expL = 0
            self.initialize_debug_info(N_samples)
        # main loop
        for n in progressBar(range(N_samples)):
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
            # main loop
            if self.method == 'RK4':
                for i in range(0, self.N_steps-1):
                    self.compute_update_RK4(i)
                    psis[n, i+1, :] = self.psi[0:self.dim]
                    if compute_debug_info:
                        self.compute_debug_info(n, i+1)
            else:
                for i in range(0, self.N_steps-1):
                    self.compute_update_Trotter(i)
                    psis[n, i+1, :] = self.psi[0:self.dim]
                    if compute_debug_info:
                        self.compute_debug_info(n, i+1)
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
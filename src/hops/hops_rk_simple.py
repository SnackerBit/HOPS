import numpy as np
import scipy.sparse

from ..util import bath_correlation_function
from ..util import noise_generator
from . import runge_kutta

class HOPS_Simple_RK4:
    """
    Class that implements the Hierarchy of Stochastic Pure States (HOPS) as described in
    https://arxiv.org/abs/1402.4647, for the simple bath correlation function

    \alpha(\tau) = g e^{-\omega \tau}
    
    (only one bath mode). Both the linear and the non-linear HOPS equation are implemented.
    For integration, Runge-Kutta (RK4) is used. The implementation uses sparse matrices
    for better performance
    """
    
    def __init__(self, g, w, h, L, duration, N_steps, N_trunc, linear, use_noise=True, use_norm_term=False):
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
        use_norm_term : bool
            wether to use the normalizgin term or not. This term was used in
            the HOPS implementation on https://github.com/dsuess/HOPS, but not
            explained in the original HOPS paper.
        """
        assert(w.size == 1)
        assert(g.size == 1)
        self.dim = h.shape[0] # dimension of the physical hilbert space
        self.h = h
        self.L = L
        self.eye = scipy.sparse.identity(self.dim)
        self.g = g
        self.w = w
        self.alpha0 = bath_correlation_function.alpha(0, g, w).item()
        self.ts = np.linspace(0, duration, 2*N_steps)
        self.dt = (self.ts[2] - self.ts[0])
        self.N_steps = N_steps
        if use_noise:
            alpha = lambda tau : bath_correlation_function.alpha(tau, g, w) 
            self.generator = noise_generator.ColoredNoiseGenerator_FourierFiltering(2*N_steps, alpha, 0, duration)
        self.N_trunc = N_trunc
        self.construct_linear_propagator()
        if use_noise:
            self.construct_noise_propagator()
        if not linear:
            self.construct_nonLinear_propagator()
        self.linear = linear
        self.use_noise = use_noise
        self.use_norm_term = use_norm_term
        self.zts = None
        
    def construct_linear_propagator(self):
        """
        Helper function that constructs the linear propagator as a sparse matrix:
        
        (-iH - kw) \Pis^{(k)} + k*alpha(0)*L*\Psi^{(k-1)} - L^*\Psi^{(k+1)}
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
        
        Note that we still need to multiply by z^*_t when updating
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
        
        Note that we still need to multiply by <L^*>_t when updating
        """
        self.non_linear_propagator = scipy.sparse.lil_matrix((self.N_trunc*self.dim, self.N_trunc*self.dim), dtype=complex)
        for k in range(self.N_trunc-1):
            self.non_linear_propagator[self.dim*k:self.dim*(k+1), self.dim*(k+1):self.dim*(k+2)] += self.eye
            
    def f_linear(self, t_index, psi):
        """
        Computes right hand side of the linear differential equation:
        
        \frac{d}{dt} \Psi_t^{(k)} = (-iH -kw + L z_t^*) \Psi_t^{(k)} + k \alpha(0) L \Psi_t{(k-1)} - L^* \Psi_t^{k+1}
        
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
        
        Additionally there is also a optional norm correction term, that was used in the fortran implementation
        of HOPS https://github.com/dsuess/HOPS, but not explained in the HOPS paper.
        
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
        expL = ((np.conj(psi[0:2])@np.conj(self.L).T@psi[0:2])/(np.conj(psi[0:2])@psi[0:2])).item()
        # compute psi update
        psi_update = self.linear_propagator.dot(psi)
        if self.use_noise:
            psi_update += (np.conj(self.zts[t_index]) + memory) * self.noise_propagator.dot(psi)    
        psi_update += expL*self.non_linear_propagator.dot(psi)
        # compute memory update
        memory_update = -np.conj(self.w.item())*memory + np.conj(self.alpha0)*expL
        # compute norm correction: I don't know where this term comes from ...
        if self.use_norm_term:
            delta = 0
            if self.use_noise:
                delta = (np.conj(self.zts[t_index]) + memory) * expL
            if psi.size >= 4:
                delta -= self.g/self.w * np.conj(psi[0:2])@np.conj(self.L).T@psi[2:4]
                delta += self.g/self.w * expL * np.conj(psi[0:2])@psi[2:4]
            psi_update -= delta * psi
        return psi_update, memory_update
    
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
            list of 2*N_steps noise values that will be used as noise instead of generating new noise.
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
            psi = np.zeros(self.dim*self.N_trunc, dtype=complex)
            psi[0:self.dim] = psi0.copy()
            # setup noise
            if self.use_noise:
                if zts_debug is None:
                    self.zts = self.generator.sample_process()
                else:
                    self.zts = zts_debug
            psis[n, 0, :] = psi[0:self.dim].copy()
            # main loop
            if self.linear:
                for i in range(0, self.N_steps-1):
                    psi = runge_kutta.integrate_RK4(psi, 2*i, self.dt, self.f_linear)
                    psis[n, i+1, :] = psi[0:self.dim]
            else:
                # initialize memory term
                memory = complex(0)
                for i in range(0, self.N_steps-1):
                    psi, memory = runge_kutta.integrate_RK4_with_memory(psi, memory, 2*i, self.dt, self.f_nonlinear)
                    psis[n, i+1, :] = psi[0:self.dim]
        return psis
    
    def compute_memory_realization(self, psi0=np.array([1, 0], dtype=complex)):
        """
        Computes a single realization of the HOPS, but returns only the memory values at each time step.
        This can be used for debugging
        
        Parameters
        ----------
        N_samples : int
            How many realizations you want to compute
        psi0 : np.ndarray
            initial state of the system. array should be of shape (self.dim,) and of dtype complex
        
        Returns
        -------
        np.ndarray
            array of shape (N_steps, dim) of dtype complex containing the physical state \Psi_t^{(k=0)}
            for discrete times t.
        np.ndarray
            array of shape (N_steps,) of dtype complex containing the memory values.
        """
        assert(self.linear == False)
        # initialize np.ndarray for storing memory
        memories = np.empty(self.N_steps, dtype=complex)
        # save psi vectors in list
        psis = np.empty((self.N_steps, self.dim), dtype=complex)
        # setup psi vector
        psi = np.zeros(self.dim*self.N_trunc, dtype=complex)
        psi[0:self.dim] = psi0.copy()
        psis[0, :] = psi[0:self.dim].copy()
        # setup noise
        if self.use_noise:
            self.zts = self.generator.sample_process()                                  
        # initialize memory term
        memory = complex(0)
        # save memory
        memories[0] = memory
        for i in range(0, self.N_steps-1):
            psi, memory = runge_kutta.integrate_RK4_with_memory(psi, memory, 2*i, self.dt, self.f_nonlinear)
            psis[i+1, :] = psi[0:self.dim].copy()
            # save memory
            memories[i+1] = memory
        return psis, memories
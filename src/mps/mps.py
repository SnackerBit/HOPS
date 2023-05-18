import numpy as np
from scipy.linalg import svd
from numpy.linalg import qr
from ..util import svd_prec

class MPS:
    """
    Class representing a Matrix Product State.
    
    Attributes
    ----------
    Bs : list of np.ndarray
        list of the tensors. One tensor for each physical site.
        Each tensor has legs 'vL vR i' (virtual left, virtual right, physical)
    L : int
        number of sites
    canonical : str
        one of {'none', 'left', 'right'}. Describes if the MPS is either not in
        canonical form, in left-canonical form, or in right-canonical form.
    norm : complex
        additional overall scalar factor of the MPS
    use_precise_svd : bool
        wether to use the slower but more precise SVD from util.svd_prec.
    dtype : np.dtype
        the np.dtype used for linear algebra. One of {np.complex128, np.complex256}
    """
    
    def __init__(self, Bs, canonical='none', norm=1., use_precise_svd=False):
        """
        Initializes an MPS. It is not checked wether the given canonicalization
        does in fact apply (if you specify eg. canonical='right', the class just
        assumes that the Bs are actually in right-canonical form)
        """
        self.Bs = Bs
        self.L = len(Bs)
        self.canonical = canonical
        self.norm = norm
        self.use_precise_svd = use_precise_svd
        if self.use_precise_svd:
            self.dtype = np.complex256
        else:
            self.dtype = np.complex128
            
    def get_theta_2(self, i):
        """
        Returns the contraction of self.Bs[i] and self.Bs[i+1]
        with shape 'vL i j vR'
        """
        result = np.tensordot(self.Bs[i], self.Bs[i+1], ([1], [0])) # vL [vR] i; [vL] vR j -> vL i vR j
        return np.transpose(result, (0, 1, 3, 2)) # vL i vR j -> vL i j vR

    def copy(self):
        """
        Creates a true copy of this MPS
        """
        return MPS([B.copy() for B in self.Bs], self.canonical, self.norm, self.use_precise_svd)
    
    def get_bond_dims(self):
        """
        Returns a list of bond dimensions.       
        """
        return [self.Bs[i].shape[1] for i in range(self.L - 1)]
    
    def get_average_bond_dim(self):
        """
        Computes and returns the average bond dimension
        """
        return np.mean(self.get_bond_dims())
        
    def get_max_bond_dim(self):
        """
        Returns the maximum bond dimension
        """
        return np.max(self.get_bond_dims())
    
    def site_expectation_value(self, op):
        """
        Computes the expectation values of a local operator at each site.
        
        Parameters
        ----------
        op : np.ndarray
            the local operator. Has to be of shape (i*, i)
        
        Returns
        -------
        list of complex
            a list of lenght self.L containing the expectation values
        """
        left_envs = self.compute_left_environments()
        right_envs = self.compute_right_environments()
        # Compute norm
        norm = np.tensordot(right_envs[0], self.Bs[0], ([0], [1])) # [vL] vL*; vL [vR] i -> vL* vL i
        norm = np.tensordot(norm, np.conj(self.Bs[0]), ([0, 2], [1, 2])) # [vL*] vL [i]; vL* [vR*] [i*] -> vL vL*
        norm = norm.item()
        # Compute expectation values
        results = []
        for i in range(self.L):
            contr = np.tensordot(left_envs[i], self.Bs[i], ([0], [0])) # [vR] vR*; [vL] vR i > vR* vR i
            contr = np.tensordot(contr, op, ([2], [0])) # vR* vR [i]; [i*] i -> vR* vR i
            contr = np.tensordot(contr, np.conj(self.Bs[i]), ([0, 2], [0, 2])) # [vR*] vR [i]; [vL*] vR*, [i*] -> vR vR*
            contr = np.tensordot(contr, right_envs[i], ([0, 1], [0, 1])) # [vR] [vR*]; [vL], [vL*]
            results.append(contr.item() / norm)
        return results
        
    def compute_left_environments(self):
        """
        Computes a list of left-environments for each site:
        
         /--           /--(B[0])--          /--(B[0])--(B[1])--                       
         |             |    |               |    |       |                     
        (1)     ,     (1)   |         ,     1)   |       |        ,    ...                            
         |             |    |               |    |       |                                    
         \--           \--(B[0])--          \--(B[0])--(B[1])--                  
         
         envs[i] is left of site i
        """
        if self.canonical == 'left':
            return [np.eye(self.Bs[i].shape[0], self.Bs[i].shape[0]) for i in range(self.L)]
        contr = np.ones((1, 1)) # vR vR*
        envs = [contr.copy()]
        for i in range(self.L - 1):
            contr = np.tensordot(contr, self.Bs[i], ([0], [0])) # [vR] vR*; [vL] vR i -> vR* vR i
            contr = np.tensordot(contr, np.conj(self.Bs[i]), ([0, 2], [0, 2])) # [vR*] vR [i]; [vL*] vR* [i*] -> vR vR*
            envs.append(contr.copy())
        return envs
    
    def compute_right_environments(self):
        """
        Computes a list of right-environments for each site:
        
                    --(B[L-2])--(B[L-1])--\          --(B[L-1])--\            --\  
                          |         |     |                |     |              |  
         ...   ,          |         |    (1)    ,          |    (1)     ,      (1) 
                          |         |     |                |     |              |  
                    --(B[L-2])--(B[L-1])--/          --(B[L-1])--/            --\  
                    
        envs[i] is the environment right of site i
        """
        if self.canonical == 'right':
            return [np.eye(self.Bs[i].shape[1], self.Bs[i].shape[1]) for i in range(self.L)]
        contr = np.ones((1, 1)) # vL vL*
        envs = [None] * self.L
        envs[self.L - 1] = contr.copy()
        for i in range(self.L - 1, 0, -1):
            contr = np.tensordot(contr, self.Bs[i], ([0], [1])) # [vL] vL*; vL [vR] i -> vL* vL i
            contr = np.tensordot(contr, np.conj(self.Bs[i]), ([0, 2], [1, 2])) # [vL*] vL [i]; vL* [vR*] [i*] -> vL vL*
            envs[i-1] = contr.copy()
        return envs

    def canonicalize(self, chi_max=0, eps=0):
        """
        canonicalizes (and optionally compresses) this MPS.
        After calling this function, the MPS will be in right-canonical form
        
        Parameters
        ----------
        chi_max : int
            maximal bond dimension. If this is set to zero,
            no limit for the bond dimension is imposed
        eps : float
            lower threshhold for the absolute size of singular values
            
        Returns
        -------
        float :
            the compression error
        """
        # Sweep left to right using QR decompositions
        B = self.Bs[0]
        for i in range(self.L-1):
            chi_vL, chi_vR, chi_i = B.shape
            B = np.transpose(B, (0, 2, 1)) # vL vR i -> vL i vR
            B = np.reshape(B, (chi_vL*chi_i, chi_vR)) # vL i vR -> (vL i) vR
            if self.use_precise_svd:
                Q, S, R = svd_prec.svd_prec(B)
                R = np.tensordot(np.diag(S), R, ([1], [0])) # DEBUG, TODO: Change!
            else:
                Q, R = qr(B)
            chi_new = Q.shape[1]
            B = np.reshape(Q, (chi_vL, chi_i, chi_new))
            B = np.transpose(B, (0, 2, 1)) # vL i vR -> vL vR i
            self.Bs[i] = B
            B = self.Bs[i+1]
            B = np.tensordot(R, B, ([1], [0])) # vL [vR]; [vL] vR i -> vL vR i
            
        # sweep right to left using SVDs to compute singular values
        error = 0.0
        for i in range(self.L - 1, 0, -1):
            chi_vL, chi_vR, chi_i = B.shape
            B = np.transpose(B, (0, 2, 1)) # vL vR i -> vL i vR
            B = np.reshape(B, (chi_vL, chi_i*chi_vR)) # vL i vR -> vL (i vR)
            # perform SVD
            U, S, V, norm_factor, error_temp = split_and_truncate(B, chi_max=chi_max, eps=eps, use_precise_svd=self.use_precise_svd)
            chi_new = S.size
            self.norm *= norm_factor
            error += error_temp
            # put back and update B
            V = np.reshape(V, (chi_new, chi_i, chi_vR)) # chi_new, (i, vR) -> chi_new, i, vR
            V = np.transpose(V, (0, 2, 1)) # vL i vR -> vL vR i
            self.Bs[i] = V
            B = self.Bs[i-1]
            B = np.tensordot(B, U, ([1], [0])) # vL [vR] i; [vL] chi_new -> vL i chi_new
            B = np.tensordot(B, np.diag(S), ([2], [0])) # vL i [chi_new]; [chi_new] chi_new -> vL i chi_new
            B = np.transpose(B, (0, 2, 1)) # vL i vR -> vL vR i
        self.Bs[0] = B
        self.canonical = 'right'
        if error > 1.e-3:
            print("[WARNING]: Large error", error, "> 1.e-3 detected")
        return error
    
    def to_state_vector(self):
        """
        Contracts the MPS to form a state vector in Hilbert space
        """
        contr = self.Bs[0][0] # i vR
        for i in range(1, self.L):
            contr = np.tensordot(contr, self.Bs[i], ([1], [0])) # i [vR]; [vL] vR j -> i vR j
            contr = np.transpose(contr, (0, 2, 1)) # i vR j -> i j vR
            contr = np.reshape(contr, (contr.shape[0]*contr.shape[1], contr.shape[2])) # i j vR -> i' vR
        return self.norm * contr[:, 0]
      
    def sanity_check(self):
        """
        Checks if all bond dimensions match up, if the dtype is correct,
        and if self.canonical is set correctly
        """
        # Check if the bond dimensions match up
        assert(self.Bs[0].shape[0] == 1)
        assert(self.Bs[-1].shape[1] == 1)
        for i in range(self.L-1):
            assert(self.Bs[i].shape[1] == self.Bs[i+1].shape[0])
        # Check if the dtype is correct
        for i in range(self.L):
            assert(self.Bs[i].dtype == self.dtype)
        # Check for canonicalization
        if self.canonical == 'left':
            assert(self.is_left_canonical())
        elif self.canonical == 'right':
            assert(self.is_right_canonical())
    
    def is_right_canonical(self):
        """
        Checks if the MPS is in right canonical form by contracting the tensors
        """
        contr = np.ones((1, 1)) # vL vL*
        for i in range(self.L - 1, 0, -1):
            contr = np.tensordot(contr, self.Bs[i], ([0], [1])) # [vL] vL*; vL [vR] i -> vL* vL i
            contr = np.tensordot(contr, np.conj(self.Bs[i]), ([0, 2], [1, 2])) # [vL*] vL [i]; vL* [vR*] [i*] -> vL vL*
            if not np.all(np.isclose(contr, np.eye(contr.shape[0]))):
                return False
        return True
        
    def is_left_canonical(self):
        """
        Checks if the MPS is in left canonical form by contracting the tensors
        """
        contr = np.ones((1, 1)) # vR vR*
        for i in range(self.L - 1):
            contr = np.tensordot(contr, self.Bs[i], ([0], [0])) # [vR] vR*; [vL] vR i -> vR* vR i
            contr = np.tensordot(contr, np.conj(self.Bs[i]), ([0, 2], [0, 2])) # [vR*] vR [i]; [vL*] vR* [i*] -> vR vR*
            if not np.all(np.isclose(contr, np.eye(contr.shape[0]))):
                return False
        return True
        
    @staticmethod
    def initialize_spinup(L, chimax=1, use_precise_svd=False):
        """
        Returns a product state with all spins up as an MPS
        """
        dtype = complex
        if use_precise_svd:
            dtype=np.complex256
        B = np.zeros([chimax, chimax, 2], dtype=dtype)
        B[0, 0, 0] = 1.
        Bs = [B.copy() for _ in range(L)]
        Bs[0] = np.expand_dims(Bs[0][0, :, :], axis=0)
        Bs[-1] = np.expand_dims(Bs[-1][:, 0, :], axis=1)
        return MPS(Bs, use_precise_svd=use_precise_svd)
    
    @staticmethod
    def initialize_from_state_vector(psi, L, chi_max=100, eps=0, d=2, use_precise_svd=False):
        """
        Initializes an MPS from a state vector in Hilbert space
        
        Parameters
        ----------
        psi : np.ndarray
            the state vector of shape (d**L,)
        L : int
            the number of physical sites
        chi_max : int
            maximal bond dimension.
        eps : float
            lower threshhold for the absolute size of singular values
        d : (list of) int
            the dimension of the local Hilbert space on each site,
            eg. d=2 for spin-1/2.
            
        Returns
        -------
        psi_mps : MPS
            the state compressed into an MPS
        """
        if type(d) != list:
            d = [d]*L
        d = np.array(d, dtype=int)
        # first, reshape the state into a single column vector (if its not already in this form)
        psi_aL = np.reshape(psi, (np.prod(d), 1))
        Ss = [None] * L 
        norm = 1.
        # now iterate over the sites of the chain
        for n in range(L-1, -1, -1):
            # compute Chi_n and R_dim. Chi_n * 2 will be the "dimension" d^(L_a) of subsystem A, 
            # R_dim//2 will be the "dimension" d^(L_b) of subsystem B
            L_dim, Chi_n = psi_aL.shape
            assert L_dim == np.prod(d[0:n+1])
            # Reshape wavefunction
            psi_LR = np.reshape(psi_aL, (L_dim//d[n], Chi_n*d[n]))
            # perform SVD
            psitilde_n, lambda_n, M_n = svd(psi_LR, full_matrices=False, lapack_driver='gesvd')
            # if necessary, truncate (keep only the schmidt vectors corresponding to the chi_max largest schmidt values)!
            if len(lambda_n) > chi_max:
                keep = np.argsort(lambda_n)[::-1][:chi_max]
                psitilde_n = psitilde_n[:, keep]
                lambda_n = lambda_n[keep]
                M_n = M_n [keep, :]
            current_norm = np.linalg.norm(lambda_n)
            norm *= current_norm
            lambda_n = lambda_n / current_norm
            # reshape M_[n]
            Chi_np1 = len(lambda_n)
            # physical index is always the dimension!
            M_n = np.reshape(M_n, (Chi_np1, d[n], Chi_n))
            # reabsorb lambda
            psi_aL = psitilde_n[:,:] * lambda_n[np.newaxis, :]
            Bs[n] = M_n
        assert(psi_aL.shape == (1,1))
        norm *= si_aL.item()
        return MPS(Bs, norm=norm, use_precise_svd=use_precise_svd)
    
    @staticmethod
    def init_HOMPS_MPS(psi0, N_bath, N_trunc, use_precise_svd=False, chi_max=1):
        """
        Returns a product state MPS that can be used in HOMPS.
        All bath modes are initially set to zero
        
        Parameters
        ----------
        psi0 : np.ndarray
            initial physical state \Psi_0^{(0)}, array of shape (d,), where d is the physical dimension
        N_bath : int
            number of bath sites
        N_trunc : int
            truncation order of the bath sites
        use_precise_svd : bool
            wether to use the slower but more precise SVD from util.svd_prec.
        
        Returns
        -------
        MPS :
            initial MPS for the HOPS algorithm. The first tensor Bs[0] has
            shape (1, 1, d), and all others have shape (1, 1, N_trunc).
            In total there are (N_bath + 1) tensors in the MPS
        """
        dtype=complex
        if use_precise_svd:
            dtype=np.complex256
        Bs = [None]*(N_bath + 1)
        chi = min(psi0.size, chi_max)
        B_physical = np.zeros([1, chi, psi0.size], dtype=dtype)
        B_physical[0, 0, :] = psi0
        Bs[0] = B_physical
        for i in range(N_bath):
            chi_prime = min(chi_max, min(chi*N_trunc, N_trunc**(N_bath-i-1)))
            B_bath = np.zeros([chi, chi_prime, N_trunc], dtype=dtype)
            B_bath[0, 0, 0] = 1.
            Bs[i+1] = B_bath
        return MPS(Bs, use_precise_svd=use_precise_svd)
    
def split_and_truncate(A, chi_max=0, eps=0, use_precise_svd=False):
    # perform SVD
    if use_precise_svd:
        U, S, V = svd_prec.svd_prec(A)
    else:
        U, S, V = svd(A, full_matrices=False, lapack_driver='gesvd')
    # truncate
    if chi_max > 0:
        chi_new = min(chi_max, np.sum(S > eps))
    else:
        chi_new = np.sum(S>=eps)
    assert chi_new >= 1
    piv = np.argsort(S)[::-1][:chi_new]  # keep the largest chi_new singular values
    error = np.sum(S[chi_new:]**2)
    U, S, V = U[:, piv], S[piv], V[piv, :]
    # renormalize
    norm = np.linalg.norm(S)
    if norm < 1.e-7:
        print("[WARNING]: Small singular values, norm(S) < 1.e-7!")
    S = S / norm
    return U, S, V, norm, error
import numpy as np
from scipy.linalg import svd
from numpy.linalg import qr

class MPS:
    """Class for a matrix product state.

    We index sites with `i` from 0 to L-1; bond `i` is left of site `i`.

    Parameters
    ----------
    Bs, Ss:
        Same as attributes.

    Attributes
    ----------
    Bs : list of np.Array[ndim=3]
        The 'matrices', one for each physical site.
        Each `B[i]` has legs (virtual left, physical, virtual right), in short ``vL i vR``
    Ss : list of np.Array[ndim=1]
        The Schmidt values at each of the bonds, ``Ss[i]`` is left of ``Bs[i]``.
    L : int
        Number of sites.
    canonical : bool
        wether the state is in (right-)canonical form. Some actions, like adding two states
        or multiplying with an MPO make the state non-canonical, and it has to be brought
        into canonical form by calling self.compress()
    norm : complex
        a scaler factor of the whole MPS. Used to make the MPS more numerically stable
    """

    def __init__(self, Bs, Ss, canonical=True, norm=1.):
        """
        Initializes a right-canonical MPS. Assumes the Bs and Ss tensors
        are given in the correct form
        """
        self.Bs = Bs
        self.Ss = Ss
        self.L = len(Bs)
        self.canonical = canonical
        self.norm = norm

    def copy(self):
        result = MPS([B.copy() for B in self.Bs], [S.copy() for S in self.Ss])
        result.canonical = self.canonical
        result.norm = self.norm
        return result

    def get_theta1(self, i):
        """Calculate effective single-site wave function on sites i in mixed canonical form.
        The returned array has legs ``vL, i, vR`` (as one of the Bs)."""
        assert(self.canonical)
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, i):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.
        The returned array has legs ``vL, i, j, vR``."""
        assert(self.canonical)
        j = i + 1
        return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR

    def get_chi(self):
        """Return bond dimensions."""
        return [self.Bs[i].shape[2] for i in range(self.L - 1)]

    def site_expectation_value(self, op):
        """Calculate expectation values of a local operator at each site."""
        assert(self.canonical)
        result = []
        for i in range(self.L):
            theta = self.get_theta1(i)  # vL i vR
            op_theta = np.tensordot(op, theta, axes=[1, 1])  # i [i*], vL [i] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2], [1, 0, 2]]))
            # [vL*] [i*] [vR*], [i] [vL] [vR]
        return np.real_if_close(result)

    def bond_expectation_value(self, op):
        """Calculate expectation values of a local operator at each bond."""
        assert(self.canonical)
        result = []
        for i in range(self.L - 1):
            theta = self.get_theta2(i)  # vL i j vR
            op_theta = np.tensordot(op[i], theta, axes=[[2, 3], [1, 2]])
            # i j [i*] [j*], vL [i] [j] vR
            result.append(np.tensordot(theta.conj(), op_theta, [[0, 1, 2, 3], [2, 0, 1, 3]]))
            # [vL*] [i*] [j*] [vR*], [i] [j] [vL] [vR]
        return np.real_if_close(result)

    def entanglement_entropy(self):
        """Return the (von-Neumann) entanglement entropy for a bipartition at any of the bonds."""
        assert(self.canonical)
        result = []
        for i in range(1, self.L):
            S = self.Ss[i].copy()
            S[S < 1.e-20] = 0.  # 0*log(0) should give 0; avoid warning or NaN.
            S2 = S * S
            assert abs(np.linalg.norm(S) - 1.) < 1.e-14
            result.append(-np.sum(S2 * np.log(S2)))
        return np.array(result)
    
    def get_average_bond_dim(self):
        """Returns the average bond dimension over all bonds"""
        return np.mean(self.get_chi())
    
    def compress(self, chi_max=0, eps=0):
        """
        Compresses and canonicalizes this MPS.
        
        Parameters
        ----------
        chi_max : int
            maximal bond dimension. If this is set to zero,
            no limit for the bond dimension is imposed
        eps : float
            lower threshhold for the absolute size of singular values
        """
        # sweep left to right using QR decompositions
        B = self.Bs[0]
        for i in range(self.L-1):
            chi_vL, chi_i, chi_vR = B.shape
            B = np.reshape(B, (chi_vL*chi_i, chi_vR)) # vL i vR -> (vL i) vR
            Q, R = qr(B)
            chi_new = Q.shape[1]
            B = np.reshape(Q, (chi_vL, chi_i, chi_new))
            self.Bs[i] = B
            B = self.Bs[i+1]
            B = np.tensordot(R, B, ([1], [0])) # vL [vR]; [vL] i vR -> vL i vR
            
        # sweep right to left using SVDs to compute singular values
        for i in range(self.L - 1, 0, -1):
            chi_vL, chi_i, chi_vR = B.shape
            B = np.reshape(B, (chi_vL, chi_i*chi_vR)) # vL i vR -> vL (i vR)
            # perform SVD
            U, S, V = svd(B, full_matrices=False, lapack_driver='gesvd')
            # truncate
            if chi_max > 0:
                chi_new = min(chi_max, np.sum(S > eps))
            else:
                chi_new = np.sum(S>eps)
            assert chi_new >= 1
            piv = np.argsort(S)[::-1][:chi_new]  # keep the largest chi_new singular values
            #if S.size - chi_new == 0:
            #    print("singular values cut: 0.")
            #else:
            #    print("singular values cut:", np.sum(S[(S.size-chi_new)::]))
            U, S, V = U[:, piv], S[piv], V[piv, :]
            # renormalize
            norm = np.linalg.norm(S)
            S = S / norm
            self.norm *= norm
            # put back and update B
            V = np.reshape(V, (chi_new, chi_i, chi_vR)) # chi_new, (i, vR) -> chi_new, i, vR
            self.Bs[i] = V
            B = self.Bs[i-1]
            B = np.tensordot(B, U, ([2], [0])) # vL i [vR]; [vL] chi_new -> vL i chi_new
            B = np.tensordot(B, np.diag(S), ([2], [0])) # vL i [chi_new]; [chi_new] chi_new -> vL i chi_new
            self.Ss[i] = S
        self.Bs[0] = B
        self.Ss[0] = np.array([1.])
        self.canonical = True
      
    @staticmethod
    def initialize_spinup(L):
        """Returns a product state with all spins up as an MPS"""
        B = np.zeros([1, 2, 1], float)
        B[0, 0, 0] = 1.
        S = np.ones([1], float)
        Bs = [B.copy() for i in range(L)]
        Ss = [S.copy() for i in range(L)]
        return MPS(Bs, Ss)
    
    @staticmethod
    def initialize_from_state_vector(psi, L, chi_max=100, eps=0, d=2):
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
        d : int
            the dimension of the local Hilbert space on each site,
            eg. d=2 for spin-1/2.
            
        Returns
        -------
        psi_mps : MPS
            the state compressed into an MPS
        """
        # first, reshape the state into a single column vector (if its not already in this form)
        psi_aL = np.reshape(psi, (d**L, 1))
        Bs = [None] * L
        Ss = [None] * L 
        norm = 1.
        # now iterate over the sites of the chain
        for n in range(L-1, -1, -1):
            # compute Chi_n and R_dim. Chi_n * 2 will be the "dimension" d^(L_a) of subsystem A, 
            # R_dim//2 will be the "dimension" d^(L_b) of subsystem B
            L_dim, Chi_n = psi_aL.shape
            assert L_dim == d**(n+1)
            # Reshape wavefunction
            psi_LR = np.reshape(psi_aL, (L_dim//2, Chi_n*2))
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
            M_n = np.reshape(M_n, (Chi_np1, d, Chi_n))
            # reabsorb lambda
            psi_aL = psitilde_n[:,:] * lambda_n[np.newaxis, :]
            Bs[n] = M_n
            Ss[n] = lambda_n
        assert(psi_aL.shape == (1,1))
        assert(Ss[0].size == 1)
        norm *= Ss[0].item() * psi_aL.item()
        Bs[0] /= Ss[0].item()
        Ss[0] = np.array([1.])
        result = MPS(Bs, Ss)
        result.norm = norm
        return result
    
    def to_state_vector(self):
        """Contracts the MPS to form a state vector in Hilbert space"""
        contr = self.Bs[0][0] # i vR
        for i in range(1, self.L):
            contr = np.tensordot(contr, self.Bs[i], ([1], [0])) # i [vR]; [vL] j vR -> i j vR
            contr = np.reshape(contr, (contr.shape[0]*contr.shape[1], contr.shape[2])) # i j vR -> i' vR
        return self.norm * contr[:, 0]

    @staticmethod
    def init_HOMPS_MPS(psi0, N_bath, N_trunc):
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
        
        Returns
        -------
        MPS :
            initial MPS for the HOPS algorithm. The first tensor Bs[0] has
            shape (1, d, 1), and all others have shape (1, N_trunc, 1).
            In total there are (N_bath + 1) tensors in the MPS
        """
        B_physical = np.zeros([1, psi0.size, 1], dtype=complex)
        B_physical[0, :, 0] = psi0
        B_bath = np.zeros([1, N_trunc, 1], dtype=complex)
        B_bath[0, 0, 0] = 1.
        Bs = [B_bath.copy() for i in range(N_bath)]
        Bs.insert(0, B_physical.copy())
        S = np.ones([1], dtype=float)
        Ss = [S.copy() for i in range(N_bath + 1)]
        return MPS(Bs, Ss)

def split_truncate_theta(theta, chi_max, eps, normalize=True):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.
    normalize : bool
        Wether to normalize the singular values after truncation

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    #print(theta.shape)
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    X, Y, Z = svd(theta, full_matrices=False, lapack_driver='gesvd')
    #print(Y)
    # truncate
    chivC = min(chi_max, np.sum(Y > eps))
    assert chivC >= 1
    piv = np.argsort(Y)[::-1][:chivC]  # keep the largest `chivC` singular values
    X, Y, Z = X[:, piv], Y[piv], Z[piv, :]
    # renormalize
    S = Y
    if normalize:
        S = Y / np.linalg.norm(Y)  # == Y/sqrt(sum(Y**2))
    # split legs of X and Z
    A = np.reshape(X, [chivL, dL, chivC])
    B = np.reshape(Z, [chivC, dR, chivR])
    return A, S, B
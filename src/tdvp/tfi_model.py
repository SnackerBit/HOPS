"""Toy code implementing the transverse-field ising model."""

import numpy as np

class TFIModel:
    """Class generating the Hamiltonian of the transverse-field Ising model.

    The Hamiltonian reads
    .. math ::
        H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i

    Parameters
    ----------
    N : int
        Number of sites.
    J, g : float
        Coupling parameters of the above defined Hamiltonian.

    Attributes
    ----------
    N : int
        Number of sites.
    d : int
        Local dimension (=2 for spin-1/2 of the transverse field ising model)
    sigmax, sigmay, sigmaz, id :
        Local operators, namely the Pauli matrices and identity.
    H_bonds : list of np.Array[ndim=4]
        The Hamiltonian written in terms of local 2-site operators, ``H = sum_i H_bonds[i]``.
        Each ``H_bonds[i]`` has (physical) legs (i out, (i+1) out, i in, (i+1) in),
        in short ``i j i* j*``.
    """

    def __init__(self, N, J, g):
        self.N, self.d = N, 2
        self.J, self.g = J, g
        self.sigmax = np.array([[0., 1.], [1., 0.]])
        self.sigmay = np.array([[0., -1j], [1j, 0.]])
        self.sigmaz = np.array([[1., 0.], [0., -1.]])
        self.id = np.eye(2)
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian. Called by __init__()."""
        sx, sz, id = self.sigmax, self.sigmaz, self.id
        d = self.d
        H_list = []
        for i in range(self.N - 1):
            gL = gR = 0.5 * self.g
            if i == 0: # first bond
                gL = self.g
            if i + 1 == self.N - 1: # last bond
                gR = self.g
            H_bond = -self.J * np.kron(sx, sx) - gL * np.kron(sz, id) - gR * np.kron(id, sz)
            # H_bond has legs ``i, j, i*, j*``
            H_list.append(np.reshape(H_bond, [d, d, d, d]))
        self.H_bonds = H_list

    def energy(self, psi):
        """Evaluate energy E = <psi|H|psi> for the given MPS."""
        assert psi.L == self.N
        return np.sum(psi.bond_expectation_value(self.H_bonds))
    
    def init_H_mpo(self):
        # shape of W: bond index left, bond index right, local physical index ket, local physical index bra
        # or in short: bl, br, p1, p2
        W = np.zeros(shape=(3, 3, self.d, self.d)) 
        W[0, 0, :, :] = self.id
        W[0, 1, :, :] = self.sigmax
        W[0, 2, :, :] = -self.g * self.sigmaz
        W[1, 2, :, :] = -self.J * self.sigmax
        W[2, 2, :, :] = self.id
        self.H_mpo = [W.copy() for i in range(self.N)]

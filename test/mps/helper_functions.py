"""
This file implements some helper functions to test the MPS implementation
"""

from scipy import sparse
import scipy.sparse.linalg
import numpy as np

def singlesite_to_full(op, i, L):
    Id = sparse.csr_matrix(np.eye(2))
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full

def generate_TFI_Hamiltonian(L, J, g):
    Id = sparse.csr_matrix(np.eye(2))
    Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
    Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])

    def gen_sx_list(L):
        return [singlesite_to_full(Sx, i, L) for i in range(L)]

    def gen_sz_list(L):
        return [singlesite_to_full(Sz, i, L) for i in range(L)]

    def gen_hamiltonian(sx_list, sz_list, g, J=1.):
        L = len(sx_list)
        H = sparse.csr_matrix((2**L, 2**L))
        # OPEN boundary conditions!
        #for j in range(L):''
        for j in range(L-1):
            #H = H - J *( sx_list[j] * sx_list[(j+1)%L])
            H = H - J *( sx_list[j] * sx_list[j+1])
            H = H - g * sz_list[j]
        # we still need to add sigma_z on the last site!
        H = H - g * sz_list[L-1]
        return H
    
    return gen_hamiltonian(gen_sx_list(L), gen_sz_list(L), g, J)

def overlap(bra, ket):
    L = len(bra.Bs)
    assert(L == len(ket.Bs))
    left = np.ones(1)
    left = np.reshape(left, (1, 1))
    for n in range(L):
        # Contract left with bra
        left = np.tensordot(left, ket.Bs[n], (0, 0)) # [vR] vR'; [vL] vR i -> vR' vR i
        # Contract left with ket
        left = np.tensordot(left, bra.Bs[n].conj(), [(0, 2), (0, 2)]) # [vR'] vR [i]; [vL'] vR' [i'] -> vR vR'
    return left[0, 0]


def uncompress(psi):
    contr = psi.Bs[0][0] # i VR
    for i in range(1, psi.L):
        contr = np.tensordot(contr, psi.Bs[i], ([1], [0])) # i [vR]; [vL] j vR -> i j vR
        contr = np.tensordot(contr, (contr.shape[0]*contr.shape[1], contr.shape[2])) # i j vR -> i' vR
    return contr[:, 0]

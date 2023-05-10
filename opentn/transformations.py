"""A module containing the transformations between Open Quantum Systems representations and the corresponding helping methods"""

import numpy as np
from scipy.linalg import expm


def lindbladian2super(H:np.ndarray = None, Li:list[np.ndarray] = [], dim:int=2)->np.ndarray:
    """
    Convert from lindbladian master equation to Liouvillian superoperator representation
    
    Note: No exponential included. A row-wise vectorization is assumed

    args:
    ---------
    H:
        Hamiltonian operator
    Li: 
        List of jump operators
    dim:
        Hilbert space dimension of the operators in H and Li
        i.e. H in Hilbert space of dimension dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts

    returns:
    ---------
        Superoperator representation of the quantum channel
    """
    super = np.zeros(shape=(dim ** 2, dim ** 2),dtype=complex)
    if H:
         super += vectorize_hamiltonian(H=H, dim=dim)
    for L in Li:
        super += vectorize_dissipative(L=L, dim=dim)

    return super

def super2choi(super:np.ndarray, dim:int=2)->np.ndarray:
    """
    Convert Superoperator to choi matrix. 
    
    Assumed to be acting on exp(super(lindbladian)). A row-wise vectorization is assumed
    args:
    ---------
    super:
        Superoperator representation of the quantum channel
    dim:
        Hilbert space dimension of the operators in H and Li
        i.e. H in Hilbert space of dimension dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts

    returns:
    ---------
        Choi Matrix representation of the quantum channel
    """
    choi = np.reshape(super, [dim] * 4)
    choi = choi.swapaxes(1, 2).reshape([dim ** 2, dim ** 2]) #see graphical proof
    return choi

def choi2kraus(choi:np.ndarray, tol:float = 1e-9)->list[np.ndarray]:
    """
    Convert a choi matrix to its kraus representation

    A tolerance is needed to avoid including small eigenvalues that are virtually zero.
    A cholesky decomposition could be used if we could get rid of the small negative 
    eigenvalues in advance.

    args:
    ---------
    choi:
        Choi Matrix representation of the quantum channel
    tol:
        Tolerance used to throw away small eigenvalues after eigendecomposition.

    returns:
    ---------
        List of kraus operators corrresponding to the quantum channel    
    """
    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus_list = [np.sqrt(eigval) * unvectorize(vec) for eigval, vec in
            zip(eigvals, eigvecs.T) if abs(eigval) > tol]
    return kraus_list


def lindbladian2kraus(H:np.ndarray = None, Li:list[np.ndarray] = [], tau:int = 1, tol:float = 1e-9)->list[np.ndarray]:
    """
    Convert operators from the lindbladian representation to a list of kraus operators
    
    Initial operators include Hamiltonian and jump operators

    args:
    ---------
    H:
        Hamiltonian operator
    Li: 
        List of jump operators
    tau:
        Time step for exponential of liouvillian (super)operator
    tol:
        Tolerance to determine which eigenvalues to take into account from diagonalization of choi matrix

    returns:
    ---------
    kraus_list:
        List contain the kraus operators corresponding to the lindbladian operators
    """
 
    # assume H and Li elements have same shape if both are passed
    if H:
        dim = H.shape[0]
    elif Li:
        dim = Li[0].shape[0]

    # TODO: generalize so that we don't have only the same level at all sites
    super = lindbladian2super(H=H, Li=Li, dim=dim)
    super_exp = expm(super*tau)

    choi = super2choi(super=super_exp, dim=dim)
    return choi2kraus(choi=choi)


def vectorize(matrix:np.ndarray)->np.array:
    "vectorize matrix in a row-wise order"
    return matrix.flatten()  

def unvectorize(vector:np.ndarray)->np.ndarray:
    "unvectorize vector in row-wise manner. Square matrix assumed"
    dim = int(np.sqrt(vector.size))
    matrix = vector.reshape((dim,dim))
    return matrix

def vectorize_hamiltonian(H:np.ndarray, dim:int=2)->np.ndarray:
    "Vectorize Hamiltonian.  Here we assume the i is included in the hamiltonian"
    I = np.eye(dim, dtype=complex)
    return np.kron(H, I) - np.kron(I, H.T)

def vectorize_dissipative(L:np.ndarray, dim:int=None)->np.ndarray:
    "Vectorize dissipative part of lindbladian"
    if not dim:
        dim = L.shape[0]
    L_vec = np.zeros(shape=(dim ** 2, dim ** 2),dtype=complex)
    I = np.eye(dim, dtype=complex)
    L_vec =  np.kron(L, L.conj()) - 0.5*np.kron(L.T.conj()@L, I) - 0.5*np.kron(I, L.T@L.conj())
    return L_vec

def op2fullspace(op:np.ndarray, i:int, N:int, num_sites:int=1):
    "convert a local operator into an operator in the full hilbert space"
    assert 0<=i<N, "i needs to be greater or equal than 0 and smaller than N!"
    assert N > 0, "N needs to be greater than 0!"
    if N == 1:
        return op
    id1 = np.eye(2**i)
    id2 = np.eye(2**(N-i-num_sites))
    return np.kron(id1, np.kron(op, id2))

def dissipative2liouvillian_full(L:np.ndarray, i:int, N:int, num_sites:int=1):
    "convert local lindbladian term to an operator in full Hilbert"
    L_full = op2fullspace(op=L, i=i, N=N, num_sites=num_sites)
    L_full_vec = vectorize_dissipative(L_full)
    return L_full_vec

def ket2dm(vec:np.ndarray)->np.ndarray:
    "Convert a statevector into a density matrix through outer product"
    assert len(vec.shape) == 1 or 1 in vec.shape, "vec is not a vector"
    return np.outer(vec, vec.conj())

def find_nonzero(matrix:np.ndarray):
    "print all the non zero elements and indices of a matrix"
    for i,j in zip(*np.nonzero(matrix)):
        print(f'{i}, {j}: {matrix[i,j]}')
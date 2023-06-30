"""
A module containing the transformations between Open Quantum Systems representations and the 
corresponding helping methods
"""

import scipy
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from opentn.states.qubits import get_ladder_operator

def exp_operator_dt(op:np.ndarray, tau:float=1, library='jax')->np.ndarray:
    "exponentiate operator with a certain time step size tau"
    if library == 'scipy':
        exp_op = scipy.linalg.expm(op*tau)
    elif library == 'jax':
        exp_op = jscipy.linalg.expm(op*tau)
    else:
        raise ValueError(f'{library} is not a valid library string')
    return exp_op

def lindbladian2super(H:np.ndarray = None, Li:list[np.ndarray] = [], dim:int=None)->np.ndarray:
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
    if not dim:
        if H:
            dim = H.shape[0]
        elif Li:
            dim = Li[0].shape[0]
    super = np.zeros(shape=(dim ** 2, dim ** 2),dtype=complex)
    if H:
         super += vectorize_hamiltonian(H=H, dim=dim)
    for L in Li:
        super += vectorize_dissipative(L=L, dim=dim)

    return super

def super2choi(super:np.ndarray, dim:int=None)->np.ndarray:
    """
    Convert Superoperator to choi matrix. 
    
    'Super' is assumed to be actually on exp(super(lindbladian)). 
    A row-wise vectorization is assumed

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
    if not dim:
         dim = int(np.sqrt(super.shape[0]))
    choi = np.reshape(super, [dim] * 4)
    choi = choi.swapaxes(1, 2).reshape([dim ** 2, dim ** 2]) #see graphical proof
    return choi

def choi2super(choi:np.ndarray, dim:int=None)->np.ndarray:
    """
    Convert a Choi Matrix to its superoperator representation. 
    
    'Super' is assumed to be actually on exp(super(lindbladian)). 
    A row-wise vectorization is assumed

    TODO: add argument N = sites. right now if dim is not passed, we assume
    that we have only 1 site with dim = sqrt(choi.shape[0]). 
    in reality we should satisfy the condition dim**(2*N) = choi.shape[0]

    args:
    ---------
    Choi:
       Choi Matrix representation of the quantum channel
    dim:
        Hilbert space dimension of the operators in H and Li
        i.e. H in Hilbert space of dimension: dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts

    returns:
    ---------
        Superoperator representation of the quantum channel
    """
    if not dim:
        dim = int(np.sqrt(choi.shape[0]))
    assert choi.shape[0] == dim
    super = np.reshape(choi, [dim] * 4)
    super = super.swapaxes(1, 2).reshape([dim ** 2, dim ** 2]) #see graphical proof
    return super

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
    print([eig for eig in eigvals if abs(eig)>tol])
    kraus_list = [np.sqrt(eigval) * unvectorize(vec) for eigval, vec in
            zip(eigvals, eigvecs.T) if abs(eigval) > tol]
    return kraus_list

def kraus2choi(kraus_list:list[np.ndarray])->np.ndarray:
    """
    Convert a list of Kraus operators into its choi matrix representation

    row-wise vectorization assumed. 
    Choi = \sum_k |Ek>> <<Ek|
    """
    if isinstance(kraus_list, np.ndarray):  # handle input of single kraus op
        if len(kraus_list[0].shape) < 2:
            kraus_list = [kraus_list]

    return sum([vectorize(Ek) @ vectorize(Ek).conj().T for Ek in kraus_list])


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
    super_exp = scipy.linalg.expm(super*tau)

    choi = super2choi(super=super_exp, dim=dim)
    return choi2kraus(choi=choi)


def vectorize(matrix:np.ndarray)->np.array:
    "vectorize matrix in a row-wise order"
    return matrix.reshape(-1,1) # column vector  

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

def factorize_psd(psd:np.ndarray, check_hermitian:bool=False, tol:float=1e-9):
    """
    factorize a positive-semidefinite-matrix (psd) into its "square root" matrix
    B and its adjoint.
    
    i.e. psd = B @ B.conj().T

    args:
    ---------
    psd:
        PSD matrix to be factorized. To save compt. speed, we will not check that this
        matrix is indeed psd. Checking hermicity is optional.
    check_hermitian:
        Wether to check if the input matrix `psd` is hermitian. np.linalg.eigh would fail if not.
    tol:
        Eigenvalues with absolute value below this number will be regarded as 0.
   
    returns:
    ---------
    B:
        "Square root" matrix of input PSD matrix.
    """
    if check_hermitian:
        assert np.allclose(psd, psd.conj().T), "Matrix is not hermitian"
    eigvals, eigvecs = np.linalg.eigh(psd)
    B = np.empty_like(eigvecs)
    for i, eig in enumerate(eigvals):
        if abs(eig) < tol:
            eig = 0
        B[:,i] = eigvecs[:,i]*np.sqrt(eig)
    return B

def create_kitaev_liouvillians(N, d, gamma):
    ""
    lowering = get_ladder_operator()
    raising = get_ladder_operator(adjoint=True)
    NN = 2
    
    Lnn = jnp.sqrt(gamma)*(op2fullspace(raising, i=0, N=NN) + op2fullspace(raising, i=1, N=NN))@(op2fullspace(lowering, i=0, N=NN) - op2fullspace(lowering, i=1, N=NN))/4

    Lvec = jnp.zeros(shape=(d**(2*N), d**(2*N)), dtype=complex)
    for i in range(0, N-1):
        Lvec += dissipative2liouvillian_full(L=Lnn, i=i, N=N, num_sites=2)    
    
    Lvec_odd = jnp.zeros(shape=(d**(2*N), d**(2*N)), dtype=complex)
    for i in range(0, N, 2):
        Lvec_odd += dissipative2liouvillian_full(L=Lnn, i=i, N=N, num_sites=2)

    Lvec_even = jnp.zeros(shape=(d**(2*N), d**(2*N)), dtype=complex)
    for i in range(1, N-1, 2):
        Lvec_even += dissipative2liouvillian_full(L=Lnn, i=i, N=N, num_sites=2)

    return Lvec, Lvec_odd, Lvec_even, Lnn
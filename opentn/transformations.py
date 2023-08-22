"""
A module containing the transformations between Open Quantum Systems representations and the
corresponding helping methods
"""

import scipy
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from itertools import chain
import cvxpy as cp
from scipy import sparse
from opentn.states.qubits import get_ladder_operator
from itertools import chain

from jax import config
config.update("jax_enable_x64", True)

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

def super2choi(superop:np.ndarray, dim:int=None)->np.ndarray:
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
        superop and choi should have dimensions = d^2n  x d^2n

    returns:
    ---------
        Choi Matrix representation of the quantum channel
    """
    if not dim:
         dim = int(np.sqrt(superop.shape[0]))
    assert superop.shape[0] == dim**2
    choi = np.reshape(superop, [dim] * 4)
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
        superop and choi should have dimensions = d^2n  x d^2n
    returns:
    ---------
        Superoperator representation of the quantum channel
    """
    if not dim:
        dim = int(np.sqrt(choi.shape[0]))
    assert choi.shape[0] == dim**2
    superop = np.reshape(choi, [dim] * 4)
    superop = superop.swapaxes(1, 2).reshape([dim ** 2, dim ** 2]) #see graphical proof
    return superop

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
    # print([eig for eig in eigvals if abs(eig)>tol])
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

    choi = super2choi(superop=super_exp, dim=dim)
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
    X:
        "Square root" matrix of input PSD matrix.
    TODO: apply what prof. Mendl said about a parameter that determines how big will the B  be.
    meaning how many eigenvalues will we keep such that: psd = X @ X+
    but with less columns in X.
    """
    if check_hermitian:
        assert np.allclose(psd, psd.conj().T), "Matrix is not hermitian"
    eigvals, eigvecs = np.linalg.eigh(psd)
    X = np.empty_like(eigvecs)
    for i, eig in enumerate(eigvals):
        if abs(eig) < tol:
            eig = 0
        with np.errstate(invalid='raise'):
            try:
                X[:,i] = eigvecs[:,i]*np.sqrt(eig)
            except:
                raise ValueError(f'invalid eigenvalue found: {eig}. input is not PSD')
            # see: https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
    return X


def create_2local_liouvillians(Lnn:np.ndarray, N:int, d:int):
    "create the liouvillians from a local two-sites lindbladian operator"

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

def get_kitaev_nn_linbladian(gamma:float):
    "Two sites libladian operator from kitaev wire noise model"
    lowering = get_ladder_operator()
    raising = get_ladder_operator(adjoint=True)
    NN = 2
    Lnn = jnp.sqrt(gamma)*(op2fullspace(raising, i=0, N=NN) + op2fullspace(raising, i=1, N=NN))@(op2fullspace(lowering, i=0, N=NN) - op2fullspace(lowering, i=1, N=NN))/4
    return Lnn

def create_kitaev_liouvillians(N:int, d:int, gamma:float):
    "create the liouvillians corresponding to the kitaev wire noise model"
    Lnn = get_kitaev_nn_linbladian(gamma)
    Lvec, Lvec_odd, Lvec_even, Lnn = create_2local_liouvillians(Lnn, N, d)
    return Lvec, Lvec_odd, Lvec_even, Lnn


def create_trotter_layers(liouvillians:list[np.ndarray], tau:float=1, order:int=2):
    """
    Create troter layers of (order)th-order by exponentiating the liouvillians.

    NOTE: We assume liouvillians = [ Lvec, Lvec_odd, Levec_even ]
    """
    exp_superop = []
    if order==2:
        for i, op in enumerate(liouvillians):
            if i == 1:
                exp_superop.append(exp_operator_dt(op, tau/2, 'jax'))
            else:
                exp_superop.append(exp_operator_dt(op, tau, 'jax'))
    else:
        raise ValueError('ATM only order 2 is implemented')
    return exp_superop


def create_supertensored_from_local(localop:np.ndarray, N:int):
    "we asssume we have only one single operator tensored accross all sites to get the full superop"
    # NOTE: we also assuming that localop is acting on two sites
    superop = localop
    for _ in range(0, N//2-1):
        superop = jnp.kron(superop, superop)
    return superop

def get_indices_supertensored2liouvillianfull(N:int):
    """
    get the source and destination indices to compare full superoperator created from local superoperators tensored with the full superoperator created from exponentiating the full vectorized lindbladians.

    Here we assume that we are going from tensored-full-liouvillian
    NOTE: for opposite convention just interchange roles of source_full <-> destination_full
    """
    # create indices to swap for a single side of the superoperator
    source_one_side = list(chain.from_iterable((2 + i*4, 3 + i*4) for i in range((N-2)//2)))
    destination_one_side = [i for i in range(N, 2*N-2)]
    # create full indices including both sides of superoperator
    source_full = source_one_side + list(np.array(source_one_side) + 2*N)
    destination_full = destination_one_side + list(np.array(destination_one_side) + 2*N)
    return source_full, destination_full

def swap_superop_indices(suoperop:np.ndarray, source_indices:list[int], destination_indices:list[int], N:int, d:int):
    "swap the indices of the superoperator"
    swaped_superop = jnp.reshape(suoperop, newshape=[d]*4*N)
    swaped_superop = jnp.moveaxis(swaped_superop, source=source_indices, destination=destination_indices)
    swaped_superop = jnp.reshape(swaped_superop, newshape=suoperop.shape)
    return swaped_superop

def convert_supertensored2liouvillianfull(local_tensored:np.ndarray, N:int, d:int):
    """
    convert full superoperator created from local superoperators tensored to the full
    superoperator created from exponentiating the full vectorized lindbladians
    """
    source_indices, destination_indices = get_indices_supertensored2liouvillianfull(N)
    return swap_superop_indices(local_tensored, source_indices, destination_indices, N, d)


def convert_liouvillianfull2supertensored(full_liouvillian:np.ndarray, N:int, d:int):
    """
    convert the full superoperator created from exponentiating the full vectorized lindbladians to the full superoperator created from local superoperators tensored
    """
    destination_indices, source_indices = get_indices_supertensored2liouvillianfull(N)
    return swap_superop_indices(full_liouvillian, source_indices, destination_indices, N, d)


def partial_transpose(op:np.ndarray, dims:list[int], idx:int=0):
    """
    Partial transpose of op over subsytem with index idx

    We assume the operator is made up of n subsystems. Each with dimension dims[i]

    args:
    ---------
    op:
        operator over which to perform partial transpose
    dims:
        list of dimensions fo the n subystems composing op.
    idx:
        index of the subsytem over which to perform the partial transpose

    returns:
    ---------
       operator with the same dimensions as op but with idx subsytem transposed
    """
    assert op.shape[0] == op.shape[1], 'op should be a sqaure matrix'
    assert op.shape == (np.prod(dims),)*2, f'dimensions {dims} do not match operator dimensions {op.shape}'
    assert 0 <= idx < len(dims), 'idx out of range of dimensions'
    return np.reshape(op, dims*2).swapaxes(idx, idx + len(dims)).reshape([np.prod(dims)]*2)

def partial_trace(op:np.ndarray, dims:list[int], idx:int=0):
    """
    partial trace of op over subsystem idx.

    We assume the operator is made up of n subsystems. Each with dimension dims[i]

    args:
    ---------
    op:
        operator over which to perform partial trace
    dims:
        list of dimensions fo the n subystems composing op.
    idx:
        index of the subsytem over which to perform the partial trace

    returns:
    ---------
       operator with the system idx traced out
    """
    assert op.shape[0] == op.shape[1], 'op should be a sqaure matrix'
    assert op.shape == (np.prod(dims),)*2, f'dimensions {dims} do not match operator dimensions {op.shape}'
    assert 0 <= idx < len(dims), 'idx out of range of dimensions'
    # TODO: fix this because we should np.prod of only the dimensions that are left without idx.
    op_traced = op.reshape(dims*2).trace(axis1=idx,axis2=idx + len(dims))
    del dims[idx]
    return op_traced.reshape([np.prod(dims)]*2)

def link_product(C1:np.ndarray, C2:np.ndarray, dim:int=None, transpose:int=0)->np.ndarray:
    """
    link product is the composition of two individual choi matrices C1 and C2

    We assume that C2 is applied after C1, i.e. C_12 = C2 o C1 (from right to left)

    sources:
    * https://arxiv.org/pdf/0904.4483.pdf
    * https://quantumcomputing.stackexchange.com/questions/10126/explicit-form-for-composition-of-choi-representation-quantum-channels/14586#14586

    NOTE: here we assume that C1 and C2 act over spaces of same dimensions

    args:
    ---------
    C1:
        PSD choi matrix representing the first linear operator to be applied
    C2:
        PSD choi matrix representing the second linear operator to be applied (i.e. after C1)
    dim:
        Hilbert space dimension over which C1 and C2 act.
        i.e. rho (density matrix) is in a Hilbert space of dimension: dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts
        chois should have dimensions = d^2n  x d^2n
    transpose:
        whether the first [0] or second [1] channel will be the one transposed in the expression

    returns:
    ---------
    C_12:
        choi representation of the composition of C1 and C2
    """
    assert C1.shape == C2.shape, 'dimensions of C1 and C2 do not coincide'
    if not dim:
        dim = int(np.sqrt(C1.shape[0]))

    IC, IA = np.eye(dim), np.eye(dim)


    if transpose == 0:
        # partial transpose over system B for first choi
        C1_TB = partial_transpose(C1, dims=[dim, dim], idx=0)
        # in theory the order of matrix multiplication here could be inverted since we have a trace at the end.
        C_12 = np.kron(IC,C1_TB) @ np.kron(C2, IA)
    elif transpose == 1:
        C2_TB = partial_transpose(C2, dims=[dim, dim], idx=1)
        C_12 = np.kron(C2_TB,IA) @ np.kron(IC, C1)
    else:
        raise ValueError('only 0, 1 are allowed as tranpose values')
    # partial trace over system B for full expression
    C_12 = partial_trace(C_12, dims=[dim, dim, dim], idx=1)
    return C_12

def choi_composition(C1:np.ndarray, C2:np.ndarray, dim:int=None)->np.ndarray:
    """
    Choi matrix of the composition of two individual choi matrices C1 and C2

    Should be equivalent to link_product()
    We assume that C2 is applied after C1.

    args:
    ---------
    C1:
        PSD choi matrix representing the first linear operator to be applied
    C2:
        PSD choi matrix representing the second linear operator to be applied (i.e. after C1)
    dim:
        Hilbert space dimension over which C1 and C2 act.
        i.e. rho (density matrix) is in a Hilbert space of dimension: dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts
        chois should have dimensions = d^2n  x d^2n

    returns:
    ---------
    C_12:
        choi representation of the composition of C1 and C2
    """
    assert C1.shape == C2.shape, 'dimensions of C1 and C2 do not coincide'
    # recall that we are trying to mimic the contrations that happen when multiplying two superoperators
    if not dim:
        dim = int(np.sqrt(C1.shape[0]))
    C_12 = jnp.tensordot(C2.reshape([dim]*4), C1.reshape([dim]*4), axes=[(1,3),(0,2)]) # out_j (in_j) out_j* (in_j*), (out_i) in_i (out_i*) in_i* -> out_j out_j* in_i in_i*
    C_12 = C_12.transpose((0,2,1,3)).reshape([dim**2]*2) # out_j in_i out_j* in_i*
    return C_12

def link_product_cvxpy(C1, C2, dim:int=None, transpose:int=0, optimization:bool=True):
    """
    link product but using only cvxpy atomic functions

    args:
    ---------
    C1:
        PSD choi matrix representing the first linear operator to be applied
    C2:
        PSD choi matrix representing the second linear operator to be applied (i.e. after C1)
    dim:
        Hilbert space dimension over which C1 and C2 act.
        i.e. rho (density matrix) is in a Hilbert space of dimension: dim x dim = d^n  x d^n
        with d the local dimension and n the number of sites over which it acts
        chois should have dimensions = d^2n  x d^2n
    transpose:
        whether the first [0] or second [1] channel will be the one transposed in the expression
        In addition, this determines implicitely which one of C1 and C2 is the variable and
        whcich one is the constant.
    optimization:
        whether we are using this expression within an optimization. If True, this would enforce
        the transpose to be applied strictly on the constant and not on the variable.

    returns:
    ---------
    C_12:
        cvxpy expression corresponding to the composition of C1 and C2

    TODO: does using sparse identity in cp.kron help in the speed of something?
    """


    I = np.eye(dim)
    if transpose == 0:
        # C1 is the constant and C2 is the variable
        if optimization:
             assert type(C1) != cp.expressions.variable.Variable, 'C1 is a variable. Only transpose constants to decrease compilation time'
        C1_tb = partial_transpose(C1, dims=[dim, dim], idx=0)
        IxC1_tb = np.kron(I, C1_tb)
        IxC1_tb = sparse.csr_matrix(IxC1_tb)
        IxC1_tb = IxC1_tb.astype(np.float64)
        C_12 =  cp.partial_trace(IxC1_tb @ cp.kron(C2, I), dims=[dim, dim, dim], axis=1)
    elif transpose == 1:
        # C1 is the variable and C2 is the constant
        if optimization:
            assert type(C2) != cp.expressions.variable.Variable, 'C2 is a variable. Only transpose constants to decrease compilation time'
        C2_tb = partial_transpose(C2, dims=[dim, dim], idx=1)
        C2_tbxI = np.kron(C2_tb, I)
        C2_tbxI = sparse.csr_matrix(C2_tbxI)
        C2_tbxI = C2_tbxI.astype(np.float64)
        C_12 =  cp.partial_trace(C2_tbxI @ cp.kron(I, C1), dims=[dim, dim, dim], axis=1)
    else:
        raise ValueError('only 0, 1 are allowed as tranpose values')
    return C_12

def choi_composition_3Y_cvxpy(C1, C2, C3, dim:int):
    """
    calculate the composition of 3 choi channels. This corresponds to the 3 layers in the 2nd order troter decomposition
    """
    I = sparse.eye(dim)
    C_13_tbtc = np.kron(partial_transpose(C3, dims=[dim]*2, idx=1), partial_transpose(C1, dims=[dim]*2, idx=0))
    C_13_tbtc = sparse.csr_matrix(C_13_tbtc).astype(np.float64)
    C_123 = cp.partial_trace(cp.partial_trace(cp.kron(cp.kron(I, C2),I) @ C_13_tbtc, dims=[dim]*4, axis=1), dims=[dim]*3, axis=1)
    return C_123
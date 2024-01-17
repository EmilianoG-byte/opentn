"""
A module containing the functions used for the optimization of a cost function defined
in the Stiefel manifold.
"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from opentn.transformations import factorize_psd_truncated, vec2list, unvectorize

from scipy.linalg import fractional_matrix_power, polar

def random_psd(dim:int, normalized:bool=True):
    "generate a random psd matrix with trace 1 if normalized =  True"
    psd_sqrt = np.random.rand(dim, dim)
    psd = psd_sqrt @ psd_sqrt.T.conj()
    if normalized:
        psd /= np.trace(psd)
    return psd

def project(X:np.ndarray, Z:np.ndarray):
    """
    Project Z matrix onto the tangent space of matrix X in Stiefel manifold

    From https://arxiv.org/pdf/2112.05176.pdf eq.26
    """
    # TODO: add back the conj
    return Z - X @ symmetrize(X.T @ Z)

def symmetrize(A:np.ndarray):
    """
    Symmetrize a matrix by projecting it onto the symmetric subspace.
    """
    # TODO: add back the conj
    return 0.5 * (A + A.T)


def antisymmetrize(A:np.ndarray):
    """
    Antisymmetrize a matrix by projecting it onto the antisymmetric (skew-symmetric) subspace.
    """
    # TODO: add back the conj
    return 0.5 * (A- A.T)

def square_to_upper_triangular(a:np.ndarray)->np.ndarray:
    "get the upper triangular of matrix `a` as a vector (does not include main diagonal)"
    return a[np.triu_indices_from(a, k=1)]

def upper_triangular_to_antisymmetric(upper:np.ndarray, p:int)->np.ndarray:
    "get the antisymmetric matrix of shape (p, p) from the upper triangular (as a 1D array)"
    assert upper.ndim == 1
    assert len(upper) == antisymmetric_dof(p=p), f"length of upper:{len(upper)} does not match with p:{p}"
    a = np.zeros((p, p))
    a[np.triu_indices(p, k=1)] = upper
    a -= a.T
    return a

def antisymmetric_dof(x:np.ndarray=None, p:int=None)->int:
    "Degrees of freedom of the antisymmetric component of the tangent space from x of the stiefel manifold"
    if not p:
        n, p = x.shape
    return p * (p - 1) // 2 # this is guaranteed to be an integer since p*(p-1) is always divisble by 2

def arbitrary_dof(x:np.ndarray=None, n:int=None, p:int=None)->int:
    "Degrees of freedom of the arbitrary component of the tangent space from x of the stiefel manifold"
    if not n or not p:
        n, p = x.shape
    return p * (n - p)

def stiefel_tangent_dof(x:np.ndarray)->int:
    "Degrees of freedom of the tangent space from x of the stiefel manifold"
    return antisymmetric_dof(x) + arbitrary_dof(x)

def canonical_metric(delta1:np.ndarray, delta2:np.ndarray, x:np.ndarray)->float:
    """
    Riemannian metric between delta1 and delta2 in tangent space of matrix x in Stiefel manifold. 
    From https://arxiv.org/abs/2112.05176 eq. 24
    """
    n = x.shape[0]
    gamma = np.eye(n) - 0.5 * (x@x.conj().T)
    return np.trace(delta1.conj().T @ gamma @ delta2).real

def euclidean_metric(delta1:np.ndarray, delta2:np.ndarray)->float:
    """
    Constant trace metric between delta1 and delta2 in tangent space of matrix x in Stiefel manifold

    Euclidean/Embedded metric independent of x
    """
    return np.trace(delta1.conj().T @ delta2).real

def tuple2int(i:int, j:int, cols:int):
    "flatten tuple (i,j) row-wise to single integer"
    assert 0 <= j < cols, 'j out of range' # alternatively could use j%cols instead  of just j
    return i*cols + j

def int2tuple(k:int, cols:int):
    "unflatten integer k to tuple (i,j) assuming row-wise traverse of matrix"
    return k//cols, k%cols

def get_k_unit_matrix(dim0:int, dim1:int, k:int=0):
    "get the k-th Eij matrix of shape `dim0` x `dim1`, i.e. matrix with zero everywhere except at (i,j) where it is 1"
    assert 0<= k < dim0*dim1, f'k:{k} value is out of range. Should be in f[0,{dim0*dim1})'
    Ek = np.zeros(shape=(dim0, dim1), dtype=np.float64)
    Ek[0,0] = 1
    return np.roll(Ek, shift=k)

def get_ij_unit_matrix(dim0:int, dim1:int, i:int=0, j:int=0):
    "get the Eij matrix of shape `dim0` x `dim1`, i.e. matrix with zero everywhere except at (i,j) where it is 1"
    return get_k_unit_matrix(dim0, dim1, k=tuple2int(i,j,dim1))

def get_unit_matrices(ops:list[np.ndarray]):
    """
    Get a list of unit matrices (matrices with zeros everywhere except on the [0,0] element with a 1 instead)
    with the same shape as the list of ops.
    """
    unit_matrices = []
    for op in ops:
        unit_matrices.append(get_ij_unit_matrix(*op.shape))
    return unit_matrices

def get_elementary_antisymmetric(k:int, x:np.ndarray):
    """
    create k-th antisymmetric elementary tangent direction of tangent space at isometry x

    x @ (Eij^A - Eji^A), for Eij^A, Eji^A two p x p real matrices

    We have updated it such that the k only traverses the upper triangle, and from it we obtain
    the antisymmetric matrix: (Eij^A - Eji^A)
    """
    p = x.shape[1]
    
    Eij = np.zeros(antisymmetric_dof(x))
    Eij[k] = 1
    return x @ upper_triangular_to_antisymmetric(upper=Eij, p=p)


def get_elementary_arbitrary(k:int, x:np.ndarray):
    """
    create k-th arbitrary elementary tangent direction of tangent space at isometry x

    X_comp @ Eij, for Eij a (n-p) x p real matrix
    """
    x_comp = get_orthogonal_complement(x)
    n, p = x.shape
    Eij = get_k_unit_matrix(dim0=(n-p), dim1=p, k=k)
    return x_comp @ Eij

def get_elementary_tangent_direction(k:int, x:np.ndarray):
    """
    Get the elementary tangent direction Eij that span the tangent space at isometry x of St(n,p)

    For a tangent vector Z = X @ A + X_comp @ B in Tx(St(n,p)) the elementary tangent directions look like
    - X_comp @ Eij^B, for Eij^B a (n-p) x p real matrix
    - X @ (Eij^A-Eji^A), for Eij^A, Eji^A two p x p real matrices such that (Eij^A-Eji^A) is skew-hermitian (antisymmetric for real case)
    
    From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 equation 20

    From `parametrization_from_tangent` we see that A is stacked before B so we will follow this convention here as well
    when using the flattened index `k` to traverse the full space n x p.

    NOTE: it is missing to explore if these are also the elementary directions also for the euclidean metric and not only the canonical one.
    """

    assert 0 <= k < stiefel_tangent_dof(x), 'k it is out of range'

    if k < antisymmetric_dof(x):
        # antisymmetric
        return get_elementary_antisymmetric(k, x)
    else:
        # arbitrary. Need to shift down the k to be in [0, (n-p)*p] not in [p**2, p**2 + (n-p)*p]
        return get_elementary_arbitrary(k - antisymmetric_dof(x), x)

def get_orthogonal_complement(x:np.ndarray):
    """
    Get the orthogonal complement matrix of X such that

    [X, X_comp].conj().T @ [X, X_comp] = I

    see 'notation' in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 lemma 8
    """
    n, p = x.shape
    # TODO: add back the conj
    if n == p:
        # unitary case
        return np.zeros_like(x)
    P_x = x @ x.T 
    P_x_comp = np.eye(n) - P_x
    return factorize_psd_truncated(P_x_comp, chi_max=n-p, eps=1e-15) # adding eps=1e-15 to make sure rank is the predominant factor to keep singular values


def get_antisymmetric(X:np.ndarray, Z:np.ndarray):
    """
    Get the anti-symmetric (skew-hermitian) component A from a general matrix Z
    
    Decomposition Z = X @ A + X_comp @ B + X @ C

    From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 lemma 8.
    """
     # TODO: add back the conj
    return antisymmetrize(X.T@Z)

def get_arbitrary(X_comp:np.ndarray, Z:np.ndarray, X:np.ndarray=None, Px_comp:np.ndarray=None):
    """
    Get the arbitrary component B from a general matrix Z
    
    Decomposition Z = X @ A + X_comp @ B + X @ C

    From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 lemma 8.
    """
    if Px_comp is None and X is not None:
        n, p = X.shape
         # TODO: add back the conj
        Px_comp = np.eye(n) - X @ X.T
    elif Px_comp is None and X is None: 
        raise('at least one of X or Px_comp should be given')
     # TODO: add back the conj
    return X_comp.T @ Px_comp @ Z

def parametrizations_from_vector(param_vec:np.ndarray, shapes:list[tuple[int]])->np.ndarray:
    """
    Convert a long vector (stacked) containing all the parametrizations of the tangent spaces onto a list 
    of tuples with the individual parametrizations.

    args:
    ---------
    param_vec:
        long 1d-array containing the parametrization of all the tangent vectors.
        It contains np.hstack([A_upper, B.reshape(-1)]) for each of the x (stiefel operator) 
        corresponding to a shape in `shapes`
    shapes:
        list of shapes of each of the stiefel operators x. These are the manifolds at which the 
        parametrizations in `param_vec` are tangent.
    returns:
    ---------
        list of tuples containing the individual parametrizations (A,B) for each of the tangent vectors
    """
    # divide the full vector into a list of equally splitted len(shapes)-arrays
    params_vec_split = np.hsplit(param_vec, len(shapes)) # 
    # now I should divide each of the elements onto 2 vectors of size p**2, (n-p)p and unvectorize them
    params = []
    for i, shape in enumerate(shapes):
        n,p = shape
        # NOTE: changed here since we are only storing the upper triangle
        upper_size = antisymmetric_dof(p=p)
        A_upper = params_vec_split[i][:upper_size]
        B_vec = params_vec_split[i][upper_size:]
        params.append((upper_triangular_to_antisymmetric(A_upper, p=p), unvectorize(B_vec, n-p, p)))
    return params

def parametrization_from_tangent_square(x:np.ndarray, z:np.ndarray, vectorized:bool=True):
    """
    parametrization for tangent space of square stiefel manifold (unitary manifold) at x
    
    Z = x @ A 
    """
    A = get_antisymmetric(x, z)
    
    if vectorized:
        A_upper = square_to_upper_triangular(A)
        return A_upper
    else:
        return A

def parametrization_from_tangent(x:np.ndarray, z:np.ndarray, x_comp:np.ndarray = None ,stack:bool=True, vectorized:bool=True):
    """
    Get the A and B matrices that parametrize the Z matrix of the tangent space of stiefel manifold at x
    
    Z = x @ A + x_comp @ B
    
    with X_comp the orthogonal complement of X.

    From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 lemma 8.

    NOTE (update): If `stack=True`, it will now only store the upper diagonal part of A.
    This corresponds to the actual degrees of freedom in A for real-valued matrices.
    To be able to stack it with B, I also need to convert B to a vector and to hstack.

    Added `vectorized` flag to keep it compatible with the non-vectorized hessian implementation
    """
    n, p = x.shape
    if n==p:
        return parametrization_from_tangent_square(x, z, vectorized)
    else:
        if x_comp is None:
            x_comp = get_orthogonal_complement(x)

        A = get_antisymmetric(x, z)
        B = get_arbitrary(X_comp=x_comp, Z=z, X=x)
        
        if stack:
            if vectorized:
                A_upper = square_to_upper_triangular(A) # ndim = 1
                # how to stack different elements: https://stackoverflow.com/questions/33356442/when-should-i-use-hstack-vstack-vs-append-vs-concatenate-vs-column-stack
                return jnp.hstack([A_upper,B.reshape(-1)], dtype=np.float64)
            else:
                return jnp.vstack([A, B], dtype=np.float64)
        else:
            return A, B
    
def tangent_from_parametrization(X:np.ndarray, A:np.ndarray, B:np.ndarray)->np.ndarray:
    "Get the tangent vector corresponding to the parametrization A (skew symmetric) and B (arbitrary)"
    n,p = X.shape
    assert A.shape == (p,p), "Wrong shape for A"
    assert B.shape == (n-p, p), "Wrong shape for B"
    if n!=p:
        return X @ A + get_orthogonal_complement(X) @ B
    else:
        return X @ A

def polar_decomposition_stiefel(X:np.ndarray, Z:np.ndarray):
    """
    Retraction based on polar decomposition of Z at tangent space of matrix X from Stiefel manifold
    
    From https://assets.press.princeton.edu/chapters/absil/Absil_Chap4.pdf eq. 4.7
    """
    assert X.shape == Z.shape, "shapes don't match"
    d = X.shape[1]
    # TODO: add back the conj
    return (X+Z)@fractional_matrix_power(np.eye(d) + Z.T@Z, -0.5)

def polar_decomposition_rectangular(X:np.ndarray, Z:np.ndarray):
    """
    Retraction based on canonical polar decomposition of scipy
    
    [1] https://page.math.tu-berlin.de/~mehl/papers/hmt1.pdf
    [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html
    """
    return polar(X+Z)[0] 

def retract_stiefel(xi:list[np.ndarray], eta:np.ndarray):
    """
    Retraction of tangent vectors to stiefel manifold

    args:
    ---------
    xi:
        list of 3n - (n-1) = 2n + 1 operators of the stiefel manifold 
    eta: 
        long 1d-array containing the parametrization of all the tangent vectors
        each of which belongs to the tangent space of each of the `x` in `xi`
        NOTE: 'parametrizations_from_vector' and 'tangent_from_parametrization' 
        should be coherent with the parametrization used to create `eta`, 
        i.e. the parametrization used in `gradient_stiefel_vec` and `riemannian_hessian_vec`.
    returns:
    ---------
        list of operators belonging to the stiefel manifold obtained from:
        retraction(x[j] + tangent[j]) for each x,tangent coming from xi,eta (respectively)

    """
    params_list = parametrizations_from_vector(eta, shapes=[op.shape for op in xi]) 
    # unitaries: project the eta[j] onto the tangent space of the vlist[j].
    dxlist = [tangent_from_parametrization(x, *params) for x,params in zip(xi, params_list)]
    return [polar_decomposition_rectangular(x, z) for x,z in zip(xi, dxlist)] # stack gives problem with tree strcuture of jax

def is_isometry_2(x:np.ndarray)->bool:
    "check if `x` belongs to the stiefel manifold"
    return np.allclose(x.conj().T@x, np.eye(x.shape[1]))

def check_isometries(x_list:list[np.ndarray], show_idx:bool=False):
    "checks if the list of operators belong to the Stiefel manifold, i.e. are isometries"
    are_isometry = [is_isometry_2(op) for op in x_list]
    if show_idx:
        false_idx = [idx for idx, value in enumerate(are_isometry) if value==False]
        if not false_idx:
            print('all elements are isometries')
        else:
            print('indices which are not isometries')
            return false_idx
    else:
        return are_isometry
    
is_isometry = check_isometries # for compatibility with "trust_region.ipynb"
    
def is_hermitian(H:np.ndarray):
    "checks if a matrix is hermitian or not. If not, return norm between H and H+"
    hermitian = np.allclose(H.conj().T, H, atol=1e-8)
    if hermitian:
        return hermitian
    else:
        return hermitian, np.linalg.norm(H - H.conj().T) 
    
def is_in_tangent_space(X:np.ndarray, Z:np.ndarray):
    "checks if the matrix Z is in the tangent space of isometry X"
    assert X.ndim == Z.ndim == 2, "only matrices allowed"
    d = X.shape[1]
    return np.allclose(X.conj().T @ Z, - Z.conj().T @ X)

def is_antisymmetric(A:np.ndarray):
    "checks if a matrix is antisymmetric or not"
    return np.allclose(A, -A.conj().T)

def is_symmetric(C:np.ndarray):
    "checks if a matrix is symmetric or not"
    return np.allclose(C, C.conj().T)

def gradient_ambient2riemannian(x:np.ndarray, gradient:np.ndarray, alpha0:int=1, alpha1:int=1):
    """
    riemannian gradient at x of Stiefel manifold. Metric determined by (alpha)_i
    
    See section 5 of https://arxiv.org/abs/2009.10159

    embedded (euclidean):
    alpha0 = alpha1 = 1
    canonical:
    alpha0 = 1, alpha1 = 1/2
    """
    return (1/alpha0) * gradient + 0.5 * ((1/alpha1) - (2/alpha0)) * x @ x.T @ gradient - 0.5 * (1/alpha1) * x @ gradient.T @ x

def gradient_stiefel_general(xi, func, alpha0=1, alpha1=1):
    """
    Compute the riemannian gradient using a general metric for the tangent space parametrized by alpha0 and alpha1

    From https://arxiv.org/abs/2009.10159 below equation 5.2

    embedded (euclidean):
    alpha0 = alpha1 = 1
    canonical:
    alpha0 = 1, alpha1 = 1/2
    """

    grads_ambient = jax.grad(func)(xi)
    return [
        gradient_ambient2riemannian(x, grad, alpha0=alpha0, alpha1=alpha1) for x,grad in zip(xi, grads_ambient)
    ]

def gradient_stiefel(xi, func):
    """
    Compute the riemannian gradient using the euclidean/embedded/constant trace metric

    From https://assets.press.princeton.edu/chapters/absil/Absil_Chap3.pdf equation 3.37

    This is equivalent to calling:
    `gradient_stiefel_general(alpha0=1, alpha1=1)`
    """
    Zi = jax.grad(func)(xi)
    return [project(X, Z)
    for X,Z in zip(xi, Zi)]

def gradient_canonical(xi, func):
    """
    Compute the riemannian gradient using the canonical metric in tangent space. 
    From https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf eq.2.53
    FY - YFYTY.

    This is equivalent to calling:
    `gradient_stiefel_general(alpha0=1, alpha1=1/2)`
    """
    zi = jax.grad(func)(xi)
    return [z - x@z.T@x for x,z in zip(xi, zi)]

def gradient_stiefel_vec(xi, func, metric='euclidean'):
    "compute the vectorized gradient for all xi"
    # NOTE: changing to store only parametrization here to adhere to trust_region function 
    if metric == 'euclidean':
        alpha0, alpha1 = 1, 1
    elif metric == 'canonical':
        alpha0, alpha1 = 1, 1/2
    else:
        raise ValueError(f'{metric} is not a valid metric value')
    zi = gradient_stiefel_general(xi, func, alpha0=alpha0, alpha1=alpha1)
    # NOTE: changed from jnp.vstack since now we are concatenating vectors.
    return jnp.hstack([
        parametrization_from_tangent(x=x, z=grad, stack=True, vectorized=True) 
    for x, grad in zip(xi, zi)])

def riemannian_connection(D_nu, nu, eta, x, alpha0:int=1, alpha1:int=1):
    """
    General parametrized riemannian connection of tangent spaces
    
    From equation 5.4 of https://arxiv.org/abs/2009.10159
    """
    In = np.eye(x.shape[0])
    return D_nu + 0.5 * x @ (eta.T @ nu + nu.T @ eta) + ((alpha0-alpha1)/alpha0)*(In - x @ x.T) @ (eta @ nu.T + nu @ eta.T) @ x

def riemannian_hessian(x, func, vector=False, metric:str='euclidean'):
    "get riemannian hessian of func evaluated at x based on metric"
    if metric == 'euclidean':
        alpha0, alpha1 = 1, 1
    elif metric == 'canonical':
        alpha0, alpha1 = 1, 1/2

    grad_func = lambda xi: gradient_stiefel_general(xi, func, alpha0=alpha0, alpha1=alpha1)

    n = len(x)
    x_comps = [get_orthogonal_complement(xi) for xi in x]
    hessian_columns = []

    if not all([(op.dtype == np.float64) for op in x]):
        print('complex value found')
        x = [op.astype(np.float64) for op in x]


    for i in range(n):
        print('column :',i)
        dxk_size = stiefel_tangent_dof(x[i]) #x[i].size
        if vector:
            xi_dxk = [np.zeros((stiefel_tangent_dof(op), dxk_size), dtype=np.float64) for op in x]
        else:
            # TODO: see if I need to deprecate this, as I don't think it makes sense anymore without vectorizing it.
            xi_dxk = [np.zeros(op.shape + (dxk_size,), dtype=np.float64) for op in x]
        
        for k in range(dxk_size):
            # see old commits for previous attempts
            tangents = [jnp.zeros_like(op, dtype=np.float64) if l!=i else get_elementary_tangent_direction(k, op) for l, op in enumerate(x)]

            grads_eval, jvp_eval = jax.jvp(grad_func, (x,), (tangents,))

            for j, element in enumerate(jvp_eval):
                # we need to project each of them and store in an array that has information about the index k. before: projection
                z = riemannian_connection(D_nu=element, nu=grads_eval[j], eta=tangents[j], x=x[j], alpha0=alpha0, alpha1=alpha1)
                if vector:            
                    # NOTE: stores only parametrization here to adhere to trust_region function
                    xi_dxk[j][:,k] += parametrization_from_tangent(x=x[j], z=z, x_comp=x_comps[j], stack=True, vectorized=True)
                else:
                    xi_dxk[j][:,:,k] += parametrization_from_tangent(x=x[j], z=z, stack=True, vectorized=False)
        hessian_columns.append(xi_dxk)
    return hessian_columns


def riemannian_hessian_vec(x, func, transpose:bool=False, metric:str='euclidean'):
    "riemannian hessian matrix of func evaluated at list of Stiefel matrices x"
    hessian_columns = riemannian_hessian(x, func, vector=True, metric=metric) 
    n = len(x)
    size_vec = hessian_columns[0][0].shape[0]
    # NOTE on shape: 0th: column, 1st: row, 2,3: from (size, size) from jvp of grad_func
    hessian_full = np.stack(hessian_columns, axis=1) # n, n, size_vec, size_vec
    hessian_full = hessian_full.swapaxes(1,2).reshape(n*size_vec, n*size_vec)
    if transpose:
        hessian_full = hessian_full.T
    return hessian_full
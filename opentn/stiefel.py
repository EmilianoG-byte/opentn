"""
A module containing the functions used for the optimization of a cost function defined
in the Stiefel manifold.
"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from opentn.transformations import factorize_psd_truncated

from scipy.linalg import fractional_matrix_power

def project(X:np.ndarray, Z:np.ndarray):
    """
    Project Z matrix onto the tangent space of matrix X in Stiefel manifold

    From https://arxiv.org/pdf/2112.05176.pdf eq.26
    """
    # TODO: add back the conj
    return Z - 0.5 * X @ (X.T @ Z + Z.T @ X)

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

def metric(delta1:np.ndarray, delta2:np.ndarray, X:np.ndarray):
    """
    Riemannian metric between delta1 and delta2 in tangent space of matrix X in Stiefel manifold. 
    From https://arxiv.org/abs/2112.05176 eq. 24
    """
    dim = X.shape[0]
    gamma = np.eye(dim) - 0.5 * (X@X.conj().T)
    return np.trace(delta1.conj().T@gamma@delta2).real

def get_unit_matrices(ops:list[np.ndarray]):
    """
    Get a list of unit matrices (matrices with zeros everywhere except on the [0,0] element with a 1 instead)
    with the same shape as the list of ops.
    """
    unit_matrices = []
    for op in ops:
        # TODO: change dtype here if I want things to be complex. Right now it is just casting it (discarding real part)
        tangent_vector = np.zeros_like(op, dtype=np.float64)
        tangent_vector[0,0] = 1 
        unit_matrices.append(tangent_vector)
    return unit_matrices

def get_orthogonal_complement(X:np.ndarray):
    """
    Get the orthogonal complement matrix of X such that

    [X, X_comp].conj().T @ [X, X_comp] = I

    see 'notation' in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753
    """
    n, p = X.shape
    # TODO: add back the conj
    P_x = X @ X.T 
    P_x_comp = np.eye(n) - P_x
    return factorize_psd_truncated(P_x_comp, chi_max=n-p)


def get_symmetric(X:np.ndarray, Z:np.ndarray):
    """
    Get the symmetric component A from a general matrix Z
    
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
        n,p = X.shape
         # TODO: add back the conj
        Px_comp = np.eye(n) - X @ X.T
    elif Px_comp is None and X is None: 
        raise('at least one of X or Px_comp should be given')
     # TODO: add back the conj
    return X_comp.T @ Px_comp @ Z

def get_tangent_parametrization(X:np.ndarray, Z:np.ndarray, X_comp:np.ndarray = None ,stack:bool=True):
    """
    Get the A and B matrices that parametrize the Z matrix of the tangent space of stiefel manifold at X
    
    Z = X @ A + X_comp @ B
    
    with X_comp the orthogonal complement of X.

    From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=984753 lemma 8.
    """
    if X_comp is None:
        X_comp = get_orthogonal_complement(X)

    A = get_symmetric(X, Z)
    B = get_arbitrary(X_comp=X_comp, Z=Z, X=X)
    if stack:
        return np.vstack([A,B], dtype=np.float64)
    else:
        return A, B

def polar_decomposition_stiefel(X, Z):
    """
    Retraction based on polar decomposition of Z at tangent space of matrix X from Stiefel manifold
    
    From https://assets.press.princeton.edu/chapters/absil/Absil_Chap4.pdf eq. 4.7
    """
    assert X.shape == Z.shape, "shapes don't match"
    d = X.shape[1]
    # TODO: add back the conj
    return (X+Z)@fractional_matrix_power(np.eye(d) + Z.T@Z, -0.5)

def retract_x(x_list, eta):
    "retraction from tanget space at x to original manifold of x"
    n = len(x_list)
    # here we have to assume that all x have the same shape 
    eta = np.reshape(eta, ((n,) + x_list[0].shape))
    # in theory if we want to emulate what is going on with unitaries, we have to:
    # project the eta[j] onto the tangent space of the vlist[j].
    return [polar_decomposition_stiefel(x_list[j], eta[j]) for j in range(n)]

def is_isometry(x_list:list[np.ndarray], show_idx:bool=False):
    "checks if the list of operators belong to the Stiefel manifold, i.e. are isometries"
    # we assume that all operators are of the same dimensions
    dim = x_list[0].shape[1]
    are_isometry = [np.allclose(op.conj().T@op, np.eye(dim)) for op in x_list]
    if show_idx:
        are_false = [idx for idx, value in enumerate(are_isometry) if value==False]
        if not are_false:
            print('all elements are isometries')
        else:
            print('indices which are not isometries')
            return are_false
    else:
        return are_isometry
    
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

def gradient_stiefel(xi, func):
    "compute riemannian gradient for all xi, returning a list"
    Zi = jax.grad(func)(xi)
    return [project(X, Z)
    for X,Z in zip(xi, Zi)]

def gradient_stiefel_vec(xi, func):
    "compute the vectorized gradient for all xi"
    # NOTE: changing to store only parametrization here to adhere to trust_region function 
    return jnp.vstack([
        get_tangent_parametrization(X=x, Z=grad, stack=True) 
    for grad, x in zip(gradient_stiefel(xi, func), xi)]).reshape(-1)


def riemannian_hessian(func, x, vector=False):
    "get riemannian hessian of func (needs to be the gradient of the actual function) evaluated at x"
    grad_func = lambda xi: gradient_stiefel(xi, func)
    n = len(x)
    assert n == 3, 'wrong input size'
    unit_matrices = get_unit_matrices(x)
    x_comps = [get_orthogonal_complement(xi) for xi in x]
    hessian_columns = []

    if not all([(op.dtype == np.float64) for op in x]):
        print('complex value found')
        x = [op.astype(np.float64) for op in x]


    for i in range(n):
        print('column :',i)
        dxk_size = x[i].size
        if vector:
            xi_dxk = [np.zeros((op.size, dxk_size), dtype=np.float64) for op in x]
        else:
            xi_dxk = [np.zeros(op.shape + (dxk_size,), dtype=np.float64) for op in x]
        
        for k in range(dxk_size):
            # printing every 100 steps to see progress
            # if k%100 == 0:
                # print('element: ', k)
            _, jvp_eval = jax.jvp(grad_func, (x,), ([jnp.zeros_like(op, dtype=np.float64) if l!=i else project(X=op,Z=jnp.roll(unit_matrices[l],k)) for l,op in enumerate(x)],))

            for j, element in enumerate(jvp_eval):
                # we need to project each of them and store in an array that has information about the index k
                if vector:            
                    # NOTE: changing to store only parametrization here to adhere to trust_region function 
                    xi_dxk[j][:,k] +=  get_tangent_parametrization(X=x[j], Z=project(x[j],element), X_comp=x_comps[j], stack=True).reshape(-1).astype(np.float64)
                else:
                    xi_dxk[j][:,:,k] += get_tangent_parametrization(X=x[j], Z=project(x[j],element), stack=True)
        hessian_columns.append(xi_dxk)
    return hessian_columns


def riemannian_hessian_vec(func, x):
    "riemannian hessian matrix of func evaluated at list of Stiefel matrices x"
    hessian_columns = riemannian_hessian(func, x, vector=True) 
    n = len(x)
    size_vec = hessian_columns[0][0].shape[0]

    hessian_full = np.stack(hessian_columns, axis=1) # n, n, size_vec, size_vec
    hessian_full = hessian_full.swapaxes(1,2).reshape(n*size_vec, n*size_vec)
    return hessian_full
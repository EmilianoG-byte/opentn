"""
A module containing the functions used for the optimization of a cost function defined
in the Stiefel manifold.
"""

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from opentn.transformations import vectorize

from scipy.linalg import fractional_matrix_power

def project(X:np.ndarray, Z:np.ndarray):
    """
    Project Z matrix onto the tangent space of matrix X in Stiefel manifold

    From https://arxiv.org/pdf/2112.05176.pdf eq.26
    """
    return Z - 0.5 * X @ (X.conj().T @ Z + Z.conj().T @ X)

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
        tangent_vector = np.zeros_like(op)
        tangent_vector[0,0] = 1 
        unit_matrices.append(tangent_vector)
    return unit_matrices

def polar_decomposition_stiefel(X, Z):
    """
    Retraction based on polar decomposition of Z at tangent space of matrix X from Stiefel manifold
    
    From https://assets.press.princeton.edu/chapters/absil/Absil_Chap4.pdf eq. 4.7
    """
    assert X.shape == Z.shape, "shapes don't match"
    d = X.shape[1]
    return (X+Z)@fractional_matrix_power(np.eye(d) + Z.conj().T@Z, -0.5)

def retract_x(x_list, eta):
    "retraction from tanget space at x to original manifold of x"
    n = len(x_list)
    # here we have to assume that all x have the same shape 
    eta = np.reshape(eta, ((n,) + x_list[0].shape))
    return [polar_decomposition_stiefel(x_list[j], eta[j]) for j in range(n)]

def check_isometry(x_list:list[np.ndarray], show_idx:bool=False):
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
    

def gradient_stiefel(xi, func):
    "compute riemannian gradient for all xi, returning a list"
    Zi = jax.grad(func)(xi)
    return [project(X, Z)
    for X,Z in zip(xi, Zi)]

def gradient_stiefel_vec(xi, func):
    "compute the vectorized gradient for all xi"
    return jnp.vstack([
            vectorize(grad) 
    for grad in gradient_stiefel(xi, func)]).reshape(-1)


def riemannian_hessian(func, x, vector=False):
    "get riemannian hessian of func (needs to be the gradient of the actual function) evaluated at x"
    grad_func = lambda xi: gradient_stiefel(xi, func)
    n = len(x)
    assert n == 3, 'wrong input size'
    unit_matrices = get_unit_matrices(x)
    hessian_columns = []

    for i in range(n):
        print('column :',i)
        dxk_size = x[i].size
        if vector:
            xi_dxk = [np.zeros((op.size, dxk_size)) for op in x]
        else:
            xi_dxk = [np.zeros(op.shape + (dxk_size,)) for op in x]
        
        for k in range(dxk_size):
            # printing every 100 steps to see progress
            # if k%100 == 0:
                # print('element: ', k)
            _, jvp_eval = jax.jvp(grad_func, (x,), ([jnp.zeros_like(op) if l!=i else jnp.roll(unit_matrices[l],k) for l,op in enumerate(x)],))

            for j, element in enumerate(jvp_eval):
                # we need to project each of them and store in an array that has information about the index k
                if vector:                    
                    xi_dxk[j][:,k] += project(x[j],element).reshape(-1)
                else:
                    xi_dxk[j][:,:,k] += project(x[j],element)
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
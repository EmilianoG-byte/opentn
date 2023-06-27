"""
A module containing optimization functions and helper functions for frequently-accessed task  in matrix
optimization such as norm calculation.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
from jax import jit
from opentn.transformations import vectorize, choi2super

def calculate_norms(Os):
    "calculates the norm of a list of operators O in Os"
    for O in Os:
        print(jnp.linalg.norm(O))

def update(old_params, grads, rate=0.1):
    assert len(old_params) == len(grads)
    new_params = []
    for param, grad in zip(old_params, grads):
        new_params.append(param - rate*grad)
    return new_params

def frobenius_norm(A:np.ndarray,B:np.ndarray):
    return jnp.linalg.norm(A-B, ord='fro')

def cosine_similarity(A:np.ndarray,B:np.ndarray):
    # NOTE: changed the function to have the most negative when A = B
    # flatten them as this is defined for vectors.
    a = vectorize(A)
    b = vectorize(B)
    # taking the real part as otherwise the grad calculation should guarantee cost_fn to be holomorphic
    return -(a@b.T.conj() / (jnp.linalg.norm(a)*jnp.linalg.norm(b))).real
    
@jit
def model(Xs):
    "Xs are assumed to be the square roots of the PSD matrices (Choi)"
    Ys = []
    for X in Xs:
        # convert each of the X into a matrix
        if X.ndim > 2:
            X = jnp.reshape(a=X, newshape=[int(np.sqrt(X.size))]*2)
        Y = jnp.dot(X, X.T.conj())
        Ys.append(Y)
    # convert to superoperators
    Ys_super = [choi2super(choi=Y) for Y in Ys]
    Y_total = Ys_super[0]
    for Y in Ys_super[1:]:
        Y_total = jnp.dot(Y, Y_total)
    return Y_total

def compute_loss(X1, X2, X3, loss_fn, exact):
    "cost function for 'troterization' of exponential. The choi matrix is the one that should be PSD"
    # NOTE: we are going to accept the Xi that correspond to the choi matrix
    # these are choi matrices
    Xs = [X1,X2,X3]
    # model will convert them to superoperators
    prediction = model(Xs)
    
    assert exact.ndim == exact.ndim == 2, 'Y_total should be a matrix'
    return jit(loss_fn)(exact, prediction)

def gds(fn:Callable, x0:list, exact:np.ndarray, rate:float = 0.01, iter:int = 10, loss_fn=None, show_cost:bool=True) -> tuple[list, list, list]:
    """
    Gradient descent (GDS) optimization workflow.

    .. math:: x(i+1) = x(i) - rate*grad(fn)

    args:
    ---------
    fn:
        function to be optimized. The signature should be: (xi:parameters_to_optimimize, loss_fn:loss function, exact:exact_value_to_compare)
    x0:
        initial list of parameters for optimization
    exact:
        exact/theoretical/reference value to use for computing the loss function
    rate:
        rate to use in optimization.
    iter:
        number of iterations to repeat optimization over
    loss_fn:
        loss function used in fn to be optimized
    show_cost:
        whether or not to print the cost function after evert iteration.
    returns:
    ---------
        Tuple containg the cost, gradient, and parameter history over optimization.
        NOTE: the cost_eval is evaluated at the current parameters xi. So we should evaluate
        the function one last time with the final xi. len(cost_list == iter + 1)
        NOTE: update: I have changed this so that we have exactly len(cost_list == iter)
        I think this makes more sense and saves evaluation time.
        NOTE: the params_list will have len == iter + 1, since it would contain the latest update as well

    """
    if not loss_fn:
        loss_fn = lambda x,y : jnp.linalg.norm(x-y, ord='fro')

    cost_list = []
    grads_list = []
    params_list = []

    xi = x0
    num_params = len(xi)
    params_list.append(xi)

    for i in range(iter):
        print(f'Starting iteration: {i}')
        cost_eval, grad_x = jax.value_and_grad(fn, list(range(num_params)))(*xi, loss_fn=loss_fn, exact=exact)
        cost_list.append(cost_eval)
        grads_list.append(grad_x)
        if show_cost:
            print('* Cost function:', cost_eval)
        xi = update(xi, grad_x, rate)
        params_list.append(xi)

    return cost_list, grads_list, params_list


# from Forest Benchmarking: 
# https://github.com/rigetti/forest-benchmarking/blob/master/forest/benchmarking/distance_measures.py

def diamond_norm_distance(choi0: np.ndarray, choi1: np.ndarray) -> float:
    """
    Return the diamond norm distance between two completely positive
    trace-preserving (CPTP) superoperators, represented as Choi matrices.

    The calculation uses the simplified semidefinite program of Watrous in [CBN]_

    .. note::

        This calculation becomes very slow for 4 or more qubits.

    .. [CBN] Semidefinite programs for completely bounded norms.
          J. Watrous.
          Theory of Computing 5, 11, pp. 217-238 (2009).
          http://theoryofcomputing.org/articles/v005a011
          http://arxiv.org/abs/0901.4709

    :param choi0: A 4**N by 4**N matrix (where N is the number of qubits)
    :param choi1: A 4**N by 4**N matrix (where N is the number of qubits)

    """
    # Kudos: Based on MatLab code written by Marcus P. da Silva
    # (https://github.com/BBN-Q/matlab-diamond-norm/)
    import cvxpy as cvx
    assert choi0.shape == choi1.shape
    assert choi0.shape[0] == choi1.shape[1]
    dim_squared = choi0.shape[0]
    dim = int(np.sqrt(dim_squared))

    delta_choi = choi0 - choi1
    delta_choi = (delta_choi.conj().T + delta_choi) / 2  # Enforce Hermiticity

    # Density matrix must be Hermitian, positive semidefinite, trace 1
    rho = cvx.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvx.trace(rho) == 1]

    # W must be Hermitian, positive semidefinite
    W = cvx.Variable([dim_squared, dim_squared], complex=True)
    constraints += [W == W.H]
    constraints += [W >> 0]

    constraints += [(W - cvx.kron(np.eye(dim), rho)) << 0]

    J = cvx.Parameter([dim_squared, dim_squared], complex=True)
    objective = cvx.Maximize(cvx.real(cvx.trace(J.H @ W)))

    prob = cvx.Problem(objective, constraints)

    J.value = delta_choi
    prob.solve()

    dnorm = prob.value * 2

    return dnorm

"""
A module containing optimization functions and helper functions for frequently-accessed task  in matrix
optimization such as norm calculation.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
from jax import jit
from opentn.transformations import vectorize, choi2super, create_supertensored_from_local, convert_supertensored2liouvillianfull, choi_composition
import cvxpy as cvx

from jax import config
config.update("jax_enable_x64", True)


def small2zero(op:np.array, tol:float=1e-8):
    "erase the elements in op smaller (abs) than tol. Does NOT change in place"
    return np.where(abs(op)>=tol, op, 0)

def calculate_norms(Os):
    "calculates the norm of a list of operators O in Os"
    for O in Os:
        print(jnp.linalg.norm(O))

def update(old_params:list, grads:list, rate:float):
    assert len(old_params) == len(grads)
    # new_params = []
    # for param, grad in zip(old_params, grads):
    #     new_params.append(param - rate*grad)
    return old_params - rate*grads

def frobenius_norm(A:np.ndarray,B:np.ndarray, squared:bool=False):
    exp = 1
    if squared:
        exp = 2
    return jnp.linalg.norm(A-B, ord='fro')**exp

def cosine_similarity(A:np.ndarray,B:np.ndarray):
    # NOTE: changed the function to have the most negative when A = B
    # flatten them as this is defined for vectors.
    a = vectorize(A)
    b = vectorize(B)
    # taking the real part as otherwise the grad calculation should guarantee cost_fn to be holomorphic
    return -(a@b.T.conj() / (jnp.linalg.norm(a)*jnp.linalg.norm(b))).real

def X2C(Xs):
    "transform a list of Cs onto a list of Cs"
    Cs = []
    for X in Xs:
        # convert each of the X into a matrix
        if X.ndim > 2:
            X = jnp.reshape(a=X, newshape=[int(np.sqrt(X.size))]*2)
        Cs.append(jnp.dot(X, X.T.conj()))
    return Cs

@jit
def model_Cs(Xs:np.ndarray):
    "Composition of choi matrices. Cs are assumed to be PSD. Will return a Choi as well"
    Cs = X2C(Xs)
    C_total  = Cs[0]
    for C in Cs[1:]:
        C_total = choi_composition(C_total, C)
    return C_total

# TODO: cannnot JIT this function due to create_supertensored_from_local
def model_Zs(Wi:np.ndarray, Xj:np.ndarray, Xk:np.ndarray, N:int, order:np.ndarray=np.array([0,1,2])):
    """
    Zi = Wi @ Wi.conj().T
    Yi_z = Zi (x) Zi (x) Zi (create_supertensored_from_local)
    Yi_z = convert_supertensored2liouvillianfull(Yi_z)
    model = Xk @ Xj @ Yi_z
    ||O - model||F

    pos: determines what is the Xi over which we are optimizing
    """
    Wi_super = create_supertensored_from_local(localop=Wi, N=N)
    d = int(Wi_super.shape[0]**(1/(2*N)))
    assert Wi_super.shape[0] == d**(2*N), f"{Wi_super.shape[0]} != {d**(2*N)}"
    Xi = convert_supertensored2liouvillianfull(Wi_super, N=N, d=d)
    Xs = jnp.array([Xi, Xj, Xk])
    return model_Ys(Xs[order])

@jit
def model_Ys(Xs:np.ndarray):
    "Xs are assumed to be the square roots of the PSD matrices (Choi). We convert them here to superoperators"
    Cs = X2C(Xs)
    # convert to superoperators
    Ys = [choi2super(choi=C) for C in Cs]
    Y_total = Ys[0]
    for Y in Ys[1:]:
        Y_total = jnp.dot(Y, Y_total)
    return Y_total

def compute_loss(xi:np.ndarray, loss_fn, model, exact:np.ndarray, **kwargs):
    """
    General function to compute the loss of the input parameters in the model agains the exact value using loss_fn as measurement
    """
    prediction = model(xi, **kwargs)
    assert exact.ndim == prediction.ndim == 2, 'not a matrix'
    return jit(loss_fn)(exact, prediction)

def gds(fn:Callable, x0:list, exact:np.ndarray, rate:float = 0.01, iter:int = 10, loss_fn=None, model=None, show_cost:bool=True, store_all:bool=True, **kwargs) -> tuple[list, list, list]:
    """
    Gradient descent (GDS) optimization workflow.

    .. math:: x(i+1) = x(i) - rate*grad(fn)

    args:
    ---------
    fn:
        function to be optimized. The signature should be: 
        (xi:parameters_to_optimimize, loss_fn:loss function, model:model for prediction, exact:exact_value_to_compare)
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
    if not model:
        model = model_Ys

    cost_list = []
    grads_list = []
    params_list = []

    xi = jnp.array(x0)
    # num_params = len(xi)
    params_list.append(xi)

    for i in range(iter):
        if show_cost:
            print(f'Starting iteration: {i}')
        cost_eval, grad_x = jax.value_and_grad(fn, 0)(xi, loss_fn=loss_fn, model=model, exact=exact, **kwargs) # NOTE: erased: list(range(num_params))
        if show_cost:
            print('* Cost function:', cost_eval)
        xi = update(xi, grad_x, rate)
        cost_list.append(cost_eval)
        if store_all:
            grads_list.append(grad_x)
            params_list.append(xi)
        else:
            if i == iter-1:
                grads_list.append(grad_x)
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


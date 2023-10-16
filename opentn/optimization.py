"""
A module containing optimization functions and helper functions for frequently-accessed task  in matrix
optimization such as norm calculation.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
from jax import jit
from opentn.transformations import vectorize, choi2super, create_supertensored_from_local, convert_supertensored2liouvillianfull, choi_composition, ortho2choi, compose_superops_list, unfactorize_psd, tensor_to_matrix, ortho2super
import cvxpy as cvx

from jax import config
config.update("jax_enable_x64", True)


def small2zero(op:np.array, tol:float=1e-8):
    "erase the elements in op smaller (abs) than tol. Does NOT change in place"
    return np.where(abs(op)>=tol, op, 0)

def calculate_norms(ops):
    "calculates the norm of a list of operators op in ops"
    for op in ops:
        print(jnp.linalg.norm(op))

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

def unfactorize_list(xs):
    "transform a list of xs onto a list of Cs"
    Cs = []
    for x in xs:
        # convert each of the X into a matrix
        if x.ndim > 2:
            x = tensor_to_matrix(x)
        Cs.append(unfactorize_psd(x))
    return Cs

@jit
def model_Cs(xs:np.ndarray):
    "Composition of choi matrices. Cs are assumed to be PSD. Will return a Choi as well"
    Cs = unfactorize_list(xs)
    C_total  = Cs[0]
    for C in Cs[1:]:
        C_total = choi_composition(C_total, C)
    return C_total

# NOTE: cannnot JIT this function due to create_supertensored_from_local
def model_Zs(Wi:np.ndarray, Xj:np.ndarray, Xk:np.ndarray, N:int, order:np.ndarray=np.array([0,1,2])):
    """
    Zi = Wi @ Wi.conj().T
    Yi_z = Zi (x) Zi (x) Zi (create_supertensored_from_local)
    Yi_z = convert_supertensored2liouvillianfull(Yi_z)
    model = Xk @ Xj @ Yi_z
    ||O - model||F
    """
    Wi_super = create_supertensored_from_local(localop=Wi, N=N)
    d = int(Wi_super.shape[0]**(1/(2*N)))
    assert Wi_super.shape[0] == d**(2*N), f"{Wi_super.shape[0]} != {d**(2*N)}"
    Xi = convert_supertensored2liouvillianfull(Wi_super, N=N, d=d)
    xs = jnp.array([Xi, Xj, Xk])
    return model_Ys(xs[order])

def model_stiefel_local(xs:np.ndarray, N:int, d:int):
    """
    Optimization model for stiefel local operators

    Pipeline is:
    stiefel -> choi_sqrt -> choi -> superop_local -> superop_full_split -> superop_full -> compose them

    Where superop_full_split is a superoperator with the pattern (i, i+1, i*, i+1*) for i in [0,N-1]
    And superop_full has the pattern (0,....,N-1, 0*, ..., N-1*)
    """
    assert len(xs)==3, 'only odd-even-odd structure allowed'
    superops_local = [ortho2super(x) for x in xs]
    # we assume the same operator acts on all sites (unlike before for even layer)
    superops_full_split = [create_supertensored_from_local(localop=op, N=N) for op in superops_local]
    # here the conversion changes for odd and even layer
    superops_full = []
    for i, op in enumerate(superops_full_split):
        if i%2 == 1:
            pbc = True
        else:
            pbc = False
        superops_full.append(convert_supertensored2liouvillianfull(op, N=N, d=d, pbc=pbc))
    return compose_superops_list(superops_full)

@jit
def model_Ys(xs:np.ndarray):
    "xs are assumed to be the square roots of the PSD matrices (Choi). We convert them here to superoperators"
    Cs = unfactorize_list(xs)
    # convert to superoperators
    Ys = [choi2super(choi=C) for C in Cs]
    return compose_superops_list(Ys)

@jit
def model_Ys_stiefel(xs:list[np.ndarray]):
    "xs are assume to be the squared roots of the PSD matrices with axis 1,2 swaped to make them orthonormal"
    xs_choi = [ortho2choi(x) for x in xs]
    return model_Ys(xs_choi)

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


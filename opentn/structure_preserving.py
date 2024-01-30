"""
A module containing the utility functions and maiin functions for the structure preserving
algorithms introduced in https://arxiv.org/abs/2103.01194
"""

import numpy as np
import jax.numpy as jnp
from opentn.transformations import op2fullspace, permute_operator_pbc, kraus2superop, exp_operator_dt, vectorize, unvectorize
from opentn.stiefel import random_psd
from opentn.optimization import frobenius_norm
from typing import Union, Optional
from itertools import combinations
from math import factorial, ceil, log
from jax import config
config.update("jax_enable_x64", True)


def effective_hamiltonian(lindbladians:Union[np.ndarray, list],  N:int, d:int, pbc:bool=True)->np.ndarray:
    """
    Get the effective hamiltonian used in the second splitting of the full lindbladian.

    See equation 5 of https://arxiv.org/abs/2103.01194.
    For now we assume that no hamiltonian is used
    # TODO: include hamiltonian terms
    """
    num_sites = int(log(lindbladians[0].shape[0], d)) # assume all lindbladians are same shape

    assert num_sites <= N, f"the numbert of sites on the lindbladians: {num_sites} should be smaller or equal than N: {N} "

    if N == num_sites or N == 1: # using pbc for a single site would not make sense
        pbc = False

    if not isinstance(lindbladians, list):
        lindbladians = [lindbladians]

    h_eff = jnp.zeros(shape=(d**N, d**N), dtype=complex)
   
    for lindbladian in lindbladians:

        for i in range(0, N-1):
            lindbladian_full = op2fullspace(op=lindbladian, i=i, N=N, num_sites=num_sites)
            h_eff += lindbladian_full.conj().T @ lindbladian_full / 2j

        if pbc: # PBC
            lindbladian_full = op2fullspace(op=lindbladian, i=N-2, N=N, num_sites=num_sites)
            lindbladian_full = permute_operator_pbc(lindbladian_full, N=N, d=d)
            h_eff += lindbladian_full.conj().T @ lindbladian_full / 2j

        if N == 1:
            # this is needed because the i loop would not enter with N = 1
            h_eff += lindbladian.conj().T @ lindbladian / 2j

    return h_eff


def kraus_lindbladians_to_superop(Ek:list[np.ndarray], N:int, d:int, pbc:bool=True)->np.ndarray:
    """
    convert a list of lindbladians into its superoperator representation.
    
    We assume each operator act as a quantum channel (kraus) on the state
    and that the same list acts on all the sites

    See L_l(rho) in equation 6.b of https://arxiv.org/abs/2103.01194.
    """
    num_sites = int(log(Ek[0].shape[0], d)) # assume all lindbladians are same shape

    assert num_sites <= N, f"the numbert of sites on the lindbladians: {num_sites} should be smaller or equal than N: {N} "

    if N == num_sites or N == 1: # using pbc for a single site would not make sense
        pbc = False

    if not isinstance(Ek, list):
        Ek = [Ek]
    superop = jnp.zeros(shape=(d**(2*N), d**(2*N)), dtype=complex)

    for i in range(0, N-1):
        Ek_full_size = [op2fullspace(op=op, i=i, N=N, num_sites=num_sites) for op in Ek]
        superop += kraus2superop(kraus_list=Ek_full_size)

    if pbc:
        Ek_full_size = [permute_operator_pbc(op2fullspace(op=op, i=N-2, N=N, num_sites=num_sites), N=N, d=d) for op in Ek]
        superop += kraus2superop(kraus_list=Ek_full_size)

    if N == 1:
        # this is needed because the i loop would not enter with N = 1
        superop += kraus2superop(kraus_list=Ek)
    return superop

def m_order_l_operator(m:int, Ek:list[np.ndarray], N:int, d:int):
    """
    Create the m-th order L_l superoperator, i.e. (L_l)^m
    """
    l_superop = kraus_lindbladians_to_superop(Ek=Ek, N=N, d=d)
    return composition(l_superop, m)

def m_order_l_full_term(m:int, dt:float, Ek:list[np.ndarray], N:int, d:int):
    """
    Create the m-th order L_l superoperator, i.e. (L_l)^m
    """
    l_superop = m_order_l_operator(m, Ek, N, d)
    return l_superop * (dt ** m) / factorial(m)

def m_order_taylor(m:int, op:np.ndarray, delta:float):
    """
    m order taylor series expansion of op with factor delta.
    
    See equation 11 of https://arxiv.org/abs/2103.01194.
    """
    rows, cols = op.shape
    assert rows == cols, "op should be a square matrix"
    taylor = identity_full(size=rows) 
    if delta != 0:
        for alpha in range(1, m+1):
            taylor += composition(op, alpha) * (delta ** alpha) / factorial(alpha)
    return taylor

def choose_nm(m:int, M:int, dt:float):
    """
    Chooses a Nm large enough to maintain the order M of the scheme based on the current number of midpoints m

    See Theorem 2 of https://arxiv.org/abs/2103.01194.
    """
    return ceil(dt ** (m - M))

def generate_rj(Nm:int, dt:float):
    """
    Generate the  Nm equally-spaced points for the time interval [0, ∆t] 

    See step III of https://arxiv.org/abs/2103.01194.
    """
    return [(j - 0.5) * (dt/Nm) for j in range(1, Nm + 1)]


def m_order_j_operator(m:int, t:float, s:float, lindbladians:list[np.ndarray], N:int, d:int, pbc:bool=True):
    """
    m order J_m(t,s) CP superoperator

    See equation 11 of https://arxiv.org/abs/2103.01194.
    """
    J = -1j * effective_hamiltonian(lindbladians=lindbladians, N=N, d=d, pbc=pbc)
    taylor_m = m_order_taylor(m=m, op=J, delta=t-s)
    return kraus2superop(taylor_m)

def m_order_f_operator(midpoints:list[float], dt:float, m:int, M:int, lindbladians:list[np.ndarray], N:int, d:int):
    """
    Create the F^M_m operator evaluated at `midpoints` with the L and J operator given by lindbladians.

    See equation 13 of https://arxiv.org/abs/2103.01194.
    """
    assert len(midpoints) == m, f"there should be exactly m = {m} midpoints"
    s_list = midpoints[::-1] + [0]
    l_superop = kraus_lindbladians_to_superop(Ek=lindbladians, N=N, d=d)
    f_superop = m_order_j_operator(m=M-m, t=dt, s=s_list[0], lindbladians=lindbladians, N=N, d=d)
    for i in range(0, m):
        # TODO: improve implementation by catching from here if t = s, we skip the term because it is just an identity
        f_superop = f_superop @ l_superop
        if s_list[i] != s_list[i+1]:
           f_superop = f_superop @ m_order_j_operator(m=M-m, t=s_list[i], s=s_list[i+1], lindbladians=lindbladians, N=N, d=d) 
    return f_superop


def identity_full(N:int=None, d:int=None, size:Optional[int]=None)->np.ndarray:
    "get the identity acting on the full hilbert space"
    if not size:
        if not N or not d:
            assert ValueError('Either `size` or `(N, d)` should be given')
        else:
            size = d**N
    return jnp.identity(n=size)


def unnormalized_scheme(lindbladians:Union[np.ndarray, list], N:int, d:int, dt:float, order:int=1, quadrature:str="trapezoidal", verbose:bool=False, pbc:bool=True):
    """
    Superoperator corresponding to a list of lindbladians following the `order` of the scheme.

    See equations 15a, 15b, 15c of https://arxiv.org/abs/2103.01194.
    The lindbladians are assumed to be acting locally on two sites
    """

    if verbose:
        print(f'currently using dt={dt}, M={order}')

    h_eff = effective_hamiltonian(lindbladians=lindbladians, N=N, d=d, pbc=pbc)
    l_superop = kraus_lindbladians_to_superop(Ek=lindbladians, N=N, d=d, pbc=pbc)
    identity = identity_full(N=N, d=d)

    superop = m_order_j_operator(m=order, t=dt, s=0, lindbladians=lindbladians, N=N, d=d, pbc=pbc)
    superop += m_order_l_full_term(m=order, dt=dt, Ek=lindbladians, N=N, d=d)

    if order == 1:
        pass
    elif order == 2:
        if quadrature == "trapezoidal":
            superop += 0.5 * dt * l_superop  @ kraus2superop(identity - 1j * dt * h_eff)
            superop += 0.5 * dt * kraus2superop(identity - 1j * dt * h_eff) @ l_superop 

        elif quadrature == "midpoint":
            superop += dt * kraus2superop(identity - 0.5j * dt * h_eff) @ l_superop @ kraus2superop(identity - 0.5j * dt * h_eff)
        else:
            raise ValueError("Only trapezoidal and midpoint quadrature methods are accepted")
    else:
        # loop m in [1, M-1]
        M = order
        for m in range(1, M):
            Nm = choose_nm(m=m, M=M, dt=dt)
            rj_all = generate_rj(Nm=Nm, dt=dt)
            # loop n in [1, min(m, Nm)]
            for n in range(1, min(m, Nm) + 1):
                ks_list = generate_indices_ks(n=n, m=m)
                # loop over all possible partitions of the m terms of length n
                for ks in ks_list:
                    rj_subsets = generate_rjs_subsets(rj_all=rj_all, n=n)
                    # loop over all possible combinations of intervals out of the original Nj
                    for rj_subset in rj_subsets:
                        factor = generate_prefactor(dt=dt, Nm=Nm, m=m, ks=ks)
                        rj_expanded = generate_rj_with_multiplicity(rj_list=rj_subset, ks=ks)
                        superop +=  m_order_f_operator(midpoints=rj_expanded, dt=dt, m=m, M=M, lindbladians=lindbladians, N=N, d=d) * factor
    return superop
        
def generate_indices_ks(n:int, m:int):
    """
    Generate all the possible ordered sets of `n` indices for order m 

    See appendix A of https://arxiv.org/abs/2103.01194.
    """
    possible_values = range(1, m) # values in [1, m-1]
    combinations_list = []
    for combo in combinations(possible_values, n - 1): # where to choose from, how many
        # NOTE: even if m = n = 1, this would loop once over the empty value and add m as single element
        combo = sorted(combo) + [m] # this guarantess the last element is always m
        combinations_list.append(combo)
    return combinations_list

def generate_differences(numbers):
    differences = []
    for i in range(len(numbers) - 1):
        differences.append(numbers[i + 1] - numbers[i])
    return differences

def check_indices_ks(ks:list[int], m:int):
    """
    Check that the indices generated by `generate_indices_ks` satisfy the suming condition

    i.e. sum_i ^ n (k_i - k_i-1 ) = m
    with k_0 = 0
    """
    ks = [0] + ks
    total = sum(generate_differences(ks))
    assert total == m, f"not a valid set of indices. {total} != {m}"

def generate_rjs_subsets(rj_all:list[float], n:int):
    """
    Gemerate all the possible ordered subsets of `rj_all` or length `n`
    """
    return list(combinations(rj_all, n))

def generate_rj_with_multiplicity(rj_list:list[float], ks:list[int]):
    """
    generate a list of rj's taking into account their multiplicity given by the ks indices

    See appendix A

    # Example usage:
    rj_list = [1, 2, 3]
    ks = [1, 2, 3]
    >> Expanded rj list: [1, 2, 2, 3, 3, 3]
    """
    differences = generate_differences([0] + ks)
    assert len(differences) == len(rj_list), f"something went wrong. len(differences) = {len(differences)} != len(rj_list) = {len(rj_list)}"
    rj_expanded = []
    for rj, multiplicity in zip(rj_list, differences):
        rj_expanded.extend([rj] * multiplicity)
    return rj_expanded

def generate_prefactor(dt:float, Nm:int, m:int, ks:list[int]):
    """
    Gemerate the prefactor of the middle term in equation 33.
    """
    # add zero to obtain the first term k1
    differences = generate_differences([0] + ks)
    denominator = 1
    for difference in differences:
        denominator *= factorial(difference)
    return (dt/Nm)**m / denominator




def composition(superop:np.ndarray, n:int)->np.ndarray:
    "composition of a superoperator repeated `n` times"
    superop_tot = superop
    for _ in range(n-1):
        superop_tot = superop @ superop_tot
    return superop_tot

def unnormalized_scheme_error(Lvec, lindbladians, N, d, tau, n, order=2, quadrature='midpoint', verbose=False):
    "get the error for the unnormalized structure preserving scheme"
    if verbose:
        print(f'currently using tau={tau}, n={n}, M={order}')
    superop_unnormalized = unnormalized_scheme(lindbladians=lindbladians, N=N, d=d, dt=tau/n, order=order, quadrature=quadrature)
    exp_Lvec = exp_operator_dt(op=Lvec, tau=tau)
    return frobenius_norm(composition(superop_unnormalized, n), exp_Lvec)

def random_psd_error(superop_exact:np.ndarray, superops_error:list[np.ndarray], iterations:int=500, normalize:bool=True, verbose:bool=False):
    """
    Calculate an array of average errors when applying each operator in `superops_error` to a random psd operator

    The average is taken over `iterations`, and the error is calculated wrt to `superop_exact` 
    """
    num_methods = len(superops_error)
    errors = np.zeros(shape=(iterations, num_methods)) 
    # create random instance:
    rows, cols = superop_exact.shape
    assert rows == cols, "this superop should be a square matrix" # Generalize this?

    for it in range(iterations):
        # sqrt since the superoperator dimensions are the original x conjugated
        rnd_psd = random_psd(dim=int(np.sqrt(rows)))
        rnd_psd_vec = vectorize(rnd_psd)
        # reference one
        vec_exact = superop_exact @ rnd_psd_vec
        
        for j, superop in enumerate(superops_error):
            # apply the superoperators from each scheme
            vec_scheme = superop @ rnd_psd_vec
            # normalize if necessary
            trace = np.trace(unvectorize(vec_scheme))
            if normalize:
                if verbose:
                    print('method no.: ', j)
                    print('unnormalized trace: ', trace)
                vec_scheme = vec_scheme/trace
            # save error of this iteraiton and operator
            errors[it,j] = frobenius_norm(unvectorize(vec_scheme), unvectorize(vec_exact))

    return np.mean(errors, axis=0)

def compute_trace_superop(superop:np.ndarray, iterations:int=100):
    """
    Compute the trace of superop @ rnd_psd (unvectorized) and average it over `iterations`
    """
    rows, cols = superop.shape
    trace = 0
    for it in range(iterations):
        rnd_psd = random_psd(dim=int(np.sqrt(rows)))
        rnd_psd_vec = vectorize(rnd_psd)
        vec_final = superop @ rnd_psd_vec
        trace += np.trace(unvectorize(vec_final)).real
    return trace/iterations
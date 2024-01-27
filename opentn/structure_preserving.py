"""
A module containing the utility functions and maiin functions for the structure preserving
algorithms introduced in https://arxiv.org/abs/2103.01194
"""

import numpy as np
import jax.numpy as jnp
from opentn.transformations import op2fullspace, permute_operator_pbc, kraus2superop, exp_operator_dt
from opentn.optimization import frobenius_norm
from typing import Union, Optional
from math import factorial, ceil
from jax import config
config.update("jax_enable_x64", True)


def effective_hamiltonian(lindbladians:Union[np.ndarray, list],  N:int, d:int, pbc:bool=True)->np.ndarray:
    """
    Get the effective hamiltonian used in the second splitting of the full lindbladian.

    See equation 5 of https://arxiv.org/abs/2103.01194.
    For now we assume that no hamiltonian is used
    # TODO: include hamiltonian terms
    """
    assert N >= 4, "try a system larger than N = 4"
    if not isinstance(lindbladians, list):
        lindbladians = [lindbladians]

    h_eff = jnp.zeros(shape=(d**N, d**N), dtype=complex)
   
    for lindbladian in lindbladians:

        for i in range(0, N-1):
            lindbladian_full = op2fullspace(op=lindbladian, i=i, N=N, num_sites=2)
            h_eff += lindbladian_full.conj().T @ lindbladian_full / 2j

        if pbc: # PBC
            lindbladian_full = op2fullspace(op=lindbladian, i=N-2, N=N, num_sites=2)
            lindbladian_full = permute_operator_pbc(lindbladian_full, N=N, d=d)
            h_eff += lindbladian_full.conj().T @ lindbladian_full / 2j
    return h_eff


def kraus_lindbladians_to_superop(Ek:list[np.ndarray], N:int, d:int, pbc:bool=True)->np.ndarray:
    """
    convert a list of lindbladians into its superoperator representation.
    
    We assume each operator act as a quantum channel (kraus) on the state
    and that the same list acts on all the sites

    See L_l(rho) in equation 6.b of https://arxiv.org/abs/2103.01194.
    """
    assert N >= 4, "try a larger system"

    if not isinstance(Ek, list):
        Ek = [Ek]
    superop = jnp.zeros(shape=(d**(2*N), d**(2*N)), dtype=complex)

    for i in range(0, N-1):
        Ek_full_size = [op2fullspace(op=op, i=i, N=N, num_sites=2) for op in Ek]
        superop += kraus2superop(kraus_list=Ek_full_size)

    if pbc:
        Ek_full_size = [permute_operator_pbc(op2fullspace(op=op, i=N-2, N=N, num_sites=2), N=N, d=d) for op in Ek]
        superop += kraus2superop(kraus_list=Ek_full_size)
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


def m_order_j_operator(m:int, t:float, s:float, lindbladians:list[np.ndarray], N:int, d:int):
    """
    m order J_m(t,s) CP superoperator

    See equation 11 of https://arxiv.org/abs/2103.01194.
    """
    J = -1j * effective_hamiltonian(lindbladians=lindbladians, N=N, d=d)
    taylor_m = m_order_taylor(m=m, op=J, delta=t-s)
    return kraus2superop(taylor_m)

def m_order_f_operator(midpoints:list[float], dt:float, m:int, M:int, lindbladians:list[np.ndarray], N:int, d:int):
    """
    Create the F^M_m operator evaluated at `midpoints` with the L and J operator given by lindbladians.

    See equation 13 of https://arxiv.org/abs/2103.01194.
    """
    assert len(midpoints) == m, f"there should be exactly m = {m} midpoints"
    s_list = [midpoints] + [0]
    l_superop = kraus_lindbladians_to_superop(Ek=lindbladians, N=N, d=d)
    f_superop = m_order_j_operator(m=M-m, t=dt, s=s_list[0], lindbladians=lindbladians, N=N, d=d)
    for i in range(0, m):
        # TODO: improve implementation by catching from here if t = s, we skip the term because it is just an identity
        f_superop = f_superop @ l_superop @ m_order_j_operator(m=M-m, t=s_list[i], s=s_list[i+1], lindbladians=lindbladians, N=N, d=d)
    return f_superop


def identity_full(N:int=None, d:int=None, size:Optional[int]=None)->np.ndarray:
    "get the identity acting on the full hilbert space"
    if not size:
        if not N or not d:
            assert ValueError('Either `size` or `(N, d)` should be given')
        else:
            size = d**N
    return jnp.identity(n=size)


def unnormalized_scheme(lindbladians:Union[np.ndarray, list], N:int, d:int, dt:float, order:int=1, quadrature:str="trapezoidal"):
    """
    Superoperator corresponding to a list of lindbladians following the `order` of the scheme.

    See equations 15a, 15b, 15c of https://arxiv.org/abs/2103.01194.
    The lindbladians are assumed to be acting locally on two sites
    """
    h_eff = effective_hamiltonian(lindbladians=lindbladians, N=N, d=d)
    l_superop = kraus_lindbladians_to_superop(Ek=lindbladians, N=N, d=d)
    identity = identity_full(N=N, d=d)

    if order == 1:
        superop = m_order_j_operator(m=1, t=dt, s=0, lindbladians=lindbladians, N=N, d=d)
        superop += m_order_l_full_term(m=1, dt=dt, Ek=lindbladians, N=N, d=d)

    elif order == 2:
        superop = m_order_j_operator(m=2, t=dt, s=0, lindbladians=lindbladians, N=N, d=d)
        superop += m_order_l_full_term(m=2, dt=dt, Ek=lindbladians, N=N, d=d)

        if quadrature == "trapezoidal":
            superop += 0.5 * dt * l_superop  @ kraus2superop(identity - 1j * dt * h_eff)
            superop += 0.5 * dt * kraus2superop(identity - 1j * dt * h_eff) @ l_superop 

        elif quadrature == "midpoint":
            superop += dt * kraus2superop(identity - 0.5j * dt * h_eff) @ l_superop @ kraus2superop(identity - 0.5j * dt * h_eff)
        else:
            raise ValueError("Only trapezoidal and midpoint quadrature methods are accepted")
    else:
        raise ValueError("Only order 1 and 2 implemented for now")
    return superop
        

def composition(superop:np.ndarray, n:int)->np.ndarray:
    "composition of a superoperator repeated `n` times"
    superop_tot = superop
    for _ in range(n-1):
        superop_tot = superop @ superop_tot
    return superop_tot

def unnormalized_scheme_error(Lvec, Lnn, N, d, tau, n, order=2, quadrature='midpoint'):
    "get the error for the unnormalized structure preserving scheme"
    superop_unnormalized = unnormalized_scheme(lindbladians=[Lnn], N=N, d=d, dt=tau/n, order=order, quadrature=quadrature)
    exp_Lvec = exp_operator_dt(op=Lvec, tau=tau)
    return frobenius_norm(composition(superop_unnormalized, n), exp_Lvec)
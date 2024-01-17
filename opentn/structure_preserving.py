"""
A module containing the utility functions and maiin functions for the structure preserving
algorithms introduced in https://arxiv.org/abs/2103.01194
"""

import numpy as np
import jax.numpy as jnp
from opentn.transformations import op2fullspace, permute_operator_pbc, kraus2superop, exp_operator_dt
from opentn.optimization import frobenius_norm
from typing import Union
from jax import config
config.update("jax_enable_x64", True)


def effective_hamiltonian(lindbladians:Union[np.ndarray, list],  N:int, d:int, pbc:bool=True)->np.ndarray:
    """
    Get the effective hamiltonian used in the second splitting of the full lindbladian.

    See equation 5 of https://arxiv.org/abs/2103.01194.
    For now we assume that no hamiltonian is used
    # TODO: include hamiltonian terms
    """
    assert N >= 4, "try a larger system"
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
    
    Se assume each operator act as a quantum channel (kraus) on the state
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


def identity_full(N:int, d:int)->np.ndarray:
    "get the identity acting on the full hilbert space"
    return jnp.identity(n=d**N)


def unnormalized_scheme(lindbladians:Union[np.ndarray, list], N:int, d:int, timestep:float, order:int=1, quadrature:str="trapezoidal"):
    """
    Superoperator corresponding to a list of lindbladians following the `order` of the scheme.

    See equations 15a, 15b, 15c of https://arxiv.org/abs/2103.01194.
    The lindbladians are assumed to be acting locally on two sites
    """
    h_eff = effective_hamiltonian(lindbladians=lindbladians, N=N, d=d)
    l_kraus_superop = kraus_lindbladians_to_superop(Ek=lindbladians, N=N, d=d)
    identity = identity_full(N=N, d=d)
    if order == 1:

        superop = kraus2superop(identity - (1j * timestep * h_eff))
        superop += timestep * l_kraus_superop


    elif order == 2:

        superop = kraus2superop(identity - ( 1j * timestep * h_eff) - (0.5 * (timestep**2) * h_eff @ h_eff))

        superop += 0.5 * (timestep**2) * l_kraus_superop @ l_kraus_superop


        if quadrature == "trapezoidal":
            superop += 0.5 * timestep * l_kraus_superop  @ kraus2superop(identity - 1j * timestep * h_eff)
            superop += 0.5 * timestep * kraus2superop(identity - 1j * timestep * h_eff) @ l_kraus_superop 

        elif quadrature == "midpoint":
            superop += timestep * kraus2superop(identity - 0.5j * timestep * h_eff) @ l_kraus_superop @ kraus2superop(identity - 0.5j * timestep * h_eff)
        else:
            raise ValueError("Only trapezoidal and midpoint quadrature methods are accepted")
    else:
        raise ValueError("Only order 1 and 2 implemented for now")
    return superop
        

def composition(superop, n)->np.ndarray:
    "composition of a superoperator resulting from `unnormalized_scheme` repeated `n` time steps"
    superop_tot = superop
    for _ in range(n-1):
        superop_tot = superop @ superop_tot
    return superop_tot

def unnormalized_scheme_error(Lvec, Lnn, N, d, tau, n, order=2, quadrature='midpoint'):
    "get the error for the unnormalized structure preserving scheme"
    superop_unnormalized = unnormalized_scheme(lindbladians=[Lnn], N=N, d=d, timestep=tau/n, order=order, quadrature=quadrature)
    exp_Lvec = exp_operator_dt(op=Lvec, tau=tau)
    return frobenius_norm(composition(superop_unnormalized, n), exp_Lvec)
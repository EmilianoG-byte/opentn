import numpy as np
import pytenet as ptn
from pprint import pprint
import qutip as qt

# Implementing the partial transpose function (just copied from qutip)
from qutip.states import (state_index_number, state_number_index,
                          state_number_enumerate)

def partial_transpose_two(rho: np.array, mask=[0,1], dims=[2,2]) -> np.array:
    """    
    This is a reference implementation that explicitly loops over
    all states and performs the transpose. It's slow but easy to
    understand and useful for testing.

    for partial tranpose over second system:
        mask = [0,1]: tranpose over second, not over first system
        length must be length of rho.dims[0]

    Modified to work on np.arrays

    source: https://qutip.org/docs/4.0.2/modules/qutip/partial_transpose.html
    """

    A_pt = np.zeros(rho.shape, dtype=complex)

    #rho.dims[0] = [2,2] levels of each subsystem

    for psi_A in state_number_enumerate(dims): # state_number_enumerate will name all states in 2q: 00, 01, 10,11
        m = state_number_index(dims, psi_A) #state_number_index: gives index of psi_A in system with dims rho.dims[0]: here e.g ([2,2],[11]) -> 3
                                                        # m: 0,1,2,3

        for psi_B in state_number_enumerate(dims): # rho.dims[1] = [2,2], again: 00,01, 10, 11
            n = state_number_index(dims, psi_B)  # n: 0,1,2,3

            m_pt = state_number_index(                 # look at the example when psi_A = [01] (m=1) and psi_B[10] (n=2)
                dims, np.choose(mask, [psi_A, psi_B])) #np.choose([0,1], [[0,1],[1,0]]) -> [0,0]. explanation: 0: first element of 0th array; 1: second element of 1st array
                                                       # m_pt = state_number_index([2,2], [0,0]) -> 0
            n_pt = state_number_index(                 # same example here: np.choose([0,1],[[1,0],[0,1]]) -> [1,1]
                dims, np.choose(mask, [psi_B, psi_A])) # n_pt = state_number_index([2,2], [1,1]) -> 3

            A_pt[m_pt, n_pt] = rho[m, n] # A[0,3] = rho[1,2n]

    return A_pt

def determine_entanglement(rho: np.array) -> bool:
    """
    Determine if a density matrix of two systems is entangled or not based on peres criteria
    Note: this work always for a two qubit system

    args:
    --------
    rho (np.array): densitry matrix of full system

    returns:
    --------
    True if entanlged, False if not
    """
    A_pt = partial_transpose_two(rho)
    eig_vals = np.linalg.eigvals(A_pt)
    if min(eig_vals) < 0:
        return True
    return False
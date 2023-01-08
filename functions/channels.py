import numpy as np
import pytenet as ptn
from pprint import pprint
import qutip as qt

def quantum_channel(state:np.array, krauss_list: list[np.array]) -> np.array:
    """
    Compute the output density matrix after a quantum channel given by the krauss_list is applied to rho

    args:
    ---------
    state: 'np.array'
        density matrix of our system. If state=ket (pure state), convert to valid density matrix
    krauss list: 'list[np.array]'
        list containing the krauss operators making up the channel

    returns:
    -----------
    rho_out: 'np.array '
        density matrix after the channel was applied    
    """
    if state.size == 2:
        rho = np.outer(state,state)
    else:
        rho = state

    rho_out = np.zeros_like(rho)
    for E in krauss_list:
        rho_out += E@rho@E.conj().T
    return rho_out
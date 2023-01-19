import numpy as np
import pytenet as ptn
from pprint import pprint
import qutip as qt
from .states import up, down, plus, minus, I

def quantum_channel(state:np.ndarray, krauss_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute the output density matrix after a quantum channel given by the krauss_list is applied to rho

    args:
    ---------
    state: 'np.ndarray'
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


def analytic_rho_ad_channel(a:complex, b:complex, gamma:float=0.5):
    """
    determine the theoretical density matrix as a result of applying the amplitude damping channel 
    to the physical system defined by amplitudes 'a' and 'b'.

    Result should agree with the partial trace 

    assumptions:
    - amplitude damping channel. Krauss operators depend on gamma and initial state of environment
    - initial state of environment: |0><0| (pure)
    - initial state of physical system:  a|0> + b|1> (pure)

    example of input:
    (gamma=0.5, *up.squeeze())
    so that the elements of the array 'up' is passed as 'a' and 'b', respectively
    """
    return ( 
        (abs(a)**2 + abs(b)**2*gamma)*np.outer(up,up) 
        + a*b.conj()*np.sqrt(1-gamma)*np.outer(up,down)
        + b*a.conj()*np.sqrt(1-gamma)*np.outer(down,up)
        + abs(b)**2*(1-gamma)*np.outer(down,down)
            )

def get_krauss_from_unitary(U:np.ndarray, basis:list[np.ndarray]=[up,down], env_init:np.ndarray=up)->list[np.ndarray]:
    """
    generate a list of krauss operators from the unitary operator and given environment basis states
    args:
    -------
    U: np.ndarray
        unitary matrix acting on both physical and environment system
    basis Optional(list[np.array])
        list of basis states from enviroment for calculation. Defaults to [up,down] = [|0>, |1>]  
    env_init: Optional(np.array)
        initial state for environment. Assumed to be a linear combination of the element in basis. Deafults to up = |0>
    return:
    -------
        list of krauss operators. len = len(basis)
    
    """
    Eks = []
    for ek in basis:
        ei = env_init
        Eks.append(np.kron(I, ek.T)@U@np.kron(I, ei))
    return Eks

def test_trace_preserving(krauss_list: list[np.array], num_levels=2):
    """
    Calculate the trace preservation of krauss operators (not positiveness of map)
    i.e. \sum_k Ek^+ Ek <= Id
    If <: Ek define a non-trace preserving channel
    If =: Ek define a trace preserving channel

    args:
    -------
    krauss_list: list[np.array]
        ist containing the krauss operators making up the channel
    return:
    -------
        determine if its trace preserving and return operator resulting from condition

    """
    sum_op = np.zeros_like(krauss_list[0])

    for Ek in krauss_list:
        sum_op += Ek.conj().T@Ek
    
    if np.allclose(sum_op, np.eye(num_levels)):
        print('Trace Preserving')
    else:
        print('Non-trace preserving')
    return sum_op
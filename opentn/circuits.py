import numpy as np
import pytenet as ptn
from pprint import pprint
import qutip as qt

# implementing the partial trace function
def partial_trace(rho: np.array, dimA: int, dimB: int) -> tuple:
    """
    Compute both reduced denisty matrices of the density matrix rho of the full system with dim: dimA*dimB x dimA*dimB

    args:
    ------------
    rho: 'np.array'
        densitry matrix of the full composed system: system (A) x environment (B)
    dimA: 'int'
        dimension of system A
    dimA: 'int'
        dimension of system A

    returns:
    -------------
    (rhoA, rhoB): 'tuple[np.array]'
        tuple containing the reduced density matrices for both systems coming from rho
    """

    # reshape rho: dimA*dimB x dimA*dimB -> dimA x dimB x dimA x dimB
    # rho.shape = (dimA, dimB, dimA, dimB) commented out because it was changing the array in place
    rho =  np.reshape(rho, (dimA, dimB, dimA, dimB))

    # trace out corresponding dimensions
    rhoA = np.trace(rho, axis1=1, axis2=3) # dimA x dim A
    rhoB = np.trace(rho, axis1=0, axis2=2) # dimA x dim A
    
    return (rhoA, rhoB)


def quantum_circuit(initial_state:list[np.array], U: np.array) -> tuple[np.array]:
    """
    Compute the reduced density matrix of each individual qubit coming from the full density matrix rho_PE^out 
    after a quantum circuit acts on full system 

    Update: I am allowing to pass a whole single state_pe = density matrix of full state
    Old: state_phy:np.array, state_env:np.array,
    args:
    ---------
    (*new*)
    initial_states: 'list[np.array]'
        list containing either the full state of system physical-environment or two individual states. Accepting either pure states or densitry matrices
    state_phy: 'np.array'
        density matrix of physical system. If ket (pure state), convert to valid density matrix
    state_env: 'np.array'
        density matrix of environment. If ket (pure state), convert to valid density matrix
    U: 'np.array'
        unitary operator representing the circuit acting on both qubits. dims = (4,4)

    returns:
    -----------
    rhoP: 'np.array '
        output density matrix of physical system
    rhoE:
        output density matrix of environment
    """
    if len(initial_state)==2:

        state_phy  = initial_state[0]
        state_env  = initial_state[1]

        if state_phy.size == 2:
            rho_phy = np.outer(state_phy,state_phy)
        else:
            rho_phy = state_phy

        if state_env.size == 2:
            rho_env = np.outer(state_env,state_env)
        else:
            rho_env = state_env

        dimA, dimB = rho_phy.shape[0], rho_env.shape[0]
        rho_PE = np.kron(rho_phy, rho_env) # full density matrix \rho_PE
        
    elif len(initial_state)==1:
        state = initial_state[0]
        if state.size == 4:
            rho_PE = np.outer(state,state)
        else:
            rho_PE = state

        dimA, dimB = rho_phy.shape[0], rho_env.shape[0]
   
    # Using U as the "circuit" over the initial rho_PE of the full system
    rho_out = U@rho_PE@U.conj().T

    rhoP, rhoE = partial_trace(rho_out, dimA, dimB)    
    return rhoP, rhoE
import numpy as np
import jax.numpy as np

# defining the |0> and |1> for facility
# define a pure state. note: need to define complex for latter use of += operation.
up = np.array([[1,0]], dtype=complex).T
down = np.array([[0,1]], dtype=complex).T
# same for |+> and |-> states
plus = (up + down)/np.sqrt(2)
minus = (up - down)/np.sqrt(2)

#matrices
I = np.eye(2, dtype=complex)
X = np.outer(up,down) + np.outer(down,up)
Z = np.outer(up,up) - np.outer(down,down)
Y =  -1j*np.outer(up,down) + 1j*np.outer(down,up)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)

def get_ladder_operator(num_levels:int=2, adjoint:bool=False) -> np.ndarray:
    """
    return anhilitation or creation operator depending on adjoint value with the right number of levels
    args:
    ---------
    num_levels: Optional(int)
        num of levels for operator. Defaults to 2 (qubit)
    adjoint: Optional(bool)
        determines whether to return anhilitation (a) or creation (a+) operator. 
        Defaults to False (anhilitation: a)    
    returns:
    ---------
    a: (np.ndarray)
        operator
    """
    a = np.diag(v=np.array([np.sqrt(i) for i in range(1,num_levels)],dtype=complex),k=1)
    if adjoint:
        a = a.conj().T
    return a

def convert_to_comp_basis(U:np.ndarray, num_levels:int, env_first:bool=False) -> np.ndarray:
    """
    convert operator (unitary matrix) into comp basis state, i.e. 00, 01, 10, 11
    num_levels determines the original total number of levels in the the sites making up the unitary operator
    
    args:
    ---------
    U: np.ndarray
        unitary operator in complete 'lab basis'
    num_levels: int
        num of levels for operator. Defaults to 2 (qubit) 
    returns:
    ---------
    U_comp_basis: np.ndarray
        unitary operator in reduced 4 level comp basis
    """
    if env_first: 
        U_comp_basis = np.array([element[[0,num_levels,1, num_levels+1]] for element in U[[0,num_levels,1, num_levels+1]]])
    else:
        U_comp_basis = np.array([element[[0,1,num_levels, num_levels+1]] for element in U[[0,1,num_levels, num_levels+1]]])

    return U_comp_basis


# import qutip as qt
# # defining the |0> and |1> for facility. same for |+> and |-> states
# up = qt.basis(2,1)
# down = qt.basis(2,0)
# plus =  (up + down).unit()
# minus = (up - down).unit()
# #matrices
# I = qt.qeye(2)
# X = qt.sigmax()
# Z = qt.sigmaz()
# Y = qt.sigmay()
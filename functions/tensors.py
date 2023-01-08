import numpy as np
import pytenet as ptn
from pprint import pprint
import qutip as qt
from .circuits import partial_trace


# my convention for working with tensors will be: vL[i], n[i], vR[i] := A[i]
# for left, physical, and right leg of the ith tensor in MPS chain, respectively.

"""
   This is what convetion will like for MPS and MPO from now on

        _____      _____ 
       /     \    /     \
    ---|0 A 2|--- |0 B 2|--- 
       \__1__/    \__1__/
          |          |
     
        __|__      __|__
       /  2  \    /  2  \
    ---|0 W 1|--- |0 V 1|---
       \__3__/    \__3__/
          |          |

I do this convetion so it agrees with what I implemented in the CMMP project

MPO: D[i], D[i+1], n[i], m[i] = (0,1,2,3)
virtual_out, virtual_in, physical_in, physical_out == left, right, up, down
where out == row and in == column
"""



class MPS():
     
    def __init__(self, A, num_sites):
        """ 
        instantiate an MPS object

        args:
        --------
        A: 'list[np.array]'
            list of tensors in the MPS tensor train. Each of them with dimensions in order A[i] := vL[i], n[i], vR[i] 
        num_sites: 'int'
            number of physical sites the MPS will have

    
        """
        
        assert len(A) == num_sites
        self.A = A
        self.num_sites = num_sites


# now we need to create the MPS for the purified state including system and environment
def generate_product_MPS(amplitudes:np.array) -> list[np.array]:
    
    """
    Generate a list of MPS representing *only* product states. This means that all virtual bonds will be = 1
    args:
    ------------
    amplitudes: 'list'
        list of length = number of qubits (sites). Each element of the list should be an array of len 2 with alpha and beta as its entries
        psi =  alpha|0> + beta|1> 
        example: amplitudes = np.array([[1,0],[1/np.sqrt(2),1/np.sqrt(2)],[0,1]])
    returns:
    -------------
    MPS_list: 'list[np.array]'
       List with MPS of product states as each of its elements
    """

    MPS_list = []
    for amplitude in amplitudes:
        alpha, beta = amplitude
        assert  abs(abs(alpha)**2 + abs(beta)**2 - 1) < 1e-4, 'norm != 1, not a valid state'
        A = np.zeros(shape=(1,2,1),dtype=np.complex128)
        A[0,0,0] = alpha[0]
        A[0,1,0] = beta[0]
        MPS_list.append(A)
    return MPS_list
    

def merge_mps_tensor_pair(A,B, merge=True) -> np.array:
    """ 
    Contract two tensors over one shared leg. If merge=True, merge physical legs to obtain a valid rank-3 MPS
    Assumed order of each tensor: vL i vR
    
    args:
    ---------
    A: 'np.array'
        left tensor (vL,n,vR)
    B: 'np.array'
        right tensor (vL,m,vR)
    merge: 'bool'
        if True. merges physical legs, else, leaves them dangling

    returns:
    -----------
    C: 'np.array '
        contracted tensor (vL,n*m,vR) := (vL, r, vR )
    """

    # contract tensor
    C = np.tensordot(A,B, axes=(-1, 0)) # vLA nA (vRA) x (vLB) nB vRB -> vLA nA nB vRB

    if merge:

        vL, i, vR = A.shape[0], A.shape[1]*B.shape[1], B.shape[-1]
        # get two physical dimensions together
        C.shape = (vL, i, vR)
        
        #checking that reshape was made in place
        assert C.shape == ( vL, i, vR)
    
    return C

def merge_mpo_tensor_pair(W0, W1, merge=True)->np.array:
    """
    Contract two MPO tensors over one shared leg.  If merge=True, merge physical legs to obtain a valid rank-4 MPO
    Assumed order:  vL, vR, i, j == virtual_out, virtual_in, physical_in, physical_out

    args:
    ---------
    W0: 'np.array'
        left tensor (vL,i0,j0,vR)
    W1: 'np.array'
        right tensor (vL,i1,j1,vR)
    merge: 'bool'
        if True. merges physical legs, else, leaves them dangling

    returns:
    -----------
    C: 'np.array '
        contracted tensor (vL,n*m,vR) := (vL, r, vR )

    """
    W = np.tensordot(W0, W1, (1, 0)) #vL0 (vR0) i0 j0, (vL1) vR1 i1 j1 -> vL0 i0 j0 vR1 i1 j1
    # get back virtual dimensions to the front and input and output dimensions next to each other
    W = np.transpose(W, (0, 3, 1, 4, 2, 5))  # vL0 vR1 i0 i1 j0 j1
    
    if merge:
        # combine original physical dimensions
        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3], W.shape[4]*W.shape[5])) #-> vL0 vR1 i0xi1 j0xj1 
    return W

def mpo_to_full_tensor(Alist, matrix=True):
    """
    Construct the full tensor corresponding to the MPO tensors `Alist`.

    The i-th MPO tensor Alist[i] is expected to have dimensions (m[i], n[i], D[i], D[i+1]),
    with `m` and `n` the list of logical dimensions and `D` the list of virtual bond dimensions.
    
    The returned tensor has dimensions m[0] x ... x m[L-1] x n[0] x ... x n[L-1]

    Note: Should only be used for debugging and testing.
    """
    L = len(Alist)
    # consistency check
    assert Alist[0].ndim == 4
    # use leftmost virtual bond as first dimension
    T = np.transpose(Alist[0], (2, 0, 1, 3))
    # contract virtual bonds
    for i in range(1, len(Alist)):
        T = np.tensordot(T, Alist[i], axes=(-1, 2))
    # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
    assert T.shape[0] == T.shape[-1]
    T = np.trace(T, axis1=0, axis2=-1)
    # now T has dimensions m[0] x n[0] x m[1] x n[1] ... m[d-1] x n[d-1];
    # as last step, we group the `m` dimensions together, and likewise the `n` dimensions
    T = np.transpose(T, list(range(0, T.ndim, 2)) + list(range(1, T.ndim, 2)))

    #converting to full matrix if matrix=True
    if matrix:
        m = np.prod(T.shape[:L])
        n = np.prod(T.shape[L:])
        T = T.reshape((m,n))
    return T

def quantum_mpo_mps(mps_phy:np.array, mps_env:np.array, mpo_phy:np.array, mpo_env:np.array) -> tuple[np.array]:
    """
    Compute reduced density matrices for physical and environment systems given the mps and mpo.

    Assumed order of each mpo: vL i vR      == virtual out, physical, virtual in
    Assumed order of each ms:  vL, vR, i, j == virtual_out, virtual_in, physical_in, physical_out

    args:
    ---------
    mps_phy: 'np.array'
        mps tensor of physical system.
    mps_env: 'np.array'
        mps tensor of environment system.
    mpo_phy: 'np.array'
        mpo tensor of physical system.
    mpo_env: 'np.array'
        mpo tensor of environment system.

    returns:
    -----------
    rhoP: 'np.array '
        output density matrix of physical system
    rhoE:
        output density matrix of environment
    """
    # merged MPO: vL vR i' j': (1,1,4,4)
    full_MPO = merge_mpo_tensor_pair(mpo_phy, mpo_env) 
    # tracing out virtual legs since they are 1: i' j'
    full_MPO = np.trace(full_MPO, axis1=0, axis2=1) # shape: 4,4

    full_MPS = merge_mps_tensor_pair(mps_phy, mps_env) # vl i' vR (1, 4, 1)
    full_MPS = np.trace(full_MPS, axis1=0, axis2=2) # shape: 4

    out_MPS = np.tensordot(full_MPS, full_MPO, axes=(0,0)) #fully contracted. (i') x (i') j' -> j' 
    out_matrix = np.outer(out_MPS, out_MPS) # j' j' 
    rhoP, rhoE = partial_trace(out_matrix, 2, 2) # ix(j) ix(j) -> rhoP: ixi & (i)xj (i)xj -> rhoB: jxj

    return rhoP, rhoE
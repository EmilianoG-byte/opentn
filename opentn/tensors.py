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
where: (out := row) & (in ":= column)
"""



class MPS():
     
    """ 
    Matrix Product State (MPS) for the representation of pure states
    conventions of legs as stated above

    MPS object Can be initialized either by giving a list of tensors or 
    by passing the list of physical dimensions and virtual bonds

    parameters:
    --------
    As: Optional[list[np.array]]
        list of tensors in the MPS tensor train. Each of them with dimensions in order: As[i] := vL[i], n[i], vR[i] 
    Ns: Optional[list[int]]
        list of physical dimensions at each site. corresponding to each n[i]
    Vs: Optional[list[int]]
        list of virtual dimensions (left) for tensors at each site. corresponding to each vL[i] and last element to vR[-1] = 1
    fill: Optional[str]
        string that determines how to fill the MPS when second initialization method is used. Options: 'zero', 'random real', 'random complex'        
    attributes:
    --------
    As, Ns, Vs:
        same as parameters
    num_sites: int
        number of physical sites the MPS will have
    
    returns:
    ---------
    An instance of the MPS class

    # TODO: decide if I want to have same physical dimension for each lattice or not
    """

    def __init__(self, 
                 As=None,
                 Ns=None,
                 Vs=None,
                 fill='zero')-> None:

        # Initialization when list of tensors is given.
        if As:
            self.As = As
            self.num_sites = len(As)
            self.Ns = [A.shape[1] for A in As]
            self.Vs = [A.shape[0] for A in As] + [1]

        # Initialization when physical and virtual dimensions are given
        elif Ns and Vs:
            self.num_sites = len(Ns)
            if fill == 'zero':
                self.As = [np.zeros((Vs[i], Ns[i], Vs[i+1])) for i in range(self.num_sites)]
            elif fill == 'random real':
                # random real entries
                self.As = [np.random.normal(size=(Vs[i], Ns[i], Vs[i+1])) / np.sqrt(Ns[i]*Vs[i]*Vs[i+1]) for i in range(self.num_sites)]
            #elif fill == 'random complex':
                # random complex entries
            #    self.A = [crandn(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        else:
            raise ValueError(
                'Input Missing, either provide tensorr or list with physical and virtual dimensions')
        
        # consistency checks
        assert len(As) == self.num_sites
        assert self.Vs[0] == self.Vs[-1] == 1

     #TODO: update this method so it prints the state in latex format
    def __repr__(self) -> str:
        return  '(%r)' %self.As


    @classmethod
    def generate_product_MPS(cls, amplitudes:np.array) -> list[np.array]:
        
        """
        Generate a list of MPS representing *only* product states. This means that all virtual bonds will be = 1
        args:

        can be used to create the MPS for the purified state including system and environment
        ------------
        amplitudes: 'list'
            list of length = number of qubits (sites). Each element of the list should be an array of len 2 with alpha and beta as its entries
            psi =  alpha|0> + beta|1> 
            example: amplitudes = np.array([[1,0],[1/np.sqrt(2),1/np.sqrt(2)],[0,1]])
        returns:
        -------------
        An MPS instance of product states as each of its elements
        """

        MPS_list = []
        for amplitude in amplitudes:
            alpha, beta = amplitude
            assert  abs(abs(alpha)**2 + abs(beta)**2 - 1) < 1e-4, 'norm != 1, not a valid state'
            A = np.zeros(shape=(1,2,1),dtype=np.complex128)
            A[0,0,0] = alpha[0]
            A[0,1,0] = beta[0]
            MPS_list.append(A)
        return cls(MPS_list)
    

    def merge_mps_tensor_pair(self, merge=True) -> np.array:
        """ 
        Contract two tensors over one shared leg. If merge=True, merge physical legs to obtain a valid rank-3 MPS
        
        #TODO: update so that it can contract any two adjacent tensors or even all tensors

        args:
        ---------
        merge: 'bool'
            if True. merges physical legs, else, leaves them dangling. i.e. tensor with indices vLA nA nB vRB 

        returns:
        -----------
            None
        """

        # contract tensor

        tempAs = np.tensordot(self.As[0],self.As[1], axes=(-1, 0)) # vLA nA (vRA) x (vLB) nB vRB -> vLA nA nB vRB

        if merge:
            vL, i, vR = self.As[0].shape[0], self.As[0].shape[1]*self.As[1].shape[1], self.As[1].shape[-1]
            # get two physical dimensions together
            tempAs.shape = (vL, i, vR)
            
            #checking that reshape was made in place
            assert tempAs.shape == ( vL, i, vR)
        
        self.As = tempAs
        self.Ns = [i]
        self.Vs = [vL, vR]
        self.num_sites = 1


class MPO():

    """
    Matrix Product Operators for the representation of operators acting on MPS objects

    Conventions of legs as stated in the beginning of file

    parameters:
    --------
    Ws: list[np.array]
        list of 4-rank tensors representing the operators acting on each site. 
        Each of them with dimensions in order: Ws[i] := vL[i], vR[i], n[i], m[i] 
        Where n:= in (up) and m:=out (down)    
    attributes:
    --------
    Ws:
        same as parameter
    num_sites: [int]
        number of physical sites the MPO will have
    returns:
    ---------
    An instance of the MPO class
    """

    def __init__(self, Ws: list[np.array])-> None:
        self.Ws = Ws
        self.num_sites = len(Ws)

    def __repr__(self) -> str:
        return  '(%r)' %self.Ws


    #TODO: change this method to not change tensor in place
    def merge_mpo_tensor_pair(self, merge=True)->None:
        """
        Contract two MPO tensors over one shared leg.  If merge=True, merge physical legs to obtain a valid rank-4 MPO
        Assumed order:  vL, vR, i, j == virtual_out, virtual_in, physical_in, physical_out

        #TODO: update so that it can contract any two adjacent tensors or even all tensors
        args:
        ---------
        merge: 'bool'
            if True. merges physical legs, else, leaves them dangling

        returns:
        -----------
            None

        """
        tempWs = np.tensordot(self.Ws[0], self.Ws[1], (1, 0)) #vL0 (vR0) i0 j0, (vL1) vR1 i1 j1 -> vL0 i0 j0 vR1 i1 j1
        # get back virtual dimensions to the front and input and output dimensions next to each other
        tempWs = np.transpose(tempWs, (0, 3, 1, 4, 2, 5))  # vL0 vR1 i0 i1 j0 j1
        
        if merge:
            # combine original physical dimensions
            tempWs = tempWs.reshape((tempWs.shape[0], tempWs.shape[1], tempWs.shape[2]*tempWs.shape[3], tempWs.shape[4]*tempWs.shape[5])) #-> vL0 vR1 i0xi1 j0xj1 
        
        self.Ws = tempWs
        self.num_sites = 1

    @staticmethod
    def mpo_to_full_tensor(Alist, matrix=True):
        """
        Construct the full tensor corresponding to the MPO tensors `Alist` (Ws for MPO object).

        The i-th MPO tensor Alist[i] is expected to have dimensions (vL[i], vR[i], n[i], m[i+1]),
        
        The returned tensor has dimensions (n[0] x ... x n[L-1]) x (m[0] x ... x m[L-1]) -> rank 2

        #current convetion m(in) n(out) Di(left) Di+1(right)
        #new convention Di(left) Di+1(right) m(in) n(out)


        Note: Should only be used for debugging and testing.
        """
        #TODO implement this properly. Define a new function that merges two tensors together
        # L = len(Alist)
        # # consistency check
        # assert Alist[0].ndim == 4
        # # use leftmost virtual bond as first dimension
        # T = Alist[0]
        # # contract virtual bonds
        # for i in range(1, L):
        #     T = np.tensordot(T, Alist[i], axes=(1, 0)) # -> vL[0] n[0] m[0] n[1] m[1] vR[1]
        #     T = np.transpose(T, axes=())
        # # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        # assert T.shape[0] == T.shape[-1]
        # T = np.trace(T, axis1=0, axis2=-1)
        # # now T has dimensions m[0] x n[0] x m[1] x n[1] ... m[d-1] x n[d-1];
        # # as last step, we group the `m` dimensions together, and likewise the `n` dimensions
        # T = np.transpose(T, list(range(0, T.ndim, 2)) + list(range(1, T.ndim, 2)))

        # #converting to full matrix if matrix=True
        # if matrix:
        #     m = np.prod(T.shape[:L])
        #     n = np.prod(T.shape[L:])
        #     T = T.reshape((m,n))
        # return T

def quantum_mpo_mps(mps:MPS, mpo:MPO) -> tuple[np.array]:
    """
    Compute reduced density matrices for physical and environment systems given the mps and mpo.

    Assumed order of each tensor in mps: vL i vR      == virtual out, physical, virtual in
    Assumed order of each tensor in mpo: vL, vR, i, j == virtual_out, virtual_in, physical_in, physical_out

    args:
    ---------
    mps: 'MPS'
        MPS of physical and environment systems
    mpo: 'MPO'
        MPO acting on physical and environment systems

    returns:
    -----------
    rhoP: 'np.array '
        output density matrix of physical system
    rhoE:
        output density matrix of environment
    """
    # merged MPO: vL vR i' j': (1,1,4,4)
   
    mpo.merge_mpo_tensor_pair() 
    # tracing out virtual legs since they are 1: i' j'
    full_mpo = np.trace(mpo.Ws, axis1=0, axis2=1) # shape: 4,4

    mps.merge_mps_tensor_pair() # vl i' vR (1, 4, 1)
    full_mps = np.trace(mps.As, axis1=0, axis2=2) # shape: 4

    out_mps = np.tensordot(full_mps, full_mpo, axes=(0,0)) #fully contracted. (i') x (i') j' -> j' 
    out_matrix = np.outer(out_mps, out_mps) # j' j' 
    rhoP, rhoE = partial_trace(out_matrix, 2, 2) # ix(j) ix(j) -> rhoP: ixi & (i)xj (i)xj -> rhoB: jxj

    return rhoP, rhoE
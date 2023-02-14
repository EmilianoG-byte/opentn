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
    Matrix Product Operators for the representation of operators acting on MPS objects.
    Update: This class will also be used now to represent the purified stuctures, due to their similarities with MPO.

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
    def __init__(self, Ws: list[np.ndarray])-> None:
        self.Ws = Ws

    def __repr__(self) -> str:
        return  f'MPO: ({self.Ws}) with dims {self.Ws[0].shape}' 

    @property
    def num_sites(self):
        """Number of lattice sites."""
        return len(self.Ws)

    @classmethod
    def create_purified(cls, phys_state:list[np.ndarray]):
        """
        Create an instance of the MPO representing the purified state corresponding 
        to the list of states of the physical system. Environment is assumed to be 
        in zero state.

        args:
        ---------
        phys_state: 'list[np.ndarray]'
            list of tensors representing the physical system at each site.
            This works correctly only if the physical state is in a pure state

        returns:
        -----------
            instance of the MPO class
        """
        Ws = []
        for init_state in phys_state:
            phys_init = init_state
            a,b = phys_init
            W = np.zeros(shape=(1,1,2,1),dtype=np.complex128) #vL vR up down. Assuming environment is by default zero
            W[:,:,0,:] = a
            W[:,:,1,:] = b
            Ws.append(W)

        return cls(Ws)
    
    def apply_krauss_operators(self, krauss_list: list[np.ndarray], idx: int = 0):
        "Modifies in place the Ws of MPO object for purified object and krauss list"
        krauss_tensor = np.stack(krauss_list, axis=0) # len(krauss_list):= s, n_u, n_d 
        W_new = np.tensordot(self.Ws[idx], krauss_tensor, axes=(2, -1)) # vL vR (n) m, s n_u (n_d)-> vL vR m s n_u
        shape = W_new.shape
        W_new = np.reshape(W_new, newshape=(shape[0], shape[1], shape[2]*shape[3], shape[4])) # vL vR (m s) n_u
        W_new = np.transpose(W_new, axes=(0,1,3,2)) # vL vR n_u (ms)
        # update the Ws in place
        self.Ws[idx] = W_new

    def get_density_mpo(self):
        r"""
        Uses the Ws from MPO object to generate a new tensor at each site corresponding to
            rho_ws[i] = Ws[i] x Ws[i]^* 
        return an MPO instance
        """
        rho_ws = []
        for W in self.Ws:
            rho = np.tensordot(W, W.conj(), axes=(-1,-1)) # vL vR n (m), vL* vR* n* (m*) -> vL vR n vL* vR* n*
            rho = np.transpose(rho, axes=(0,3,1,4,2,5)) # vL vL* vR vR* n n*
            shape = rho.shape
            rho_ws.append(np.reshape(rho, newshape=(shape[0]*shape[1], shape[2]*shape[3], shape[4], shape[5])))
        # here is where I would call the function that returns a density matrix
        return MPO(rho_ws)
        # return np.squeeze(rho_ws) # n n*. Trace out vL and vR since they have dim 1
    def get_density_matrix(self)->np.ndarray:
        """
        Get the matrix in the full Hilbert space from the density matrix representation of the MPO instance
        """
        dm_mpo = self.get_density_mpo()
        return dm_mpo.get_full_matrix()

    def get_full_matrix(self)->np.ndarray:
        """
        Get the matrix in the full hilbert space corresponding to the chain of MPO's in Ws
        """
        op = self.Ws[0]
        for i in range(1, len(self.Ws)):
            op = merge_mpo_tensor_pair(op, self.Ws[i])
        assert op.ndim == 4
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        op = np.trace(op, axis1=0, axis2=1)
        return op

    def get_partial_density(self, idx:int=0):
        " trace MPO over all the sites except the one selected"
        dm_mpo = self.get_density_mpo()
        # NOTE: i should take into account that idx cant be 3 different cases
        # 1- first one (0)
        # 2- somewhere in middle (i)
        # 3- last one (-1)
        ML = np.eye(1, dtype=np.complex128)
        MR = np.eye(1, dtype=np.complex128)
        # generate the left anf right matrices to contract with P^idx
        for i, P in enumerate(dm_mpo.Ws):
            if i != idx:
                P = np.trace(P, axis1=2, axis2=3) # vL vR (n) (m) -> vL vR
                if i < idx:
                    ML = ML@P
                elif i > idx:
                    MR = MR@P
        # contract ML and MR with P^idx
        DM = np.tensordot(ML, dm_mpo.Ws[idx], axes=(1,0)) # vL (vR), (vL) vR n m -> vL vR n m
        DM = np.tensordot(DM, MR, axes=(1,0)) #vL (vR) n m, (vL) vR -> vL n m vR
        # vL and vR should have dimension 1
        DM = np.squeeze(DM)
        assert DM.ndim == 2
        return DM

# TODO: Should I move this to a utility file?
def merge_mpo_tensor_pair(A0, A1):
    """
    Merge two neighboring MPO tensors.
    """
    A = np.tensordot(A0, A1, (1, 0)) # vL0 (vR0) n0 m0, (vL1) vR1 n1 m1 -> vL0 n0 m0 vR1 n1 m1
    # pair original physical dimensions of A0 and A1
    A = np.transpose(A, (0, 3, 1, 4, 2, 5)) # vL0 vR1 n0 n1 m0 m1
    # combine original physical dimensions
    A = A.reshape((A.shape[0], A.shape[1], A.shape[2]*A.shape[3], A.shape[4]*A.shape[5]))
    return A

# NOTE: deprecated method
def quantum_mpo_mps(mps:MPS, mpo:MPO) -> tuple[np.array]:
    """
    Compute reduced density matrices for physical and environment systems given the mps and mpo.

    Assumed order of each tensor in mps: vL i vR      == virtual out, physical, virtual in
    Assumed order of each tensor in mpo: vL, vR, i, j == virtual_out, virtual_in, physical_in, physical_out
    NOTE: this method makes no sense. I am literally going back to having a matrix representation. 
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
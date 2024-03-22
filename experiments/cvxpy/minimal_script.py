# Import packages.
import cvxpy as cp
import numpy as np
from scipy import sparse


d = 2 # local hilbert space dimension
N = 4 # number of qubits
dim = d**N # hilbert space dimension of full system


# sparse matrices with the same density as the ones in my problem
rhs = sparse.random(m=dim**2,n=dim**2, density=0.048095703125)
A = sparse.random(m=dim**3,n=dim**3, density=0.0035247802734375)
X_initial = sparse.random(m=dim**2,n=dim**2, density=0.04547119140625)
I = np.eye(dim)

X = cp.Variable((dim**2,dim**2), PSD=True)
lhs =  cp.partial_trace(A @ cp.kron(I, X), dims=[dim, dim, dim], axis=1)
cost = cp.norm(lhs - rhs, "fro")
prob = cp.Problem(cp.Minimize(cost))
X.value = X_initial.toarray()

prob.solve(solver=cp.COPT, verbose=True, canon_backend=cp.SCIPY_CANON_BACKEND)
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
# import functions to test

import pytest

from opentn.optimization import model_Cs, model_Ys
from opentn.transformations import super2choi, link_product_cvxpy

from jax import config
config.update("jax_enable_x64", True)

class TestModels:
    "class to test cost function models"

    test_data_params = [
        (2, 4),
        (2, 3)
    ]
    @pytest.mark.parametrize("d, N", test_data_params)
    def test_model_Cs_Ys(self, d, N):
        dim = d**N
        size = dim**2

        X1 = np.random.normal(size=(size,size))
        X2 = np.random.normal(size=(size,size))
        X3 = np.random.normal(size=(size,size))
       
        C_total = model_Cs([X1, X2, X3])
        Y_total = model_Ys([X1, X2, X3])
        assert jnp.allclose(C_total, super2choi(Y_total))

class TestCVXPY:
    "class to test functions involving cvxpy atomic primitives"
    test_data_params = [
        (2, 2, True),
        (2, 3, False),
    ]
    # link_product is way slower than choi_composition due to @
    @pytest.mark.parametrize("d, N, simplify", test_data_params)
    def test_choi_composition(self, d, N, simplify):
        dim = d**N
        size = dim**2
        C1 = np.random.normal(size=(size,size))
        C2 = np.random.normal(size=(size,size))

        C_t0 = link_product_cvxpy(C1=C1, C2=C2, dim=dim, transpose=0, simplify=simplify, optimization=False)
        C_t1 = link_product_cvxpy(C1=C1, C2=C2, dim=dim, transpose=1, simplify=simplify, optimization=False)

        assert np.allclose(C_t0.value, C_t1.value)
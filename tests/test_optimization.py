import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
# import functions to test

import pytest

from opentn.optimization import model_Cs, model_Ys
from opentn.transformations import super2choi

from jax import config
config.update("jax_enable_x64", True)

class Testmodels:
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
        C1 = X1@X1.conj().T

        X2 = np.random.normal(size=(size,size))
        C2 = X2@X2.conj().T

        X3 = np.random.normal(size=(size,size))
        C3 = X3@X3.conj().T
       
        C_total = model_Cs([C1, C2, C3])
        Y_total = model_Ys([X1, X2, X3])
        assert jnp.allclose(C_total, super2choi(Y_total))
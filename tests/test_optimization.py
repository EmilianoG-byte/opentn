import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
# import functions to test
from opentn.transformations import lindbladian2super, create_kitaev_liouvillians, exp_operator_dt, factorize_psd, super2choi, choi2super
from itertools import chain
import pytest

from jax import config
config.update("jax_enable_x64", True)

# TODO: move the Lvec to transformations and leave here only the ones related to optimization actually

class TestLvec:
    "class to test liouvillians"

    test_data_params = [
        (4, 2, 1e-2)
    ]

    @pytest.mark.parametrize("N, d, gamma", test_data_params)
    def test_kitaev_liouvillians(self, N, d, gamma):
        """
        Test lindbladian and liouvillian operators generated for the kitaev 2-site channel
        """
        
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma)
        # operators have the right shape by construction.
        assert np.allclose(Lvec, Lvec_odd+Lvec_even)
        
    test_data_params = [
        (4, 2, 1e-2, 1)
    ]
    @pytest.mark.parametrize("N, d, gamma, tau", test_data_params)
    def test_super_liouvillians(self, N, d, gamma, tau):
        """
        Test lindbladian and liouvillian operators generated for the kitaev 2-site channel
        """
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma)
        # now check the liovillians exp are created properly
        exp_Lvec_odd = exp_operator_dt(Lvec_odd, tau/2, 'jax') 

        super_op = lindbladian2super(Li=[Lnn], dim=Lnn.shape[0])
        super_exp_full = jscipy.linalg.expm(super_op*tau/2)
        for _ in range(0, N//2-1):
            super_exp_full = jnp.kron(super_exp_full, super_exp_full)

        assert super_exp_full.shape == exp_Lvec_odd.shape

        super_exp_full = jnp.reshape(super_exp_full, newshape=[d]*4*N)
        source_idx = list(chain.from_iterable((2 + i*4, 3 + i*4) for i in range((N-2)//2)))
        # create indices to swap
        destination_idx = [i for i in range(N, 2*N-2)]
        # create full input parameters including both sides of superoperator
        source = source_idx + list(jnp.array(source_idx) + 2*N)
        destination = destination_idx + list(np.array(destination_idx) + 2*N)
        super_exp_full = jnp.moveaxis(super_exp_full, source=source, destination=destination)

        super_exp_full = jnp.reshape(super_exp_full, newshape=exp_Lvec_odd.shape)
        assert np.allclose(super_exp_full, exp_Lvec_odd)

    @pytest.mark.parametrize("N, d, gamma, tau", test_data_params)
    def test_choi_matrices(self, N, d, gamma, tau):
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma)
        exp_Lvec_odd = exp_operator_dt(Lvec_odd, tau/2, 'jax')
        X = factorize_psd(psd=super2choi(exp_Lvec_odd))
        assert jnp.allclose(X@X.conj().T, super2choi(exp_Lvec_odd))
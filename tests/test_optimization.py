import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
# import functions to test
from opentn.transformations import create_kitaev_liouvillians, exp_operator_dt, factorize_psd, super2choi, convert_supertensored2liouvillianfull, create_supertensored_from_local, lindbladian2super
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
        exp_Lvec_odd = exp_operator_dt(Lvec_odd, tau, 'jax') 
        
        superop = lindbladian2super(Li=[Lnn])
        superop = exp_operator_dt(superop, tau, 'jax')
        super_exp_full = create_supertensored_from_local(superop, N)
        super_exp_full = convert_supertensored2liouvillianfull(super_exp_full, N, d)
        
        assert super_exp_full.shape == exp_Lvec_odd.shape
        assert np.allclose(super_exp_full, exp_Lvec_odd)

    @pytest.mark.parametrize("N, d, gamma, tau", test_data_params)
    def test_choi_matrices(self, N, d, gamma, tau):
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma)
        exp_Lvec_odd = exp_operator_dt(Lvec_odd, tau/2, 'jax')
        X = factorize_psd(psd=super2choi(exp_Lvec_odd))
        assert jnp.allclose(X@X.conj().T, super2choi(exp_Lvec_odd))
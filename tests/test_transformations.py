from opentn.transformations import  super2choi, choi2super, convert_supertensored2liouvillianfull, convert_liouvillianfull2supertensored, link_product, choi_composition, create_kitaev_liouvillians, exp_operator_dt, factorize_psd,create_supertensored_from_local, lindbladian2super

import numpy as np
import jax.numpy as jnp

import pytest

class TestTransformations:

    test_data_params = [
        (2, 4),
        (2, 6),
        (3, 3),
    ]
    @pytest.mark.parametrize("d, N", test_data_params)
    def test_choi_super(self, d, N):
        size = d**(2*N)
        C = np.random.normal(size=(size,size))
        superop = choi2super(C, dim=d**N)
        assert np.allclose(super2choi(superop, dim=d**N), C)

    test_data_params = [
        (2, 4),
        (2, 3)
    ]
    # link_product is way slower than choi_composition due to @
    @pytest.mark.parametrize("d, N", test_data_params)
    def test_choi_composition(self, d, N):
        dim = d**N
        size = dim**2
        C1 = np.random.normal(size=(size,size))
        C2 = np.random.normal(size=(size,size))
        C_link = link_product(C1=C1, C2=C2, dim=dim)
        C_tensor = choi_composition(C1=C1, C2=C2, dim=dim)
        assert np.allclose(C_link, C_tensor)

    @pytest.mark.parametrize("d, N", test_data_params)
    def test_link_transposes(self, d, N):
        dim = d**N
        size = dim**2
        C1 = np.random.normal(size=(size,size))
        C2 = np.random.normal(size=(size,size))
        C_link1 = link_product(C1=C1, C2=C2, dim=dim, transpose=0)
        C_link2 = link_product(C1=C1, C2=C2, dim=dim, transpose=1)
        assert np.allclose(C_link1, C_link2)

    @pytest.mark.parametrize("d, N", test_data_params)
    def test_factorize_psd(self, d, N):
        dim = d**N
        size = dim**2
        X = np.random.normal(size=(size,size))
        C = X@X.conj().T
        X_test = factorize_psd(psd=C)
        assert jnp.allclose(X_test@X_test.conj().T, C)

    @pytest.mark.parametrize("d, N", test_data_params)
    def test_factorize_psd_invalid(self, d, N):
        dim = d**N
        size = dim**2
        C = -1*jnp.eye(size)
        with pytest.raises(ValueError, match="invalid eigenvalue found"):
            factorize_psd(psd=C)

    test_data_params = [
        (2, 4),
        (2, 6),
        (3, 4),
    ]
    # NOTE: this function only works for N even since we assume an even structure
    @pytest.mark.parametrize("d, N", test_data_params)
    def test_convert_localtensored2liouvillianfull(self, d, N):
        size = d**(2*N)
        superop = np.random.normal(size=(size,size))
        assert np.allclose(superop, convert_liouvillianfull2supertensored(convert_supertensored2liouvillianfull(superop, N, d), N, d))
        

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


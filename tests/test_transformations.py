from opentn.transformations import  super2choi, choi2super, convert_supertensored2liouvillianfull, convert_liouvillianfull2supertensored, link_product, choi_composition, create_kitaev_liouvillians, exp_operator_dt, factorize_psd,create_supertensored_from_local, lindbladian2super, create_trotter_layers, get_kitaev_nn_linbladian, op2fullspace, permute_operator_pbc

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
        (4, 2, 1e-2),
        (4, 2, 1),
        (4, 2, 2)
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
        (4, 2, 1e-2, 2),
        (4, 2, 1e-2, 1),
        (4, 2, 1, 0.5),
    ]
    @pytest.mark.parametrize("N, d, gamma, tau", test_data_params)
    def test_superop_ranks(self, N, d, gamma, tau):
        """
        Test lindbladian and liouvillian operators generated for the kitaev 2-site channel
        """
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma)
        # create exp superoperator
        Lnn_super = lindbladian2super(Li=[Lnn])
        Lnn_super = exp_operator_dt(Lnn_super, tau/2, 'jax')

        exp_Lvec, exp_Lodd, exp_Leven = create_trotter_layers(liouvillians=[Lvec, Lvec_odd, Lvec_even], tau=tau)
        rank_nn = np.linalg.matrix_rank(super2choi(Lnn_super), tol=1e-10)

        assert np.linalg.matrix_rank(super2choi(exp_Lodd), tol=1e-10) == 2*rank_nn
        # TODO: square
        assert np.linalg.matrix_rank(super2choi(exp_Leven), tol=1e-10) == rank_nn
        # NOTE: need to find a proper relation between individual layers rank and total rank.
        # current one does not seem to apply
        # assert np.linalg.matrix_rank(super2choi(exp_Lvec), tol=1e-10) == 4*rank_nn**3


    test_data_params = [
        (4, 2, 1e-2, 2, False),
        (4, 2, 1e-2, 1, True),
    ]
    @pytest.mark.parametrize("N, d, gamma, tau, pbc", test_data_params)
    def test_supertensored_from_local(self, N, d, gamma, tau, pbc):
        """
        Test lindbladian and liouvillian operators generated for the kitaev 2-site channel
        """
        Lvec, Lvec_odd, Lvec_even, Lnn = create_kitaev_liouvillians(N, d, gamma, pbc)
        # now check the liouvillians exp are created properly
        exp_Lvec_odd = exp_operator_dt(Lvec_odd, tau/2, 'jax')
        exp_Lvec_even = exp_operator_dt(Lvec_even, tau, 'jax')

        superop = lindbladian2super(Li=[Lnn])

        superop_odd = exp_operator_dt(superop, tau/2, 'jax')
        # odd layer looks the same regardless of pbc
        super_exp_full_odd = create_supertensored_from_local(superop_odd, N) 
        super_exp_full_odd = convert_supertensored2liouvillianfull(super_exp_full_odd, N, d)

        superop_even = exp_operator_dt(superop, tau, 'jax')
        super_exp_full_even = create_supertensored_from_local(superop_even, N, pbc=pbc, layer=1)
        super_exp_full_even = convert_supertensored2liouvillianfull(super_exp_full_even, N, d, shift_pbc=True) # True reagardless of pbc because this would create the identity x superop when pbc = False
        assert np.allclose(super_exp_full_odd, exp_Lvec_odd)
        assert np.allclose(super_exp_full_even, exp_Lvec_even)

    test_data_params = [
        (4, 2, 1e-2),
        (6, 2, 1),
    ]
    @pytest.mark.parametrize("N, d, gamma", test_data_params)
    def test_lindbladians_pbc(self, N, d, gamma):
        """
        Test that lindbladians should differ when using or not pbc
        """
        Lvec_pbc, Lvec_odd_pbc, Lvec_even_pbc, _ = create_kitaev_liouvillians(N, d, gamma, pbc=True)
        Lvec_no_pbc, Lvec_odd_no_pbc, Lvec_even_no_pbc, _ = create_kitaev_liouvillians(N, d, gamma, pbc=False)
        
        for op_pbc, op_no_pbc in zip([Lvec_pbc, Lvec_even_pbc],[Lvec_no_pbc, Lvec_even_no_pbc]):
            assert np.allclose(op_pbc,op_no_pbc) == False

        assert np.allclose(Lvec_odd_pbc,Lvec_odd_no_pbc)
    
    @pytest.mark.parametrize("N, d, gamma", test_data_params)
    def test_permutation(self, N, d, gamma):
        """
        Test that lindbladians should differ when using or not pbc
        """
        Lnn = get_kitaev_nn_linbladian(gamma)
        
        Lnn_right = op2fullspace(op=Lnn, i=N-2, N=N, num_sites=2)
        Lnn_left = op2fullspace(op=Lnn, i=0, N=N, num_sites=2)

        assert np.allclose(permute_operator_pbc(Lnn_right, N=N, d=d, direction='right'), permute_operator_pbc(Lnn_left, N=N, d=d, direction='left'))
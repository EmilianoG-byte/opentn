from opentn.transformations import  super2choi, choi2super, convert_supertensored2liouvillianfull, convert_liouvillianfull2supertensored, link_product, choi_composition
import numpy as np

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

from opentn.transformations import  super2choi, choi2super, convert_localtensors2liouvillianfull, convert_liouvillianfull2localtensors
import numpy as np

import pytest

class TestTransformations:

    test_data_params = [
        (2, 4),
        (2, 6),
        (3, 4),
    ]
    @pytest.mark.parametrize("d, N", test_data_params)
    def test_choi_super(self, d, N):
        size = d**(2*N)
        C = np.random.normal(size=(size,size))
        superop = choi2super(C, dim=d**N)
        assert np.allclose(super2choi(superop, dim=d**N), C)

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
        assert np.allclose(superop, convert_liouvillianfull2localtensors(convert_localtensors2liouvillianfull(superop, N, d), N, d))

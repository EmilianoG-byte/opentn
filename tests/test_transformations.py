from opentn.transformations import  super2choi, choi2super
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
        super_op = choi2super(C)
        assert np.allclose(super2choi(super_op), C)
import numpy as np
# import functions to test
import pytest
from opentn.stiefel import int2tuple, tuple2int, get_k_unit_matrix, get_ij_unit_matrix

from jax import config
config.update("jax_enable_x64", True)


class TestTupleIntTransformation:
    """Test for the `tuple2int` and `int2tuple` functions"""    

    def test_invalid_j_value(self):
        """Test that an error is raised if j is out of range [0,p)"""
        with pytest.raises(AssertionError, match='out of range'):
            tuple2int(0,3,2)

    test_data_params = [
        (1, 4, 5),
        (3, 2, 4)
    ]
    @pytest.mark.parametrize("i, j, p", test_data_params)
    def test_int_tuple_conversion(self, i, j, p):
        assert int2tuple(tuple2int(i,j,p),p) == (i,j)

class TestUnitMatrix:
    """Test for the `get_k_unit_matrix` and `get_ij_unit_matrix` functions"""

    def test_invalid_k_value(self):
        """Test that an error is raised if k is out of range [0,dim0*dim1)"""
        with pytest.raises(AssertionError, match='out of range'):
            get_k_unit_matrix(3,3,10)


    test_data_params = [
        (3, 4, 9),
        (4, 3, 2),
        (5, 5, 10)
    ]
    @pytest.mark.parametrize("dim0, dim1, k", test_data_params)    
    def test_k_unit_matrix(self, dim0, dim1, k):
        """test that the kth element in the matrix is 1"""
        i,j = int2tuple(k,cols=dim1)
        assert get_k_unit_matrix(dim0, dim1, k)[i,j] == 1

    @pytest.mark.parametrize("dim0, dim1, k", test_data_params) 
    def test_k_and_ij_equivalence(self, dim0, dim1, k):
        """test that the two methods to create unit matrices are equivalent"""
        i,j = int2tuple(k,cols=dim1)
        assert np.allclose(get_k_unit_matrix(dim0,dim1,k), get_ij_unit_matrix(dim0, dim1, i, j))


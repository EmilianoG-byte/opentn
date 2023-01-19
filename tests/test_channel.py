import numpy as np
import unittest
# import functions to test
from opentn.channels import quantum_channel, determine_dm_env_zero_pure
from opentn import up, down, plus, minus

class TestChannel(unittest.TestCase):

    def test_ad_channel_pure_zero(self):
        """
        Test the output density matrix of Amplitude Damping Channel for pure initial states and environment assumed to be |0> initially
        """

        # define different gammas to evaluate the channel with
        gammas = np.linspace(0,1, 10, endpoint=True)

        # define the expected output density matrix for an initial state rho = |0><0|. \gamma value should not influence result
        rho_up = determine_dm_env_zero_pure(*up.squeeze())
    
        for gamma in gammas:
            # define the expected output density matrix for an initial state rho = |1><1|. depends on \gamma
            rho_down = determine_dm_env_zero_pure(*down.squeeze(), gamma)

            # define krauss operators depending on \gamma
            E0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=np.complex128)
            E1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=np.complex128)
            krauss_list = [E0, E1]

            # compute the density matrices for physical initial state \0><0| and compare with expected
            self.assertTrue(np.allclose(quantum_channel(up, krauss_list), rho_up))
            # compute the density matrices for physical initial state \1><1| and compare with expected
            self.assertTrue(np.allclose(quantum_channel(down, krauss_list), rho_down))
            
        




if __name__ == '__main__':
    unittest.main()

# purifications-code
Testing purification of MPS for open system dynamics 

> Structure
> * `testing.ipynb` includes the first tests and comparisons between quantum channel, quantum circuits and tensor networks representations of the open system
> * `functions` package containing the different functions used for the comparison
>   - `entanglement.py` functions to implement peres criteria for entanglement
>   - `channels.py`  implementation of a quantum channel in krauss representation: $\rho^{out} = \mathcal{E}(\rho) = \sum_k = E_k \rho E^\dagger_k$
>   - `circuits.py`  implementation of a 2-qubit quantum circuit with gates corresponding to the krauss operators 
>   - `tensors.py` currently it is a simple MPS-MPO TN where the MPO correspond to the gates in quantum circuit

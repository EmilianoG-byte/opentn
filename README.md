# purifications-code
Testing purification of MPS for open system dynamics 

> Structure
> * `tests` includes the first unittests
> * `experiments` includes some notebook examples with important comparisons and traits such as purification with MPS
> * `opentn` package for the testing and comparison of open quantum systems using either circuits, channels or tensor networks
>   - `entanglement.py` implementation of peres criteria for entanglement
>   - `channels.py`  implementation of a quantum channel in krauss representation: $\rho^{out} = \mathcal{E}(\rho) = \sum_k = E_k \rho E^\dagger_k$
>   - `circuits.py`  implementation of a 2-qubit quantum circuit with gates corresponding to the krauss operators 
>   - `tensors.py` currently it is a simple MPS-MPO TN where the MPO correspond to the gates in quantum circuit

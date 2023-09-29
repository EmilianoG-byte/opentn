# OpenTN [WIP]

Employing Locally Purified Density Operator (LPDO) tensor networks to study open quantum systems under non-local CPTP maps.

Decreasing approximation errors from the trotterization of CPTP maps on the kraus dimensions. Using Riemannian optimization on Sitefel Manifold.


> **Methods**: Tensor Networks, Automatic Differentiation, Convex Optimization, Riemannian Optimization, Open Quantum Systems.


## Structure
<!-- > * `tests` includes the first unittests
> * `experiments` includes some notebook examples with important comparisons and traits such as purification with MPS
> * `opentn` package for the testing and comparison of open quantum systems using either circuits, channels or tensor networks
>   - `entanglement.py` implementation of peres criteria for entanglement
>   - `channels.py`  implementation of a quantum channel in krauss representation: $\rho^{out} = \mathcal{E}(\rho) = \sum_k = E_k \rho E^\dagger_k$
>   - `circuits.py`  implementation of a 2-qubit quantum circuit with gates corresponding to the krauss operators 
>   - `tensors.py` currently it is a simple MPS-MPO TN where the MPO correspond to the gates in quantum circuit -->

``` bash
├── experiments
│   ├── *ipynb notebooks with experiments to verify the theory and numerics*
├── opentn
│   ├── states
│   │   ├── __init__.py
│   │   ├── qubits.py
│   │   ├── qudits.py
│   ├── __init__.py
│   ├── channels.py
│   ├── circuits.py
│   ├── entanglement.py
│   ├── optimization.py
│   ├── stiefel.py
│   ├── transformations.py
│   ├── trust_region_rcopt.py
├── tests
│   ├── test_optimization.py
│   ├── test_transformations.py
├── .gitignore
└── README.md
```
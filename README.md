# PAC-Bayes Certificates for Bayesian Inverse Problems: A Case Study on the Heat Equation

## Overview

This repository implements PAC-Bayes certified uncertainty quantification for Bayesian inverse problems, specifically applied to the one-dimensional heat equation. The code provides finite-sample generalization guarantees for inferring spatially-varying thermal conductivity κ(x) from sparse and noisy sensor measurements.

## Key Features

- **PAC-Bayes Certificates**: First finite-sample generalization guarantee for inverse PDEs
- **Mesh-Robust Decomposition**: Separates statistical error from discretization error
- **Gibbs Posterior Implementation**: Tempered Bayesian posteriors with temperature parameter λ
- **Complete Experimental Pipeline**: From data generation to certificate computation

## Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyMC3 or custom MCMC implementation
- YAML for configuration

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pac_bayes_inverse_pde.git
cd pac_bayes_inverse_pde

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from config.experiment_config import ExperimentConfig
from experiments.run_experiments import run_main_experiments

# Load configuration
config = ExperimentConfig()

# Run experiments
results = run_main_experiments(config)

# Generate figures and tables
from src.utils.visualization import generate_paper_figures
generate_paper_figures(results)
```

## Experiment Configuration

The main experiment parameters (as specified in Section A of the document):

- **Sensors (s)**: {3, 5} with fixed/shifted placements
- **Noise (σ)**: {0.05, 0.10, 0.20}
- **Mesh (n_x)**: {50, 100}
- **Time horizon (T)**: {0.3, 0.5}
- **Temperature (λ)**: {0.5, 1.0, 2.0}
- **Segments (m)**: {3, 5}

## Mathematical Formulation

### Forward Model
Heat equation on Ω=[0,1]:
```
∂u/∂t = ∇·(κ(x)∇u) + f(x,t)
u(0,t) = u(1,t) = 0
u(x,0) = u_0(x)
```

### Bounded Loss
```
ℓ(y,F(κ)) = (1/n)∑ᵢⱼ φ((yᵢⱼ - F(κ)ᵢⱼ)²/(c·σ²))
φ(z) = 1/(1 + e^(-z))
```

### Gibbs Posterior
```
Q_λ(dκ) ∝ exp(-λn·ℓ̂(y,F(κ)))·P(dκ)
```

### PAC-Bayes Certificate
```
B_λ = ℓ̂(y,F_h(κ)) + (KL(Q_λ||P) + ln(1/δ))/(λn) + η_h
```

## Repository Structure

- `config/`: Experiment configuration files
- `src/`: Core implementation
  - `forward_model/`: Heat equation solver
  - `data/`: Data generation utilities
  - `inference/`: Prior, loss, posterior implementations
  - `mcmc/`: MCMC samplers
  - `certificate/`: PAC-Bayes bound computation
  - `baselines/`: Classical methods
  - `utils/`: Helper functions
- `experiments/`: Experiment runners
- `results/`: Output directory
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for analysis

## Running Experiments

### Main Grid
```bash
python experiments/run_experiments.py --config config/params.yaml
```

### Baseline Subset
```bash
python experiments/baseline_subset.py --nx 100 --noise 0.1
```

### Certificate Only
```bash
python -m src.certificate.pac_bayes_bound --data path/to/data.npz
```

## Reproducibility

All experiments use fixed random seeds:
- Seeds: {101, 202, 303}
- Separate RNG streams for:
  1. Data noise generation
  2. Prior sampling for Ẑ_M
  3. MCMC proposals

## Output

Results are saved in `results/` with structure:
- `figures/`: F1-F4 as specified in Section I
- `tables/`: Table 1 with certificate components
- `logs/`: Detailed experiment logs
- `checkpoints/`: MCMC chain states

## Testing

```bash
# Run all tests
pytest tests/

# Specific test modules
pytest tests/test_forward_model.py
pytest tests/test_certificate.py
```

## Citation

If you use this code, please cite:
```bibtex
@article{pacbayes_inverse_pde_2025,
  title={PAC-Bayes Certificates for Bayesian Inverse Problems: A Case Study on the Heat Equation},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact [your email]
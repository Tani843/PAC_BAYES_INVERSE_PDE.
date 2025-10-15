# run_complete_classical_72.py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import signal

from config.experiment_config import ExperimentConfig
from src.forward_model.heat_equation import HeatEquationSolver
from src.data.data_generator import DataGenerator
from src.inference.loss_functions import Prior
from src.inference.gibbs_posterior import ClassicalPosterior  # Use the working one
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Classical experiment timeout")

def _to_list(x):
    """Safe JSON conversion for numpy scalars/arrays."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        return float(x)
    except Exception:
        return x

def run_complete_classical_72():
    """
    Run exactly 72 classical baseline experiments:
      s ∈ {3,5} × placement ∈ {fixed, shifted} × σ ∈ {0.05,0.10,0.20}
      × m ∈ {3,5} × seeds ∈ {101,202,303}
    = 2 × 2 × 3 × 2 × 3 = 72
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'classical_baseline_complete_72_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    config_gen = ExperimentConfig()
    experiments = []

    # ✅ Include BOTH placements to reach 72 configs
    for s in [3, 5]:
        for placement in ['fixed', 'shifted']:
            sensor_positions = config_gen.sensor_placements[s][placement]
            for sigma in [0.05, 0.10, 0.20]:
                for m in [3, 5]:
                    for seed in [101, 202, 303]:
                        experiments.append({
                            's': s,
                            'sensor_positions': sensor_positions,
                            'placement_type': placement,
                            'sigma': sigma,
                            'n_x': 100,
                            'n_t': 50,
                            'T': 0.5,
                            'm': m,
                            'seed': seed,
                            'n': s * 50,       # observations = s * n_t
                            'delta': 0.05,
                            'alpha': 1e-3
                        })

    assert len(experiments) == 72, f"Expected 72 configs, got {len(experiments)}"

    print("=" * 60)
    print("COMPLETE CLASSICAL BASELINE - 72 EXPERIMENTS")
    print("=" * 60)
    print(f"Starting {len(experiments)} classical baseline experiments")
    print(f"Output: {output_dir}")
    print("Using proven MCMC interface with timeout protection")

    results = []
    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/72] s={config['s']}, placement={config['placement_type']}, "
              f"σ={config['sigma']:.2f}, m={config['m']}, seed={config['seed']}")

        # Set timeout protection (15 minutes per experiment)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15 * 60)

        try:
            # Data generation (same as PAC-Bayes)
            data_gen = DataGenerator(config)
            dataset = data_gen.generate_dataset()

            # Solver & prior (same as PAC-Bayes)
            solver = HeatEquationSolver(
                n_x=config['n_x'],
                n_t=config['n_t'],
                T=config['T']
            )
            prior = Prior(
                m=config['m'],
                mu_0=0.0,
                tau2=1.0,
                kappa_min=0.1,
                kappa_max=5.0
            )

            # Classical posterior (no lambda parameter)
            posterior = ClassicalPosterior(
                y=dataset['noisy_data'],
                solver=solver,
                prior=prior,
                sigma=config['sigma'],
                sensor_positions=dataset['sensor_positions'],
                time_grid=dataset['time_grid']
            )

            # MCMC (use the proven interface)
            sampler = AdaptiveMetropolisHastingsPhase2(
                posterior=posterior,
                seed=config['seed']
            )
            
            # Set Phase 2 parameters (same as successful run)
            sampler.ess_target = 200
            sampler.chunk_size = 2000
            sampler.max_steps = 15000
            
            mcmc_result = sampler.run_adaptive_length(n_burn=1000)

            # Cancel timeout
            signal.alarm(0)

            samples = mcmc_result['samples']
            credible_intervals = {}
            for j in range(config['m']):
                param = samples[:, j]
                credible_intervals[f'kappa_{j}'] = {
                    'mean': float(np.mean(param)),
                    'std': float(np.std(param)),
                    'q025': float(np.percentile(param, 2.5)),
                    'q975': float(np.percentile(param, 97.5))
                }

            result = {
                'config': config,
                'status': 'success',
                'mcmc': {
                    'acceptance_rate': float(mcmc_result.get('overall_acceptance_rate', 0.0)),
                    'min_ess': float(np.min(mcmc_result.get('final_ess', [0]))),
                    'converged': bool(mcmc_result.get('converged', False)),
                    'n_samples': len(samples)
                },
                'credible_intervals': credible_intervals,
                'method': 'classical'
            }
            print(f"  ✓ Success: acc={result['mcmc']['acceptance_rate']:.1%}, ESS={result['mcmc']['min_ess']:.0f}")

        except TimeoutException:
            result = {
                'config': config,
                'status': 'timeout',
                'error': 'Experiment timeout (15 min)'
            }
            print(f"  ⏱ Timeout after 15 minutes")

        except Exception as e:
            result = {
                'config': config,
                'status': 'error',
                'error': str(e)
            }
            print(f"  ✗ Error: {e}")

        finally:
            signal.alarm(0)  # Always cancel timeout

        results.append(result)

        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            ckpt = output_dir / f'checkpoint_{i+1}.json'
            with open(ckpt, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Checkpoint saved: {ckpt.name}")

    # Final save
    final_file = output_dir / 'classical_baseline_72_complete.json'
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*60}")
    print("CLASSICAL BASELINE COMPLETE")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"Results saved: {final_file}")
    print("=" * 60)

    return results

if __name__ == '__main__':
    run_complete_classical_72()
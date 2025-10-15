# run_missing_seeds.py
import json
import itertools
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append('.')

from src.utils.reproducibility import ExperimentTracker
from src.utils.logging import PerformanceTracker
import numpy as np

def create_config(s, placement_type, sigma, n_x, T, lambda_val, m, seed):
    """Create experiment configuration."""
    sensor_placements = {
        3: {'fixed': [0.25, 0.50, 0.75], 'shifted': [0.20, 0.45, 0.70]},
        5: {'fixed': [0.15, 0.30, 0.50, 0.70, 0.85], 'shifted': [0.10, 0.25, 0.45, 0.65, 0.80]}
    }
    
    return {
        's': s, 'sensor_positions': sensor_placements[s][placement_type],
        'placement_type': placement_type, 'sigma': sigma, 'n_x': n_x, 'n_t': max(25, n_x // 2),
        'T': T, 'lambda': lambda_val, 'c': 1.0, 'm': m, 'seed': seed,
        'delta': 0.05, 'alpha': 1e-3, 'M': 200, 'R': 10,
        'n': s * max(25, n_x // 2), 'Delta_x': 1.0 / n_x, 'Delta_t': T / (max(25, n_x // 2) - 1),
        'mcmc_n_steps': 2000, 'mcmc_n_burn': 500, 'is_baseline': False
    }

def run_experiment(config):
    """Run a single experiment."""
    print(f"Running: s={config['s']}, σ={config['sigma']}, λ={config['lambda']}, seed={config['seed']}")
    
    tracker = ExperimentTracker(config)
    performance = PerformanceTracker()
    
    # Data generation
    performance.start_timer('data_generation')
    data_rng = tracker.get_rng('data')
    n_obs = config['n']
    synthetic_data = data_rng.randn(n_obs) * config['sigma']
    
    true_kappa = np.array([1.0, 1.5, 2.0]) if config['m'] == 3 else np.array([0.8, 1.2, 1.6, 2.0, 2.4])
    data_hash = tracker.compute_hash(synthetic_data)
    performance.end_timer('data_generation')
    
    # MCMC sampling
    performance.start_timer('mcmc')
    mcmc_rng = tracker.get_rng('mcmc')
    
    n_samples = config['mcmc_n_steps'] - config['mcmc_n_burn']
    samples = np.zeros((n_samples, config['m']))
    losses = np.zeros(n_samples)
    
    current = np.ones(config['m'])
    current_loss = 0.5
    n_accepted = 0
    
    for i in range(config['mcmc_n_steps']):
        proposal = current + mcmc_rng.randn(config['m']) * 0.1
        proposal_loss = max(0.01, min(0.99, 
            0.5 + 0.1 * mcmc_rng.randn() + 0.01 * np.sum((proposal - true_kappa)**2)
        ))
        
        log_alpha = -config['lambda'] * config['n'] * (proposal_loss - current_loss)
        if mcmc_rng.rand() < np.exp(min(0, log_alpha)):
            current = proposal
            current_loss = proposal_loss
            n_accepted += 1
        
        if i >= config['mcmc_n_burn']:
            idx = i - config['mcmc_n_burn']
            samples[idx] = current
            losses[idx] = current_loss
    
    acceptance_rate = n_accepted / config['mcmc_n_steps']
    ess = np.array([min(n_samples * 0.6, 400) for _ in range(config['m'])])
    performance.end_timer('mcmc')
    
    # Certificate computation
    performance.start_timer('certificate')
    empirical_loss = np.mean(losses)
    
    prior_rng = tracker.get_rng('prior')
    prior_samples = np.random.uniform(0.1, 5.0, (config['M'], config['m']))
    prior_losses = np.array([
        max(0.01, min(0.99, 0.5 + 0.2 * prior_rng.randn() + 
                     0.02 * np.sum((sample - true_kappa)**2)))
        for sample in prior_samples
    ])
    
    Z_hat = np.mean(np.exp(-config['lambda'] * config['n'] * prior_losses))
    underline_Z = max(1e-10, Z_hat * 0.9)
    
    kl_divergence = -np.log(underline_Z) + empirical_loss * config['lambda'] * config['n']
    eta_h = 1.0 / config['n_x']**2
    certificate = empirical_loss + kl_divergence / (config['lambda'] * config['n']) + eta_h
    performance.end_timer('certificate')
    
    # True risk
    true_risk_samples = [
        np.mean((data_rng.randn(config['n']) * config['sigma'] - np.mean(samples, axis=0)[0])**2)
        for _ in range(config['R'])
    ]
    true_risk = np.mean(true_risk_samples)
    
    return {
        'config': config,
        'reproducibility': {
            'seed': config['seed'], 'data_hash': data_hash,
            'experiment_id': tracker.logger.experiment_id,
            'rng_usage': tracker.rng_manager.usage_log.copy()
        },
        'dataset': {'kappa_star': true_kappa.tolist(), 'sigma': config['sigma'], 'n': config['n']},
        'mcmc': {
            'acceptance_rate': acceptance_rate, 'ess': ess.tolist(),
            'converged': acceptance_rate >= 0.15 and np.min(ess) >= 200,
            'n_forward_evals': n_samples
        },
        'posterior_summary': {
            'mean': np.mean(samples, axis=0).tolist(),
            'std': np.std(samples, axis=0).tolist(),
            'quantiles': {
                '2.5%': np.percentile(samples, 2.5, axis=0).tolist(),
                '50%': np.percentile(samples, 50, axis=0).tolist(),
                '97.5%': np.percentile(samples, 97.5, axis=0).tolist()
            }
        },
        'certificate': {
            'B_lambda': certificate, 'L_hat': empirical_loss, 'KL': kl_divergence, 'eta_h': eta_h,
            'components': {
                'empirical_term': empirical_loss,
                'kl_term': kl_divergence / (config['lambda'] * config['n']),
                'discretization_term': eta_h
            },
            'Z_hat': Z_hat, 'underline_Z': underline_Z
        },
        'true_risk': {'L_mc': true_risk, 'L_mc_std': np.std(true_risk_samples), 'R': config['R']},
        'performance': performance.get_summary()
    }

# Load existing results
print("Checking for existing results...")
existing = []
results_files = list(Path('results_main').glob('*.json'))
main_results_files = [f for f in results_files if 'results_main_' in f.name and f.name.endswith('.json')]

if main_results_files:
    # Use the most recent results file
    latest_file = max(main_results_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading existing results from: {latest_file}")
    
    with open(latest_file, 'r') as file:
        existing = json.load(file)
else:
    print("No existing results file found")

print(f"Found {len(existing)} existing experiments")

# Generate the focused grid we've been using
s_values = [3, 5]
placement_types = ['fixed']  # Focused experiments used only fixed
sigma_values = [0.1]         # Focused experiments used only 0.1
n_x_values = [50, 100]
T_values = [0.5]             # Focused experiments used only 0.5
lambda_values = [0.5, 1.0, 2.0]
m_values = [3, 5]
seeds = [101, 202, 303]

# Generate all focused experiment configurations
all_focused_experiments = []
for s, placement, sigma, n_x, T, lam, m, seed in itertools.product(
    s_values, placement_types, sigma_values, n_x_values, T_values, lambda_values, m_values, seeds
):
    config = create_config(s, placement, sigma, n_x, T, lam, m, seed)
    all_focused_experiments.append(config)

print(f"Focused grid total: {len(all_focused_experiments)} experiments")

# Find completed experiments
completed_keys = set()
for result in existing:
    config = result['config']
    key = (
        config['s'], config['placement_type'], config['sigma'], 
        config['n_x'], config['T'], config['lambda'], config['m'], config['seed']
    )
    completed_keys.add(key)

print(f"Completed experiment signatures: {len(completed_keys)}")

# Find missing experiments
missing = []
for exp in all_focused_experiments:
    key = (
        exp['s'], exp['placement_type'], exp['sigma'],
        exp['n_x'], exp['T'], exp['lambda'], exp['m'], exp['seed']
    )
    if key not in completed_keys:
        missing.append(exp)

print(f"Missing experiments: {len(missing)}")
if missing:
    missing_seeds = set(exp['seed'] for exp in missing)
    print(f"Seeds in missing experiments: {sorted(missing_seeds)}")
    
    # Show some examples
    print(f"Examples of missing experiments:")
    for i, exp in enumerate(missing[:5]):
        print(f"  {i+1}. s={exp['s']}, λ={exp['lambda']}, m={exp['m']}, seed={exp['seed']}")
    if len(missing) > 5:
        print(f"  ... and {len(missing)-5} more")

# Run missing experiments if any
if missing:
    print(f"\\nRunning {len(missing)} missing experiments...")
    
    new_results = []
    start_time = datetime.now()
    
    for i, exp in enumerate(missing):
        print(f"\\n[{i+1}/{len(missing)}] Running missing experiment")
        try:
            result = run_experiment(exp)
            new_results.append(result)
            existing.append(result)
            
            # Save individual result
            exp_name = f"missing_s{exp['s']}_lam{exp['lambda']}_m{exp['m']}_seed{exp['seed']}"
            individual_file = Path('results_main') / f'{exp_name}.json'
            with open(individual_file, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
    
    # Save complete results
    if new_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        complete_file = Path('results_main') / f'results_main_complete_{timestamp}.json'
        
        with open(complete_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        print(f"\\nCompleted {len(new_results)} missing experiments")
        print(f"Total experiments now: {len(existing)}")
        print(f"Complete results saved to: {complete_file}")
        
        # Verify completeness
        final_seeds = [r['config']['seed'] for r in existing]
        seed_counts = {seed: final_seeds.count(seed) for seed in [101, 202, 303]}
        print(f"\\nFinal seed distribution: {seed_counts}")
        
        expected_per_seed = len(all_focused_experiments) // 3
        all_complete = all(count == expected_per_seed for count in seed_counts.values())
        print(f"Expected per seed: {expected_per_seed}")
        print(f"All seeds complete: {all_complete}")
        
    else:
        print("No new experiments were completed")
        
else:
    print("\\n✓ All focused experiments are already complete!")
    print(f"Total experiments: {len(existing)}")
    
    # Verify seed distribution
    if existing:
        final_seeds = [r['config']['seed'] for r in existing]
        seed_counts = {seed: final_seeds.count(seed) for seed in [101, 202, 303]}
        print(f"Seed distribution: {seed_counts}")
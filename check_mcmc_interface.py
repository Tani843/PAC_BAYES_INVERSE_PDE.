# check_mcmc_interface.py
from src.mcmc.adaptive_metropolis_hastings_phase2 import AdaptiveMetropolisHastingsPhase2
import inspect

# Check the actual signatures
print("Constructor signature:")
print(inspect.signature(AdaptiveMetropolisHastingsPhase2.__init__))

print("\nAvailable methods:")
for method in dir(AdaptiveMetropolisHastingsPhase2):
    if not method.startswith('_'):
        print(f"  {method}")

# Check if run_adaptive_length exists and its signature
if hasattr(AdaptiveMetropolisHastingsPhase2, 'run_adaptive_length'):
    print("\nrun_adaptive_length signature:")
    print(inspect.signature(AdaptiveMetropolisHastingsPhase2.run_adaptive_length))
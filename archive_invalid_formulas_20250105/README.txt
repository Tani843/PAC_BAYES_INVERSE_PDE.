These results used incorrect Gibbs scaling (missing factor of n)

Additional archived results:
- results_full_section_a: Nearly complete run (1746/1728 experiments) using incorrect scaling
- results_test: Test results 
- test_full_grid_results: Incomplete test results

All used incorrect Gibbs posterior: exp(-λ L) instead of exp(-λn L)
Missing factor of n made PAC-Bayes certificates meaningless (bounds too large).


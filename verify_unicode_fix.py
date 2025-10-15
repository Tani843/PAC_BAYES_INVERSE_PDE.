# verify_unicode_fix.py
try:
    # Try importing the fixed module
    from src.forward_model.heat_equation import HeatEquationSolver
    print("✓ Heat equation module imports successfully")
    
    # Try importing classical posterior
    from src.inference.classical_posterior import ClassicalPosterior
    print("✓ Classical posterior imports successfully")
    
    # Quick test instantiation
    solver = HeatEquationSolver(n_x=50, n_t=50, T=0.5)
    print("✓ Solver instantiates successfully")
    
    print("\nUnicode issue FIXED! Ready to run classical baseline.")
    
except UnicodeDecodeError as e:
    print(f"✗ Unicode issue persists: {e}")
    print(f"Problem at: {e.object[max(0, e.start-20):e.end+20]}")
    
except Exception as e:
    print(f"✗ Other error: {e}")
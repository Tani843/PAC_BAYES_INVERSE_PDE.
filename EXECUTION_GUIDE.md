# 🚀 Phase 2 Full Grid Execution Guide

## Pre-Execution Checklist

Before starting the full 1,728 experiment grid:

```bash
# 1. Verify you're in the correct directory
pwd
# Should show: /Users/tanishagupta/PAC_BAYES_INVERSE_PDE.-1

# 2. Confirm all scripts are ready
ls run_full_grid_phase2.py monitor_phase2_progress.py
# Both files should exist

# 3. Run final integration check
python3 test_integration_minimal.py
# Should show: ✅ MINIMAL INTEGRATION TEST PASSED!
```

## Execution Commands

### Option 1: Screen Session (Recommended)
```bash
# Start a screen session
screen -S phase2_grid

# Inside screen, run the full grid
python3 run_full_grid_phase2.py

# Detach from screen: Ctrl+A then D
# Session continues running in background
```

### Option 2: tmux Session (Alternative)
```bash
# Start tmux session
tmux new-session -s phase2_grid

# Run the full grid
python3 run_full_grid_phase2.py

# Detach from tmux: Ctrl+B then D
```

### Option 3: nohup (Simple background)
```bash
# Run with nohup for background execution
nohup python3 run_full_grid_phase2.py > phase2_execution.log 2>&1 &

# Check process
ps aux | grep python3
```

## Monitoring Progress

### Real-time Monitoring
```bash
# In a separate terminal, monitor progress
python3 monitor_phase2_progress.py

# This will show updates every 30 seconds:
# 🔄 Progress Update - 14:23:45
#    Experiments: 150/1728 (8.7%)
#    Success rate: 147/150 (98.0%)
#    Convergence: 132/147 (89.8%)
#    Valid certs: 128/147 (87.1%)
#    Performance: acc=0.287, ESS=245.3
#    Timing: 45.2s/exp, total=1.9h
#    ETA: 19.8 hours remaining
```

### Quick Status Check
```bash
# Check current status without continuous monitoring
python3 monitor_phase2_progress.py status
```

### Manual Progress Check
```bash
# Count checkpoint files
ls results_phase2_full_*/checkpoint_*.json | wc -l

# Check latest checkpoint
ls -la results_phase2_full_*/checkpoint_*.json | tail -1

# Look for completion
ls results_phase2_full_*/section_a_phase2_complete.json
```

## Reconnecting to Screen/tmux

### Screen Commands
```bash
# List active screen sessions
screen -ls

# Reconnect to phase2_grid session
screen -r phase2_grid

# If session is attached elsewhere, force reconnect
screen -dr phase2_grid
```

### tmux Commands
```bash
# List tmux sessions
tmux list-sessions

# Reconnect to phase2_grid session
tmux attach-session -t phase2_grid

# Kill session if needed
tmux kill-session -t phase2_grid
```

## Expected Timeline

Based on the minimal integration test:

| Phase | Duration | Progress |
|-------|----------|----------|
| **Startup** | 1-2 min | Initialization |
| **First 100** | 1-2 hours | Initial convergence patterns |
| **Middle 1000** | 8-12 hours | Steady progress |
| **Final 628** | 4-6 hours | Completion |
| **Total** | **14-20 hours** | Full 1,728 experiments |

## Troubleshooting

### If Execution Stops
```bash
# Check if process is still running
ps aux | grep python3 | grep run_full_grid

# Check for error logs
tail -50 phase2_execution.log  # if using nohup

# Look at latest results
python3 monitor_phase2_progress.py status
```

### Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f run_full_grid_phase2)

# Check disk space
df -h .
```

### Restart from Checkpoint
The script automatically handles restarts. If it stops, simply rerun:
```bash
python3 run_full_grid_phase2.py
```
It will skip completed experiments based on existing checkpoints.

## Success Indicators

### Healthy Execution Signs:
- ✅ **Progress updates** every 50 experiments
- ✅ **Convergence rate** > 70%
- ✅ **Valid certificate rate** > 60%
- ✅ **Acceptance rates** 20-40%
- ✅ **ESS values** > 200

### Warning Signs:
- ⚠️ **No progress** for > 30 minutes
- ⚠️ **Convergence rate** < 50%
- ⚠️ **Many errors** in succession
- ⚠️ **Very low acceptance** < 10%

## Output Files

Upon completion, you'll have:
```
results_phase2_full_YYYYMMDD_HHMMSS/
├── section_a_phase2_complete.json      # 🎯 Main results file
├── phase2_summary.json                 # 📊 Summary statistics  
├── checkpoint_0050.json                # 💾 Periodic saves
├── checkpoint_0100.json
├── ...
└── checkpoint_1728.json
```

## Expected Final Results

Based on Phase 2 improvements, expect:

- **🎯 Convergence Rate:** 80-90% (vs 0% original)
- **🎯 Valid Certificates:** 70-85% (vs 1.4% original)  
- **🎯 Mean Acceptance:** 25-35% (vs 0.1% original)
- **🎯 Mean ESS:** 300-500 (vs never achieved originally)
- **🎯 Total Runtime:** 14-20 hours (adaptive termination)

## Next Steps After Completion

Once the execution completes:

1. **Generate updated figures:**
   ```bash
   python3 generate_figures.py  # Using new results
   ```

2. **Compare with original:**
   ```bash
   python3 analyze_results.py  # Updated analysis
   ```

3. **Create publication materials:**
   - Phase 2 results show dramatic improvements
   - Ready for manuscript figures
   - Reliable PAC-Bayes certificates achieved

---

## 🚀 EXECUTE NOW

You are ready to execute the full 1,728 experiment grid with Phase 2 Adaptive MCMC:

```bash
# Start execution in screen
screen -S phase2_grid
python3 run_full_grid_phase2.py
# Ctrl+A then D to detach

# Monitor in separate terminal
python3 monitor_phase2_progress.py
```

**This will transform your PAC-Bayes inverse PDE experiments from convergence failure to publication-ready success.**
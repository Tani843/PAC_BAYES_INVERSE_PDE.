#!/bin/bash
echo "=== Phase 2 Process Status ==="
if screen -ls | grep -q "phase2_grid"; then
    echo "✓ Screen session: ACTIVE"
else
    echo "✗ Screen session: NOT FOUND"
fi

if ps aux | grep -q "[r]un_full_grid_phase2"; then
    echo "✓ Python process: RUNNING"
else
    echo "✗ Python process: NOT RUNNING"
fi

LATEST=$(ls -t results_phase2_*/checkpoint_*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "✓ Latest checkpoint: $(basename $LATEST)"
    echo "  Modified: $(stat -f "%Sm" $LATEST 2>/dev/null || stat -c "%y" $LATEST 2>/dev/null | cut -d' ' -f1-2)"
else
    echo "✗ No checkpoint files found"
fi

echo ""
echo "=== Results Directory Status ==="
if [ -d "results_phase2_full_20250919_085939" ]; then
    file_count=$(find results_phase2_full_20250919_085939 -type f | wc -l)
    echo "✓ Results directory exists with $file_count files"
    if [ $file_count -gt 0 ]; then
        echo "  Recent files:"
        find results_phase2_full_20250919_085939 -type f -exec ls -lt {} + | head -3
    fi
else
    echo "✗ Results directory not found"
fi
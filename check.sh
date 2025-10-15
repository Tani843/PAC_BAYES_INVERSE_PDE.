#!/bin/bash
if screen -ls | grep -q phase2; then
    echo "✓ Phase 2 Running"
    echo "  First checkpoint expected: ~45 min"
else
    echo "✗ Phase 2 NOT running"
fi
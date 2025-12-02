#!/bin/bash
# Launch script for MuJoCo + CPG communication test

echo "=== MuJoCo + CPG Shared Memory Communication Test ==="
echo ""
echo "This script will:"
echo "1. Clean up any existing shared memory"
echo "2. Start MuJoCo simulator (500Hz) in background"
echo "3. Wait 2 seconds for initialization"
echo "4. Start CPG planner (100Hz) in foreground"
echo ""
echo "Press Ctrl+C to stop both processes"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    python3 -c "from shared_sim_data import cleanup_shared_memory; cleanup_shared_memory()"
    echo "Done"
    exit 0
}

trap cleanup INT TERM

# Clean up existing shared memory
echo "Cleaning up existing shared memory..."
python3 -c "from shared_sim_data import cleanup_shared_memory; cleanup_shared_memory()" 2>/dev/null

# Start MuJoCo sim in background
echo "Starting MuJoCo simulator (500Hz)..."
python3 mujoco_sim.py > /tmp/mujoco_sim.log 2>&1 &
MUJOCO_PID=$!
echo "MuJoCo PID: $MUJOCO_PID (log: /tmp/mujoco_sim.log)"

# Wait for MuJoCo to initialize
echo "Waiting for MuJoCo to initialize..."
sleep 2

# Check if MuJoCo is still running
if ! kill -0 $MUJOCO_PID 2>/dev/null; then
    echo "ERROR: MuJoCo failed to start. Check /tmp/mujoco_sim.log"
    cat /tmp/mujoco_sim.log
    exit 1
fi

echo ""
echo "Starting CPG planner (100Hz)..."
python3 spider_cpg_qp.py

# Cleanup on exit
cleanup

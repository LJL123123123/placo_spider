from shared_sim_data import SimToCPGData, CPGToSimData, cleanup_shared_memory
import time

dt = 1/500.0  # 500Hz update rate
sim_to_cpg = None
cpg_to_sim = None

for attempt in range(30):  # Wait up to 3 seconds
    try:
        sim_to_cpg = SimToCPGData(create=False)
        cpg_to_sim = CPGToSimData(create=False)
        break
    except FileNotFoundError:
        time.sleep(0.1)
            
if sim_to_cpg is None or cpg_to_sim is None:
    print("ERROR: MuJoCo sim not found. Start mujoco_sim.py first!", flush=True)
    exit(1)

print("Connected to shared memory. Reading data...", flush=True)
iteration = 0

try:
    while True:
        if iteration % 100 == 0:  # Print every 100 iterations (0.2 seconds at 500Hz)
            try:
                qpos, ctrl, sim_ts = sim_to_cpg.read()
                qpos_desired, kp, kd, cpg_ts = cpg_to_sim.read()
                print(f"\n=== Iteration {iteration} ===", flush=True)
                print(f"Sim->CPG (timestamp={sim_ts}):", flush=True)
                print(f"  qpos[0:3]: {qpos[0:3]}", flush=True)
                print(f"  qpos[6:9]: {qpos[6:9]}", flush=True)
                print(f"  ctrl[0:3]: {ctrl[0:3]}", flush=True)
                print(f"\nCPG->Sim (timestamp={cpg_ts}):", flush=True)
                print(f"  qpos_desired[0:3]: {qpos_desired[0:3]}", flush=True)
                print(f"  kp[0:3]: {kp[0:3]}", flush=True)
                print(f"  kd[0:3]: {kd[0:3]}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to read from shared memory: {e}", flush=True)
        
        iteration += 1
        time.sleep(dt)
        
except KeyboardInterrupt:
    print("\nStopping read_data.py...", flush=True)
finally:
    if sim_to_cpg is not None:
        sim_to_cpg.close()
    if cpg_to_sim is not None:
        cpg_to_sim.close()
    print("Cleanup complete", flush=True)
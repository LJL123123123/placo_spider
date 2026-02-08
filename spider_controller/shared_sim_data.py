#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Memory Data Structure for MuJoCo Sim <-> CPG QP Communication

Data Flow:
- MuJoCo (500Hz) -> CPG: qpos(19), ctrl(12), timestamp
- CPG (100Hz) -> MuJoCo: qpos_desired(19), kp(12), kd(12), timestamp

Memory Layout:
- sim_to_cpg: 32 floats (19 qpos + 12 ctrl + 1 timestamp)
- cpg_to_sim: 44 floats (19 qpos_desired + 12 kp + 12 kd + 1 timestamp)
"""
import numpy as np
from multiprocessing import shared_memory
import time

# Constants
QPOS_SIZE = 19  # 7 (free joint) + 12 (actuated joints)
CTRL_SIZE = 12  # 12 actuators
KP_SIZE = 12
KD_SIZE = 12

# Shared memory names
SHM_SIM_TO_CPG = "mujoco_sim_to_cpg"
SHM_CPG_TO_SIM = "cpg_to_mujoco_sim"

# Data sizes (in float64)
SIM_TO_CPG_SIZE = QPOS_SIZE + CTRL_SIZE + 1  # qpos + ctrl + timestamp = 32
CPG_TO_SIM_SIZE = QPOS_SIZE + CTRL_SIZE + KP_SIZE + KD_SIZE + 1  # qpos_desired + ctrl + kp + kd + timestamp = 56


class SimToCPGData:
    """MuJoCo -> CPG data structure"""
    
    def __init__(self, create=False):
        """
        Args:
            create: If True, create new shared memory; else attach to existing
        """
        self.shm = None
        self.data = None
        
        if create:
            try:
                # Try to unlink existing memory first
                try:
                    old_shm = shared_memory.SharedMemory(name=SHM_SIM_TO_CPG)
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                
                self.shm = shared_memory.SharedMemory(
                    name=SHM_SIM_TO_CPG,
                    create=True,
                    size=SIM_TO_CPG_SIZE * 8  # 8 bytes per float64
                )
            except FileExistsError:
                # Memory already exists, attach to it
                self.shm = shared_memory.SharedMemory(name=SHM_SIM_TO_CPG)
        else:
            # Attach to existing memory
            self.shm = shared_memory.SharedMemory(name=SHM_SIM_TO_CPG)
        
        # Create numpy view
        self.data = np.ndarray((SIM_TO_CPG_SIZE,), dtype=np.float64, buffer=self.shm.buf)
        
        # Initialize to zero if creating
        if create:
            self.data[:] = 0.0
    
    def write(self, qpos, ctrl):
        """Write qpos and ctrl to shared memory"""
        qpos_arr = np.asarray(qpos, dtype=np.float64).flatten()
        ctrl_arr = np.asarray(ctrl, dtype=np.float64).flatten()
        
        # Pad if necessary
        if qpos_arr.size < QPOS_SIZE:
            qpos_arr = np.pad(qpos_arr, (0, QPOS_SIZE - qpos_arr.size))
        if ctrl_arr.size < CTRL_SIZE:
            ctrl_arr = np.pad(ctrl_arr, (0, CTRL_SIZE - ctrl_arr.size))
        
        self.data[0:QPOS_SIZE] = qpos_arr[:QPOS_SIZE]
        self.data[QPOS_SIZE:QPOS_SIZE+CTRL_SIZE] = ctrl_arr[:CTRL_SIZE]
        self.data[-1] = time.time()  # timestamp
    
    def read(self):
        """Read qpos and ctrl from shared memory"""
        qpos = self.data[0:QPOS_SIZE].copy()
        ctrl = self.data[QPOS_SIZE:QPOS_SIZE+CTRL_SIZE].copy()
        timestamp = self.data[-1]
        return qpos, ctrl, timestamp
    
    def close(self):
        """Close shared memory (don't unlink)"""
        if self.shm is not None:
            self.shm.close()
    
    def unlink(self):
        """Unlink (delete) shared memory - only call from creator"""
        if self.shm is not None:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class CPGToSimData:
    """CPG -> MuJoCo data structure"""
    
    def __init__(self, create=False):
        """
        Args:
            create: If True, create new shared memory; else attach to existing
        """
        self.shm = None
        self.data = None
        
        if create:
            try:
                # Try to unlink existing memory first
                try:
                    old_shm = shared_memory.SharedMemory(name=SHM_CPG_TO_SIM)
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                
                self.shm = shared_memory.SharedMemory(
                    name=SHM_CPG_TO_SIM,
                    create=True,
                    size=CPG_TO_SIM_SIZE * 8  # 8 bytes per float64
                )
            except FileExistsError:
                # Memory already exists, attach to it
                self.shm = shared_memory.SharedMemory(name=SHM_CPG_TO_SIM)
        else:
            # Attach to existing memory
            self.shm = shared_memory.SharedMemory(name=SHM_CPG_TO_SIM)
        
        # Create numpy view
        self.data = np.ndarray((CPG_TO_SIM_SIZE,), dtype=np.float64, buffer=self.shm.buf)
        
        # Initialize to default values if creating
        if create:
            self.data[:] = 0.0
            # Default gains
            self.data[QPOS_SIZE+CTRL_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE] = 0.0  # default kp
            self.data[QPOS_SIZE+CTRL_SIZE+KP_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE+KD_SIZE] = 0.0  # default kd

    def write(self, qpos_desired, ctrl_desired, kp, kd):
        """Write desired qpos and PD gains to shared memory"""
        qpos_arr = np.asarray(qpos_desired, dtype=np.float64).flatten()
        ctrl_arr = np.asarray(ctrl_desired, dtype=np.float64).flatten()
        kp_arr = np.asarray(kp, dtype=np.float64).flatten()
        kd_arr = np.asarray(kd, dtype=np.float64).flatten()
        
        # Pad if necessary
        if qpos_arr.size < QPOS_SIZE:
            qpos_arr = np.pad(qpos_arr, (0, QPOS_SIZE - qpos_arr.size))
        if ctrl_arr.size < CTRL_SIZE:
            ctrl_arr = np.pad(ctrl_arr, (0, CTRL_SIZE - ctrl_arr.size))
        if kp_arr.size < KP_SIZE:
            kp_arr = np.pad(kp_arr, (0, KP_SIZE - kp_arr.size))
        if kd_arr.size < KD_SIZE:
            kd_arr = np.pad(kd_arr, (0, KD_SIZE - kd_arr.size))
        
        self.data[0:QPOS_SIZE] = qpos_arr[:QPOS_SIZE]
        self.data[QPOS_SIZE:QPOS_SIZE+CTRL_SIZE] = ctrl_arr[:CTRL_SIZE]
        self.data[QPOS_SIZE+CTRL_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE] = kp_arr[:KP_SIZE]
        self.data[QPOS_SIZE+CTRL_SIZE+KP_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE+KD_SIZE] = kd_arr[:KD_SIZE]
        self.data[-1] = time.time()  # timestamp
    
    def read(self):
        """Read desired qpos and PD gains from shared memory"""
        qpos_desired = self.data[0:QPOS_SIZE].copy()
        ctrl = self.data[QPOS_SIZE:QPOS_SIZE+CTRL_SIZE].copy()
        kp = self.data[QPOS_SIZE+CTRL_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE].copy()
        kd = self.data[QPOS_SIZE+CTRL_SIZE+KP_SIZE:QPOS_SIZE+CTRL_SIZE+KP_SIZE+KD_SIZE].copy()
        timestamp = self.data[-1]
        return qpos_desired, ctrl, kp, kd, timestamp

    def close(self):
        """Close shared memory (don't unlink)"""
        if self.shm is not None:
            self.shm.close()
    
    def unlink(self):
        """Unlink (delete) shared memory - only call from creator"""
        if self.shm is not None:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


# Cleanup function to call at exit
def cleanup_shared_memory():
    """Clean up all shared memory segments"""
    for name in [SHM_SIM_TO_CPG, SHM_CPG_TO_SIM]:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
            print(f"Cleaned up shared memory: {name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Failed to cleanup {name}: {e}")


if __name__ == "__main__":
    # Test the data structures
    print("Testing shared memory data structures...")
    
    # Create data structures
    sim_data = SimToCPGData(create=True)
    cpg_data = CPGToSimData(create=True)
    
    # Write test data
    test_qpos = np.random.randn(19)
    test_ctrl = np.random.randn(12)
    sim_data.write(test_qpos, test_ctrl)
    
    test_qpos_desired = np.random.randn(19)
    test_ctrl_desired = np.random.randn(12)
    test_kp = np.ones(12) * 50.0
    test_kd = np.ones(12) * 5.0
    cpg_data.write(test_qpos_desired, test_ctrl_desired, test_kp, test_kd)
    time.sleep(0.1)  # ensure timestamps differ
    # Read back
    qpos, ctrl, ts1 = sim_data.read()
    qpos_d, ctrl_d, kp, kd, ts2 = cpg_data.read()
    
    print(f"Sim->CPG: qpos={qpos[:3]}..., ctrl={ctrl[:3]}..., timestamp={ts1}")
    print(f"CPG->Sim: qpos_desired={qpos_d[:3]}..., ctrl={ctrl_d[:3]}..., kp={kp[:3]}, kd={kd[:3]}, timestamp={ts2}")

    # Verify
    assert np.allclose(qpos, test_qpos), "qpos mismatch"
    assert np.allclose(ctrl, test_ctrl), "ctrl mismatch"
    assert np.allclose(qpos_d, test_qpos_desired), "qpos_desired mismatch"
    assert np.allclose(ctrl_d, test_ctrl_desired), "ctrl_desired mismatch"
    assert np.allclose(kp, test_kp), "kp mismatch"
    assert np.allclose(kd, test_kd), "kd mismatch"
    
    print("✓ All tests passed!")
    
    # Cleanup
    sim_data.close()
    cpg_data.close()
    sim_data.unlink()
    cpg_data.unlink()
    print("✓ Cleanup complete")

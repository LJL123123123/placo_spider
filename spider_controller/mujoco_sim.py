import argparse
import time
import numpy as np
import atexit
import mujoco

from shared_sim_data import SimToCPGData, CPGToSimData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--xml",
        type=str,
        default="../spider_SLDASM_2m6d/urdf/scene.xml",
        help="MuJoCo MJCF XML path",
    )
    p.add_argument(
        "--viewer",
        action="store_true",
        help="Open official MuJoCo viewer window (mujoco.viewer).",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep dt each step to run roughly in real time.",
    )
    return p.parse_args()


def _as_vec(x, n: int, default: float = 0.0) -> np.ndarray:
    """Convert x to float64 vector of length n. Broadcast scalar if needed."""
    if x is None:
        return np.full(n, default, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        return np.full(n, default, dtype=np.float64)
    return arr


def main() -> None:
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Shared memory
    sim_to_cpg = SimToCPGData(create=True)
    cpg_to_sim = CPGToSimData(create=True)

    atexit.register(lambda: sim_to_cpg.close())
    atexit.register(lambda: sim_to_cpg.unlink())
    atexit.register(lambda: cpg_to_sim.close())
    # 不在这里 unlink cpg_to_sim，让 CPG 端负责

    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()

    dt = float(model.opt.timestep) if float(model.opt.timestep) > 0 else 0.001
    print(
        f"[mujoco_sim] physics started: nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={dt}, viewer={args.viewer}"
    )

    def step_once() -> None:
        # 1) read CPG command
        try:
            qpos_desired, ctrl_desired, kp_raw, kd_raw, cpg_timestamp = cpg_to_sim.read()
        except Exception:
            qpos_desired = initial_qpos
            ctrl_desired = np.zeros(12, dtype=np.float64)
            kp_raw = 0.0
            kd_raw = 0.0
            cpg_timestamp = 0.0

        # 2) sanitize gains
        kp = _as_vec(kp_raw, model.nu, default=0.0)
        kd = _as_vec(kd_raw, model.nu, default=0.0)

        # 3) sanitize target qpos
        qd = np.asarray(qpos_desired, dtype=np.float64).reshape(-1)
        tau_desired = np.asarray(ctrl_desired, dtype=np.float64).reshape(-1)
        if qd.size == model.nq:
            target_qpos = qd
            target_ctrl = tau_desired
        elif qd.size == model.nu:
            # 只给了受控关节目标：塞到最后 nu 个关节
            target_qpos = data.qpos.copy()
            target_qpos[model.nq - model.nu : model.nq] = qd
            target_ctrl = tau_desired
        else:
            # 尺寸不匹配：退回初始姿态，避免崩
            target_qpos = initial_qpos
            target_ctrl = np.zeros(model.nu, dtype=np.float64)

        # 4) PD control -> data.ctrl
        # data.ctrl[:] = data.qfrc_gravcomp[6:]  # gravity compensation for all joints\
        for a in range(model.nu):
            qidx = model.nq - model.nu + a
            vidx = model.nv - model.nu + a
            data.ctrl[a] = kp[a] * (target_qpos[qidx] - data.qpos[qidx]) - kd[a] * data.qvel[vidx] + target_ctrl[a]
            if data.ctrl[a] > 50.:
                data.ctrl[a] = 50.
            elif data.ctrl[a] < -50.:
                data.ctrl[a] = -50.

        # 5) step physics
        mujoco.mj_step(model, data)

        # 6) write sim state back
        try:
            sim_to_cpg.write(data.qpos, data.ctrl)
        except Exception as e:
            print(f"Warning: Failed to write to shared memory: {e}", flush=True)

    if args.viewer:
        # 官方 viewer（不需要第三方 mujoco_viewer）
        from mujoco import viewer as mj_viewer  # type: ignore

        with mj_viewer.launch_passive(model, data) as v:
            while v.is_running():
                step_once()
                v.sync()
                if args.realtime:
                    time.sleep(dt)
    else:
        try:
            while True:
                step_once()
                if args.realtime:
                    time.sleep(dt)
        except KeyboardInterrupt:
            pass

    print("[mujoco_sim] exited.")


if __name__ == "__main__":
    main()

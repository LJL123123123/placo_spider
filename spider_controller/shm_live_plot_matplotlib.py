#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight real-time plotting from shared memory using matplotlib.

This is a fallback when you don't want to depend on PlotJuggler plugins/Qt.

Examples:
  python shm_live_plot_matplotlib.py --duration 10 --hz 50
  python shm_live_plot_matplotlib.py --signals qpos_06 qpos_07 qpos_08 ctrl_00 ctrl_01

Signals:
  - qpos_00..qpos_18
  - ctrl_00..ctrl_11
  - qpos_des_00..qpos_des_18
  - tau_des_00..tau_des_11
  - kp_00..kp_11
  - kd_00..kd_11
  - sim_ts, cpg_ts
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from shared_sim_data import SimToCPGData, CPGToSimData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=float, default=50.0, help="Plot refresh rate")
    p.add_argument("--duration", type=float, default=10.0, help="Window length (seconds)")
    p.add_argument(
        "--signals",
        nargs="*",
        default=["qpos_06", "qpos_07", "qpos_08", "ctrl_00", "ctrl_01", "ctrl_02"],
        help="Signal names to plot",
    )
    p.add_argument("--wait", action="store_true", help="Wait for SHM to appear")
    return p.parse_args()


def _connect_shm(wait: bool) -> Tuple[SimToCPGData, CPGToSimData]:
    printed = False
    while True:
        try:
            return SimToCPGData(create=False), CPGToSimData(create=False)
        except FileNotFoundError:
            if not wait:
                raise
            if not printed:
                print("[shm_live_plot] waiting for SHM (start mujoco_sim.py)...", flush=True)
                printed = True
            time.sleep(0.2)


def _get_from_sample(sample: Dict[str, float], key: str) -> float:
    return float(sample.get(key, float("nan")))


def _build_sample(sim: SimToCPGData, cpg: CPGToSimData) -> Dict[str, float]:
    out: Dict[str, float] = {}

    qpos, ctrl, sim_ts = sim.read()
    out["sim_ts"] = float(sim_ts)
    for i, v in enumerate(np.asarray(qpos).reshape(-1)):
        out[f"qpos_{i:02d}"] = float(v)
    for i, v in enumerate(np.asarray(ctrl).reshape(-1)):
        out[f"ctrl_{i:02d}"] = float(v)

    qpos_des, tau_des, kp, kd, cpg_ts = cpg.read()
    out["cpg_ts"] = float(cpg_ts)
    for i, v in enumerate(np.asarray(qpos_des).reshape(-1)):
        out[f"qpos_des_{i:02d}"] = float(v)
    for i, v in enumerate(np.asarray(tau_des).reshape(-1)):
        out[f"tau_des_{i:02d}"] = float(v)
    for i, v in enumerate(np.asarray(kp).reshape(-1)):
        out[f"kp_{i:02d}"] = float(v)
    for i, v in enumerate(np.asarray(kd).reshape(-1)):
        out[f"kd_{i:02d}"] = float(v)

    return out


def main() -> None:
    args = parse_args()

    sim_to_cpg, cpg_to_sim = _connect_shm(wait=args.wait)

    hz = float(args.hz) if float(args.hz) > 0 else 50.0
    period = 1.0 / hz

    duration = max(2.0, float(args.duration))
    capacity = int(duration * hz) + 1

    signals: List[str] = list(args.signals)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("SHM live plot")
    ax.set_xlabel("t (s, relative)")
    ax.grid(True)

    t0 = time.time()
    ts: List[float] = []
    ys: Dict[str, List[float]] = {s: [] for s in signals}

    lines = {}
    for s in signals:
        (line,) = ax.plot([], [], label=s)
        lines[s] = line
    ax.legend(loc="upper right")

    try:
        while True:
            now = time.time()
            sample = _build_sample(sim_to_cpg, cpg_to_sim)

            t_rel = now - t0
            ts.append(t_rel)
            if len(ts) > capacity:
                ts.pop(0)

            for s in signals:
                ys[s].append(_get_from_sample(sample, s))
                if len(ys[s]) > capacity:
                    ys[s].pop(0)

            # update plot
            for s in signals:
                lines[s].set_data(ts, ys[s])

            if ts:
                ax.set_xlim(max(0.0, ts[-1] - duration), ts[-1])

            # autoscale y based on visible window
            all_vals = []
            for s in signals:
                all_vals.extend([v for v in ys[s] if np.isfinite(v)])
            if all_vals:
                vmin = float(np.min(all_vals))
                vmax = float(np.max(all_vals))
                if vmin == vmax:
                    vmin -= 1.0
                    vmax += 1.0
                margin = 0.05 * (vmax - vmin)
                ax.set_ylim(vmin - margin, vmax + margin)

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            sim_to_cpg.close()
        except Exception:
            pass
        try:
            cpg_to_sim.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

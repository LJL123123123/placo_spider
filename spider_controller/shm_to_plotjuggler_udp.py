#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream shared-memory data to PlotJuggler via UDP (JSON).

Why this exists
---------------
PlotJuggler has a built-in streaming plugin called "UDP Server".
It accepts UDP datagrams containing JSON, then flattens the JSON into time series.

This script reads your MuJoCo<->controller shared memory and sends samples to PlotJuggler.

Usage
-----
1) Start MuJoCo sim (creates SHM):
   python mujoco_sim.py --realtime

2) Start PlotJuggler, then enable streamer:
   - GUI: Streaming -> "UDP Server" (select address/port, default port is often 9870)
   - (Optional) CLI auto-start streamer (depends on your PJ build):
       plotjuggler --start_streamer DataStreamUDP

3) Run this script:
   python shm_to_plotjuggler_udp.py --hz 50 --address 127.0.0.1 --port 9870

Notes
-----
- PlotJuggler UDP Server timestamps points using *arrival time* (by default),
  but it will still plot your embedded timestamps as regular series.
- We keep keys flat-ish to avoid ambiguous array flattening.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Optional, Tuple

import numpy as np

from shared_sim_data import SimToCPGData, CPGToSimData


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--address", type=str, default="127.0.0.1", help="PlotJuggler UDP Server address")
    p.add_argument("--port", type=int, default=9870, help="PlotJuggler UDP Server port")
    p.add_argument("--hz", type=float, default=50.0, help="Send rate in Hz")
    p.add_argument(
        "--mode",
        type=str,
        choices=["sim", "cpg", "both"],
        default="both",
        help="Which SHM direction(s) to stream",
    )
    p.add_argument(
        "--wait",
        action="store_true",
        help="Wait for SHM to appear instead of failing immediately.",
    )
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
                print("[shm_to_plotjuggler_udp] waiting for SHM (start mujoco_sim.py)...", flush=True)
                printed = True
            time.sleep(0.2)


def _flat_prefix(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:02d}"


def main() -> None:
    args = parse_args()

    sim_to_cpg, cpg_to_sim = _connect_shm(wait=args.wait)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    period = 1.0 / float(args.hz) if float(args.hz) > 0 else 0.02

    print(
        f"[shm_to_plotjuggler_udp] streaming to udp://{args.address}:{args.port} @ {args.hz:.1f} Hz (mode={args.mode})",
        flush=True,
    )

    try:
        while True:
            payload: dict = {
                "wall_time": time.time(),
            }

            if args.mode in ("sim", "both"):
                try:
                    qpos, ctrl, sim_ts = sim_to_cpg.read()
                    sim_obj: dict = {
                        "sim_ts": float(sim_ts),
                    }
                    for i, v in enumerate(np.asarray(qpos).reshape(-1)):
                        sim_obj[_flat_prefix("qpos", i)] = float(v)
                    for i, v in enumerate(np.asarray(ctrl).reshape(-1)):
                        sim_obj[_flat_prefix("ctrl", i)] = float(v)
                    payload["sim"] = sim_obj
                except Exception:
                    # If MuJoCo restarted, SHM may disappear/reappear.
                    # We keep running; user can restart this tool too.
                    pass

            if args.mode in ("cpg", "both"):
                try:
                    qpos_desired, ctrl_desired, kp, kd, cpg_ts = cpg_to_sim.read()
                    cpg_obj: dict = {
                        "cpg_ts": float(cpg_ts),
                    }
                    for i, v in enumerate(np.asarray(qpos_desired).reshape(-1)):
                        cpg_obj[_flat_prefix("qpos_des", i)] = float(v)
                    for i, v in enumerate(np.asarray(ctrl_desired).reshape(-1)):
                        cpg_obj[_flat_prefix("tau_des", i)] = float(v)
                    for i, v in enumerate(np.asarray(kp).reshape(-1)):
                        cpg_obj[_flat_prefix("kp", i)] = float(v)
                    for i, v in enumerate(np.asarray(kd).reshape(-1)):
                        cpg_obj[_flat_prefix("kd", i)] = float(v)
                    payload["cpg"] = cpg_obj
                except Exception:
                    pass

            sock.sendto(json.dumps(payload).encode("utf-8"), (args.address, int(args.port)))
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
        try:
            sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

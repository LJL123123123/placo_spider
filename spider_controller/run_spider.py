"""run_spider.py

对应 run_wbc.py：提供一个可直接运行的主程序。

- 使用键盘输入生成 cmd_vxyz / yaw_rate
- 调用 SpiderIK.step() 完成规划+QP+通信+logger+可视化

运行：
    python3 run_spider.py

注意：
- 需要先启动 mujoco_sim.py（它会创建 shared memory）。
"""

from __future__ import annotations

import time
import sys
import select
import termios
import tty
import atexit
import numpy as np

from spider_ik import SpiderIK, SpiderIkConfig


class Keyboard:
    def __init__(self):
        self._old = None
        try:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self._old = None

        atexit.register(self.restore)

        self.last_pressed = {}
        self.press_timeout = 0.18

        # speed params
        self.speed_forward = 0.1
        self.speed_lateral = 0.1
        self.yaw_speed = 0.1
        self.height_speed = 0.05

    def restore(self):
        if self._old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None

    def poll(self):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                return
            if ch:
                ch = ch.lower()
                now = time.time()
                self.last_pressed[ch] = now
                # make movement keys exclusive
                movement = set(['w', 'a', 's', 'd', 'q', 'e', 'r', 'f'])
                if ch in movement:
                    for k in list(self.last_pressed.keys()):
                        if k != ch and k in movement:
                            del self.last_pressed[k]

    def held(self, key: str) -> bool:
        now = time.time()
        return (key in self.last_pressed) and (now - self.last_pressed[key] < self.press_timeout)

    def get_cmd(self):
        vx = 0.0
        if self.held('w'):
            vx += self.speed_forward
        if self.held('s'):
            vx -= self.speed_forward

        vy = 0.0
        if self.held('a'):
            vy += self.speed_lateral
        if self.held('d'):
            vy -= self.speed_lateral

        yaw = 0.0
        if self.held('q'):
            yaw += self.yaw_speed
        if self.held('e'):
            yaw -= self.yaw_speed

        vz = 0.0
        if self.held('r'):
            vz += self.height_speed
        if self.held('f'):
            vz -= self.height_speed

        return np.array([vx, vy, vz], dtype=np.float64), float(yaw)


def main():
    cfg = SpiderIkConfig(
        dt=0.001,
        gait_mode='quasi_static',
        cycle_period=2.0,
        swing_height=0.1,
        stand_transition_duration=0.6,
    )

    spider = SpiderIK(cfg)

    print(
        "SpiderIK running (keyboard)\n"
        "  W/S: vx  A/D: vy  Q/E: yaw  R/F: vz  Ctrl-C to quit\n",
        flush=True,
    )

    kb = Keyboard()
    try:
        while True:
            kb.poll()
            cmd_vxyz, cmd_yaw = kb.get_cmd()
            spider.step(cmd_vxyz, cmd_yaw)
            time.sleep(cfg.dt)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

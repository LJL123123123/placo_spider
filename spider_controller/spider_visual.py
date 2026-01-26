"""spider_visual.py

可视化封装：把 placo_utils 的 viz/line_viz/point_viz 做一层薄封装，
使 run_spider.py / spider_ik.py 只需要调用类方法。

说明：
- 如果环境里没有 placo_utils，本模块会降级为空操作（不崩溃）。
- 目前主要服务于：
  - 机器人姿态显示（robot_viz）
  - COM 点显示（point_viz）
  - 支撑多边形显示（line_viz）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class SpiderVisualizer:
    robot: object
    enabled: bool = True

    def __post_init__(self):
        self._viz = None
        self._point_viz = None
        self._line_viz = None

        if not self.enabled:
            return

        try:
            from placo_utils.visualization import robot_viz, point_viz, line_viz
            self._viz = robot_viz(self.robot)
            self._point_viz = point_viz
            self._line_viz = line_viz
        except Exception:
            # placo_utils 不可用 -> 直接降级
            self._viz = None
            self._point_viz = None
            self._line_viz = None

    def display_robot(self, q):
        if self._viz is None:
            return
        try:
            self._viz.display(q)
        except Exception:
            pass

    def display_com_xy(self, com_world_xyz: Sequence[float], name: str = "com", color: int = 0xFF0000):
        if self._point_viz is None:
            return
        try:
            p = np.asarray(com_world_xyz, dtype=float).reshape(3).copy()
            p[2] = 0.0
            self._point_viz(name, p, color=color)
        except Exception:
            pass

    def display_support_polygon(self, points_xy: Sequence[Sequence[float]], prefix: str = "support", color: int = 0xFFAA00):
        """画多边形边（points_xy 为 Nx2，N>=2）。"""
        if self._line_viz is None:
            return
        try:
            pts = [np.asarray(p, dtype=float).reshape(2) for p in points_xy]
            if len(pts) < 2:
                return
            for k in range(len(pts)):
                a = pts[k]
                b = pts[(k + 1) % len(pts)]
                self._line_viz(f"{prefix}_{k}", np.array([a, b]), color=color)
        except Exception:
            pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""spider_gait_manager_qp.py

兼容入口（legacy）。

原先本文件包含完整的 gait + QP + SHM + logger + viz 循环；
为了对齐你 GO2 WBIK 的结构，现在已经拆成：

- `spider_ik.py` / `run_spider.py` / `spider_visual.py` / `spider_logger.py`

因此这里保留为一个薄 wrapper：直接转到 `run_spider.py`。

运行方式（两个终端）：
- Terminal 1: python3 mujoco_sim.py
- Terminal 2: python3 spider_gait_manager_qp.py   (等价于 python3 run_spider.py)
"""

from run_spider import main


if __name__ == '__main__':
    main()

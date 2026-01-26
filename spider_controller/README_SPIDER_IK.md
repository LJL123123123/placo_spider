# Spider IK/QP runner (WBIK-style)

这个目录把原先的 `spider_gait_manager_qp.py` 拆成类似你 GO2 WBIK 的结构：

- `spider_ik.py`：核心类（类似 `wbc.py`）
- `run_spider.py`：主程序入口（类似 `run_wbc.py`）
- `spider_visual.py`：可视化封装（类似 `ik_visualization.py` 的角色，但这里用 placo_utils）
- `spider_gait_manager.py`：gait manager 的导入 shim（对应 `gait_manager_cuda.py` 的角色）
- `spider_logger.py`：CSV logger（类似 `wbc_logger.py`）

日志输出：默认写到 `./debug/*.csv`。

## 如何运行

1. 先启动 MuJoCo（创建 shared memory）

```bash
python3 mujoco_sim.py
```

2. 再启动 SpiderIK

```bash
python3 run_spider.py
```

## 输出 CSV

- `debug/cmd_data.csv`
- `debug/com_target_data.csv`
- `debug/LF_target_data.csv` / `RF_target_data.csv` / `LH_target_data.csv` / `RH_target_data.csv`
- `debug/contact_state_data.csv`
- `debug/support_polygon_data.csv`

你可以用 pandas/matplotlib 做复现绘图。

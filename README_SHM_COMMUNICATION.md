# MuJoCo + CPG 进程间通信使用说明

## 概述

实现了 `mujoco_sim.py` (500Hz) 和 `spider_cpg_qp.py` (100Hz) 之间的无阻塞进程间通信。

### 数据流

```
MuJoCo Sim (500Hz)          Shared Memory           CPG Planner (100Hz)
┌──────────────────┐                               ┌───────────────────┐
│                  │   qpos[19], ctrl[12]         │                   │
│  - 物理仿真      │ ─────────────────────────>   │  - 步态规划       │
│  - PD控制器      │                               │  - 逆运动学QP     │
│  - 重力补偿      │   qpos_desired[19]           │  - 足端轨迹生成   │
│                  │   kp[12], kd[12]             │                   │
│                  │ <─────────────────────────   │                   │
└──────────────────┘                               └───────────────────┘
```

### 共享内存结构

1. **sim_to_cpg** (MuJoCo → CPG)
   - `qpos[19]`: 当前关节位置 (7自由度基座 + 12关节)
   - `ctrl[12]`: 当前执行器控制力矩
   - `timestamp`: 时间戳

2. **cpg_to_sim** (CPG → MuJoCo)
   - `qpos_desired[19]`: 期望关节位置
   - `kp[12]`: 每个执行器的比例增益
   - `kd[12]`: 每个执行器的微分增益
   - `timestamp`: 时间戳

## 使用方法

### 方法1: 使用启动脚本(推荐)

```bash
cd /home/placo_cpg
./run_sim_cpg.sh
```

该脚本会:
1. 清理旧的共享内存
2. 后台启动 MuJoCo (日志: `/tmp/mujoco_sim.log`)
3. 前台启动 CPG planner
4. Ctrl+C 自动清理两个进程

### 方法2: 手动启动

**终端1: 启动 MuJoCo 仿真**
```bash
cd /home/placo_cpg
# 清理旧共享内存
python3 -c "from shared_sim_data import cleanup_shared_memory; cleanup_shared_memory()"
# 启动仿真
python3 mujoco_sim.py
```

**终端2: 启动 CPG 规划器**
```bash
cd /home/placo_cpg
python3 spider_cpg_qp.py
```

### 方法3: 测试通信(不运行完整CPG)

```bash
# 终端1
python3 mujoco_sim.py

# 终端2
python3 test_shm_communication.py
```

## 文件说明

- `shared_sim_data.py`: 共享内存数据结构定义
  - `SimToCPGData`: MuJoCo→CPG数据
  - `CPGToSimData`: CPG→MuJoCo数据
  - `cleanup_shared_memory()`: 清理函数

- `mujoco_sim.py`: MuJoCo仿真器 (500Hz)
  - 读取 `qpos_desired, kp, kd` 从共享内存
  - PD控制器: `ctrl = kp * (qpos_desired - qpos) - kd * qvel`
  - 重力补偿: 通过 `actuator-bias` 回调
  - 写入 `qpos, ctrl` 到共享内存

- `spider_cpg_qp.py`: CPG步态规划器 (100Hz)
  - 读取 `qpos, ctrl` 从共享内存
  - 使用 FootTrajectoryCPG 生成足端轨迹
  - 使用 placo QP求解器计算期望关节角度
  - 写入 `qpos_desired, kp, kd` 到共享内存

- `test_shm_communication.py`: 简单通信测试
  - 不需要完整CPG,只发送测试指令

- `run_sim_cpg.sh`: 一键启动脚本

## PD增益调节

在 `spider_cpg_qp.py` 中修改:

```python
# 在 loop() 函数中
kp_gains = np.ones(12) * 100.0  # 比例增益
kd_gains = np.ones(12) * 10.0   # 微分增益
```

**调节建议:**
- `kp` 太大 → 震荡,不稳定
- `kp` 太小 → 跟踪慢,机器人下沉
- `kd` 太大 → 过阻尼,运动缓慢
- `kd` 太小 → 欠阻尼,震荡

推荐范围:
- `kp`: 50-200
- `kd`: 5-20
- 关系: `kd ≈ kp / 10`

## 性能监控

### MuJoCo输出 (每0.2秒)
```
t=0.20s, qpos[2]=0.1929, ctrl_rms=94.15
```
- `t`: 仿真时间
- `qpos[2]`: 机器人重心高度(z坐标)
- `ctrl_rms`: 控制力矩均方根

### CPG输出 (每1秒)
```
CPG t=1.00s, body_pos=[0.120, 0.000, 0.200], sim_z=0.193
```
- `t`: CPG时间
- `body_pos`: 期望机体位置
- `sim_z`: 仿真实际高度

## 故障排查

### 问题1: "MuJoCo sim not found"
**原因:** CPG先于MuJoCo启动
**解决:** 先启动 `mujoco_sim.py`,等2秒再启动 `spider_cpg_qp.py`

### 问题2: "There appear to be leaked shared memory"
**原因:** 程序非正常退出,共享内存未清理
**解决:**
```bash
python3 -c "from shared_sim_data import cleanup_shared_memory; cleanup_shared_memory()"
```

### 问题3: 机器人剧烈震荡或爆炸
**原因:** PD增益过大或期望位置突变
**解决:**
1. 降低 `kp`, `kd` 增益
2. 检查 `qpos_desired` 是否连续
3. 确保初始姿态合理

### 问题4: 机器人持续下沉
**原因:** 
1. PD增益太小
2. 期望轨迹不合理(足端无支撑)
**解决:**
1. 增加 `kp` 增益
2. 检查CPG生成的足端轨迹

## 调试技巧

### 1. 查看共享内存实时数据
```python
from shared_sim_data import SimToCPGData, CPGToSimData
import time

sim_data = SimToCPGData(create=False)
cpg_data = CPGToSimData(create=False)

while True:
    qpos, ctrl, ts1 = sim_data.read()
    qpos_d, kp, kd, ts2 = cpg_data.read()
    print(f"Sim: qpos[2]={qpos[2]:.3f}, CPG: kp[0]={kp[0]:.1f}")
    time.sleep(0.1)
```

### 2. 监控日志
```bash
# MuJoCo日志 (如果后台运行)
tail -f /tmp/mujoco_sim.log

# 实时CSV数据
tail -f /home/placo_cpg/debug/robot_data.csv
```

### 3. 性能分析
```bash
# 检查进程CPU占用
top -p $(pgrep -f mujoco_sim) -p $(pgrep -f spider_cpg)

# 检查通信延迟
python3 -c "
from shared_sim_data import *
import time
s = SimToCPGData(create=False)
_, _, ts = s.read()
print(f'Latency: {(time.time() - ts)*1000:.2f}ms')
"
```

## 扩展开发

### 添加新的共享数据

1. 修改 `shared_sim_data.py` 中的大小常量
2. 更新 `write()` 和 `read()` 方法
3. 重新启动两个进程

### 修改控制频率

**MuJoCo频率** (`mujoco_sim.py`):
```python
dt = 1.0 / 500.0  # 改为 1/1000 = 1000Hz
```

**CPG频率** (`spider_cpg_qp.py`):
```python
dt = 0.01  # 改为 0.005 = 200Hz
@schedule(interval=dt)
```

## 已知限制

1. **单机通信**: 仅支持同一台机器上的进程
2. **无同步机制**: 使用无锁共享内存,可能有数据撕裂(概率极低)
3. **固定大小**: qpos=19, ctrl/kp/kd=12,修改需重新定义结构

## 相关资源

- MuJoCo文档: https://mujoco.readthedocs.io/
- Placo文档: https://github.com/Rhoban/placo
- Python multiprocessing.shared_memory: https://docs.python.org/3/library/multiprocessing.shared_memory.html

# Placo Spider - 四足机器人全身控制系统

## 项目简介

Placo Spider 是一个基于 [Placo](https://github.com/Placo-Team/placo) 运动学求解器的四足机器人全身控制（Whole-Body Control, WBC）系统。该项目实现了完整的四足机器人步态规划、逆运动学求解、力控制和仿真验证功能。

### 主要特性

- 🤖 **全身运动控制**：基于 Placo KinematicsSolver 的 QP（二次规划）求解器
- 🚶 **步态管理**：支持准静态步态（quasi-static）和动态步态
- 🦶 **接触力优化**：自动处理足端接触状态和支撑多边形约束
- 📊 **实时可视化**：集成 3D 机器人可视化和数据监控
- 📈 **数据记录**：完整的 CSV 数据日志系统
- 🎮 **交互控制**：支持键盘实时控制和命令输入
- 🔧 **模块化设计**：清晰的类结构，便于扩展和调试

## 系统架构

```
placo_spider/
├── spider_controller/           # 控制器核心模块
│   ├── spider_ik.py            # 主控制器类（SpiderIK）
│   ├── run_spider.py           # 主程序入口
│   ├── mujoco_sim.py           # MuJoCo 仿真环境
│   ├── gait_manager.py         # 步态管理器
│   ├── spider_comp.py          # 补偿控制
│   ├── spider_visual.py        # 可视化模块
│   ├── spider_logger.py        # 数据日志记录
│   ├── shared_sim_data.py      # 仿真数据通信
│   └── debug/                  # 数据输出目录
├── spider_SLDASM_2m6d/         # 机器人 URDF 模型
│   └── urdf/
│       ├── robot.urdf          # 机器人描述文件
│       └── meshes/             # 3D 网格文件
└── spider_sldasm/              # 备用模型
```

### 核心组件

#### 1. SpiderIK 控制器 (`spider_ik.py`)
- **功能**：主控制器类，整合步态规划、QP求解、力控制
- **特性**：
  - 基于 Placo KinematicsSolver 的 QP 优化
  - 支持 COM（质心）约束和支撑多边形约束
  - 集成重力补偿和接触力计算
  - 可配置的控制滤波器
  - 力矩限幅和安全保护

#### 2. 步态管理器 (`gait_manager.py`)
- **功能**：处理步态规划和足端轨迹生成
- **支持步态**：
  - 准静态步态（quasi-static）
  - 站立模式（stand）
  - 可扩展的动态步态

#### 3. MuJoCo 仿真 (`mujoco_sim.py`)
- **功能**：物理仿真环境和数据通信
- **特性**：
  - 实时物理仿真
  - 共享内存通信
  - 可视化渲染
  - 传感器数据模拟

## 快速开始

### 环境要求

- Python 3.8+
- [Placo](https://github.com/Placo-Team/placo) 运动学库
- MuJoCo 物理仿真引擎
- NumPy, matplotlib（用于数据处理和可视化）

### 安装依赖

```bash
# 安装 Placo（参考官方文档）
pip install placo

# 安装其他依赖
pip install numpy matplotlib mujoco
```

### 运行步骤

1. **启动 MuJoCo 仿真环境**
   ```bash
   cd spider_controller
   python mujoco_sim.py --viewer --realtime
   ```

2. **启动控制器**（新开终端）
   ```bash
   cd spider_controller
   # 如果仅用meshcat可视化
   python run_spider.py

   # 如果要仿真，请先启用仿真
   python run_spider.py --enable-shm
   ```

3. **交互控制**
   使用键盘控制机器人运动：
   - `w/s`：前进/后退
   - `a/d`：左转/右转
   - `q/e`：左右平移
   - `空格`：停止
   - `Ctrl+C`：退出

### 配置参数

主要配置在 `SpiderIkConfig` 类中：

```python
@dataclass
class SpiderIkConfig:
    dt: float = 0.01                    # 控制周期
    gait_mode: str = 'quasi_static'     # 步态模式
    cycle_period: float = 2.0           # 步态周期
    swing_height: float = 0.12          # 摆动腿高度
    body_height: float = 0.26           # 机身高度
    
    # 控制滤波
    enable_ctrl_filter: bool = True     # 启用控制滤波
    ctrl_filter_alpha: float = 0.8      # 滤波系数
    
    # QP 权重
    leg_task_weight: float = 1e3        # 足端任务权重
    body_task_weight: float = 1e1       # 机身任务权重
```

## 数据监控

系统自动记录以下数据到 `debug/` 目录：

- `cmd_data.csv`：控制命令历史
- `com_target_data.csv`：质心目标轨迹
- `com_world_data.csv`：实际质心位置
- `contact_state_data.csv`：足端接触状态
- `LH_target_data.csv`：各足端目标位置
- `tauff_data.csv`：控制力矩输出

### 数据可视化

```bash
cd spider_controller/debug
bash plot.sh  # 生成所有数据图表
```

## 技术细节

### 控制架构

1. **步态规划**：GaitCycleManager 生成足端轨迹和接触状态
2. **QP 求解**：Placo KinematicsSolver 优化关节速度
3. **力控制**：计算重力补偿和接触力
4. **滤波输出**：平滑控制信号，减少高频噪声

### 约束处理

- **COM 约束**：质心保持在支撑多边形内
- **速度约束**：关节速度限制
- **力矩约束**：输出力矩限幅（±30 N⋅m）
- **接触约束**：足端法向约束（锥约束）

### 安全机制

- 控制力矩限幅
- 速度限制
- 异常检测和错误处理
- 滤波器减少控制噪声

## 开发指南

### 扩展步态

在 `gait_manager.py` 中添加新的步态模式：

```python
def update_trot_gait(self, t: float, dt: float):
    # 实现 trot 步态逻辑
    pass
```

### 添加传感器

在 `shared_sim_data.py` 中扩展数据结构：

```python
@dataclass
class SimToCPGData:
    # 添加新的传感器数据
    imu_data: np.ndarray = field(default_factory=lambda: np.zeros(6))
```

### 调试技巧

1. **可视化调试**：使用 `spider_visual.py` 显示机器人状态
2. **数据分析**：查看 `debug/` 目录下的 CSV 文件
3. **日志输出**：在代码中添加 print 语句跟踪执行
4. **参数调优**：修改 `SpiderIkConfig` 中的权重和参数

## 常见问题

### Q: MuJoCo 启动失败
A: 确保已正确安装 MuJoCo 并设置环境变量，检查 URDF 文件路径。

### Q: 机器人不稳定
A: 尝试调整 QP 权重，增加 `body_task_weight` 或减小 `leg_task_weight`。

### Q: 控制延迟
A: 检查控制频率设置，确保 `dt` 参数合理（建议 0.01-0.02）。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证。

## 相关链接

- [Placo 官方文档](https://github.com/Placo-Team/placo)
- [MuJoCo 物理引擎](https://mujoco.org/)
- [四足机器人控制理论](https://manipulation.csail.mit.edu/pick.html)

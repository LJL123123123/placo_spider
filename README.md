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

#### 1. 主程序入口 (`run_spider.py`)
- **功能**：主程序以及通信
- **特性**：
  - class SHM2DDSData
  ```python
    #CTRL_SIZE = 24 , KP_SIZE = 24
    q: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    qdd: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(KP_SIZE, dtype=np.float64))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(KD_SIZE, dtype=np.float64))
    ```
  - 函数def SpiderIkData2SHM2DDSData(data: SpiderIkData) -> SHM2DDSData
    - 将`spider_ik.py`中传出的 `SpiderIkData` 转换成 `SHM2DDSData` 适配实机24个电机，同时已经将 `ankle_roll_joint` 的偏置设置好了
  - `ikdata = spider.step(cmd_vxyz, cmd_yaw)` 更新 WBIK 求解器， `data = SpiderIkData2SHM2DDSData(ikdata)` 来转化数据格式

#### 2. SpiderIK 控制器 (`spider_ik.py`)
- **功能**：主控制器类，创建`self.gait_params` & `self.gait` 管理步态 、创建 `self.solver` 管理WBIK求解器
- **特性**：
  - 基于 Placo KinematicsSolver 的 QP 优化
    - 构建
  - 支持 COM（质心）约束和支撑多边形约束
- **接口**：
  - class SpiderIkData
    ```python
    class SpiderIkData:
      q: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
      qd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
      qdd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64)) 
    ```
  - `step(self, cmd_vxyz: np.ndarray, cmd_yaw_rate: float)` 
    - 输入线速度与 yaw 角速度
    - 调用 `plan = self.gait.update(……)` 更新步态管理器
    - 构建机身与足端的 `add_position_task` 任务来实现跟随步态管理器输出的机身与足端位置
    - 构建机身的 `add_orientation_task` 任务与足端的 `add_axisalign_task` 任务来约束姿态
    - 构建 `add_regularization_task` & `add_com_polygon_constraint` 约束来现在质心的瞬移以及约束质心在支撑多边形内
    - 返回参数 `data：SpiderIkData`
  - class SpiderIkConfig
    - solver weights
    ```python
    dt: float = 0.01                              # 控制周期
    polygon_margin: float = 0.05                  # 支撑多边形的余量
    leg_task_weight: float = 1e3                  # 足端位置任务权重
    leg_task_ori_weight: float = 1e1+50           # 足端姿态权重
    body_task_weight: float = 1e1                 # 机身位置权重
    body_task_ori_weight: float = 1e2             # 机身姿态权重
    com_constraint_weight: float = 1e2+50         # 支撑多边形权重
    regularization_weight: float = 1e-1           # 正则化权重
    ```
    - `leg_foot_name_map` URDF 中足端 name
     ```python
     {
      'LH': 'LR_wheel_link',
      'RH': 'RR_wheel_link',
      'RF': 'RF_wheel_link',
      'LF': 'LF_wheel_link',
      }
    ```
    - `body_link_name` URDF 中机身 name
    - `cycle_period` 步态周期
    - `swing_height` 抬腿高度
    - `enable_logger` 数据打印开关
    - `enable_visual` meshcat可视化开关
    - `leg_init_state` 机器人行走模式默认 `SpiderIkData：q`
     ```python
     lambda: np.array([0,0,0.26,
                        0,0,0,1,
                        0.7899997871565971,0.01658366118250419,-0.004088938263042,
                        -0.7867905831961373,-0.016585433996785828,0.004100463013502682,
                        -0.7867905831961349,0.016585433996786143,-0.00410046301350216,
                        0.7899997871565964,-0.016583661182505084,0.004088938263039494
                    ])
    ```


#### 3. 步态管理器 (`gait_manager.py`)
- **功能**：处理步态规划和足端轨迹生成
- **接口**：
  - class GaitPlan
    ```python
    in_cycle: bool 
    stance_legs: List[str]              # 支撑足列表
    swing_legs: List[str]               # 摆动足列表
    contact_state: Dict[str, bool]      # 触地状态
    target_pos: Dict[str, np.ndarray]   # 目标位置列表
    target_ori: Dict[str, np.ndarray]   # 目标姿态列表

    is_stand: bool = False 
    ```
  - def update(……) -> GaitPlan 步态规划器更新接口，接收机器人当前位置列表、姿态列表、目标线速度、目标yaw角速度规划步态，输出 `plan=GaitPlan`

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
   使用XboxController控制机器人运动：


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
### 26.3.14 添加shm_to_plotjuggler_udp
Usage：

### 26.3.16 更换 urdf 为 sqr_description
#### 区别
0位 && 默认姿态
##### robot.urdf修改
1.`*ankle*joint & *wheel_joint `改成了`fixed`
2.`RR_ankle_roll_joint & RF_ankle_roll_joint` 改成了 `rpy="1.57 0 0"`；`LR_ankle_roll_joint & LF_ankle_roll_joint` 改成了 `rpy="-1.57 0 0"`

##### 控制修改
1.将`leg_orien_tasks`的`add_cone_constraint`换成`add_axisalign_task（"frame-name", array_at_frame, world_target_array）`
```
add_axisalign_task
硬任务会强制误差趋近于0，可能导致求解失败如果约束不可满足
软任务的权重是相对值，需要根据系统中其他任务的权重来平衡
可以通过 solver.dump_status() 查看所有任务的实时误差状态 solver_status.rst:46-48
误差计算基于轴线的角度偏差，使用弧度作为单位
```
2.`class SpiderIkConfig:`中添加
```
leg_init_state: np.ndarray = field(default_factory=lambda: np.array([
            0,0,0.26,
            0,0,0,1,
            0.7899997871565971,0.01658366118250419,-0.004088938263042,
            -0.7867905831961373,-0.016585433996785828,0.004100463013502682,
            -0.7867905831961349,0.016585433996786143,-0.00410046301350216,
            0.7899997871565964,-0.016583661182505084,0.004088938263039494]))
```
### 26.3.18 调高zmp权重
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

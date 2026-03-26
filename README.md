# Placo Spider - 四足机器人全身控制系统

## 系统架构

```
placo_spider/
├── spider_controller/           # 控制器核心模块
│   ├── spider_ik.py            # 主控制器类（SpiderIK）
│   ├── run_spider.py           # 主程序入口
│   ├── gait_manager.py         # 步态管理器
│   ├── spider_comp.py          # 补偿控制
│   ├── spider_visual.py        # 可视化模块
│   ├── spider_logger.py        # 数据日志记录
│   ├── debug/                  # 数据输出目录
│   └──robot.urdf          # 机器人描述文件(12维 joint 备份)
├── sqr_a1_description/         # 机器人 URDF 模型
│   ├── urdf/
│   │   └── robot.urdf          # 机器人描述文件
│   └── meshes/             # 3D 网格文件
└── spider_sldasm/              # 备用模型
```
ps:需要将 `spider_controller/robot.urdf` 复制到 `sqr_a1_description/urdf/` 中

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
  - 计划在 `data = SpiderIkData2SHM2DDSData(ikdata)` 后进行数据通信

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
pip install numpy matplotlib meshcat pygame
```

### 运行步骤

1. **启动控制器**（新开终端）
   ```bash
   cd spider_controller
   # 如果仅用meshcat可视化
   python run_spider.py

   # 如果要仿真，请先启用仿真
   python run_spider.py --enable-shm
   ```

2. **交互控制**
   使用XboxController控制机器人运动：
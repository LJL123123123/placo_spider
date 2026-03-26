# Spider IK SharedMemory to DDS Publisher

这个程序负责将 Spider IK (使用 Placo QP 求解器) 通过共享内存输出的控制命令转换成 DDS 消息，发布给 MuJoCo 仿真模块。

## 架构概览

```
┌─────────────────┐
│  spider_ik.py   │ (在 spider_controller 中运行)
│ (Placo QP 求解) │
└────────┬────────┘
         │ 写入共享内存
         │ (CPG_TO_SIM)
         ▼
┌─────────────────────────┐
│  Shared Memory          │
│ - qpos_desired (19)    │
│ - ctrl_desired (12)    │
│ - kp gains (12)        │
│ - kd gains (12)        │
│ - timestamp            │
└────────┬────────────────┘
         │ 读取
         ▼
┌──────────────────────────────────┐
│ shm_to_dds_publisher (本程序)   │
│ - SharedMemoryToDdsPublisher    │
│ - 转换成 DDS PdCmd 消息         │
└────────┬─────────────────────────┘
         │ DDS 发布
         │ (~/mujoco_sim_zbsdk 订阅)
         ▼
┌──────────────────┐
│  mujoco_sim_zbsdk│
│  (C++ 仿真环境)  │
└──────────────────┘
```

## 文件说明

### SharedMemoryClient.h
- 头文件（位于 `spider_c/include/`）
- 封装了与共享内存的交互
- 提供 `read_cpg_cmd()` 读取 CPG 发往仿真的命令
- 提供 `write_sim_state()` 写入仿真的观测状态

### SharedMemoryToDdsPublisher.h / .cpp
- 主要的发布者类，负责：
  1. 初始化 DDS 参与者、主题、发布者、数据写入者
  2. 启动后台发布线程
  3. 定期从共享内存读取数据并通过 DDS 发布

### shm_to_dds_main.cpp
- 程序入口点
- 创建发布者实例，启动发布线程
- 处理信号以优雅关闭

## 编译指令

### 前置条件
1. 已安装 DDS 库（通常来自 HWdds）
2. 已安装 Eigen3
3. spider-controller 正在运行，已创建共享内存

### 编译
```bash
cd /home/zenbot-ljl/placo_spider/spider_c
mkdir -p build
cd build
cmake ..
make shm_to_dds_publisher
```

编译后的可执行文件会在 `build/` 目录中。

## 运行指令

### 基本用法
```bash
./build/shm_to_dds_publisher
```

### 指定发布频率
```bash
# 发布频率为 200Hz（而不是默认 100Hz）
./build/shm_to_dds_publisher 200
```

### 指定 DDS Topic 名称
```bash
./build/shm_to_dds_publisher 100 "custom_topic_name"
```

## 工作流程

### 启动顺序
1. **首先启动 spider-controller**
   ```bash
   cd /home/zenbot-ljl/placo_spider/spider_controller
   python run_spider.py  # 或其他入口脚本
   ```

2. **然后启动 MuJoCo 仿真**
   ```bash
   cd /home/zenbot-ljl/placo_spider/mujoco_sim_zbsdk/build
   ./install/bin/mujoco_sim_zbsdk/mujoco_sim_zbsdk sqr_a1
   ```

3. **最后启动本程序**
   ```bash
   ./build/shm_to_dds_publisher 100
   ```

### 数据流向
1. `spider_ik.py` 计算控制命令，写入 CPG_TO_SIM 共享内存
2. `shm_to_dds_publisher` 每 10ms（100Hz）读一次共享内存
3. 读取的数据转换成 DDS `robot_msgs::msg::PdCmd` 消息
4. 通过 DDS 发布给 MuJoCo 仿真
5. MuJoCo 接收命令，更新机器人控制并反馈状态到 SIM_TO_CPG 共享内存

## 共享内存布局

### CPG_TO_SIM 内存（本程序读取）
```
[0..18]   : qpos_desired    (19个 float64，关节期望位置)
[19..30]  : ctrl_desired    (12个 float64，前馈力矩)
[31..42]  : kp              (12个 float64，P型增益)
[43..54]  : kd              (12个 float64，D型增益)
[55]      : timestamp       (1个 float64，秒)
```

## DDS 消息结构（robot_msgs::msg::PdCmd）

```
struct PdCmd {
    int64 ts;                    // 时间戳（纳秒）
    sequence<PdMotorCmd> motors; // 电机控制命令数组
};

struct PdMotorCmd {
    float q;      // 期望关节角度（rad）
    float qd;     // 期望关节速度（rad/s）
    float tau;    // 前馈力矩（Nm）
    float kp;     // P型增益
    float kd;     // D型增益
    float kt;     // ?（reserved）
    int32 mode;   // 控制模式
};
```

## 常见问题

### 共享内存不能打开？
- 确保 spider-controller 正在运行（会创建共享内存）
- 检查 named semaphore `/mujoco_sim_to_cpg` 和 `/cpg_to_mujoco_sim` 是否存在
  ```bash
  ls -la /dev/shm/ | grep mujoco
  ```

### DDS 发布失败？
- 确保 DDS 配置文件存在（通常在 `dds/config/` 目录）
- 检查 MuJoCo 仿真程序是否运行和订阅了相同的 Topic

### 程序崩溃?
- 检查 EigenVectorXd 的大小是否为 19
- 检查 `std::vector<double>` 的大小是否为 12（ctrl、kp、kd）
- 查看程序输出的错误信息

## 集成建议

本程序可以集成到以下较大的系统中：
1. **自动测试框架**：自动启动 MuJoCo、spider-controller 和本程序
2. **实硬件衔接**：修改本程序改为订阅 DDS MuJoCo 状态并写回硬件
3. **可视化**：结合 PlotJuggler 可视化实时数据流

## 代码修改指南

如需修改消息发送逻辑，主要修改点在 `SharedMemoryToDdsPublisher::publish_once()` 方法中的 TODO 部分。

具体步骤：
1. 检查实际生成的 `robot_msgs.h` 消息类型
2. 根据 IDL 生成的 C++ 代码调整消息构建部分
3. 确保关节索引映射正确（特别是与浮动基的偏移）
4. 编译并测试

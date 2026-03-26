#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#if __has_include(<eigen3/Eigen/Dense>)
#include <eigen3/Eigen/Dense>
#elif __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#else
#error "Eigen headers not found. Please install Eigen and configure include paths."
#endif

// 对应 Python 中的 Size
constexpr int QPOS_SIZE = 24;
constexpr int CTRL_SIZE = 24;
constexpr int KP_SIZE = 24;
constexpr int KD_SIZE = 24;
constexpr int SIM_TO_CPG_SIZE = QPOS_SIZE + CTRL_SIZE + 1; // 56
constexpr int CPG_TO_SIM_SIZE = QPOS_SIZE + CTRL_SIZE + KP_SIZE + KD_SIZE + 1; // 104

class SharedMemoryClient {
public:
    double* sim_to_cpg_data = nullptr;
    double* cpg_to_sim_data = nullptr;

    SharedMemoryClient() {
        // Python的 shared_memory 在底层通常会加上前缀 "/"
        const char* shm_sim_to_cpg = "/mujoco_sim_to_cpg";
        const char* shm_cpg_to_sim = "/cpg_to_mujoco_sim";

        // 1. 映射 Sim -> CPG 内存 (读取用)
        int fd_sim = shm_open(shm_sim_to_cpg, O_RDWR, 0666);
        if (fd_sim == -1) {
            std::cerr << "无法打开共享内存 " << shm_sim_to_cpg 
                      << " (请确保 Python MuJoCo 仿真已启动!)" << std::endl;
            exit(1);
        }
        sim_to_cpg_data = (double*)mmap(0, SIM_TO_CPG_SIZE * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_sim, 0);

        // 2. 映射 CPG -> Sim 内存 (写入用)
        int fd_cpg = shm_open(shm_cpg_to_sim, O_RDWR, 0666);
        if (fd_cpg == -1) {
            std::cerr << "无法打开共享内存 " << shm_cpg_to_sim << std::endl;
            exit(1);
        }
        cpg_to_sim_data = (double*)mmap(0, CPG_TO_SIM_SIZE * sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd_cpg, 0);
    }

    ~SharedMemoryClient() {
        if (sim_to_cpg_data) munmap(sim_to_cpg_data, SIM_TO_CPG_SIZE * sizeof(double));
        if (cpg_to_sim_data) munmap(cpg_to_sim_data, CPG_TO_SIM_SIZE * sizeof(double));
    }

    // 读取 MuJoCo 发来的状态 (这里主要用于获取实测质心状态，如果需要的话)
    void read_sim_state(std::vector<double>& qpos, std::vector<double>& ctrl, double& timestamp) {
        qpos.assign(sim_to_cpg_data, sim_to_cpg_data + QPOS_SIZE);
        ctrl.assign(sim_to_cpg_data + QPOS_SIZE, sim_to_cpg_data + QPOS_SIZE + CTRL_SIZE);
        timestamp = sim_to_cpg_data[SIM_TO_CPG_SIZE - 1];
    }

    // 读取 CPG 发来的期望指令
    void read_cpg_cmd(Eigen::VectorXd& qpos_desired,
                      std::vector<double>& ctrl_desired,
                      std::vector<double>& kp,
                      std::vector<double>& kd,
                      double& timestamp)
    {
        qpos_desired.resize(QPOS_SIZE);
        for (int i = 0; i < QPOS_SIZE; ++i) {
            qpos_desired[i] = cpg_to_sim_data[i];
        }

        ctrl_desired.assign(cpg_to_sim_data + QPOS_SIZE,
                            cpg_to_sim_data + QPOS_SIZE + CTRL_SIZE);
        kp.assign(cpg_to_sim_data + QPOS_SIZE + CTRL_SIZE,
                  cpg_to_sim_data + QPOS_SIZE + CTRL_SIZE + KP_SIZE);
        kd.assign(cpg_to_sim_data + QPOS_SIZE + CTRL_SIZE + KP_SIZE,
                  cpg_to_sim_data + QPOS_SIZE + CTRL_SIZE + KP_SIZE + KD_SIZE);

        timestamp = cpg_to_sim_data[CPG_TO_SIM_SIZE - 1];
    }

    // 写入 MuJoCo 状态，发送给 CPG
    void write_sim_state(const std::vector<double>& qpos,
                         const std::vector<double>& ctrl,
                         double timestamp)
    {
        const int qpos_count = std::min<int>(QPOS_SIZE, static_cast<int>(qpos.size()));
        const int ctrl_count = std::min<int>(CTRL_SIZE, static_cast<int>(ctrl.size()));

        for (int i = 0; i < qpos_count; ++i) {
            sim_to_cpg_data[i] = qpos[i];
        }
        for (int i = qpos_count; i < QPOS_SIZE; ++i) {
            sim_to_cpg_data[i] = 0.0;
        }

        for (int i = 0; i < ctrl_count; ++i) {
            sim_to_cpg_data[QPOS_SIZE + i] = ctrl[i];
        }
        for (int i = ctrl_count; i < CTRL_SIZE; ++i) {
            sim_to_cpg_data[QPOS_SIZE + i] = 0.0;
        }

        sim_to_cpg_data[SIM_TO_CPG_SIZE - 1] = timestamp;
    }

    // 写入求解出的期望指令发送给 MuJoCo
    void write_cpg_cmd(const Eigen::VectorXd& qpos_desired, 
                       const std::vector<double>& ctrl_desired, 
                       const std::vector<double>& kp, 
                       const std::vector<double>& kd) 
    {
        // 写入期望关节角度
        const int qpos_count = std::min((int)qpos_desired.size(), QPOS_SIZE);
        for (int i = 0; i < qpos_count; ++i) {
            cpg_to_sim_data[i] = qpos_desired[i];
        }
        for (int i = qpos_count; i < QPOS_SIZE; ++i) {
            cpg_to_sim_data[i] = 0.0;
        }
        
        // 写入前馈力矩
        const int ctrl_count = std::min((int)ctrl_desired.size(), CTRL_SIZE);
        for (int i = 0; i < ctrl_count; ++i) cpg_to_sim_data[QPOS_SIZE + i] = ctrl_desired[i];
        for (int i = ctrl_count; i < CTRL_SIZE; ++i) cpg_to_sim_data[QPOS_SIZE + i] = 0.0;
        
        // 写入 Kp, Kd
        const int kp_count = std::min((int)kp.size(), KP_SIZE);
        const int kd_count = std::min((int)kd.size(), KD_SIZE);
        for (int i = 0; i < kp_count; ++i) cpg_to_sim_data[QPOS_SIZE + CTRL_SIZE + i] = kp[i];
        for (int i = kp_count; i < KP_SIZE; ++i) cpg_to_sim_data[QPOS_SIZE + CTRL_SIZE + i] = 0.0;

        for (int i = 0; i < kd_count; ++i) cpg_to_sim_data[QPOS_SIZE + CTRL_SIZE + KP_SIZE + i] = kd[i];
        for (int i = kd_count; i < KD_SIZE; ++i) cpg_to_sim_data[QPOS_SIZE + CTRL_SIZE + KP_SIZE + i] = 0.0;

        // 写入时间戳
        auto now = std::chrono::system_clock::now().time_since_epoch();
        cpg_to_sim_data[CPG_TO_SIM_SIZE - 1] = std::chrono::duration<double>(now).count();
    }
};
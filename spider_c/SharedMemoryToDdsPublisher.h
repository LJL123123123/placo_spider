#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include "SharedMemoryClient.h"

/**
 * @brief 从共享内存读取 Spider IK 控制数据并通过 DDS 发布的发布者类
 *
 * 功能流程：
 * 1. 通过 SharedMemoryClient 读取 CPG_TO_SIM 共享内存中的控制命令
 *    - qpos_desired (31维，其中前24维用于腿部)
 *    - ctrl_desired (24维)
 *    - kp 增益 (24维)
 *    - kd 增益 (24维)
 *    - timestamp
 *
 * 2. 将数据转换成 DDS PdCmd 消息格式
 *
 * 3. 通过 DDS 发布给 mujoco-sim-zbsdk
 */
class SharedMemoryToDdsPublisher {
public:
    SharedMemoryToDdsPublisher();
    ~SharedMemoryToDdsPublisher();

    /**
     * @brief 初始化 DDS 发布者
     * @param topicName DDS 里的 topic 名称，默认为 "robot_msgs::msg::PdCmd"
     * @return 成功则返回 true
     */
    bool init(const std::string& topicName = "robot_msgs::msg::PdCmd");

    /**
     * @brief 启动发布线程，持续从共享内存读取并发布
     * @param frequency 发布频率（Hz），默认为 100Hz
     * @return 成功则返回 true
     */
    bool start_publisher(int frequency = 100);

    /**
     * @brief 停止发布线程
     */
    void stop_publisher();

    /**
     * @brief 清理 DDS 资源
     */
    void destroy();

private:
    /**
     * @brief 发布线程的主函数
     * @param frequency 发布频率（Hz）
     */
    void publisher_thread_main(int frequency);

    /**
     * @brief 从共享内存读取一次数据并发布
     * @return 成功则返回 true
     */
    bool publish_once();

    std::string resolve_config_path() const;

    // zb_sdk 相关状态
    bool m_sdk_initialized{false};
    std::string m_topic_name;

    // 共享内存客户端
    std::unique_ptr<SharedMemoryClient> m_shm_client;

    // 发布线程控制
    std::atomic<bool> m_running{false};
    std::unique_ptr<std::thread> m_publisher_thread;

    // 统计信息
    unsigned long m_publish_count;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_last_log_time;
};

#include <iostream>
#include <csignal>
#include <cstdlib>
#include <atomic>
#include <thread>
#include "SharedMemoryToDdsPublisher.h"

std::atomic<bool> g_shutdown{false};

void signal_handler(int signal)
{
    std::cout << "\n[MAIN] Received signal " << signal << ", shutting down..." << std::endl;
    g_shutdown.store(true);
}

int main(int argc, char* argv[])
{
    // 注册信号处理
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=== Spider IK SharedMemory to DDS Publisher ===" << std::endl;
    std::cout << "Reads CPG control commands from shared memory and publishes via zb_sdk(DDS) to MuJoCo"
              << std::endl;

    // 解析命令行参数
    int publish_frequency = 100;  // 默认 100Hz
    std::string topic_name = "rt/desire/joint";

    if (argc > 1) {
        try {
            publish_frequency = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid frequency argument: " << argv[1]
                      << ", using default 100 Hz" << std::endl;
            publish_frequency = 100;
        }
    }

    if (argc > 2) {
        topic_name = argv[2];
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Publish frequency: " << publish_frequency << " Hz" << std::endl;
    std::cout << "  - Topic name: " << topic_name << std::endl;
    std::cout << "  - Optional env ZB_SDK_CONFIG: "
              << (std::getenv("ZB_SDK_CONFIG") ? std::getenv("ZB_SDK_CONFIG") : "<not set>")
              << std::endl;

    // 创建并初始化发布者
    SharedMemoryToDdsPublisher publisher;

    std::cout << "\n[MAIN] Initializing DDS publisher..." << std::endl;
    if (!publisher.init(topic_name)) {
        std::cerr << "[MAIN] Failed to initialize publisher" << std::endl;
        return 1;
    }

    std::cout << "[MAIN] Starting publisher thread..." << std::endl;
    if (!publisher.start_publisher(publish_frequency)) {
        std::cerr << "[MAIN] Failed to start publisher" << std::endl;
        return 1;
    }

    std::cout << "[MAIN] Publisher is running. Press Ctrl+C to stop." << std::endl;

    // 保持主线程运行直到收到关闭信号
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "[MAIN] Shutting down..." << std::endl;
    publisher.stop_publisher();
    publisher.destroy();

    std::cout << "[MAIN] Exit successfully" << std::endl;
    return 0;
}

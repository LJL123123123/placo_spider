#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <poll.h>
#include <unistd.h>

#include "zbsdk/zb_sdk_topics.h"

namespace {
std::atomic<bool> g_running{true};

int64_t now_ns()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

bool file_readable(const std::string& path)
{
    return !path.empty() && ::access(path.c_str(), R_OK) == 0;
}

std::string resolve_config_path()
{
    const char* env = std::getenv("ZB_SDK_CONFIG");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }

    const char* candidates[] = {
        "./dds/config/config.json",
        "../dds/config/config.json",
        "../../spider_c/dds/config/config.json",
        "../mujoco_sim_zbsdk/config/config.json",
        "../../mujoco_sim_zbsdk/config/config.json",
    };
    for (const auto* p : candidates) {
        if (file_readable(p)) {
            return std::string(p);
        }
    }

    return "./dds/config/config.json";
}

void on_signal(int)
{
    g_running.store(false);
}

int run_pub(const std::string& topic)
{
    int rc = zb_pdcmd_writer_init(topic.c_str());
    if (rc < 0) {
        std::cerr << "[robotcmdtest] zb_pdcmd_writer_init(" << topic << ") failed: " << rc << std::endl;
        return 1;
    }

    std::cout << "[robotcmdtest] publishing PdCmd on topic: " << topic << std::endl;
    std::cout << "[robotcmdtest] Ctrl+C to stop" << std::endl;

    const int kMotors = 12;
    robot_msgs_PdMotorCmd_C motors[kMotors]{};
    robot_msgs_PdCmd_View view{};
    view.motors = motors;
    view.motors_len = kMotors;

    auto start = std::chrono::steady_clock::now();
    uint64_t seq = 0;

    while (g_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - start).count();

        view.ts = now_ns();
        for (int i = 0; i < kMotors; ++i) {
            const double w = 2.0 * M_PI * (0.4 + 0.15 * i);
            motors[i].q = static_cast<float>(0.3 * std::sin(w * t));
            motors[i].qd = static_cast<float>(0.3 * w * std::cos(w * t));
            motors[i].tau = 0.0f;
            motors[i].kp = 50.0f;
            motors[i].kd = 1.5f;
            motors[i].kt = 0.0f;
            motors[i].mode = 0;
        }

        rc = zb_pdcmd_write(&view);
        if (rc < 0) {
            std::cerr << "[robotcmdtest] zb_pdcmd_write failed: " << rc << std::endl;
        }

        if ((seq % 200) == 0) {
            std::cout << "[PUB] M0 q=" << motors[0].q
                      << " qd=" << motors[0].qd
                      << " kp=" << motors[0].kp
                      << " kd=" << motors[0].kd << std::endl;
        }
        ++seq;

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    return 0;
}

int run_sub(const std::string& topic)
{
    int rc = zb_pdcmd_reader_init(topic.c_str());
    if (rc < 0) {
        std::cerr << "[robotcmdtest] zb_pdcmd_reader_init(" << topic << ") failed: " << rc << std::endl;
        return 1;
    }

    std::cout << "[robotcmdtest] subscribing PdCmd from topic: " << topic << std::endl;
    std::cout << "[robotcmdtest] Ctrl+C to stop" << std::endl;

    const int kCap = 64;
    robot_msgs_PdMotorCmd_C motors[kCap]{};
    robot_msgs_PdCmd_View view{};
    view.motors = motors;
    view.motors_len = 0;

    const int notify_fd = zb_pdcmd_get_notify_fd();
    uint64_t recv_cnt = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (g_running.load()) {
        int n = 0;
        if (notify_fd >= 0) {
            pollfd pfd{};
            pfd.fd = notify_fd;
            pfd.events = POLLIN;
            const int pr = ::poll(&pfd, 1, 200);
            if (pr > 0 && (pfd.revents & POLLIN)) {
                n = zb_pdcmd_try_read(&view, kCap);
            }
        } else {
            if (zb_pdcmd_has_new()) {
                n = zb_pdcmd_try_read(&view, kCap);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (n < 0) {
            std::cerr << "[robotcmdtest] zb_pdcmd_try_read failed: " << n << std::endl;
            continue;
        }
        if (n == 0 || view.motors_len <= 0) {
            continue;
        }

        ++recv_cnt;
        if ((recv_cnt % 200) == 0) {
            const auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double hz = dt > 1e-6 ? static_cast<double>(recv_cnt) / dt : 0.0;
            std::cout << "[SUB] M0 q=" << view.motors[0].q
                      << " qd=" << view.motors[0].qd
                      << " tau=" << view.motors[0].tau
                      << " kp=" << view.motors[0].kp
                      << " kd=" << view.motors[0].kd
                      << " motors=" << view.motors_len
                      << " rate=" << hz << " Hz" << std::endl;
        }
    }

    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [pub|sub] [topic]\n"
                  << "  default topic: rt/desire/joint" << std::endl;
        return 1;
    }

    const std::string mode(argv[1]);
    const std::string topic = (argc > 2) ? std::string(argv[2]) : std::string("rt/desire/joint");

    const std::string cfg = resolve_config_path();
    const int rc_init = zb_sdk_init(ZB_BACKEND_DDS, cfg.c_str());
    if (rc_init < 0) {
        std::cerr << "[robotcmdtest] zb_sdk_init failed: " << rc_init
                  << ", config=" << cfg << std::endl;
        return 1;
    }
    std::cout << "[robotcmdtest] zb_sdk initialized with config: " << cfg << std::endl;

    int ret = 0;
    if (mode == "pub") {
        ret = run_pub(topic);
    } else if (mode == "sub") {
        ret = run_sub(topic);
    } else {
        std::cerr << "Invalid mode: " << mode << " (expected pub/sub)" << std::endl;
        ret = 1;
    }

    zb_sdk_shutdown();
    return ret;
}

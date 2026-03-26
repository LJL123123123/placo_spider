#include "SharedMemoryToDdsPublisher.h"
#include "zbsdk/zb_sdk_topics.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sys/stat.h>

namespace {
constexpr int kActuatedJoints = 24;

bool file_exists(const std::string& path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}
} // namespace

SharedMemoryToDdsPublisher::SharedMemoryToDdsPublisher()
    : m_publish_count(0)
{
}

SharedMemoryToDdsPublisher::~SharedMemoryToDdsPublisher()
{
    stop_publisher();
    destroy();
}

bool SharedMemoryToDdsPublisher::init(const std::string& topicName)
{
    m_topic_name = topicName;

    try {
        m_shm_client = std::make_unique<SharedMemoryClient>();
        std::cout << "[SHM2DDS] SharedMemoryClient initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[SHM2DDS] Failed to initialize SharedMemoryClient: " << e.what() << std::endl;
        return false;
    }

    const std::string cfg_path = resolve_config_path();
    const int rc_init = zb_sdk_init(ZB_BACKEND_DDS, cfg_path.c_str());
    if (rc_init < 0) {
        std::cerr << "[SHM2DDS] zb_sdk_init failed: " << rc_init
                  << ", config=" << cfg_path << std::endl;
        return false;
    }
    m_sdk_initialized = true;
    std::cout << "[SHM2DDS] zb_sdk initialized with config: " << cfg_path << std::endl;

    const int rc_writer = zb_pdcmd_writer_init(m_topic_name.c_str());
    if (rc_writer < 0) {
        std::cerr << "[SHM2DDS] zb_pdcmd_writer_init('" << m_topic_name
                  << "') failed: " << rc_writer << std::endl;
        return false;
    }
    std::cout << "[SHM2DDS] PdCmd writer initialized on topic: " << m_topic_name << std::endl;

    m_last_log_time = std::chrono::high_resolution_clock::now();
    return true;
}

bool SharedMemoryToDdsPublisher::start_publisher(int frequency)
{
    if (frequency <= 0) {
        std::cerr << "[SHM2DDS] Invalid frequency: " << frequency << std::endl;
        return false;
    }

    if (m_running.load()) {
        std::cout << "[SHM2DDS] Publisher is already running" << std::endl;
        return false;
    }

    m_running.store(true);
    m_publisher_thread = std::make_unique<std::thread>(
        &SharedMemoryToDdsPublisher::publisher_thread_main, this, frequency);
    std::cout << "[SHM2DDS] Publisher thread started at " << frequency << " Hz" << std::endl;

    return true;
}

void SharedMemoryToDdsPublisher::stop_publisher()
{
    if (m_running.load()) {
        m_running.store(false);
        if (m_publisher_thread && m_publisher_thread->joinable()) {
            m_publisher_thread->join();
            std::cout << "[SHM2DDS] Publisher thread stopped" << std::endl;
        }
    }
}

void SharedMemoryToDdsPublisher::publisher_thread_main(int frequency)
{
    const int sleep_ms = 1000 / frequency;
    auto next_wake_time = std::chrono::high_resolution_clock::now();

    std::cout << "[SHM2DDS] Publisher loop started (period: " << sleep_ms << "ms)" << std::endl;

    while (m_running.load()) {
        // 发布一次
        if (!publish_once()) {
            std::cerr << "[SHM2DDS] Publish failed in iteration " << m_publish_count << std::endl;
            // 继续尝试，不中断循环
        }

        // 精确定时
        next_wake_time += std::chrono::milliseconds(sleep_ms);
        auto now = std::chrono::high_resolution_clock::now();
        if (now < next_wake_time) {
            std::this_thread::sleep_until(next_wake_time);
        }

        // 每秒打印一次统计信息
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - m_last_log_time).count();
        if (elapsed_ms >= 1000) {
            std::cout << "[SHM2DDS] Published " << m_publish_count
                      << " messages (freq: ~" << (m_publish_count * 1000 / elapsed_ms) << " Hz)"
                      << std::endl;
            m_publish_count = 0;
            m_last_log_time = current_time;
        }
    }

    std::cout << "[SHM2DDS] Publisher loop ended" << std::endl;
}

bool SharedMemoryToDdsPublisher::publish_once()
{
    if (!m_shm_client || !m_sdk_initialized) {
        return false;
    }

    try {
        Eigen::VectorXd qpos_desired;
        std::vector<double> ctrl_desired;
        std::vector<double> kp;
        std::vector<double> kd;
        double timestamp;

        m_shm_client->read_cpg_cmd(qpos_desired, ctrl_desired, kp, kd, timestamp);

        std::vector<robot_msgs_PdMotorCmd_C> motors(kActuatedJoints);
        const int q_count = std::min(kActuatedJoints, static_cast<int>(qpos_desired.size()));
        const int tau_count = std::min(kActuatedJoints, static_cast<int>(ctrl_desired.size()));
        const int kp_count = std::min(kActuatedJoints, static_cast<int>(kp.size()));
        const int kd_count = std::min(kActuatedJoints, static_cast<int>(kd.size()));

        for (int i = 0; i < kActuatedJoints; ++i) {
            robot_msgs_PdMotorCmd_C cmd{};
            if (i < q_count) {
                cmd.q = static_cast<float>(qpos_desired[i]);
            }
            cmd.qd = 0.0f;
            if (i < tau_count) {
                cmd.tau = static_cast<float>(ctrl_desired[i]);
            }
            if (i < kp_count) {
                cmd.kp = static_cast<float>(kp[i]);
            }
            if (i < kd_count) {
                cmd.kd = static_cast<float>(kd[i]);
            }
            cmd.kt = 0.0f;
            cmd.mode = 0;
            motors[i] = cmd;
        }

        robot_msgs_PdCmd_View view{};
        const double ts_ns_double = timestamp * 1e9;
        if (ts_ns_double > static_cast<double>(std::numeric_limits<int64_t>::max())) {
            view.ts = std::numeric_limits<int64_t>::max();
        } else if (ts_ns_double < static_cast<double>(std::numeric_limits<int64_t>::min())) {
            view.ts = std::numeric_limits<int64_t>::min();
        } else {
            view.ts = static_cast<int64_t>(ts_ns_double);
        }
        view.motors = motors.data();
        view.motors_len = static_cast<int>(motors.size());

        const int rc = zb_pdcmd_write(&view);
        if (rc < 0) {
            std::cerr << "[SHM2DDS] zb_pdcmd_write failed: " << rc << std::endl;
            return false;
        }

        m_publish_count++;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[SHM2DDS] Exception in publish_once: " << e.what() << std::endl;
        return false;
    }
}

void SharedMemoryToDdsPublisher::destroy()
{
    if (m_sdk_initialized) {
        zb_sdk_shutdown();
        m_sdk_initialized = false;
        std::cout << "[SHM2DDS] zb_sdk shutdown" << std::endl;
    }

    m_shm_client.reset();
    std::cout << "[SHM2DDS] resources destroyed" << std::endl;
}

std::string SharedMemoryToDdsPublisher::resolve_config_path() const
{
    if (const char* env = std::getenv("ZB_SDK_CONFIG"); env != nullptr && env[0] != '\0') {
        return std::string(env);
    }

    const std::vector<std::string> candidates = {
        "./dds/config/config.json",
        "../mujoco_sim_zbsdk/config/config.json",
        "../mujoco_sim_zbsdk/build/install/etc/mujoco_sim_zbsdk/config.json"
    };

    for (const auto& path : candidates) {
        if (file_exists(path)) {
            return path;
        }
    }

    return "./dds/config/config.json";
}

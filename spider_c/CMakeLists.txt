# CMake Minumum Version
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
project(shm_to_dds_publisher)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Specify c++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(/usr/include/eigen3)
include_directories(./include)
# include_directories(../dds)

set(ZBSDK_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mujoco_sim_zbsdk/lib/zbsdk")

# 删除 find_package(qpOASES REQUIRED)
# 改为手动设置 qpOASES 路径和链接
# Set executable
# ADD_EXECUTABLE(${EXE_NAME} src/main.cpp)

# Target link libraries
 
 

# Set directory of the executable
# SET_TARGET_PROPERTIES(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ============ SHM to DDS Publisher ============
# 将 Spider IK 共享内存中的数据通过 DDS 发布给 MuJoCo 仿真
SET(SHM_DDS_EXE_NAME shm_to_dds_publisher)

ADD_EXECUTABLE(${SHM_DDS_EXE_NAME}
                shm_to_dds_main.cpp
                SharedMemoryToDdsPublisher.cpp
                )

TARGET_LINK_LIBRARIES(${SHM_DDS_EXE_NAME}
                        "${ZBSDK_LIB_DIR}/libzb_sdk.a"
                        "${ZBSDK_LIB_DIR}/libHWdds.a"
                        "${ZBSDK_LIB_DIR}/libgreenstone-DCPS.so"
                        pthread
                        util
                        dl
                        rt
                        m
                    )

set_target_properties(${SHM_DDS_EXE_NAME} PROPERTIES
        BUILD_RPATH "${ZBSDK_LIB_DIR}"
)

# ============ zbsdk RobotStateTest / RobotCmdTest ============
add_executable(robotstatetest_zbsdk
        robotstatetest_zbsdk.cpp
)

target_link_libraries(robotstatetest_zbsdk
        "${ZBSDK_LIB_DIR}/libzb_sdk.a"
        "${ZBSDK_LIB_DIR}/libHWdds.a"
        "${ZBSDK_LIB_DIR}/libgreenstone-DCPS.so"
        pthread
        util
        dl
        rt
        m
)

set_target_properties(robotstatetest_zbsdk PROPERTIES
        BUILD_RPATH "${ZBSDK_LIB_DIR}"
)

add_executable(robotcmdtest_zbsdk
        robotcmdtest_zbsdk.cpp
)

target_link_libraries(robotcmdtest_zbsdk
        "${ZBSDK_LIB_DIR}/libzb_sdk.a"
        "${ZBSDK_LIB_DIR}/libHWdds.a"
        "${ZBSDK_LIB_DIR}/libgreenstone-DCPS.so"
        pthread
        util
        dl
        rt
        m
)

set_target_properties(robotcmdtest_zbsdk PROPERTIES
        BUILD_RPATH "${ZBSDK_LIB_DIR}"
)
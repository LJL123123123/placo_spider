import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, line_viz, point_viz, frame_viz, robot_frame_viz
from placo_utils.tf import tf
import pandas as pd
import time
import math
import sys
import termios
import tty
import select
import atexit
from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG
"""
Quadruped robot:
- Standing on three legs (leg1, leg2 and leg4) - hard priority
- Trying to reach targets with leg3 (randomized every 3s) - high priority
- Trying to keep its body - low priority
- Avoiding tilting (CoM is constrained in the support) - hard priority
- Velocity are constrained to 1 rad/s
"""

robot = placo.RobotWrapper("/home/placo_cpg/spider_sldasm/urdf", placo.Flags.ignore_collisions)

for leg in ["Joint1", "Joint2", "Joint3", "Joint4"]:
    robot.set_joint_limits(leg + "-1", -np.pi/2, np.pi/2)
    robot.set_joint_limits(leg + "-3", 0.0, np.pi)

solver = placo.KinematicsSolver(robot)
trajectory_df = pd.read_csv("./trajectory.csv", header=None, names=["time", "x", "y", "z"])

high = [0.0, 0.0, 0.15]

# initialize the robot to a standing pose

T_world_body = robot.get_T_world_frame("base_link")
T_world_leg1 = robot.get_T_world_frame("Link1-6")
T_world_leg2 = robot.get_T_world_frame("Link2-6")
T_world_leg3 = robot.get_T_world_frame("Link3-6")
T_world_leg4 = robot.get_T_world_frame("Link4-6")

# leg1_init = solver.add_frame_task("Link1-6", T_world_leg1)
# leg2_init = solver.add_frame_task("Link2-6", T_world_leg2)
# leg3_init = solver.add_frame_task("Link3-6", T_world_leg3)
# leg4_init = solver.add_frame_task("Link4-6", T_world_leg4)

# leg1_init.configure("Link1-6", "soft", 1.0, 1e2)
# leg2_init.configure("Link2-6", "soft", 1.0, 1e2)
# leg3_init.configure("Link3-6", "soft", 1.0, 1e2)
# leg4_init.configure("Link4-6", "soft", 1.0, 1e2)

body_init_task = solver.add_frame_task("base_link", tf.translation_matrix(high))
body_init_task.configure("base_link", "soft", 1.0, 1e6)
# print("T_world_leg1:\n", T_world_leg1)
# # 提取T_world_leg1的orientation信息（旋转矩阵）
# orientation_leg1 = T_world_leg1[:3, :3]
# print("Orientation (rotation matrix) of leg1:\n", orientation_leg1)

# 提取T_world_leg1的position信息
position_body = T_world_body[:3, 3]
print("Position of body:\n", position_body)
position_leg1 = T_world_leg1[:3, 3]+np.array(high)
# print("Position of leg1:\n", position_leg1)
position_leg2 = T_world_leg2[:3, 3]+np.array(high)
position_leg3 = T_world_leg3[:3, 3]+np.array(high)
position_leg4 = T_world_leg4[:3, 3]+np.array(high)

# 提取position_leg1的前两项
position_body_xy = position_body[:2]
position_leg1_xy = position_leg1[:2]
# print("position_leg1_xy:\n", position_leg1_xy)
position_leg2_xy = position_leg2[:2]
position_leg3_xy = position_leg3[:2]
position_leg4_xy = position_leg4[:2]


leg1 = solver.add_position_task("Link1-6", position_leg1)
leg2 = solver.add_position_task("Link2-6", position_leg2)
leg3 = solver.add_position_task("Link3-6", position_leg3)
leg4 = solver.add_position_task("Link4-6", position_leg4)

leg1.configure("Link1-6", "soft", 1.0)
leg2.configure("Link2-6", "soft", 1.0)
leg3.configure("Link3-6", "soft", 1.0)
leg4.configure("Link4-6", "soft", 1.0)

# Using some steps to initialize the robot
for _ in range(32):
    robot.update_kinematics()
    solver.solve(True)


# Limiting velocities to 1 rad/s
robot.set_velocity_limits(1.)

solver.enable_velocity_limits(True)
viz = robot_viz(robot)

t = 0.0
dt = 0.01
solver.dt = dt

leg1_swim = solver.add_relative_position_task("base_link","Link1-6", T_world_leg1[:3, 3])
leg2_swim = solver.add_relative_position_task("base_link","Link2-6", T_world_leg2[:3, 3])
leg3_swim = solver.add_relative_position_task("base_link","Link3-6", T_world_leg3[:3, 3])
leg4_swim = solver.add_relative_position_task("base_link","Link4-6", T_world_leg4[:3, 3])

leg1_swim.configure("Link1-6", "soft", 0)
leg2_swim.configure("Link2-6", "soft", 0)
leg3_swim.configure("Link3-6", "soft", 0)
leg4_swim.configure("Link4-6", "soft", 0)

base_task = solver.add_position_task("base_link", np.array([0,0, 0.193492]))
base_task.configure("base_link", "soft", 0)

def get_xyz_from_trajectory(t):
    times = pd.to_numeric(trajectory_df["time"], errors='coerce').dropna().values
    x_vals = pd.to_numeric(trajectory_df["x"], errors='coerce').dropna().values
    y_vals = pd.to_numeric(trajectory_df["y"], errors='coerce').dropna().values
    z_vals = pd.to_numeric(trajectory_df["z"], errors='coerce').dropna().values

    if t <= times[0]:
        x, y, z = x_vals[0], y_vals[0], z_vals[0]
    elif t >= times[-1]:
        x, y, z = x_vals[-1], y_vals[-1], z_vals[-1]
    else:
        idx = np.searchsorted(times, t)
        t0, t1 = times[idx - 1], times[idx]
        alpha = (t - t0) / (t1 - t0)
        x = x_vals[idx - 1] + alpha * (x_vals[idx] - x_vals[idx - 1])
        y = y_vals[idx - 1] + alpha * (y_vals[idx] - y_vals[idx - 1])
        z = z_vals[idx - 1] + alpha * (z_vals[idx] - z_vals[idx - 1])
    return np.array([x, y, z])

def sample_target(t):
    return np.random.uniform(np.array([0.1, -0.1, 0.0]), np.array([0.3, 0.1, 0.2]))


target = sample_target(t)
last_sample_t = 0.0


def state_update():
    global position_body, position_leg1, position_leg2, position_leg3, position_leg4
    global position_body_xy, position_leg1_xy, position_leg2_xy, position_leg3_xy, position_leg4_xy
    T_world_body = robot.get_T_world_frame("base_link")
    T_world_leg1 = robot.get_T_world_frame("Link1-6")
    T_world_leg2 = robot.get_T_world_frame("Link2-6")
    T_world_leg3 = robot.get_T_world_frame("Link3-6")
    T_world_leg4 = robot.get_T_world_frame("Link4-6")

    position_body = T_world_body[:3, 3]
    # print("Position of body:\n", position_body)
    position_leg1 = T_world_leg1[:3, 3]
    position_leg2 = T_world_leg2[:3, 3]
    position_leg3 = T_world_leg3[:3, 3]
    position_leg4 = T_world_leg4[:3, 3]

    position_body_xy = position_body[:2]
    position_leg1_xy = position_leg1[:2]
    position_leg2_xy = position_leg2[:2]
    position_leg3_xy = position_leg3[:2]
    position_leg4_xy = position_leg4[:2]

leg1_support_target = np.append(position_leg1_xy, 0.0)
leg2_support_target = np.append(position_leg2_xy, 0.0)
leg3_support_target = np.append(position_leg3_xy, 0.0)
leg4_support_target = np.append(position_leg4_xy, 0.0)
base_target = np.append(position_body_xy, 0.0) + np.array(high)
v_vector=np.array([1.0, 0.0 , 0.0])
# 确保初始方向为单位向量（只影响模长）
norm_v = np.linalg.norm(v_vector)
if norm_v > 0:
    v_vector = v_vector / norm_v
step_length=0.1
# 非阻塞键盘输入设置 (Unix)
try:
    _stdin_fd = sys.stdin.fileno()
    _stdin_orig_settings = termios.tcgetattr(_stdin_fd)
    tty.setcbreak(_stdin_fd)

    def _restore_stdin():
        try:
            termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _stdin_orig_settings)
        except Exception:
            pass

    atexit.register(_restore_stdin)
except Exception:
    _stdin_fd = None
    _stdin_orig_settings = None


def get_key_nonblocking():
    """返回单个按键字符或 None（非阻塞）。仅在支持 fileno/termios 的终端有效。"""
    if _stdin_fd is None:
        return None
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        try:
            return sys.stdin.read(1)
        except Exception:
            return None
    return None


def rotate_vector_z(v, angle_rad):
    """绕 Z 轴旋转 v（仅作用于 x,y 分量）。"""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y = v[0], v[1]
    xr = c * x - s * y
    yr = s * x + c * y
    vec = np.array([xr, yr, v[2]])
    # 旋转后归一化模长为1（仅在非零向量时）
    n = np.linalg.norm(vec)
    if n > 0:
        return vec / n
    return vec


# 键盘控制参数
SPEED_DELTA = 0.1
STEP_DELTA = 0.01         # 每次按 w/s 改变的 step_length
ROT_DEG = 10.0            # 每次按 a/d 旋转角度（度）
ROT_RAD = math.radians(ROT_DEG)
speed = 1.0

def process_key(key):
    """根据按键调整全局 v_vector 和 step_length。"""
    global v_vector, step_length, speed
    if key is None:
        return
    k = key.lower()
    if k == 'w':
        step_length += STEP_DELTA
        print(f"step_length -> {step_length:.3f}")
    elif k == 's':
        step_length -= STEP_DELTA
        print(f"step_length -> {step_length:.3f}")
    elif k == 'a':
        v_vector = rotate_vector_z(v_vector, ROT_RAD)
        print(f"v_vector -> [{v_vector[0]:.3f}, {v_vector[1]:.3f}]")
    elif k == 'd':
        v_vector = rotate_vector_z(v_vector, -ROT_RAD)
        print(f"v_vector -> [{v_vector[0]:.3f}, {v_vector[1]:.3f}]")
    elif k == '+':
        speed += SPEED_DELTA
        print(f"speed -> {speed:.3f}")
    elif k == '-':
        speed -= SPEED_DELTA
        print(f"speed -> {speed:.3f}")
    # 吞掉换行等不可见字符
def target_update(dt,t,flying_leg):
    global target,base_target,leg1_support_target,leg2_support_target,leg3_support_target,leg4_support_target,v
    leg1_support_target = np.append(position_leg1_xy, 0.0)
    leg2_support_target = np.append(position_leg2_xy, 0.0)
    leg3_support_target = np.append(position_leg3_xy, 0.0)
    leg4_support_target = np.append(position_leg4_xy, 0.0)

    match flying_leg:
        case 1:
            leg_target_base = np.array([0.371678, -0.372385, 0.0])
        case 2:
            leg_target_base = np.array([0.372385, 0.371678, 0.0])
        case 3:
            leg_target_base = np.array([-0.371678, 0.372385, 0.0])
        case 4:
            leg_target_base = np.array([-0.361251, -0.360544, 0.0])
    # print("leg_target_base:", leg_target_base)
    #求v_vector与[1,0,0]的夹角
    angle = np.arctan2(v_vector[1], v_vector[0])
    # print("angle:", angle)
    #根据夹角计算旋转矩阵

    target = leg_target_base + np.array([(t%3)/3*step_length*math.cos(angle),(t%3)/3*step_length*math.sin(angle),1.5*get_xyz_from_trajectory(t%3)[2]])
    base_target = np.append(position_body_xy, 0.0) + np.array(high) + dt*v_vector*step_length

set_com_flag = True
def task_update(flying_leg):
    global set_com_flag
    
    # print("target:", id(target))

    base_task.configure("base_link", "soft", 1e6)
    base_task.target_world = base_target
    # print("base_target:", base_target)

    match flying_leg:
        case 1:
            if(set_com_flag):
                polygon = np.array([
                    position_leg3_xy,
                    position_leg2_xy,
                    position_leg4_xy
                ])
                # com = solver.add_com_polygon_constraint(polygon, 0.015)
                # com.configure("com_constraint", "soft", 1.0)
                set_com_flag = False

                leg1.configure("Link1-6", "soft", 0.0)
                leg2.configure("Link2-6", "soft", 1e6)
                leg3.configure("Link3-6", "soft", 1e6)
                leg4.configure("Link4-6", "soft", 1e6)

                leg1_swim.configure("Link1-6", "soft", 1e1)
                leg2_swim.configure("Link2-6", "soft", 0)
                leg3_swim.configure("Link3-6", "soft", 0)
                leg4_swim.configure("Link4-6", "soft", 0)

                leg2.target_world = leg2_support_target
                leg3.target_world = leg3_support_target
                leg4.target_world = leg4_support_target

                support_tasks = [leg3, leg4, leg2]
                for k in range(3):
                    line_from = support_tasks[k].target_world
                    line_to = support_tasks[(k + 1) % 3].target_world
                    line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)


            leg1_swim.target = target - np.array(high)
            # leg1.target_world = target

        case 2:
            if(set_com_flag):
                polygon = np.array([
                position_leg3_xy,
                position_leg1_xy,
                position_leg4_xy
                ])
                # com = solver.add_com_polygon_constraint(polygon, 0.015)
                # com.configure("com_constraint", "soft", 1.0)
                set_com_flag = False

                leg1.configure("Link1-6", "soft", 1e6)
                leg2.configure("Link2-6", "soft", 0.0)
                leg3.configure("Link3-6", "soft", 1e6)
                leg4.configure("Link4-6", "soft", 1e6)

                leg1_swim.configure("Link1-6", "soft", 0)
                leg2_swim.configure("Link2-6", "soft", 1e3)
                leg3_swim.configure("Link3-6", "soft", 0)
                leg4_swim.configure("Link4-6", "soft", 0)

                leg1.target_world = leg1_support_target
                leg3.target_world = leg3_support_target
                leg4.target_world = leg4_support_target

                support_tasks = [leg3, leg4, leg1]
                for k in range(3):
                    line_from = support_tasks[k].target_world
                    line_to = support_tasks[(k + 1) % 3].target_world
                    line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

            leg2_swim.target = target - np.array(high)
            # leg2.target_world = target

        case 3:
            if(set_com_flag):
                polygon = np.array([
                position_leg4_xy,
                position_leg1_xy,
                position_leg2_xy
                ])
                # com = solver.add_com_polygon_constraint(polygon, 0.015)
                # com.configure("com_constraint", "soft", 1.0)
                set_com_flag = False

                leg1.configure("Link1-6", "soft", 1e6)
                leg2.configure("Link2-6", "soft", 1e6)
                leg3.configure("Link3-6", "soft", 0.)
                leg4.configure("Link4-6", "soft", 1e6)

                leg1_swim.configure("Link1-6", "soft", 0)
                leg2_swim.configure("Link2-6", "soft", 0)
                leg3_swim.configure("Link3-6", "soft", 1e1)
                leg4_swim.configure("Link4-6", "soft", 0)

                leg1.target_world = leg1_support_target
                leg2.target_world = leg2_support_target
                leg4.target_world = leg4_support_target

                support_tasks = [leg2, leg4, leg1]
                for k in range(3):
                    line_from = support_tasks[k].target_world
                    line_to = support_tasks[(k + 1) % 3].target_world
                    line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

            leg3_swim.target = target - np.array(high)
            # leg3.target_world = target
            # print("leg3.target_world:", leg3.target_world)

        case 4:
            if(set_com_flag):
                polygon = np.array([
                position_leg3_xy,
                position_leg1_xy,
                position_leg2_xy
                ])
                # com = solver.add_com_polygon_constraint(polygon, 0.015)
                # com.configure("com_constraint", "soft", 1.0)
                set_com_flag = False

                leg1.configure("Link1-6", "soft", 1e6)
                leg2.configure("Link2-6", "soft", 1e6)
                leg3.configure("Link3-6", "soft", 1e6)
                leg4.configure("Link4-6", "soft", 0.)

                leg1_swim.configure("Link1-6", "soft", 0)
                leg2_swim.configure("Link2-6", "soft", 0)
                leg3_swim.configure("Link3-6", "soft", 0)
                leg4_swim.configure("Link4-6", "soft", 1e1)

                leg1.target_world = leg1_support_target
                leg2.target_world = leg2_support_target
                leg3.target_world = leg3_support_target

                support_tasks = [leg3, leg2, leg1]
                for k in range(3):
                    line_from = support_tasks[k].target_world
                    line_to = support_tasks[(k + 1) % 3].target_world
                    line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

            leg4_swim.target = target - np.array(high)
            # leg4.target_world = target
            
flying_leg = 1
def gait(t):
    global flying_leg,set_com_flag
    if t % 12 < 3:
        flying_leg = 1
    elif t % 12 < 6:
        flying_leg = 2
    elif t % 12 < 9:
        flying_leg = 3
    else:
        flying_leg = 4
    if t%3<0.05:
        set_com_flag = True
        # print("set_com_flag:", set_com_flag)
@schedule(interval=dt)
def loop():
    global t, target, last_sample_t,flying_leg
    t += speed*dt
    # 非阻塞查询键盘输入并处理
    try:
        key = get_key_nonblocking()
        if key:
            process_key(key)
    except Exception:
        # 若终端不支持非阻塞读取，忽略
        pass

    gait(t)
    # Updating target every 3 seconds
    if last_sample_t + 0.1 < t:
        last_sample_t = t
        # target = np.array([-0.371678, 0.372385, 0.0]) + 5*get_xyz_from_trajectory(t%3)
        state_update()
        target_update(dt,t,flying_leg)
        # print("target1:", id(target))
        # print("time:", t)

    # Showing target
    point_viz("target", target, color=0x00FF00)
    task_update(flying_leg)

    # Showing the center of mass (on the ground)
    com_world = robot.com_world()
    com_world[2] = 0.0
    point_viz("com", com_world, color=0xFF0000)

    # Showing body frame and its target
    robot_frame_viz(robot, "base_link")
    frame_viz("body_target", body_init_task.T_world_frame, opacity=.25)

    solver.solve(True)
    robot.update_kinematics()

    viz.display(robot.state.q)


run_loop()

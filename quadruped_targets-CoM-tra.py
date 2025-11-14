import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, line_viz, point_viz, frame_viz, robot_frame_viz
from placo_utils.tf import tf
import pandas as pd


"""
Quadruped robot:
- Standing on three legs (leg1, leg2 and leg4) - hard priority
- Trying to reach targets with leg3 (randomized every 3s) - high priority
- Trying to keep its body - low priority
- Avoiding tilting (CoM is constrained in the support) - hard priority
- Velocity are constrained to 1 rad/s
"""

robot = placo.RobotWrapper("/home/placo-examples/models/quadruped", placo.Flags.ignore_collisions)

for leg in ["leg1", "leg2", "leg3", "leg4"]:
    robot.set_joint_limits(leg + "_a", -np.pi/2, np.pi/2)
    robot.set_joint_limits(leg + "_c", 0.0, np.pi)

solver = placo.KinematicsSolver(robot)

leg1 = solver.add_position_task("leg1", np.array([-0.15, 0.0, 0.0]))
leg2 = solver.add_position_task("leg2", np.array([0.02, -0.15, 0.0]))
leg3 = solver.add_position_task("leg3", np.array([0.15, 0.0, 0.0]))
leg4 = solver.add_position_task("leg4", np.array([0.02, 0.15, 0.0]))

# body_task = solver.add_frame_task("body", tf.translation_matrix([0.0, 0.0, 0.05]))
body_task = solver.add_position_task("body", np.array([0,0, 0.05]))
body_task.configure("body"
                    , "soft"                    
                    , 1e3
                    # , 1.0
                    # ,"hard"
                    )

# Using some steps to initialize the robot
for _ in range(32):
    robot.update_kinematics()
    solver.solve(True)

# Support legs should not move (hard constraint)
# leg1.configure("leg1", "hard")
# leg2.configure("leg2", "hard")
# leg4.configure("leg4", "hard")
# support_tasks = [leg4, leg2]
# for k in range(2):
#     line_from = support_tasks[k].target_world
#     line_to = support_tasks[(k + 1) % 2].target_world
#     line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

# The body should remain (soft constraint)
# body_task.configure("body", "soft", 1.0, 1.0)

# The leg3 should reach its targets (soft constraint, higher priority than body)
# leg3.configure("leg3", "soft", 1e3)
# leg1.configure("leg1", "soft", 1e3)

# Adding a polygon constraint to avoid robot tilting
# polygon = np.array([
#     [-0.15, 0.],
#     [0.02, 0.15],
#     [0.02, -0.15]
# ])
# com = solver.add_com_polygon_constraint(polygon, 0.015)
# com.configure("com_constraint", "hard")

# Limiting velocities to 1 rad/s
robot.set_velocity_limits(1.)

solver.enable_velocity_limits(True)
viz = robot_viz(robot)

t = 0.0
dt = 0.01
solver.dt = dt

# 读取CSV文件
trajectory_df = pd.read_csv("./trajectory.csv", header=None, names=["time", "x", "y", "z"])

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

def sample_target(t, x, y, z):
    # return np.random.uniform(np.array([0.1, -0.1, 0.0]), np.array([0.3, 0.1, 0.2]))
    return np.array([x, y, z])

x, y, z = 0, 0, 0
target = sample_target(t, x, y, z)
target1 = sample_target(t, x, y, z)
last_sample_t = 0.0

def get_body_line_target(t, start, end, duration):
    # t: 当前时间
    # start, end: 起点和终点坐标
    # duration: 运动总时长
    alpha = min(max(t / duration, 0), 1)
    return start + alpha * (end - start)


@schedule(interval=dt)
def loop():
    global t, target, last_sample_t, target1
    t += dt

    # Updating target every 3 seconds
    if last_sample_t + 0.1 < t:
        last_sample_t = t
        if t % 6 < 3:
            leg2.configure("leg2"
                           , "soft"
                           , 1e6
                        #    , "hard"
                           )
            leg4.configure("leg4"
                           , "soft"
                           , 1e6
                        #    , "hard"
                           )

            support_tasks = [leg4, leg2]
            for k in range(2):
                line_from = support_tasks[k].target_world
                line_to = support_tasks[(k + 1) % 2].target_world
                line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

            leg3.configure("leg3"
                           , "soft"
                           , 1e3
                           )
            leg1.configure("leg1"
                           , "soft"
                           , 1e3
                           )

            # polygon = np.array([
            #     [-0.15, 0.],
            #     [0.02, 0.15],
            #     [0.02, -0.15]
            # ])
            # com = solver.add_com_polygon_constraint(polygon, 0.015)
            # com.configure("com_constraint", "hard")

            x, y, z = get_xyz_from_trajectory(t % 3) + 1.0 * int(t/6) * np.array([0.05, 0, 0]) + np.array([0.15, 0.0, 0.0])
            target = sample_target(t, x, y, z)

            x1, y1, z1 = get_xyz_from_trajectory(t % 3) + 1.0 * int(t/6) * np.array([0.05, 0, 0]) + np.array([-0.15, 0.0, 0.0])
            target1 = sample_target(t, x1, y1, z1)

            # Showing target
            point_viz("target", target, color=0x00FF00)
            leg3.target_world = target
            point_viz("target1", target1, color=0x00FF01)
            leg1.target_world = target1
        else:
            leg1.configure("leg1"
                           , "soft"
                           , 1e6
                        # , "hard"
                           )
            leg3.configure("leg3"
                           , "soft"
                           , 1e6
                        # , "hard"
                           )

            support_tasks = [leg3, leg1]
            for k in range(2):
                line_from = support_tasks[k].target_world
                line_to = support_tasks[(k + 1) % 2].target_world
                line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

            leg4.configure("leg4"
                           , "soft"
                           , 1e3
                           )
            leg2.configure("leg2"
                           , "soft"
                           , 1e3
                           )

            x, y, z = get_xyz_from_trajectory(t % 6-3) + 1.0 * int(t/6) * np.array([0.05, 0, 0]) + np.array([0.02, 0.15, 0.0])
            target = sample_target(t, x, y, z)

            x1, y1, z1 = get_xyz_from_trajectory(t % 6-3) + 1.0 * int(t/6) * np.array([0.05, 0, 0]) + np.array([0.02, -0.15, 0.0])
            target1 = sample_target(t, x1, y1, z1)

            # Showing target
            point_viz("target", target, color=0x00FF00)
            leg4.target_world = target
            point_viz("target1", target1, color=0x00FF01)
            leg2.target_world = target1

    # 设置body_task沿直线运动
    start = np.array([0.0, 0.0, 0.05])
    end = np.array([0.2*10., 0.0, 0.05])
    duration = 24.0*10.  # 5秒完成一次运动
    body_target = get_body_line_target(t % duration, start, end, duration)
    body_task.target_world = body_target
    ################################################################################

    # Showing the center of mass (on the ground)
    com_world = robot.com_world()
    com_world[2] = 0.0
    point_viz("com", com_world, color=0xFF0000)

    # # Showing body frame and its target
    # robot_frame_viz(robot, "body")
    # frame_viz("body_target", body_task.T_world_frame, opacity=.25)

    solver.solve(True)
    robot.update_kinematics()

    viz.display(robot.state.q)


run_loop()

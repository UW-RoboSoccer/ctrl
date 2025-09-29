'''
This script is used to fine tune different gait parameters
'''

import numpy as np
import eigenpy, pinocchio 
import placo
import time

DT = 0.005
REPLAN_DT = 0.1

model_path = "./robot/urdf/robot_mod.urdf"

robot = placo.HumanoidRobot(model_path, 1) 

params = placo.HumanoidParameters()

# Timing parameters
params.single_support_duration = 0.38
params.single_support_timesteps = 10
params.double_support_ratio = 0.20
params.startend_double_support_ratio = 1.0
params.planned_timesteps = 48

# Posture parameters
params.walk_com_height = 0.32
params.walk_foot_height = 0.04
params.walk_trunk_pitch = 0.15
params.walk_foot_rise_ratio = 0.2

# Feet parameters
params.foot_length = 0.2
params.foot_width = 0.1

T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
params.feet_spacing = abs(T_world_left[1,3] - T_world_right[1,3]) or 0.18

params.zmp_margin = 0.01
params.foot_zmp_target_x = 0.0
params.foot_zmp_target_y = 0.0

# Limit parameters
params.walk_max_dtheta = 1.0
params.walk_max_dy = 0.04
params.walk_max_dx_forward = 0.08
params.walk_max_dx_backward = 0.03

solver = placo.KinematicsSolver(robot)
solver.enable_velocity_limits(True)
solver.dt = DT

tasks = placo.WalkTasks()
tasks.initialize_tasks(solver, robot)

elbow = -50 * np.pi / 180
shoulder_roll = 0 * np.pi / 180
shoulder_pitch = 20 * np.pi / 180
joints_task = solver.add_joints_task()
joints_task.set_joints(
    {
        "left_shoulder_roll": shoulder_roll,
        "left_shoulder_pitch": shoulder_pitch,
        "left_elbow": elbow,
        "right_shoulder_roll": -shoulder_roll,
        "right_shoulder_pitch": shoulder_pitch,
        "right_elbow": elbow,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
    }
)
joints_task.configure("joints", "soft", 1.0)

print("Reaching initial position")
tasks.reach_initial_pose(
    np.eye(4),
    params.feet_spacing,
    params.walk_com_height,
    params.walk_trunk_pitch,
)
print("Initial position reached")

footstep_planner = placo.FootstepsPlannerRepetitive(params)
d_x = 0.1
d_y = 0.0
d_theta = 0.0
np_steps = 10

footstep_planner.configure(d_x, d_y, d_theta, np_steps)


footsteps = footstep_planner.plan(placo.HumanoidRobot_Side.left, T_world_left, T_world_right)

supports = placo.FootstepsPlanner.make_supports(footsteps, 0.0, True, params.has_double_support(), True)

# —— 轨迹生成后 ——
walk = placo.WalkPatternGenerator(robot, params)
traj = walk.plan(supports, robot.com_world(), 0.0)

# 复位到中立（可选）
if hasattr(robot, "reset"):
    robot.reset()

# 关节名列表（不包含自由基座）
names = list(robot.joint_names())
# 当前关节位置字典
qmap = {n: robot.get_joint(n) for n in names}

T_END = getattr(traj, "duration", lambda: 6.0)()
t = 0.0
start_t = time.time()

while t < T_END:
    tasks.update_tasks_from_trajectory(traj, t)
    robot.update_kinematics()

    ret = None
    try:
        ret = solver.solve(False)   # 有的版本要 bool 参数
    except TypeError:
        ret = solver.solve(True)

    import numpy as np
    if isinstance(ret, (list, tuple, np.ndarray)):
        qd = np.asarray(ret)
    elif hasattr(solver, "qd"):
        qd = np.asarray(solver.qd)
    elif hasattr(solver, "get_qd"):
        qd = np.asarray(solver.get_qd())
    else:
        raise RuntimeError("Cannot retrieve qd from solver")

    # --- 对齐：去掉自由基座项 ---
    offset = len(qd) - len(names)   # 预期是 6
    if offset not in (6, 7, 0):
        print(f"WARN: unexpected offset {offset}")
    qd_j = qd[offset:]              # 取后 20 个

    # 欧拉积分 + 写回
    for n, dq in zip(names, qd_j):
        qmap[n] += float(dq) * DT
        robot.set_joint(n, qmap[n])


    t += DT
    while time.time() < start_t + t:
        time.sleep(0.001)

print(f"gait_playground: done (T={T_END:.2f}s)")

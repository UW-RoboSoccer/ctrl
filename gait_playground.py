# -*- coding: utf-8 -*-
"""
Placo Walk: +Y Straight Walk (Stable Alternation + Dynamic Knee Control)
- Foot Orientation: mask="yz"/local + AxisAlign Z->worldZ (Soft)
- Footsteps: Strict Y increase, fixed X (Version B post-processing)
- Joint Limits: Enabled
- Dynamic Knee: Auto-sign + Phase-driven (Greater flexion during swing)
"""

import time
import numpy as np
import pinocchio
import placo
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

# ------------------ Robot setup & visualization ------------------
MODEL_PATH = "./robot/urdf/robot_mod.urdf"
MESH_DIR   = "./robot/urdf"

robot = placo.HumanoidRobot(MODEL_PATH, 1)
model, cm, vm = pinocchio.buildModelsFromUrdf(MODEL_PATH, MESH_DIR, pinocchio.JointModelFreeFlyer())

viz = MeshcatVisualizer(model, cm, vm)
viz.initViewer(loadModel=True)
viz.viewer.delete()
viz.loadViewerModel("robot")

names = list(robot.joint_names())
print("Joint names:", names)

print("\n[Joint order check: name -> offset]")
for n in robot.joint_names():
    try:
        print(f"{n:>24s}  q_off={robot.get_joint_offset(n)}  v_off={robot.get_joint_v_offset(n)}")
    except Exception:
        pass

# ------------------ Gait parameters ------------------
params = placo.HumanoidParameters()

# Timing
params.single_support_duration = 0.40
params.single_support_timesteps = 12
params.double_support_ratio = 0.0
params.startend_double_support_ratio = 1.5
params.planned_timesteps = 48

# Posture
params.walk_com_height   = 0.32
params.walk_foot_height  = 0.05
params.walk_trunk_pitch  = 0.12
params.walk_foot_rise_ratio = 0.2

# Feet & ZMP
params.foot_length = 0.20
params.foot_width  = 0.10
params.zmp_margin  = 0.01

# Step limits
params.walk_max_dtheta      = 1.0
params.walk_max_dy          = 0.06
params.walk_max_dx_forward  = 0.08
params.walk_max_dx_backward = 0.03

# ------------------ IK solver and tasks setup ------------------
solver = placo.KinematicsSolver(robot)
solver.enable_velocity_limits(True)
robot.set_velocity_limits(10.0)
solver.enable_joint_limits(True)     # Prevent knee inversion
DT = 0.01
solver.dt = DT

tasks = placo.WalkTasks()
if hasattr(params, "trunk_mode"):
    tasks.trunk_mode = params.trunk_mode
if hasattr(tasks, "com_x"):
    tasks.com_x = 0.0
tasks.initialize_tasks(solver, robot)

# Foot orientation: local 'yz'
try:
    tasks.left_foot_task.orientation().mask.set_axises("yz", "local")
    tasks.right_foot_task.orientation().mask.set_axises("yz", "local")
    print("[OK] foot orientation mask: local 'yz'")
except Exception:
    pass

# AxisAlign: Foot Z -> World Z (Soft)
try:
    z_align_L = solver.add_axisalign_task("left_foot",  np.array([0., 0., 1.]), np.array([0., 0., 1.]))
    z_align_L.configure("L_foot_upright", "soft", 2.0)
    z_align_R = solver.add_axisalign_task("right_foot", np.array([0., 0., 1.]), np.array([0., 0., 1.]))
    z_align_R.configure("R_foot_upright", "soft", 2.0)
    print("[OK] AxisAlign: foot-Z -> world-Z (soft w=2.0)")
except Exception as e:
    print("[WARN] AxisAlignTask not available:", e)

# Lightweight Frontal Symmetry
sym = solver.add_joints_task()
sym.set_joints({
    "left_ankle_roll":  0.0,
    "right_ankle_roll": 0.0,
    "left_hip_roll":    0.0,
    "right_hip_roll":   0.0,
})
sym.configure("frontal_symmetry", "soft", 0.7)


# ------------------ Upper Body Configuration ------------------
# Elbows: Forward flexion (signs differ due to axis definition)
ELBOW_FLEX = 15 * np.pi / 180

LEFT_ELBOW_Q  = -ELBOW_FLEX        # Left: negative = forward
RIGHT_ELBOW_Q =  ELBOW_FLEX        # Right: positive = forward

# Shoulders
SHOULDER_ROLL = 10 * np.pi / 180   # Slight abduction
INIT_SHOULDER_PITCH = 0.0          # Vertical

# Arm Swing Amplitude (Joint space)
ARM_SWING = 0.4

joints_task = solver.add_joints_task()
joints_task.set_joints({
    "left_shoulder_roll":   SHOULDER_ROLL,
    "right_shoulder_roll": -SHOULDER_ROLL,
    "left_shoulder_pitch":  INIT_SHOULDER_PITCH,
    "right_shoulder_pitch": INIT_SHOULDER_PITCH,
    "left_elbow":           LEFT_ELBOW_Q,
    "right_elbow":          RIGHT_ELBOW_Q,
    "head_yaw":             0.0,
    "head_pitch":           0.0,
})
joints_task.configure("joints", "soft", 1.0)


# ------------------ Initial pose (double support) ------------------
print("Reaching initial position...")
tasks.reach_initial_pose(np.eye(4), 0.18, params.walk_com_height, params.walk_trunk_pitch)
robot.update_kinematics()
print("Initial position reached.")

# Set feet spacing based on current Y-difference
T_L0 = placo.flatten_on_floor(robot.get_T_world_left())
T_R0 = placo.flatten_on_floor(robot.get_T_world_right())
feet_spacing_y = float(abs(T_L0[1, 3] - T_R0[1, 3]))
if feet_spacing_y < 0.10:
    print(f"[WARN] feet_spacing_y {feet_spacing_y:.3f} < 0.10, fallback to 0.18")
    feet_spacing_y = 0.18
params.feet_spacing = feet_spacing_y
print(f"[FIX] feet_spacing set by Y-diff = {params.feet_spacing:.3f} m")

# ------------------ Footstep planning ------------------
footstep_planner = placo.FootstepsPlannerRepetitive(params)

# Configuration: Walk along +Y (d_x=0)
d_x, d_y, d_theta = 0.00, -0.06, 0.0
num_steps = 10
footstep_planner.configure(d_x, d_y, d_theta, num_steps)

T_world_left  = placo.flatten_on_floor(robot.get_T_world_left())
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())

footsteps = footstep_planner.plan(placo.HumanoidRobot_Side.left, T_world_left, T_world_right)

# Post-processing (Version B): Enforce strict Y increase, lock X
for i in range(1, len(footsteps)):
    fs, prev = footsteps[i], footsteps[i-1]
    if fs.side != prev.side:
        new_y = prev.frame[1, 3] + d_y
        new_x = prev.frame[0, 3]
        fs.set_frame_xy(new_x, new_y)

# Generate supports
supports = placo.FootstepsPlanner.make_supports(
    footsteps, 0.0, True, params.has_double_support(), True
)

# ------------------ Gait pattern generation ------------------
walk = placo.WalkPatternGenerator(robot, params)

# Initialize CoM at the midpoint of feet
lf = robot.get_T_world_left()[:3, 3]
rf = robot.get_T_world_right()[:3, 3]
com0 = 0.5 * (lf + rf)
robot.update_kinematics()

T_L = placo.flatten_on_floor(robot.get_T_world_left())
T_R = placo.flatten_on_floor(robot.get_T_world_right())
print("Y_L, Y_R =", T_L[1,3], T_R[1,3], "ΔY =", abs(T_L[1,3]-T_R[1,3]))

trajectory = walk.plan(supports, com0, 0.0)

# ------------------ Dynamic Knee Control ------------------
# 1. Identify knee joints
left_knees, right_knees = [], []
for n in names:
    ln = n.lower()
    if "knee" in ln:
        if ("left" in ln) or ("l_" in ln):
            left_knees.append(n)
        elif ("right" in ln) or ("r_" in ln):
            right_knees.append(n)

# 2. Determine initial angles and bending signs
knee_init = {}
knee_sign = {}

# Left Knees: Maintain current bending direction
for n in left_knees:
    a0 = robot.get_joint(n)
    knee_init[n] = a0
    if abs(a0) < 1e-3:
        knee_sign[n] = -1.0
    else:
        knee_sign[n] = -np.sign(a0)

# Right Knees: Invert direction logic for symmetry
for n in right_knees:
    a0 = robot.get_joint(n)
    knee_init[n] = a0
    if abs(a0) < 1e-3:
        knee_sign[n] = -1.0
    else:
        knee_sign[n] = -np.sign(a0)

print("[knee_init] (per-knee initial angle):", knee_init)
print("[knee_sign] (per-knee sign by side):", knee_sign)

# 3. Setup soft task for knee dynamics
knee_dyn = None
if left_knees or right_knees:
    knee_dyn = solver.add_joints_task()
    knee_dyn.configure("knee_dyn", "soft", 0.6)

# 4. Helper: Safe support side detection
def support_label_at(tval):
    if trajectory.support_is_both(tval):
        return "both"
    side = trajectory.support_side(tval)
    return "left" if side == placo.HumanoidRobot_Side.left else "right"

last_support = support_label_at(0.0)
phase_time = 0.0
SS_T = params.single_support_duration

# Knee Profile: Sine wave peaking in middle of phase
def knee_profile(s, k_max):
    return k_max * np.sin(np.pi * max(0.0, min(1.0, s)))

K_SWING  = 0.28   # Swing leg flexion (~16 deg)
K_STANCE = 0.10   # Stance leg flexion (~6 deg)


# ------------------ Main Loop: Simulation ------------------
initial_ds = params.startend_double_support_duration()
final_ds   = params.startend_double_support_duration()
T_END = (initial_ds + final_ds) + num_steps * params.single_support_duration

t = 0.0
start_time = time.time()
print(f"Start walking for {T_END:.2f} seconds.")

while t < T_END + 1e-6:
    tasks.update_tasks_from_trajectory(trajectory, t)

    # ------- Calculate support phase -------
    support = support_label_at(t)

    # Reset phase time on support switch
    if support != last_support:
        phase_time = 0.0
        last_support = support

    # Phase 's': [0, 1] during Single Support, 0 during Double Support
    if support == "both":
        s = 0.0
    else:
        s = min(max(phase_time / max(SS_T, 1e-3), 0.0), 1.0)

    # ================= Dynamic Knee Control =================
    if knee_dyn is not None:
        swing_left  = (support == "right")  # Right stance -> Left swing
        swing_right = (support == "left")   # Left stance -> Right swing

        knee_targets = {}
        
        # Calculate targets based on Swing vs Stance magnitude
        if left_knees:
            k = knee_profile(s, K_SWING if swing_left else K_STANCE)
            for jn in left_knees:
                base  = knee_init.get(jn, 0.0)
                delta = knee_sign.get(jn, 1.0) * k
                knee_targets[jn] = base + delta

        if right_knees:
            k = knee_profile(s, K_SWING if swing_right else K_STANCE)
            for jn in right_knees:
                base  = knee_init.get(jn, 0.0)
                delta = knee_sign.get(jn, 1.0) * k
                knee_targets[jn] = base + delta

        if knee_targets:
            knee_dyn.set_joints(knee_targets)

    # ================= Arm Swing (Anti-phase) =================
    # Arm phase: 0 -> max -> 0
    if support == "both":
        arm_phase = 0.0
    else:
        arm_phase = ARM_SWING * np.sin(np.pi * s)

    # Define swing direction based on support side
    if support == "left":
        # Left Stance (Right leg fwd) -> Left arm fwd, Right arm back
        left_shoulder_pitch  = INIT_SHOULDER_PITCH + arm_phase
        right_shoulder_pitch = INIT_SHOULDER_PITCH + arm_phase # Note: Axis signs may require tuning (+/-)
    elif support == "right":
        # Right Stance (Left leg fwd) -> Right arm fwd, Left arm back
        left_shoulder_pitch  = INIT_SHOULDER_PITCH - arm_phase
        right_shoulder_pitch = INIT_SHOULDER_PITCH - arm_phase
    else:
        left_shoulder_pitch  = INIT_SHOULDER_PITCH
        right_shoulder_pitch = INIT_SHOULDER_PITCH

    joints_task.set_joints({
        "left_shoulder_roll":   SHOULDER_ROLL,
        "right_shoulder_roll": -SHOULDER_ROLL,
        "left_shoulder_pitch":  left_shoulder_pitch,
        "right_shoulder_pitch": right_shoulder_pitch,
        "left_elbow":           LEFT_ELBOW_Q,
        "right_elbow":          RIGHT_ELBOW_Q,
        "head_yaw":             0.0,
        "head_pitch":           0.0,
    })

    # Solve IK
    solver.solve(True)
    robot.update_kinematics()

    phase_time += DT

    # Update Meshcat
    q_pin = pinocchio.neutral(model)
    for name in names:
        try:
            jid = model.getJointId(name)
            if jid > 0:
                q_pin[model.idx_qs[jid]] = robot.get_joint(name)
        except Exception:
            pass
    try:
        T_base = robot.get_T_world_fbase()
        quat = pinocchio.Quaternion(T_base[:3, :3])
        q_pin[0:3] = T_base[:3, 3]
        q_pin[3:7] = [quat.x(), quat.y(), quat.z(), quat.w()]
    except Exception:
        pass
    viz.display(q_pin)

    # Print Status (Every 0.5s)
    if (t % 0.5) < DT:
        phase = "DS" if trajectory.support_is_both(t) else support_label_at(t)
        try:
            pos_l = robot.get_T_world_left()[:3, 3]
            pos_r = robot.get_T_world_right()[:3, 3]
        except Exception:
            pos_l = pos_r = np.zeros(3)
        com = robot.com_world()
        print(f"t={t:4.2f}s | phase={phase} | L={pos_l.round(3)} | R={pos_r.round(3)} | COM={com.round(3)}")

    t += DT
    elapsed = time.time() - start_time
    if elapsed < t:
        time.sleep(t - elapsed)

print("✅ Walking gait generation with Placo completed.")
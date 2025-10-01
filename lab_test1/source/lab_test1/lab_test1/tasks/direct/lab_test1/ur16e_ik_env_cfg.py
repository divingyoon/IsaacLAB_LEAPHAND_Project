# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from lab_test1.robots.ur16e import robot

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg as CtrlCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg





@configclass
class Ur16eIKEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6
    observation_space = 18
    state_space = 0

    ctrl: CtrlCfg = CtrlCfg(
        command_type="pose",          # "position" 또는 "pose"
        ik_method="dls",              # "pinv" | "svd" | "trans" | "dls"
        ik_params={"lambda_val": 0.01},  # 방법별 파라미터(옵션). dls면 lambda_val 사용
        use_relative_mode=False       # 필요시 True
    )
    # simulation
    # sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

        # scene


    # custom parameters/scales
    # - action scale
    action_scale = 7.5
    dof_vel_scale = 0.1
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_action = 0.01
    rew_scale_dist = 2.5
    rew_scale_rot = 0.1

        # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )
        # robot(s)
    robot_cfg: ArticulationCfg = robot.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.0,
        replicate_physics=True,
        
    )
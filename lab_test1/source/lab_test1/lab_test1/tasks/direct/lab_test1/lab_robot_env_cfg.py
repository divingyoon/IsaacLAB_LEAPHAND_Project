# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from lab_test1.robots.lab_robot import robot

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


from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


@configclass
class LabRobotEnvCfg(DirectRLEnvCfg):
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



    # custom parameters/scales
    # - action scale
    action_scale = 7.5
    dof_vel_scale = 0.01
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_action = 0.01
    rew_scale_dist = 2.5
    rew_scale_rot = 0.3

        # goal object
#    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
#        prim_path="/Visuals/goal_marker",
#        markers={
#            "goal": sim_utils.UsdFileCfg(
#                usd_path=f"/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/cup.usd",
#                scale=(1.0, 1.0, 1.0),
#            )
#        },
#    )


#    goal_object_cfg: RigidObjectCfg = RigidObjectCfg(
#        prim_path="/World/envs/env_.*/goal_cup",         # (여러 env면 {ENV_REGEX_NS}/cup 사용)
#        spawn=sim_utils.UsdFileCfg(
#            usd_path="/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/cup.usd",
#            scale=(1.0, 1.0, 1.0),
#
#            # 물리 속성 (질량, 중력, 속도 제한 등)
#            rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                disable_gravity=True,       # 중력 받게
#                kinematic_enabled=False,     # 동적으로 움직이는 물체
#                max_linear_velocity=100.0,
#                max_angular_velocity=100.0,
#                linear_damping=0.01,
#                angular_damping=0.01,
#                enable_gyroscopic_forces=True,
#            ),
#
#            # 충돌(콜리전) 속성
#            collision_props=sim_utils.CollisionPropertiesCfg(
#                collision_enabled=True,
#                contact_offset=0.01,
#                rest_offset=0.0,
#                # USD에 콜리전이 없으면 아래처럼 근사 생성 (옵션)
#                # approximation_shape="convexDecomposition",
#                # decompose_mesh=True,
#            ),
#
#
#        ),
#
#
#    )   


    CUP_USD = "/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/cup.usd"
    
    goal_object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/goal_cup",   # 루트(ArticulationRoot)
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUP_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,     # ★ contact reporter 켜짐
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,         # 중력 받게
                kinematic_enabled=False,       # 물리반응 원하면 False
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                linear_damping=0.01,
                angular_damping=0.01,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            
            joint_pos={},          # ★ 조인트 없음 → 빈 dict
            joint_vel={},          # ★ 조인트 없음 → 빈 dict
        ),
        actuators={},   # ★ 추가: 필수
    )




    contact_forces=ContactSensorCfg(
        prim_path="/World/envs/env_.*/goal_cup/left_cup/contact_sensor",
        
    )
    move_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/beads",         # (여러 env면 {ENV_REGEX_NS}/cup 사용)
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/bead.usd",
            scale=(1.0, 1.0, 1.0),

            # 물리 속성 (질량, 중력, 속도 제한 등)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,       # 중력 받게
                kinematic_enabled=False,     # 동적으로 움직이는 물체
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                linear_damping=0.01,
                angular_damping=0.01,
                enable_gyroscopic_forces=True,
            ),

            # 충돌(콜리전) 속성
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
                # USD에 콜리전이 없으면 아래처럼 근사 생성 (옵션)
                # approximation_shape="convexDecomposition",
                # decompose_mesh=True,
            ),


        ),


    )   

        # robot(s)
    robot_cfg: ArticulationCfg = robot.replace(prim_path="/World/envs/env_.*/Robot")
        # scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.0,
        replicate_physics=True,
        
    )




    show_eef_axes: bool = False
    show_goal_axes: bool = False

    # 축 마커 CFG (FRAME_MARKER_CFG를 복사해서 사용)
    eef_axes_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
    goal_axes_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()

#    def __post_init__(self):
#        #super().__post_init__()
#
#        # EEF 축
#        self.eef_axes_cfg = self.eef_axes_cfg.copy()
#        self.eef_axes_cfg.prim_path = "/Visuals/EEFAxes"
#        self.eef_axes_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
#        self.eef_axes_cfg.markers["frame"].count = self.scene.num_envs  # env 개수만큼 인스턴스
#
#        # Goal 축
#        self.goal_axes_cfg = self.goal_axes_cfg.copy()
#        self.goal_axes_cfg.prim_path = "/Visuals/GoalAxes"
#        self.goal_axes_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
#        self.goal_axes_cfg.markers["frame"].count = self.scene.num_envs
    
    #eef_axes_cfg = eef_axes_cfg.copy()
    eef_axes_cfg.prim_path = "/Visuals/EEFAxes"
    eef_axes_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    #eef_axes_cfg.markers["frame"].count = scene.num_envs 
    #goal_axes_cfg = goal_axes_cfg.copy()
    goal_axes_cfg.prim_path = "/Visuals/GoalAxes"
    goal_axes_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    #goal_axes_cfg.markers["frame"].count = scene.num_envs
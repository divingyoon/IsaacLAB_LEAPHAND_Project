# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .lab_robot_env_cfg import LabRobotEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul, quat_conjugate, euler_xyz_from_quat
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from lab_test1.utils import rotationMatrixToQuaternion1

import yaml, numpy as np
import os, yaml

from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

class LabRobotEnv(DirectRLEnv):
    cfg: LabRobotEnvCfg

    def __init__(self, cfg: LabRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # init data

        self._init_tensors()
        self._compute_intermediate_values()


        # initialize goal marker
        #self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        #self.goal_cup = RigidObject(self.cfg.goal_object_cfg)


        if self.cfg.show_goal_axes:
            self.goal_axes = VisualizationMarkers(self.cfg.goal_axes_cfg)
        if self.cfg.show_eef_axes:
            self.eef_axes = VisualizationMarkers(self.cfg.eef_axes_cfg)

        #print(self._robot.joint_names)
        #print(self._robot.num_joints)

    def _init_tensors(self):

        """Initialize tensors once."""

        self.arm_ids = torch.arange(0, 6, device=self.device, dtype=torch.long)
        #init이 실행되면서 _scene이 실행되고 DirectRLEnv 안의 property에서 num_envs를 반환함, numjoint는 usd로 반환
        self.piper_joint_pos = torch.zeros(self.num_envs, 6, device=self.device)
        self.piper_joint_vel = torch.zeros((self.num_envs, 6), device=self.device)
        #usd에서 자동으로 각 관절의 리미트값을 가져옴
        self.piper_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, 0:6, 0].to(device=self.device)
        self.piper_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, 0:6, 1].to(device=self.device)
        #print("====================================================================================")
        #print(self.piper_dof_lower_limits)
        #주어진 tensor와 같은 shape으로, 모든 값이 1.0인 텐서를 만듦. 관절속도 스케일링. 1이면 똑같 2이면 빠 0.5이면 느린거임. 크기 6
        self.piper_dof_speed_scales = torch.ones_like(self.piper_dof_lower_limits)
        self.num_piper_dofs = 6
        self.piper_dof_targets = torch.zeros(
            (self.num_envs, self.num_piper_dofs), dtype=torch.float, device=self.device
        )

        #self.peg_pos = torch.zeros((self.num_envs, 3), device=self.device)
        #self.peg_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_cup_pos=torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.to_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.to_target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.to_target_rot = torch.zeros((self.num_envs, 3), device=self.device)
        self.cup_pos_now=torch.zeros((self.num_envs, 3), device=self.device)

        self.piper_default_dof_pos = torch.tensor(
            [1.33, 1.19, -1.26, -0.1, -0.57, -0.23], device=self.device
        )
        self.dof_pos_scaled = torch.zeros((self.num_envs, 6), device=self.device)


        # Computer body indices
        self.eef_link_idx = self._robot.body_names.index("right_mcp_joint_2")
        #print("eef idx=",self.eef_link_idx)
        #print(self._robot.num_joints)
        jnames = self._robot.joint_names

        target_vals = {
            "right_0": 0.0,
            "right_1": 0.65,
            "right_2": 0.65,
            "right_3": 0.65,
            "right_4": 0.0,
            "right_5": 0.65,
            "right_6": 0.65,
            "right_7": 0.65,
            "right_8": 0.0,
            "right_9": 0.65,
            "right_10": 0.65,
            "right_11": 0.65,
            "right_12": 1.5,
            "right_13": 0.0,
            "right_14": 0.2,
            "right_15": 0.6,
        }
        lut = {n: i for i, n in enumerate(jnames)}
        #print(lut,"lut")

        idxs, vals = [], []
        for name, val in target_vals.items():
            if name in lut:
                idxs.append(lut[name])
                vals.append(val)

        idxs = torch.as_tensor(idxs, device=self.device, dtype=torch.long)
        vals = torch.as_tensor(vals, device=self.device, dtype=torch.float32)
        # (5) 현재 상태 가져오기
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()

        joint_pos[:, idxs] = vals  # 브로드캐스트됨
        joint_vel[:, idxs] = 0.0

        # (7) 시뮬레이터에 반영
        # 방법 1: 리셋/즉시 적용
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.hand_ids = idxs
        self.hand_pos_targets = vals.unsqueeze(0).repeat(self.num_envs, 1).contiguous()
        self._robot.set_joint_position_target(self.hand_pos_targets, joint_ids=self.hand_ids)
        self.cup_idx=self._robot.body_names.index("right_cup")
        
    def _compute_intermediate_values(self):

        # self.to_target_pos = self._robot.data.body_pos_w[:, self.eef_link_idx] - self.scene.env_origins  # local option ?
        self.peg_pos = self._robot.data.body_pos_w[:, self.eef_link_idx]
        self.peg_quat = self._robot.data.body_quat_w[:, self.eef_link_idx]
        #print("peg_pos=",self.peg_pos[0])
        #print("upper limit=",self.piper_dof_upper_limits)
        #print("lower limit=",self.piper_dof_lower_limits)
        self.piper_joint_pos = self._robot.data.joint_pos[:,0:6]
        self.piper_joint_vel = self._robot.data.joint_vel[:,0:6]
        #print("piper_vel=",self.piper_joint_vel)

        #1~-1로 범위제한
        self.dof_pos_scaled = (
            2.0
            * (self.piper_joint_pos - self.piper_dof_lower_limits)
            / (self.piper_dof_upper_limits - self.piper_dof_lower_limits)
            - 1.0
        )

        self.to_target_pos = self.goal_pos - self.peg_pos
        self.to_target_quat = quat_mul(self.peg_quat, quat_conjugate(self.goal_quat))
        _roll, _pitch, _yaw = euler_xyz_from_quat(self.to_target_quat)
        self.to_target_rot = torch.stack((_roll, _pitch, _yaw), dim=-1)


        self.cup_pos_now = self.goal_cup.data.root_pos_w                # (num_envs, 3)
        self.cup_distance  = self.cup_pos_now - self.goal_cup_pos

        self.bead_in_cup=self._contact_sensor.data.net_forces_w.squeeze(1)
        #print(self.bead_in_cup.shape,"bead shape")

        





        if hasattr(self, "eef_axes") and self.cfg.show_eef_axes:
            self.eef_axes.visualize(self.peg_pos, self.peg_quat)          # EEF 축

        if hasattr(self, "goal_axes") and self.cfg.show_goal_axes:
            self.goal_axes.visualize(self.goal_pos, self.goal_quat)   # Goal 축(또는 goal_pos)



    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.goal_cup = Articulation(self.cfg.goal_object_cfg)
        #self.goal_cup = RigidObject(self.cfg.goal_object_cfg)
        self.goal_bead = RigidObject(self.cfg.move_object_cfg)
        self._contact_sensor=ContactSensor(self.cfg.contact_forces)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["goal_cup"] = self.goal_cup
        #self.scene.rigid_objects["goal_cup"] = self.goal_cup
        self.scene.rigid_objects["goal_bead"] = self.goal_bead
        self.scene.sensors["contact_forces"]=self._contact_sensor
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        #print("1111111111111111111_pre_physics_step=================================================================================================")
        self.actions = actions.clone()


    # 원래 action은 -1~1사이의 아무런의미도 없는 값이지만 여기서 target을 정해주는 동시에 자기자신에게 의미를 부여.
    #기존 joint값에 action값에 따라 joint값을 추가함.
    def _apply_action(self) -> None:
        #print("222222222222222222_apply_action=================================================================================================")
        targets = self.piper_dof_targets + self.piper_dof_speed_scales * self.cfg.sim.dt * self.actions * self.cfg.action_scale
        self.piper_dof_targets[:] = tensor_clamp(targets, self.piper_dof_lower_limits, self.piper_dof_upper_limits)


        self._robot.set_joint_position_target(self.piper_dof_targets, joint_ids=self.arm_ids)
        self._robot.set_joint_position_target(self.hand_pos_targets, joint_ids=self.hand_ids) 


    def _get_observations(self) -> dict:
        #print("333333333333333_get_observations=================================================================================================")
        obs = torch.cat(
            (
                self.dof_pos_scaled,
                self.piper_joint_vel * self.cfg.dof_vel_scale,
                self.to_target_pos,
                self.to_target_rot,
            ),
            dim=-1,
        )
        #print("ob=",self.piper_joint_vel * self.cfg.dof_vel_scale)
        observations = {"policy": obs}
        return observations

    #reward에서 action이 크면 reward를 작게 주게 만듦.
    def _get_rewards(self) -> torch.Tensor:
        #print("444444444444444444_get_rewards=================================================================================================")

        total_reward = compute_rewards(
            self.to_target_pos,
            self.to_target_rot,
            self.actions,
            self.cfg.rew_scale_dist,
            self.cfg.rew_scale_action,
            self.cfg.rew_scale_rot,
            self.cup_distance,
            self.bead_in_cup,
        )
        return total_reward

    #_pre_physics_step → (시뮬레이트) → _apply_action → _get_observations → _get_rewards → _get_dones 이런 식으로 매 스탭마다 돌음
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        #print("55555555555555555555_get_dones=================================================================================================")


        self._compute_intermediate_values()
        #print("glagla",self._contact_sensor.data.net_forces_w)


        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # reset if peg is closed goal or max length reached
        arrive_target = torch.norm(self.to_target_pos, p=2, dim=-1) < 0.015
        cup_kicked = torch.norm(self.cup_distance, p=2, dim=-1) > 0.02
        goal_bead=torch.norm(self.bead_in_cup, p=2, dim=-1) > 0.2
        
        #print(arrive_target.shape)  # 기대: (N,)
        #print(cup_kicked.shape)     # 기대: (N,)
        #print(goal_bead.shape)      # 기대: (N,)

        #terminal_flag=arrive_target
        terminal_flag = arrive_target | cup_kicked | goal_bead



        return terminal_flag, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        #print("6666666666666666666_reset_idx=================================================================================================")

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)


        #print(env_ids)
        pos = tensor_clamp(
            self.piper_default_dof_pos.unsqueeze(0)
            + 0.50 * (torch.rand((len(env_ids), self.num_piper_dofs), device=self.device) - 0.5),
            self.piper_dof_lower_limits,
            self.piper_dof_upper_limits,
        )
        vel = torch.zeros((1, self.num_piper_dofs), device=self.device)

        self.piper_dof_targets[env_ids, :] = pos
        self.piper_joint_pos[env_ids, :] = pos
        self.piper_joint_vel[env_ids, :] = vel

        #self._robot.set_joint_position_target(self.ur_dof_targets[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(self.piper_joint_pos[env_ids, :], self.piper_joint_vel[env_ids, :], joint_ids=self.arm_ids, env_ids=env_ids)






        robot_description_path = "/home/user/tools/lab_test1/lab_test1/source/lab_test1/lab_test1/robots/lab_robot_description.yaml"
        urdf_path = "/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/lab_right_hand_attatch_piper_description.urdf"
        #target_link = "wrist_3_link"

        # 1) Solver 생성
        lula_kinematics_solver = LulaKinematicsSolver(
        	robot_description_path=robot_description_path,
        	urdf_path=urdf_path,
        )
        temp_joints_ = torch.clamp(
            6.00 * (torch.rand((len(env_ids), self.num_piper_dofs), device=self.device) - 0.5),
            self.piper_dof_lower_limits, self.piper_dof_upper_limits)
        num = 0
        # z > 0.10
        peg_p = []
        peg_m = []

        for joints_ in temp_joints_:
            while True:

                peg_p, peg_m = lula_kinematics_solver.compute_forward_kinematics("right_mcp_joint_2", joints_.cpu().numpy())
                
                
                # 1) 높이 조건
                if peg_p[2] < 0.2 or peg_p[2]>0.5 or peg_p[1]<0.15:
                    re_joints_ = torch.clamp(
                        6.00 * (torch.rand((1, self.num_piper_dofs), device=self.device) - 0.5),
                        self.piper_dof_lower_limits, self.piper_dof_upper_limits
                    )
                    joints_ = re_joints_[0, :]
                    continue
                
                # 2) 자세 조건: 로컬 z축이 아래(-Z)를 보는가?
                # 회전행렬 peg_m의 3번째 열이 로컬 z축 → z_dir
                z_dir = peg_m[:, 2] / np.linalg.norm(peg_m[:, 2])
                dot_val = np.dot(z_dir, np.array([0, 0, -1]))
                if dot_val < 0.9367:   # cos(20°) ≈ 0.9367
                    re_joints_ = torch.clamp(
                        6.00 * (torch.rand((1, self.num_piper_dofs), device=self.device) - 0.5),
                        self.piper_dof_lower_limits, self.piper_dof_upper_limits
                    )
                    joints_ = re_joints_[0, :]
                    continue
                
                # 두 조건을 모두 만족했으면 탈출
                break

            peg_m_q = rotationMatrixToQuaternion1.rotationMatrixToQuaternion1(peg_m)
            #print("peg_m",peg_m[0])
            

            peg_p = torch.from_numpy(peg_p)
            peg_m_q = torch.from_numpy(peg_m_q)

            self.goal_pos[env_ids[num], :] = peg_p.to(torch.device(self.device))
            #print("goal",self.goal_pos)
            self.goal_quat[env_ids[num], :] = peg_m_q.to(torch.device(self.device))
            num = num + 1

        # 월드 원점 보정 + 마커 일괄 업데이트(모든 env에 대해)
        ur_pos = self.scene.env_origins
        ur_rot = self._robot.data.root_quat_w  # (사용하진 않지만 남겨둠)
        self.goal_pos[env_ids] = self.goal_pos[env_ids] + ur_pos[env_ids]
        self.goal_cup_pos[env_ids]=self.goal_pos[env_ids]
        # z축만 -0.1 내려주기
        self.goal_cup_pos[env_ids, 2] -= 0.15
        #print(self.goal_pos[env_ids],"org")
        #print(self.goal_cup_pos[env_ids])
        #print("gd")
        #self.goal_markers.visualize(self.goal_cup_pos, self.goal_quat)
        

        #root_pose = torch.cat((self.goal_cup_pos[env_ids], self.goal_quat[env_ids]), dim=-1)
        #self.goal_cup.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        # pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = (N, 13)
        pos  = self.goal_cup_pos[env_ids]          # (N,3)
        quat = self.goal_quat[env_ids]             # (N,4)  (qw, qx, qy, qz)
        zeros = torch.zeros_like(pos)              # (N,3)
        
        #pos_test = pos.clone()
        #pos_test[:, 2] += 0.05
        
        cup_pos_all  = self._robot.data.body_pos_w[:, self.cup_idx, :]   # (num_envs, 3)
        cup_pos  = cup_pos_all[env_ids]
        pos_bead = cup_pos.clone()
        pos_bead[:, 2] += 0.05
        
        root_state_cup = torch.cat([pos, quat, zeros, zeros], dim=-1)  # (N,13)
        
        root_state_bead = torch.cat([pos_bead, quat, zeros, zeros], dim=-1)  # (N,13)
        
        #root_state_test = torch.cat([pos_test, quat, zeros, zeros], dim=-1)

        self.goal_cup.write_root_state_to_sim(root_state_cup, env_ids=env_ids)
        
        self.goal_bead.write_root_state_to_sim(root_state_bead, env_ids=env_ids)
















@torch.jit.script

def compute_rewards(
    to_target_pos: torch.Tensor,
    to_target_rot: torch.Tensor,
    actions: torch.Tensor,
    rew_scale_dist: float,
    rew_scale_action: float,
    rew_scale_rot: float,
    cup_distance: torch.Tensor, 
    bead_in_cup:torch.Tensor,
):
    # distance from peg to the goal
    d = torch.norm(to_target_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 1, dist_reward * 2, dist_reward)
    #print(to_target_rot[0])

    # the difference in radian form peg to the goal
    d_r = torch.norm(to_target_rot, p=2, dim=-1)
    rot_reward = 1.0 / (1.0 + d_r ** 2)
    rot_reward *= rot_reward
    rot_reward = torch.where(d >= 1.0, rot_reward * 2, rot_reward)
    rot_reward = torch.where(d <= 1.0, 0, rot_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = rew_scale_dist * dist_reward \
        - rew_scale_action * action_penalty \
        + rew_scale_rot * rot_reward
    

    # bonus for finish option
    rewards = torch.where(torch.norm(to_target_pos, p=2, dim=-1) < 2, rewards + 1, rewards)
    rewards = torch.where(torch.norm(to_target_pos, p=2, dim=-1) < 1, rewards + 8, rewards)
    
    #rewards = torch.where((torch.norm(to_target_rot, p=2, dim=-1) < 1.0)&(torch.norm(to_target_pos, p=2, dim=-1) < 0.2), rewards + 1, rewards)
    #rewards = torch.where((torch.norm(to_target_rot, p=2, dim=-1) < 0.2)&(torch.norm(to_target_pos, p=2, dim=-1) < 0.02), rewards + 30, rewards)
    
    #rewards = torch.where((torch.norm(to_target_rot, p=2, dim=-1) > 1.0)&(torch.norm(to_target_pos, p=2, dim=-1) < 0.2), rewards - 1, rewards)
    #rewards = torch.where((torch.norm(to_target_rot, p=2, dim=-1) > 0.2)&(torch.norm(to_target_pos, p=2, dim=-1) < 0.02), rewards - 30, rewards)
    
    rewards = torch.where(torch.norm(cup_distance, p=2, dim=-1) > 0.015, rewards - 50, rewards)
    rewards = torch.where(torch.norm(bead_in_cup, p=2, dim=-1) > 0.1, rewards +111, rewards)    
    
    
    
    return rewards

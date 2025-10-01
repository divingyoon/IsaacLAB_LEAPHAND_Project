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

from .ur16e_ik_env_cfg import Ur16eIKEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_mul, quat_conjugate, euler_xyz_from_quat
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from lab_test1.utils import rotationMatrixToQuaternion1

import yaml, numpy as np
import os, yaml

def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

class Ur16eIKEnv(DirectRLEnv):
    cfg: Ur16eIKEnvCfg

    def __init__(self, cfg: Ur16eIKEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # init data
        # self._set_body_inertias()
        self._init_tensors()
        self._compute_intermediate_values()

        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel
        # initialize goal marker
        #self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        
    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.ur_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        
        self.ur_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.ur_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.ur_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.ur_dof_speed_scales = torch.ones_like(self.ur_dof_lower_limits)
        self.num_ur_dofs = self._robot.num_joints
        self.ur_dof_targets = torch.zeros(
            (self.num_envs, self.num_ur_dofs), dtype=torch.float, device=self.device
        )

        self.peg_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.to_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.to_target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.to_target_rot = torch.zeros((self.num_envs, 3), device=self.device)

        self.ur_default_dof_pos = torch.tensor(
            [1.33, -0.99, 1.26, -1.84, -1.57, -0.23], device=self.device
        )
        self.dof_pos_scaled = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Computer body indices
        self.eef_link_idx = self._robot.body_names.index("wrist_3_link")

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
    def _joints_for_lula(self, joints_np, coordinate_names):
        """
        IsaacLab 로봇의 joint_names 순서로 들어있는 joints_np(길이>=6)를
        Lula의 cspace.coordinate_names 순서(정확히 6개)로 재정렬하여 반환.
        """
        # IsaacLab에서 보고 있는 조인트 이름 목록(순서 중요)
        robot_joint_names = list(self._robot.joint_names)  # Articulation의 조인트 이름들
        name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}

        ordered = []
        for cname in coordinate_names:
            if cname not in name_to_idx:
                raise RuntimeError(f"[Lula mapping] '{cname}' 가 IsaacLab joint_names에 없습니다. "
                                   f"IsaacLab joint_names={robot_joint_names}")
            ordered.append(joints_np[name_to_idx[cname]])
        return np.asarray(ordered, dtype=np.float64)

    def _compute_intermediate_values(self):

        # self.to_target_pos = self._robot.data.body_pos_w[:, self.eef_link_idx] - self.scene.env_origins  # local option ?
        self.peg_pos = self._robot.data.body_pos_w[:, self.eef_link_idx]
        self.peg_quat = self._robot.data.body_quat_w[:, self.eef_link_idx]

        self.ur_joint_pos = self._robot.data.joint_pos
        self.ur_joint_vel = self._robot.data.joint_vel

        self.dof_pos_scaled = (
            2.0
            * (self.ur_joint_pos - self.ur_dof_lower_limits)
            / (self.ur_dof_upper_limits - self.ur_dof_lower_limits)
            - 1.0
        )

        self.to_target_pos = self.goal_pos - self.peg_pos
        self.to_target_quat = quat_mul(self.peg_quat, quat_conjugate(self.goal_quat))
        _roll, _pitch, _yaw = euler_xyz_from_quat(self.to_target_quat)
        self.to_target_rot = torch.stack((_roll, _pitch, _yaw), dim=-1)







    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:      
        self.actions = actions.clone()
        
    def _apply_action(self) -> None:
        targets = self.ur_dof_targets + self.ur_dof_speed_scales * self.cfg.sim.dt * self.actions * self.cfg.action_scale
        self.ur_dof_targets[:] = tensor_clamp(targets, self.ur_dof_lower_limits, self.ur_dof_upper_limits)

        self._robot.set_joint_position_target(self.ur_dof_targets)
#    def _get_observations(self) -> dict:
#        obs = torch.cat(
#            (
#                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
#                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
#                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
#                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
#            ),
#            dim=-1,
#        )
#        observations = {"policy": obs}
#        return observations
#
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.dof_pos_scaled,
                self.ur_joint_vel * self.cfg.dof_vel_scale,
                self.to_target_pos,
                self.to_target_rot,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations











    #def _get_rewards(self) -> torch.Tensor:
    #    total_reward = compute_rewards(
    #        self.cfg.rew_scale_alive,
    #        self.cfg.rew_scale_terminated,
    #        self.cfg.rew_scale_pole_pos,
    #        self.cfg.rew_scale_cart_vel,
    #        self.cfg.rew_scale_pole_vel,
    #        self.joint_pos[:, self._pole_dof_idx[0]],
    #        self.joint_vel[:, self._pole_dof_idx[0]],
    #        self.joint_pos[:, self._cart_dof_idx[0]],
    #        self.joint_vel[:, self._cart_dof_idx[0]],
    #        self.reset_terminated,
    #    )
    #    return total_reward
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.to_target_pos,
            self.to_target_rot,
            self.actions,
            self.cfg.rew_scale_dist,
            self.cfg.rew_scale_action,
            self.cfg.rew_scale_rot,
        )
        return total_reward








#
#    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#        self.joint_pos = self.robot.data.joint_pos
#        self.joint_vel = self.robot.data.joint_vel
#
#        time_out = self.episode_length_buf >= self.max_episode_length - 1
#        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
#        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
#        return out_of_bounds, time_out
#
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # reset if peg is closed goal or max length reached
        terminal_flag = torch.norm(self.to_target_pos, p=2, dim=-1) < 0.015

        return terminal_flag, time_out   
   
   
#    def _reset_idx(self, env_ids: Sequence[int] | None):
#        if env_ids is None:
#            env_ids = self.robot._ALL_INDICES
#        super()._reset_idx(env_ids)
#
#        joint_pos = self.robot.data.default_joint_pos[env_ids]
#        joint_pos[:, self._pole_dof_idx] += sample_uniform(
#            self.cfg.initial_pole_angle_range[0] * math.pi,
#            self.cfg.initial_pole_angle_range[1] * math.pi,
#            joint_pos[:, self._pole_dof_idx].shape,
#            joint_pos.device,
#        )
#        joint_vel = self.robot.data.default_joint_vel[env_ids]
#
#        default_root_state = self.robot.data.default_root_state[env_ids]
#        default_root_state[:, :3] += self.scene.env_origins[env_ids]
#
#        self.joint_pos[env_ids] = joint_pos
#        self.joint_vel[env_ids] = joint_vel
#
#        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
#



    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_indices = len(env_ids)

        pos = tensor_clamp(
            self.ur_default_dof_pos.unsqueeze(0)
            + 0.50 * (torch.rand((len(env_ids), self.num_ur_dofs), device=self.device) - 0.5),
            self.ur_dof_lower_limits,
            self.ur_dof_upper_limits,
        )
        vel = torch.zeros((1, self.num_ur_dofs), device=self.device)

        self.ur_dof_targets[env_ids, :] = pos
        self.ur_joint_pos[env_ids, :] = pos
        self.ur_joint_vel[env_ids, :] = vel

        self._robot.set_joint_position_target(self.ur_dof_targets[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(self.ur_joint_pos, self.ur_joint_vel)

        robot_description_path = "/home/user/ur_urdf/ur_description/config/ur10e/ur16e_robot_description.yaml"
        urdf_path = "/home/user/ur_urdf/ur16e.urdf"
        target_link = "wrist_3_link"
    
        # 1) Solver 생성
        lula_kinematics_solver = LulaKinematicsSolver(
        	robot_description_path=robot_description_path,
        	urdf_path=urdf_path,
        )
        temp_joints_ = torch.clamp(
            2.00 * (torch.rand((len(env_ids), self.num_ur_dofs), device=self.device) - 0.5),
            self.ur_dof_lower_limits, self.ur_dof_upper_limits)
        num = 0
        # z > 0.10
        peg_p = []
        peg_m = []

        for joints_ in temp_joints_:
            # print(joints_.cpu().numpy())
            # peg_p, peg_m = lula_kinematics_solver.compute_forward_kinematics("peg1", joints_.cpu().numpy())
            while True:
                
                #with open(robot_description_path, "r") as f:
                #    robot_desc = yaml.safe_load(f)
#
#
                ##cspace_names = robot_desc["cspace"]["coordinate_names"]  # 길이 6이어야 함
                ## cspace가 dict(예: {"coordinate_names": [...]})이거나 list(예: ["joint1", ...]) 둘 다 지원
                #cspace_field = robot_desc.get("cspace")
                #if isinstance(cspace_field, dict):
                #    cspace_names = cspace_field.get("coordinate_names")
                #else:
                #    cspace_names = cspace_field  # list인 경우
#
                #if not isinstance(cspace_names, (list, tuple)) or len(cspace_names) == 0:
                #    raise RuntimeError(f"[YAML] cspace가 비정상입니다: {cspace_names}")
#
                ## joints_: torch.Tensor(shape=(self.num_ur_dofs,))
                #ilab_joint_names = list(self._robot.joint_names)
                #j_np = joints_.detach().cpu().numpy()
#
#
                ## IsaacLab joint_names → Lula coordinate_names 순서로 재정렬(정확히 6개)
                #j_for_lula = self._joints_for_lula(j_np, cspace_names)
#
                ## 이제 FK는 반드시 길이 6짜리, 올바른 순서의 벡터를 받음
                #peg_p, peg_m = lula_kinematics_solver.compute_forward_kinematics(
                #    target_link, j_for_lula
                #)
                peg_p, peg_m = lula_kinematics_solver.compute_forward_kinematics("wrist_3_link", joints_.cpu().numpy())
                if peg_p[2] < 0.2:
                    # re_joints_ = torch.clamp(
                    #     self.ur_default_dof_pos.unsqueeze(0)
                    #     + 0.00 * (torch.rand((1, self.num_ur_dofs), device=self._device) - 0.5),
                    #     self.ur_dof_lower_limits, self.ur_dof_upper_limits)
                    re_joints_ = torch.clamp(
                        2.00 * (torch.rand((1, self.num_ur_dofs), device=self.device) - 0.5),
                        self.ur_dof_lower_limits, self.ur_dof_upper_limits)
                    joints_ = re_joints_[0, :]
                else:
                    break

            peg_m_q = rotationMatrixToQuaternion1.rotationMatrixToQuaternion1(peg_m)

            peg_p = torch.from_numpy(peg_p)
            peg_m_q = torch.from_numpy(peg_m_q)

            self.goal_pos[env_ids[num], :] = peg_p.to(torch.device(self.device))
            self.goal_quat[env_ids[num], :] = peg_m_q.to(torch.device(self.device))
            num = num + 1

        # 월드 원점 보정 + 마커 일괄 업데이트(모든 env에 대해)
        ur_pos = self.scene.env_origins
        ur_rot = self._robot.data.root_quat_w  # (사용하진 않지만 남겨둠)
        self.goal_pos[env_ids] = self.goal_pos[env_ids] + ur_pos[env_ids]
        #self.goal_markers.visualize(self.goal_pos, self.goal_quat)














@torch.jit.script
#def compute_rewards(
#    rew_scale_alive: float,
#    rew_scale_terminated: float,
#    rew_scale_pole_pos: float,
#    rew_scale_cart_vel: float,
#    rew_scale_pole_vel: float,
#    pole_pos: torch.Tensor,
#    pole_vel: torch.Tensor,
#    cart_pos: torch.Tensor,
#    cart_vel: torch.Tensor,
#    reset_terminated: torch.Tensor,
#):
#    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#    rew_termination = rew_scale_terminated * reset_terminated.float()
#    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#    return total_reward
def compute_rewards(
    to_target_pos: torch.Tensor,
    to_target_rot: torch.Tensor,
    actions: torch.Tensor,
    rew_scale_dist: float,
    rew_scale_action: float,
    rew_scale_rot: float
):
    # distance from peg to the goal
    d = torch.norm(to_target_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    # the difference in radian form peg to the goal
    d_r = torch.norm(to_target_rot, p=2, dim=-1)
    rot_reward = 1.0 / (1.0 + d_r ** 2)
    rot_reward *= rot_reward

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = rew_scale_dist * dist_reward \
        - rew_scale_action * action_penalty \
        + rew_scale_rot * rot_reward

    # bonus for finish option
    rewards = torch.where(torch.norm(to_target_pos, p=2, dim=-1) < 0.15, rewards + 1, rewards)
    rewards = torch.where(torch.norm(to_target_pos, p=2, dim=-1) < 0.015, rewards + 30, rewards)
    return rewards 
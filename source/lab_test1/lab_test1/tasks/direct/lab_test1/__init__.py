# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Lab-Test1-Direct-v0",
    entry_point=f"{__name__}.lab_test1_env:LabTest1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lab_test1_env_cfg:LabTest1EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
gym.register(
    id="Template-Lab-Test1-Lab_Robot-v0",
    entry_point=f"{__name__}.lab_robot_env:LabRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lab_robot_env_cfg:LabRobotEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_lab_robot_ppo_cfg.yaml",
    },
)
gym.register(
    id="Template-Lab-Test1-Ur16e_ik-v0",
    entry_point=f"{__name__}.ur16e_ik_env:Ur16eIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur16e_ik_env_cfg:Ur16eIKEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ur16e_ik_ppo_cfg.yaml",
    },
)
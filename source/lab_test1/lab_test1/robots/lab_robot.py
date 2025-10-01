
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# robot(s)
# robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
_usd_path = "/home/user/humble_ws/src/piper_isaac_sim/piper_description/urdf/lab_robot.usd"
robot = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        usd_path=_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
        "right_joint1": 1.4,
        "right_joint2": 1.2,
        "right_joint3": -1.1,
        "right_joint4": 0.0,
        "right_joint5": 0.1,
        "right_joint6": 0.0,

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
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "right_arm_actuactor": ImplicitActuatorCfg(
            joint_names_expr=[r"^right_joint[1-6]$"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=660,
            velocity_limit=124.6,
        ),
        "right_hand_actuactor": ImplicitActuatorCfg(
            joint_names_expr=[r"^right_(?:\d|1[0-5])$"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=660,
            velocity_limit=124.6,
        ),
    },
    # actuators={
    # "arm": ImplicitActuatorCfg(
    #     joint_names_expr=[".*"],
    #     velocity_limit=100.0,
    #     effort_limit=87.0,
    #     stiffness=800.0,
    #     damping=40.0,
    #     ),
    # },
)
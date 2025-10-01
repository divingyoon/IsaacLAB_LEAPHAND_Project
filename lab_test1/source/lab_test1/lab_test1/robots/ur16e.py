
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# robot(s)
# robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
_usd_path = "/home/user/ur_urdf/ur16e/ur16e.usd"
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
        "shoulder_pan_joint": 1.33,
        "shoulder_lift_joint": -0.99,
        "elbow_joint": 1.26,
        "wrist_1_joint": -1.84,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": -0.23,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder_pan_actuactor": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=660,
            velocity_limit=124.6,
        ),
        "shoulder_lift_actuactor": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=660,
            velocity_limit=124.6,
        ),
        "elbow_actuactor": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=300,
            velocity_limit=124.6,
        ),
        "wrist_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wrist_[1-3]_joint"],
            stiffness=1745.3302,
            damping=174.53293,
            friction=0.0,
            armature=0.0,
            effort_limit=108,
            velocity_limit=149.5,
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
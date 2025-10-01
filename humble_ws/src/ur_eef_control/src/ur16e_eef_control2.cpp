#include <chrono>
#include <map>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "kdl/chainiksolverpos_nr_jl.hpp"
#include "rclcpp/rclcpp.hpp"
#include "trac_ik/trac_ik.hpp"

#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"

#define DEG2RAD(x)                          (x * 0.01745329252)  // *PI/180
#define RAD2DEG(x)                          (x * 57.2957795131)  // *180/PI

using std::placeholders::_1;

class URjointTest : public rclcpp::Node{
  public:
    URjointTest();
    //pub function
    void publish_target_joints(sensor_msgs::msg::JointState target_j);
    void publish_ft_data(geometry_msgs::msg::WrenchStamped ft_data);
    void publish_peg_ft_data(geometry_msgs::msg::WrenchStamped ft_data);
    void publish_result(std_msgs::msg::Float64MultiArray result);

    //sub function
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr joint_);
    void dposeCallback(const geometry_msgs::msg::Twist::SharedPtr dPose);
    void ft_1Callback(const geometry_msgs::msg::WrenchStamped::SharedPtr ft_1);

    // Function

    // Data
    std::vector<std::string> joint_name;
    std::vector<double> joint_position, joint_velocity, joint_effort;
    geometry_msgs::msg::Twist desired_pose;
    geometry_msgs::msg::Wrench ft_1;
    geometry_msgs::msg::WrenchStamped ft_data;

  private:
    // Publsher
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr ur_jointarget_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_data_pub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr peg_ft_data_pub;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr result_pub;

    // Subscribe
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr ur_jointstate_sub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr ur_desired_pose_sub;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_1_sub;

};

static const rclcpp::Logger LOGGER = rclcpp::get_logger("ur16e_ft_test1");

URjointTest::URjointTest()
  : rclcpp::Node("ur16e_ft_test1")
{
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
  // Pub
  ur_jointarget_pub = this->create_publisher<sensor_msgs::msg::JointState>("joint_command", 10);
  ft_data_pub = this->create_publisher<geometry_msgs::msg::WrenchStamped>("wr_ft_data", qos_profile);
  peg_ft_data_pub = this->create_publisher<geometry_msgs::msg::WrenchStamped>("peg_ft_data", qos_profile);
  result_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>("ur_result", qos_profile);

  // Sub
  ur_jointstate_sub = this->create_subscription<sensor_msgs::msg::JointState>(
              "joint_states",
              qos_profile,
              std::bind(&URjointTest::jointCallback, this, _1)
              );
  ur_desired_pose_sub = this->create_subscription<geometry_msgs::msg::Twist>(
    "eef_target_d_twist", qos_profile, std::bind(&URjointTest::dposeCallback, this, _1));
  ft_1_sub = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "ft_data", qos_profile, std::bind(&URjointTest::ft_1Callback, this, _1));

  // Init data
  std::vector<std::string> check_ur_j = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  joint_name = check_ur_j;

  joint_position.resize(6);
  joint_velocity.resize(6);
  joint_effort.resize(6);

}

void URjointTest::publish_target_joints(sensor_msgs::msg::JointState target_j){

  this->ur_jointarget_pub->publish(target_j);

}

void URjointTest::publish_ft_data(geometry_msgs::msg::WrenchStamped ft){
  ft.header.stamp = this->now();
  this->ft_data_pub->publish(ft);
}

void URjointTest::publish_peg_ft_data(geometry_msgs::msg::WrenchStamped ft){
  ft.header.stamp = this->now();
  this->peg_ft_data_pub->publish(ft);
}

void URjointTest::publish_result(std_msgs::msg::Float64MultiArray rs){
  this->result_pub->publish(rs);
}

void URjointTest::jointCallback(const sensor_msgs::msg::JointState::SharedPtr joint_){
//    RCLCPP_INFO(LOGGER, "joint %s", joint_->name);

    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){
            if(joint_name[i] == joint_->name[j]){
                joint_position[i] = joint_->position[j];
                joint_velocity[i] = joint_->velocity[j];
                joint_effort[i] = joint_->effort[j];
            }
        }
//        RCLCPP_INFO(LOGGER, "joint [%d] %lf deg",i, RAD2DEG(joint_position[i]));
    }

}

void URjointTest::dposeCallback(const geometry_msgs::msg::Twist::SharedPtr dPose){
  this->desired_pose = *dPose;
  // RCLCPP_INFO_STREAM(LOGGER,
  // "d Pose: linear x: " << this->desired_pose.linear.x<<" y: "<<this->desired_pose.linear.y<<" z: "<<this->desired_pose.linear.z);
  // RCLCPP_INFO_STREAM(LOGGER,
  // "d Pose: angular x: " << this->desired_pose.angular.x<<" y: "<<this->desired_pose.angular.y<<" z: "<<this->desired_pose.angular.z);
}

void URjointTest::ft_1Callback(const geometry_msgs::msg::WrenchStamped::SharedPtr ft_1_data){

  this->ft_1 = ft_1_data->wrench;
}

double constrain_v(double v, double v_max, double v_min){
  if (v > v_max){
    v = v_max;
  }
  else if(v < v_min){
    v = v_min;
  }
  return v;
}

// Main script *************************************************************************************

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // auto node = rclcpp::Node::make_shared("ur16e_joint_test");
  auto sub_node = std::make_shared<URjointTest>();
  auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

  // Init data
  std::vector<std::string> joint_name;
  std::vector<double> joint_position, joint_velocity, joint_effort;
  std::string chain_start, chain_end, urdf_xml;
  double timeout;

  sub_node->declare_parameter<double>("timeout", 0.005);
  sub_node->declare_parameters<std::string>(
    std::string(),       // parameters are not namespaced
    std::map<std::string, std::string>{
    {"chain_start", std::string()},
    {"chain_end", std::string()},
    {"robot_description", std::string()},
  });

  sub_node->get_parameter("timeout", timeout);
  sub_node->get_parameter("chain_start", chain_start);
  sub_node->get_parameter("chain_end", chain_end);
  sub_node->get_parameter("robot_description", urdf_xml);

  if (chain_start.empty() || chain_end.empty()) {
    RCLCPP_FATAL(LOGGER, "Missing chain info in launch file");
    exit(-1);
  }

  // Create a JointState message
  sensor_msgs::msg::JointState joint_state;
  double current_joints[6] = {0,};
  double target_joints[6] = {0,};

  joint_state.name.resize(6);
  joint_state.position.resize(6);
  joint_state.effort.resize(6);
  joint_state.velocity.resize(6);

  std::vector<std::string> check_ur_j = {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  // std::vector<double> default_joints = {1.4198, -0.9986, 1.3143, -1.8892, -1.6622, 0.166};  //1.158, -1.912, -1.765, 0.0
  std::vector<double> default_joints = {-1.29955968, -1.96147873, -1.62985075, -1.1210595,   1.57079633,  0.27123665};  //1.158, -1.912, -1.765, 0.0

  joint_state.name = check_ur_j;
  joint_state.position = default_joints;

  // TRAC-IK setting start
  double eps = 1e-5;
  // This constructor parses the URDF loaded in rosparm urdf_xml into the
  // needed KDL structures.  We then pull these out to compare against the KDL
  // IK solver.
  TRAC_IK::TRAC_IK tracik_solver(chain_start, chain_end, urdf_xml, timeout, eps);

  KDL::Chain chain;
  KDL::JntArray ll, ul;  // lower joint limits, upper joint limits

  bool valid = tracik_solver.getKDLChain(chain);

  if (!valid) {
    RCLCPP_ERROR(sub_node->get_logger(), "There was no valid KDL chain found");
    return 0;
  }

  valid = tracik_solver.getKDLLimits(ll, ul);

  if (!valid) {
    RCLCPP_ERROR(sub_node->get_logger(), "There were no valid KDL joint limits found");
    return 0;
  }

  //check Dof
  assert(chain.getNrOfJoints() == ll.data.size());
  assert(chain.getNrOfJoints() == ul.data.size());

  RCLCPP_INFO(sub_node->get_logger(), "Using %d joints", chain.getNrOfJoints());

  // Set up KDL IK
  KDL::ChainFkSolverPos_recursive fk_solver(chain);  // Forward kin. solver

  // Create Nominal chain configuration midway between all joint limits
  KDL::JntArray nominal(chain.getNrOfJoints());

  for (uint j = 0; j < nominal.data.size(); j++) {
    nominal(j) = (ll(j) + ul(j)) / 2.0;
  }

  KDL::JntArray result;
  KDL::Frame end_effector_pose; // Desired eef - Pose
  KDL::Frame current_effector_pose;
  KDL::Frame current_effector_pose_r;
  KDL::Frame desired_effector_pose;
  int rc;
  KDL::JntArray q(chain.getNrOfJoints()); // Initial joints
  KDL::JntArray q_current(chain.getNrOfJoints()); // Initial joints

  for (int i=0; i<chain.getNrOfJoints(); i++){
    q(i) = default_joints[i];
  }

  std::vector<double> vec2(q.data.data(), q.data.data()+q.data.size());
  RCLCPP_INFO_STREAM(sub_node->get_logger(),"Init Joint q: ");
  for(int i=0; i<chain.getNrOfJoints(); i++){
    RCLCPP_INFO_STREAM(sub_node->get_logger(),"joint " << i+1<<" "<<vec2[i]);
  }

  fk_solver.JntToCart(q, end_effector_pose); // Set initial joint pose

  // Solve Inverse kinematics using Trac-IK
  rc = tracik_solver.CartToJnt(nominal, end_effector_pose, result);

  if (rc < 0){
    RCLCPP_ERROR(sub_node->get_logger(), "There were no valid end effector pose");
  }

  // Wrench set
  KDL::Frame frame_bf;  //frame of base to ft sensor
  KDL::Vector Pfp(0.62607, 0.00, 0);
  KDL::Rotation Rfp = KDL::Rotation::RPY(0,0,0);  //M_PI_2
  KDL::Frame frame_fp(Rfp, Pfp);  //frame of f/t sensor to peg
  KDL::Wrench ft_sensor_wrench;
  KDL::Wrench base_wrench;
  KDL::Wrench peg_wrench;

  // Init setting

  int loop_c = 0;

  // test code
  // KDL::Vector pBA(-1,0,0);
  // KDL::Rotation R_BA = KDL::Rotation::Identity();
  // KDL::Vector fA(10, 15, 20);
  // KDL::Vector fB;
  // KDL::Vector tA(1,2,3);
  // KDL::Vector tB;
  // KDL::Wrench wA(fA,tA);
  // KDL::Wrench wB;

  // KDL::Frame Tba(R_BA,pBA);

  // wB = Tba * wA;

  // KDL::Wrench wA_dot;
  // wA_dot = Tba.Inverse() * wB;

  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "T BA");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "T BA's rotation");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), Tba.M.data[0]<<" "<<Tba.M.data[1]<<" "<<Tba.M.data[2]);
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), Tba.M.data[3]<<" "<<Tba.M.data[4]<<" "<<Tba.M.data[5]);
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), Tba.M.data[6]<<" "<<Tba.M.data[7]<<" "<<Tba.M.data[8]);
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "T BA's P");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), Tba.p.data[0]<< " " << Tba.p.data[1]<< " " << Tba.p.data[2]);

  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Wrench B");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Force "<< wB.force.data[0]<< " "<< wB.force.data[1]<< " "<< wB.force.data[2]<< " ");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Torque " << wB.torque.data[0]<<" " << wB.torque.data[1]<<" " << wB.torque.data[2]<<" ");

  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Wrench A'");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Force "<< wA_dot.force.data[0]<< " "<< wA_dot.force.data[1]<< " "<< wA_dot.force.data[2]<< " ");
  // RCLCPP_INFO_STREAM(sub_node->get_logger(), "Torque " << wA_dot.torque.data[0]<<" " << wA_dot.torque.data[1]<<" " << wA_dot.torque.data[2]<<" ");

  rclcpp::WallRate loop_rate(100);
  while(rclcpp::ok()){
    rclcpp::spin_some(sub_node);

    // Get F/T Sensor data <- this data is the compensation of Compliant solver forces
    sub_node->ft_data.wrench.force.x  = sub_node->ft_1.force.x  ;
    sub_node->ft_data.wrench.force.y  = sub_node->ft_1.force.y  ;
    sub_node->ft_data.wrench.force.z  = sub_node->ft_1.force.z  ;
    sub_node->ft_data.wrench.torque.x = sub_node->ft_1.torque.x ;
    sub_node->ft_data.wrench.torque.y = sub_node->ft_1.torque.y ;
    sub_node->ft_data.wrench.torque.z = sub_node->ft_1.torque.z ;

    // Set base wrench
    base_wrench.force.x(sub_node->ft_data.wrench.force.x);
    base_wrench.force.y(sub_node->ft_data.wrench.force.y);
    base_wrench.force.z(sub_node->ft_data.wrench.force.z);
    base_wrench.torque.x(sub_node->ft_data.wrench.torque.x);
    base_wrench.torque.y(sub_node->ft_data.wrench.torque.y);
    base_wrench.torque.z(sub_node->ft_data.wrench.torque.z);

    // Get current Pose
    for(int i=0; i<chain.getNrOfJoints(); i++){
      q_current(i) = sub_node->joint_position[i];
    }
    fk_solver.JntToCart(q_current, current_effector_pose_r); // Set current joint pose
    // temp test!!
    fk_solver.JntToCart(q, current_effector_pose); // Set current joint pose

    // Set frame_bf
    frame_bf.M = current_effector_pose.M;
    frame_bf.p = current_effector_pose.p;

    // Get ft_sensor_wrench
    ft_sensor_wrench = frame_bf.Inverse() * base_wrench;

    // Get peg wrench
    peg_wrench = frame_fp.Inverse() * ft_sensor_wrench;

    // Get ft_sensor_wrench (version global)
    KDL::Wrench ft_sensor_wrench_gb(frame_bf.M * ft_sensor_wrench.force, frame_bf.M * ft_sensor_wrench.torque);

    // Get peg wrench (version global)
    KDL::Rotation roation_bp = frame_bf.M * frame_fp.M;
    KDL::Wrench peg_wrench_gb(roation_bp * peg_wrench.force, roation_bp * peg_wrench.torque);


    geometry_msgs::msg::WrenchStamped ft_s_w;
    ft_s_w.wrench.force.x = ft_sensor_wrench_gb(0);
    ft_s_w.wrench.force.y = ft_sensor_wrench_gb(1);
    ft_s_w.wrench.force.z = ft_sensor_wrench_gb(2);
    ft_s_w.wrench.torque.x = ft_sensor_wrench_gb(3);
    ft_s_w.wrench.torque.y = ft_sensor_wrench_gb(4);
    ft_s_w.wrench.torque.z = ft_sensor_wrench_gb(5);
    sub_node->publish_ft_data(ft_s_w);

    geometry_msgs::msg::WrenchStamped ft_peg;
    ft_peg.wrench.force.x = peg_wrench_gb(0);
    ft_peg.wrench.force.y = peg_wrench_gb(1);
    ft_peg.wrench.force.z = peg_wrench_gb(2);
    ft_peg.wrench.torque.x = peg_wrench_gb(3);
    ft_peg.wrench.torque.y = peg_wrench_gb(4);
    ft_peg.wrench.torque.z = peg_wrench_gb(5);
    sub_node->publish_peg_ft_data(ft_peg);

    // data publish
    if(loop_c >= 100){//-- loop_rate x 100 = 5hz
      //ros2 result topic publish
      std_msgs::msg::Float64MultiArray ft_result;
      ft_result.data.push_back(1);
      ft_result.data.push_back(ft_sensor_wrench_gb.torque(0));
      ft_result.data.push_back(ft_sensor_wrench_gb.torque(1));
      ft_result.data.push_back(ft_sensor_wrench_gb.torque(2));
      ft_result.data.push_back(ft_sensor_wrench_gb.force(0));
      ft_result.data.push_back(ft_sensor_wrench_gb.force(1));
      ft_result.data.push_back(ft_sensor_wrench_gb.force(2));
      ft_result.data.push_back(2);
      ft_result.data.push_back(peg_wrench.torque(0));
      ft_result.data.push_back(peg_wrench.torque(1));
      ft_result.data.push_back(peg_wrench.torque(2));
      ft_result.data.push_back(peg_wrench.force(0));
      ft_result.data.push_back(peg_wrench.force(1));
      ft_result.data.push_back(peg_wrench.force(2));
      ft_result.data.push_back(3);
      ft_result.data.push_back(peg_wrench_gb.torque(0));
      ft_result.data.push_back(peg_wrench_gb.torque(1));
      ft_result.data.push_back(peg_wrench_gb.torque(2));
      ft_result.data.push_back(peg_wrench_gb.force(0));
      ft_result.data.push_back(peg_wrench_gb.force(1));
      ft_result.data.push_back(peg_wrench_gb.force(2));


      sub_node->publish_result(ft_result);

      loop_c = 0;
    }

    // Get desired Pose
    // KDL::Rotation dep_r = KDL::Rotation::RPY(0,0, (-1) * M_PI / 2);
    KDL::Rotation dep_r = KDL::Rotation::RPY(0,0,0);
    KDL::Vector dp(sub_node->desired_pose.linear.x, sub_node->desired_pose.linear.y, sub_node->desired_pose.linear.z);
    KDL::Vector dr_v(sub_node->desired_pose.angular.x, sub_node->desired_pose.angular.y, sub_node->desired_pose.angular.z);
    KDL::Rotation dr = KDL::Rotation::RPY( 2.0 * sub_node->desired_pose.angular.z, (-2,0)*sub_node->desired_pose.angular.y, 2.0 * sub_node->desired_pose.angular.x);
    // desired_effector_pose.p = dep_r * dp;
    desired_effector_pose.p = dp;
    desired_effector_pose.M = dr;

    // Get target Pose
    // end_effector_pose.p = current_effector_pose.p + current_effector_pose.M * desired_effector_pose.p;
    end_effector_pose.M = current_effector_pose.M * desired_effector_pose.M;
    end_effector_pose.p = current_effector_pose.p + desired_effector_pose.p;
    // end_effector_pose.M = current_effector_pose.M.RPY(sub_node->desired_pose.angular.x, sub_node->desired_pose.angular.y, sub_node->desired_pose.angular.z);

    // Get target joints q
    // Solve Inverse kinematics using Trac-IK
    rc = tracik_solver.CartToJnt(q, end_effector_pose, result);

    if (rc < 0){
      RCLCPP_ERROR(sub_node->get_logger(), "There were no valid end effector pose");
    }

    joint_state.header.stamp = sub_node->get_clock()->now();
    for(int i=0;i <6;i++){
      joint_state.position[i] = result(i);
      // RCLCPP_INFO_STREAM(LOGGER, i+1<<" q: "<<q(i)<<" result: "<<result(i));
    }
    // RCLCPP_INFO_STREAM(LOGGER,"T Pose x :" << end_effector_pose.p.x() << " y :"<< end_effector_pose.p.y() <<" z :"<<end_effector_pose.p.z());
    q = result;

    sub_node->publish_target_joints(joint_state);

    loop_rate.sleep();
    loop_c++;
  }



  return 0;
}

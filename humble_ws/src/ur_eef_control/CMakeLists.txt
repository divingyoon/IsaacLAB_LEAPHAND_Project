cmake_minimum_required(VERSION 3.8)
project(ur_eef_control)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
 add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(joint_state_broadcaster REQUIRED)
find_package(joint_trajectory_controller REQUIRED)
find_package(rviz2 REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(xacro REQUIRED)
find_package(urdf REQUIRED)
find_package(NLopt REQUIRED)
find_package(rclcpp REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(trac_ik_lib REQUIRED)
set(THIS_PACKAGE_INCLUDE_DEPENDS
 ament_cmake
 std_msgs
 joint_state_broadcaster
 joint_trajectory_controller
 robot_state_publisher
 xacro
 urdf
 NLopt
 rclcpp
 orocos_kdl
 trac_ik_lib
)
add_executable(ur16e_eef_control2 src/ur16e_eef_control2.cpp)
#add_executable(ur16e_compliance2 src/ur16e_compliance2.cpp)
set_target_properties(ur16e_eef_control2 PROPERTIES
 CXX_STANDARD 20
 CXX_STANDARD_REQUIRED YES
 CXX_EXTENSIONS NO
)

target_include_directories(ur16e_eef_control2 PUBLIC
 ${NLOPT_INCLUDE_DIRS}
)
# Needed since NLopt is not an ament package
target_link_libraries(ur16e_eef_control2
 ${NLOPT_LIBRARIES}
)
ament_target_dependencies(ur16e_eef_control2
 ${THIS_PACKAGE_INCLUDE_DEPENDS}
)
install(TARGETS ur16e_eef_control2
 ARCHIVE DESTINATION lib
 LIBRARY DESTINATION lib
 RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
 DIRECTORY
 launch
 rviz
 DESTINATION share/${PROJECT_NAME}
)
if(BUILD_TESTING)
 find_package(ament_lint_auto REQUIRED)
 set(ament_cmake_copyright_FOUND TRUE)
 set(ament_cmake_cpplint_FOUND TRUE)
 ament_lint_auto_find_test_dependencies()
endif()
ament_package()

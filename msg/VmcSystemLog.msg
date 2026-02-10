std_msgs/Header header

# Robot State
float64[] joint_positions
float64[] joint_torques
float64[] link_velocities
geometry_msgs/Point tcp_position
geometry_msgs/Point camera_position

# Internal VMC Forces (Vector3 per x,y,z)
geometry_msgs/Vector3 force_attraction_spring
geometry_msgs/Vector3 force_attraction_damper
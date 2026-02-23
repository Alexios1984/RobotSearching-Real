# Message with the joint positions and velocities to send to the map node

std_msgs/Header header
geometry_msgs/Pose cam_pose     # Position e Orientation of TCP computed by VMC
float64[] joint_positions       # q present
float64[] joint_velocities      # q_dot present

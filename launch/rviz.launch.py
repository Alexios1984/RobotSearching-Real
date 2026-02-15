import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    # Path to the folder
    startup_pkg_share = FindPackageShare('vmc_startup')

    # --- Rviz Node ---
    
    # Use the saved default configuration file 
    rviz_config_file = PathJoinSubstitution(
        [startup_pkg_share, 'config', 'rviz2_config_rosbag.rviz']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file], 
        output='screen'
    )


    # --- Timing Launch ---
    
    delayed_rviz = TimerAction(
        period=1.0,
        actions=[rviz_node]
    )

    return LaunchDescription([
        delayed_rviz,   
    ])
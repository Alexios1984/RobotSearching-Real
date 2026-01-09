import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():

    # Percorso al pacchetto vmc_startup
    startup_pkg_share = FindPackageShare('vmc_startup')

    # --- 0. RVIZ CONFIGURATION ---
    # Costruiamo il percorso al file .rviz che hai salvato
    rviz_config_file = PathJoinSubstitution(
        [startup_pkg_share, 'config', 'rviz2_config.rviz']
    )

    # --- 1. REALSENSE CAMERA ---
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'align_depth.enable': 'true',
            'pointcloud.enable': 'false',
            'depth_module.depth_profile':'848x480x90',
            'depth_module.color_profile':'848x480x90'

        }.items()
    )

    # --- 2. FRANKA BRINGUP ---
    franka_cmd = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'franka_bringup', 'example.launch.py',
            'controller_name:=julia_torque_controller'
        ],
        output='screen',
        respawn_delay=4.0 
    )

    # --- 3. VISION NODE ---
    vision_node = Node(
        package='vmc_vision',
        executable='vision_node',
        output='screen'
    )

    # --- 4. MAP NODE (IL CERVELLO) ---
    map_node = Node(
        package='vmc_map',
        executable='map_node',
        output='screen'
    )

    # --- 5. RVIZ NODE ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file], # Carica la tua config
        output='screen'
    )

    # --- GESTIONE TEMPISTICA ---
    
    # RViz parte subito (o con leggero ritardo se preferisci)
    delayed_rviz = TimerAction(
        period=1.0,
        actions=[rviz_node]
    )

    delayed_vision = TimerAction(
        period=3.0,
        actions=[vision_node]
    )

    delayed_map = TimerAction(
        period=5.0,
        actions=[map_node]
    )

    return LaunchDescription([
        realsense_launch,
        franka_cmd,
        delayed_rviz,   # Aggiunto alla lista
        delayed_vision,
        delayed_map
    ])
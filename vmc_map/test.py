import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import message_filters 
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

# ROS Messages
from vmc_interfaces.msg import ObjectDetection3DArray, VmcRobotState, VmcObstacles, VmcTarget, VmcMapConfig
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, ColorRGBA, Float64MultiArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

# Threads Executions
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading

"""
Node that represents an ignored zone in the voxel grid.
"""
class IgnoredZone:
    def __init__(self, indices):
        self.member_indices = indices       # List of (ix, iy, iz) tuples
        self.total_count = len(indices)     # Total number of voxels in the zone


"""
Main node to manage the voxel grid mapping from point clouds.
"""
class MapLogicNode(Node):
    def __init__(self):
        
        super().__init__('map_pcl_rand_esc')

        self.node_start_time = None                 # Used not to go in deadlock at time 0.0

        self.current_target_idx = None              # Current target voxel index (ix, iy, iz)
        
        self.decision_lock = threading.Lock()       # Lock for decision loop
        
        self.type_experiment = "boxes"
        
        # ============================== 
        # --- Workspace Configuration ---
        # ==============================
        
        # self.WS_BOUNDS = {                  # Test Workspace
        #     'x': [-0.15, 0.44],   
        #     'y': [ 0.68, 1.18],
        #     'z': [ 0.03, 0.8]
        # }
        

        # ------------- Shelves Case ---------------------------------------------------------
        
        if self.type_experiment == "shelves":

            self.WS_BOUNDS = {                  # Bounds for Shelves Case
                'x': [-0.25, 0.55],   
                'y': [ 0.4, 1.2],
                'z': [ 0.03, 0.9]
            }

            self.TARGET_VOXEL_SIZE = 0.05             # Voxel size in meters (Shelves Case)
        
        
        
        # ------------- Booxes Case ----------------------------------------------------------

        if self.type_experiment == "boxes":

            self.WS_BOUNDS = {                  # Bounds for Boxes Case
                'x': [-0.24, 0.55],   
                'y': [ 0.5, 1.0],
                'z': [ 0.03, 0.40]
            }

            self.TARGET_VOXEL_SIZE = 0.04             # Voxel size in meters (Boxes Case)

        
        
        
        # ------------- Strawberries Case -----------------------------------------------------

        if self.type_experiment == "strawberries":

            self.WS_BOUNDS = {                  # Bounds for Strawberries Case
                'x': [-0.15, 0.55],   
                'y': [ 0.75, 1.1],
                'z': [ 0.45, 0.85]
            }

            self.TARGET_VOXEL_SIZE = 0.015             # Voxel size in meters (Strawberries Case)

        # -------------------------------------------------------------------------------------


        # Dimensions of the workspace
        self.dim_x = self.WS_BOUNDS['x'][1] - self.WS_BOUNDS['x'][0]
        self.dim_y = self.WS_BOUNDS['y'][1] - self.WS_BOUNDS['y'][0]
        self.dim_z = self.WS_BOUNDS['z'][1] - self.WS_BOUNDS['z'][0]
        
        # Calculate number of voxels in each dimension
        self.N_VOXELS_X = int(np.ceil(self.dim_x / self.TARGET_VOXEL_SIZE))
        self.N_VOXELS_Y = int(np.ceil(self.dim_y / self.TARGET_VOXEL_SIZE))
        self.N_VOXELS_Z = int(np.ceil(self.dim_z / self.TARGET_VOXEL_SIZE))
        
        # Voxels' dimensions
        self.res_x = self.dim_x / self.N_VOXELS_X
        self.res_y = self.dim_y / self.N_VOXELS_Y
        self.res_z = self.dim_z / self.N_VOXELS_Z

        self.get_logger().info(f"Workspace: {self.dim_x:.2f}x{self.dim_y:.2f}x{self.dim_z:.2f} m")
        self.get_logger().info(f"Voxel Res: {self.res_x:.3f}x{self.res_y:.3f}x{self.res_z:.3f} m")
        self.get_logger().info(f"Grid Size: {self.N_VOXELS_X}x{self.N_VOXELS_Y}x{self.N_VOXELS_Z}")


        # ==================================================
        # --- Probabilistic Configuration (The Buckets) ---
        #
        # We use a "water bucket" analogy:
        # - Each voxel is a bucket that can hold water (confidence).
        # - When we see an obstacle, we pour some water in (increase confidence).
        # - When we see free space, we let some water evaporate (decrease confidence).  
        # ==================================================
        
        # --- Bucket Tresholds ---
        
        self.VAL_MIN = 0                    # Empty bucket
        self.VAL_MAX = 300                  # Full bucket
        self.VAL_UNKNOWN = 55               # Default uncertainty level
        self.VAL_OCCUPIED = 90              # Min Threshold to say "It's occupied!"
        self.VAL_FREE = 20                  # Max Threshold to say "It's free for sure"
        self.VAL_LOCK = 301                 # Once full, we don't decrease it anymore (for stability)
        
        
        # --- Bucket Quantities ---
        
        self.HIT_INC = 20                  # How much water we add if we see an obstacle (Rapid Increment)
        self.MISS_DEC = 5                   # How much water we lose if we see free space (Slow Decay)


        # --- Voxel Grid Initialization --- 
       
        self.grid = np.full(
            (self.N_VOXELS_X, self.N_VOXELS_Y, self.N_VOXELS_Z), 
            self.VAL_UNKNOWN, 
            dtype=np.int16
        )        
        
        
        # --- Camera Configuration ---
        
        # Field of View from the Realsense D405 specs
        self.GRID_W = 87
        self.GRID_H = 58
        self.FOV_H = np.deg2rad(float(self.GRID_W)) 
        self.FOV_V = np.deg2rad(float(self.GRID_W))
        self.MAX_DEPTH = 0.40               # Max depth to consider (meters). Less than the sensor max for performance (for safety) 
        self.MIN_DEPTH = 0.07               # Min depth to consider (meters). Same as the sensor min for performance  

        # --- Logic Configuration ---
        
        self.MAX_OBSTACLES = 100                        # Max number of repulsors to consider (for performance). Selected among the points of the point cloud.

        self.latest_cam_pose = None                     # Latest Camera Pose (from Robot State)
        
        self.current_visible_obstacles = []             # Current visible obstacles (updated from PCL callback)

        self.cb_group = ReentrantCallbackGroup()        # Lock for the group where all can access at the same time (for parallelization). The default group is the MutuallyExsclusiveCallbackGroup that means that just one per time can be executed

        self.min_distance_rep = 0.06                  # Minimum distance between repulsors 

        self.min_points_per_voxel = 2                  # Minimun number of points to be in a voxel to be considered occupied, otherwise it's just noise
        
        self.MINIMUM_EXPLORED_PERCENTAGE = 95.0         # Minimum completion of the grid to compare the result

        # Scores for the next target selection
        self.SCORE_FRONTALITY = 1.0
        self.SCORE_RELEVANCE = 2.0
        self.SCORE_DISTANCE = 1.0
        
        # --- Deadlock Configuration ---
        
        self.TIME_TRESHOLD_DEADLOCK = 5.0               # Time to wait before starting checking for deadlocks (to avoid initial ignored zone)

        self.DEADLOCK_VEL_EPS = 0.02                    # Min treshold for the links velocity
        self.DEADLOCK_TORQUE_EPS = 0.20                 # Min torque for the joints
        self.DEADLOCK_TIME_THRESHOLD = 3.0              # Time below the tresholds needed to trigger the deadlock
        self.DEADLOCK_ZONE_RADIUS = 0.10                # Radius of the ignored zone to consider from the deadlock target

        self.DEADLOCK_MIN_DIST = 0.40                   # Minimum distance from last deadlock to pick a new target (for avoiding stucking again in the same area)
        self.last_deadlock_pos = None                   # Last deadlock target coordinates
        
        
        self.last_moving_time = time.time()             # Last time we moved (used to compute the time spent in a configuration)
        
        self.last_discovery_check_time = time.time()    # Last time we checked for discovery rate
        self.DISCOVERY_CHECK_INTERVAL = 5.0             # Interval between discovery rate checks (seconds)
        self.MIN_DISCOVERY_RATE = 0.1                   # Minimum discovery rate (new unknown voxels per second) to consider that we are moving
        
        self.is_deadlocked = False                      # Flag for the deadlock
        
        self.ignored_targets = set()                    # Set for the ignored target voxels
        self.ignored_targets.clear()

        self.ignored_zones = []                         # List of ignored zones
        
        # --- Deadlock Recovery Configuration ---
        self.is_recovering = False                  # Flag for intermediate target phase
        self.has_reached_intermediate = False       # Flag to trigger the retry to the original target
        
        self.original_stuck_target = None           # Store the target that caused the deadlock
        
        self.RECOVERY_MIN_DIST = 0.15               # Min dist for random recovery voxel (meters)
        self.RECOVERY_MAX_DIST = 0.40               # Max dist for random recovery voxel (meters)
        self.RECOVERY_REACH_TOLERANCE = 0.1        # Distance to consider the intermediate target reached

        # Check the initial number of unknown voxels for the discovery rate
        mask_initial_unknown = (self.grid >= self.VAL_FREE) & (self.grid <= self.VAL_OCCUPIED)
        self.unknown_count_last_check = np.count_nonzero(mask_initial_unknown)
        
        
        # ============================
        # --- Subscribers ---
        # ============================
        
        # --- Synchronized Subscribers ---
        self.sub_vision_filter = message_filters.Subscriber(
            self, PointCloud2, '/vision/obstacle_cloud',
            callback_group=self.cb_group)
        
        self.sub_robot_filter = message_filters.Subscriber(
            self, VmcRobotState, '/vmc/robot_state', 
            callback_group=self.cb_group)
                
        # Create the Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_vision_filter, self.sub_robot_filter], 
            queue_size=25,          # queue_size=10: how many message to put in queue to search for a synchronization
            slop=0.001               # slop=0.05: tollerance time
        )
        self.ts.registerCallback(self.cb_synced_data)  # When the timestamps from the two subscribers a single synchronized callback is launched
        
        
        # --- Asynchronized Subscribers ---
        self.sub_deadlock = self.create_subscription(                        # Subscriber to the topic with max velocity and torque  
            Float64MultiArray, '/vmc/deadlock_data', self.cb_deadlock, 10, 
            callback_group=self.cb_group
        )


        # ============================
        # --- Publishers --- 
        # ============================

        self.pub_target = self.create_publisher(                    # Target voxel and set of obstacles
            VmcTarget, '/vmc/target_point', 10)
        
        self.pub_obstacles = self.create_publisher(                 # Set of obstacles
            VmcObstacles, '/vmc/active_obstacles', 10)

        self.pub_repulsors_viz = self.create_publisher(             # Markers that represent the repulsor used by VMC
            Marker, '/vmc/repulsors_viz', 10)
        
        self.pub_viz = self.create_publisher(                       # Voxel Grid Visualization
            Marker, '/vmc/voxel_grid_viz', 10)
        
        self.pub_frustum = self.create_publisher(                   # Frustum Visualization
            Marker, '/vmc/camera_frustum', 10)

        self.pub_ws_bounds = self.create_publisher(                 # Workspace Bounds
            Marker, '/vmc/workspace_bounds', 10)

        self.pub_status_text = self.create_publisher(               # Completion Text
            Marker, '/vmc/status_text', 10)
        
        self.pub_completion_time = self.create_publisher(           # Completion Time
            Marker, '/vmc/completion_time', 10)

        self.pub_log_config = self.create_publisher(                # Log Config
            VmcMapConfig, '/vmc/log/config', 1) 

        self.publish_experiment_config()                            # Publish the experiment configuration



        # ============================
        # --- Timers ---
        # ============================
        
        # Timer Visualization 
        self.timer_grid_viz = self.create_timer(0.5, self.publish_grid_worker, callback_group=self.cb_group)
        self.timer_fast_viz = self.create_timer(0.5, self.publish_fast_viz, callback_group=self.cb_group)

        # Logic Timer 
        self.timer = self.create_timer(0.001, self.decision_loop, callback_group=self.cb_group)
        
        # Obstacles Timer
        self.timer_obstacles = self.create_timer(0.1, self.publish_obstacles_worker, callback_group=self.cb_group)
    # -----------------------------------------------------------------------------------------------------------------------


    # =================================================================================================================================
    # --- Callbacks ---
    # =================================================================================================================================
    
    """
    This function is the core logic loop that decides the robot's next exploration target.
    It performs the following steps:
    1. Acquires a lock to prevent race conditions in a multi-threaded environment.
    2. Checks for and handles deadlock conditions, potentially creating "ignored zones" around the deadlock point.
    3. Verifies the current target: if it has been "resolved" (seen as free or occupied), it clears the target. Otherwise, it publishes the command to move towards it.
    4. If no active target exists, it searches for the best new unknown target, considering factors like frontality, distance, and relevance.
    5. If a new target is found, it sets it as the current target and publishes the command.
    """
    def decision_loop(self):

        if not self.decision_lock.acquire(blocking=False):
            return

        try:
            
            # Take the camera position to compute the distance from the voxels around it
            if self.latest_cam_pose is None: return
            cam_pos = self.latest_cam_pose['pos']
            
            # Copy in a local variable the current target index
            curr_idx = self.current_target_idx 

            # Before starting searching the best target check if some ignored zone needs to be free 
            self.check_and_liberate_zones()
            
            
            # --------------------------------------
            # --- Discovery Rate Check ---
            # --------------------------------------
            
            dt_check = time.time() - self.last_discovery_check_time
            
            if dt_check > self.DISCOVERY_CHECK_INTERVAL:
                
                mask_unknown = (self.grid > self.VAL_FREE) & (self.grid < self.VAL_OCCUPIED)
                current_unknown_count = np.sum(mask_unknown)
                
                voxels_discovered = self.unknown_count_last_check - current_unknown_count
                
                discovery_rate = (voxels_discovered / dt_check) / max(1, current_unknown_count)
                
                if self.current_target_idx is not None and current_unknown_count > 0:
                    
                    if discovery_rate < self.MIN_DISCOVERY_RATE:
                        self.is_deadlocked = True
                
                # 5. Aggiorna lo stato per il prossimo giro
                self.last_discovery_check_time = time.time()
                self.unknown_count_last_check = current_unknown_count
            
            # --------------------------------------
            # --- Recovery Monitoring ---
            # --------------------------------------
            # Check if we reached the intermediate recovery voxel
            if self.is_recovering and not self.has_reached_intermediate and curr_idx is not None:
                curr_voxel_pos = self.grid_to_world(*curr_idx)
                
                # Compute NOSE position (0.25m offset along Camera Z-axis)
                r_cam = R.from_quat(self.latest_cam_pose['quat'])
                nose_offset_cam = np.array([0.0, 0.0, 0.25])
                nose_pos_world = cam_pos + r_cam.apply(nose_offset_cam)
                
                # Check distance using the nose, not the camera base
                dist_to_intermediate = np.linalg.norm(nose_pos_world - curr_voxel_pos)
                
                if dist_to_intermediate < self.RECOVERY_REACH_TOLERANCE:
                    self.get_logger().info("🔄 Reached intermediate recovery voxel. Retrying original target!")
                    
                    # Switch back to the original target
                    self.current_target_idx = self.original_stuck_target
                    curr_idx = self.current_target_idx # Update local var
                    
                    self.has_reached_intermediate = True
                    self.is_deadlocked = False 
                    self.last_moving_time = time.time()


            # --------------------------------------
            # --- Deadlock Management ---
            # --------------------------------------
            if self.is_deadlocked:
                if curr_idx is not None:
                    
                    # --- CASE 1: First Deadlock (Try Recovery) ---
                    if not self.is_recovering:
                        
                        intermediate_coords = self.find_recovery_voxel(curr_idx, self.RECOVERY_MIN_DIST, self.RECOVERY_MAX_DIST)


                        if intermediate_coords is not None:
                            self.get_logger().warn(f"⚠️ Deadlock! Trying recovery at {intermediate_coords}")
                            
                            # Setup Recovery
                            self.is_recovering = True
                            self.has_reached_intermediate = False
                            self.original_stuck_target = curr_idx
                            self.current_target_idx = intermediate_coords
                            
                            self.is_deadlocked = False
                            self.last_moving_time = time.time()
                            return # Let the loop publish the new target on the next cycle
                        
                        # If it fails to find a recovery voxel, it falls through to Case 2 (Ban)
                        
                    # --- CASE 2: Deadlock during Recovery (Ban Original Target) ---
                    # Use the original target for the ban, not the intermediate one
                    target_to_ban = self.original_stuck_target if self.original_stuck_target is not None else curr_idx
                    self.get_logger().warn(f"🚫 Target {target_to_ban} in DEADLOCK (Recovery failed). Banning zone!")
                    
                    zone_indices = self.create_ignored_zone(target_to_ban)
                    for idx in zone_indices:
                        self.ignored_targets.add(idx)
                    
                    self.last_deadlock_pos = self.grid_to_world(*target_to_ban)
                    
                    # Reset COMPLETE Recovery State
                    self.is_recovering = False
                    self.has_reached_intermediate = False
                    self.original_stuck_target = None
                    self.current_target_idx = None
                    
                else:
                    self.get_logger().warn("WARNING Deadlock Phantom. Reset.")
                    
                self.is_deadlocked = False
                self.last_moving_time = time.time()
                return 
                
                
            # --------------------------------------
            # --- Check Present Target ---
            # --------------------------------------
            
            if curr_idx is not None:
                current_val = self.grid[curr_idx]       
                
                # Normal behavior: clear target if it becomes Free or Occupied
                if not self.is_recovering:
                    is_resolved = (current_val < self.VAL_FREE) or (current_val > self.VAL_OCCUPIED)
                # Recovery behavior: allow going to Free voxels, clear only if it's an Obstacle
                else:
                    is_resolved = (current_val > self.VAL_OCCUPIED)
                
                if is_resolved:
                    self.get_logger().info(f"✅ TARGET {curr_idx} Solved! (Val: {current_val}). Finding a new one.")
                    self.current_target_idx = None      
                else:
                    # Publish the target and wait
                    target_pos = self.grid_to_world(*curr_idx)
                    
                    msg = VmcTarget() 
                    msg.header.frame_id = "fr3_link0"
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.target_position.x = float(target_pos[0])
                    msg.target_position.y = float(target_pos[1])
                    msg.target_position.z = float(target_pos[2])

                    self.pub_target.publish(msg)
                    return
            
            
            # --------------------------------------
            # --- Select New Target ---
            # --------------------------------------
            
            # If we get here means that we do not have a target and we need to find one.
            # Search now a new one calling the ad-hoc function 
            best_idx, best_scores = self.find_best_unknown_target(cam_pos, allow_risky_skin=False)

            # Whenever we cannot find a target means that we should search in the ignored list
            if best_idx is None:
                
                if len(self.ignored_targets) > 0:           # Check if there is any ignored voxel
                    
                    self.get_logger().warn(f"♻️ No targets. Reset (ignore_list + Deadlock Pos).")
                    
                    # Reset the ignored voxels list and the target 
                    self.ignored_targets.clear()
                    self.last_deadlock_pos = None  
                    
                    # Now that we have done a reset we can search for a new target 
                    best_idx = self.find_best_unknown_target(cam_pos, allow_risky_skin=False)

                if best_idx is None:
                     
                    self.get_logger().warn("⚠️ No safe or ignored voxels available: Wall Skin Filter Off.")
                    best_idx = self.find_best_unknown_target(cam_pos, allow_risky_skin=True)    

            if best_idx:
                self.current_target_idx = best_idx
                target_pos = self.grid_to_world(*best_idx)
                
                
                msg = VmcTarget()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "fr3_link0" 

                msg.target_position.x = target_pos[0]
                msg.target_position.y = target_pos[1]
                msg.target_position.z = target_pos[2]
                
                msg.score_total = best_scores['total']
                msg.score_frontality = best_scores['frontality']
                msg.score_relevance = best_scores['relevance']
                msg.score_distance = best_scores['distance']
                
                self.pub_target.publish(msg)
                    
            else:
                self.get_logger().info("🎉 Completed Exploration! (Davvero finito tutto, anche i risky).", throttle_duration_sec=5.0)

        finally:
            self.decision_lock.release()


    """
    Callback for the synchronized subscribers. It receives the robot state and point cloud with aligned timestamps.
    Filters, transforms and updates the map.
    """
    def cb_synced_data(self, pcl_msg, state_msg):

        # -- 1. Setup the Transformation for the Pointcloud ---
        
        # Extract the synchronized pose 
        current_cam_pose = {
            'pos': np.array([state_msg.cam_pose.position.x, state_msg.cam_pose.position.y, state_msg.cam_pose.position.z]),
            'quat': np.array([state_msg.cam_pose.orientation.x, state_msg.cam_pose.orientation.y, state_msg.cam_pose.orientation.z, state_msg.cam_pose.orientation.w])
        }

        # Update the global variable for the other threads (Visualization Frustum, Target Logic)
        self.latest_cam_pose = current_cam_pose 

        # Setup Transformation Matrix (Camera -> World)
        r_cam = R.from_quat(current_cam_pose['quat'])
        p_cam = current_cam_pose['pos']
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = r_cam.as_matrix()
        T_world_cam[:3, 3] = p_cam
        
        
        # --- 2. Analyze Pointcloud ---
        
        # Read the points. skip_nans=True removes invalid points (NaN/Inf)
        cloud_data = pc2.read_points_numpy(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        if cloud_data.shape[0] == 0:
            return

        # Handle structured vs. unstructured array types
        if cloud_data.dtype.names:
            points_cam = np.column_stack([cloud_data['x'], cloud_data['y'], cloud_data['z']])   # Structured
        else:
            points_cam = cloud_data                                                             # Unstructured
        
        
        # --- 3. Clean Empty Space ---
        
        # "Empty" the cone in front of the camera before filling it with new obstacles.
        self.update_free_space_in_fov(T_world_cam, points_cam)

        
        # --- 4. Tranformation Points in World Frame ---
        
        # Transform (considering homogeneous vectors) and take just the vector of 3 coordinates
        # ones = np.ones((points_cam.shape[0], 1))
        # points_cam_h = np.hstack([points_cam, ones])
        
        # points_world = (T_world_cam @ points_cam_h.T).T[:, :3]
        points_world = points_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


        # --- 5. Filter Points Inside Workspace ---
        
        # Get bounds
        x_min, x_max = self.WS_BOUNDS['x']
        y_min, y_max = self.WS_BOUNDS['y']
        z_min, z_max = self.WS_BOUNDS['z']

        # Create Boolean Mask
        mask_ws = (
            (points_world[:, 0] >= x_min) & (points_world[:, 0] <= x_max) &
            (points_world[:, 1] >= y_min) & (points_world[:, 1] <= y_max) &
            (points_world[:, 2] >= z_min) & (points_world[:, 2] <= z_max)
        )

        # Apply mask to keep only points inside workspace
        valid_points = points_world[mask_ws]

        if valid_points.shape[0] == 0:
            return
        
        # --- 6. Filter Isolated Points ---
        
        res_filter = self.res_x         # Resolution of the map (can be decreased to have a better resolution)

        # Compute the indexes for the points of the pointcloud, this is the new method of distance computation. The voxel indexes are compared instead of euclidean distance computation 
        ix_f = ((valid_points[:, 0] - x_min) / res_filter).astype(int)
        iy_f = ((valid_points[:, 1] - y_min) / res_filter).astype(int)
        iz_f = ((valid_points[:, 2] - z_min) / res_filter).astype(int)
        
        # Create a matrix with these
        voxel_keys = np.column_stack((ix_f, iy_f, iz_f))
        
        # Counts the number of times the same voxel contains a point 
        _, inverse_indices, counts = np.unique(
            voxel_keys,                             # Check this matrix
            axis=0,                                 # Compare triplets: (ix, iy, iz)
            return_inverse=True,                    # Returns an array as long as the original matrix (number of voxels) 
            return_counts=True                      # Returns the count for the unique voxel occurrencies
        ) 
        
        # Mask for the voxels with a minimum number of points
        mask_keep = counts[inverse_indices] >= self.min_points_per_voxel

        # Mask applied
        valid_points = valid_points[mask_keep]

        if valid_points.shape[0] == 0:
            return     
        
        
        # --- 7. Debug Visualization ---
        
        debug_header = Header()
        debug_header.frame_id = "fr3_link0" 
        debug_header.stamp = self.get_clock().now().to_msg()
        
        # Create the pointcloud with all the filtered points
        pc2_msg = pc2.create_cloud_xyz32(debug_header, valid_points)

        
        
        # --- 8. Downsampling ---
        
        step = 1                                # Take a point each 'step' points  
        final_points = valid_points[::step]     # Downsalmpled set of points
        
        # Update repulsors' list ############################################################################################################################################################################################################################
        self.current_visible_obstacles = final_points 


        # --- 9. Mapping (Update Grid) ---
        
        # Convert in indexes for the grid
        ix = ((final_points[:, 0] - x_min) / self.res_x).astype(int)
        iy = ((final_points[:, 1] - y_min) / self.res_y).astype(int)
        iz = ((final_points[:, 2] - z_min) / self.res_z).astype(int)

        # Security clipping
        ix = np.clip(ix, 0, self.N_VOXELS_X - 1)
        iy = np.clip(iy, 0, self.N_VOXELS_Y - 1)
        iz = np.clip(iz, 0, self.N_VOXELS_Z - 1)

        # Get the unique indecex so that the we don't have redundancy
        unique_indices = np.unique(np.column_stack((ix, iy, iz)), axis=0)
        uix, uiy, uiz = unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]
        
        # Add a bucket (Rapid Increment)
        current_vals = self.grid[uix, uiy, uiz]
        new_vals = current_vals + self.HIT_INC
        
        # Ensure values stay within the defined range 
        new_vals = np.clip(new_vals, self.VAL_MIN, self.VAL_MAX)
        
        # Update the grid
        self.grid[uix, uiy, uiz] = new_vals

    
    """
    Select the best best unknown voxel to go to.
    Includes: 
    - Hysteresis: ignore voxels that have been just seen, they have to be above a safe treshold to be interesting (this ensure not to fall into the deadlock loop: Seen - Noise - New Target - Seen).
    - Wall Skin Filter: we don't want target attached to the walls (occupied voxels) because often these voxels are just noise or walls.
    - Ignore List Check
    - Safe/Risky Logic for Deadlock: divide the candidates in two groups, the safe ones are far from the deadlock target and the risky ones are close.
    - Score Logic: the target is chosen as the voxel with the highest score, which is given by distance from camera position, relevance and frontality.
    """
    def find_best_unknown_target(self, robot_pos, allow_risky_skin=False):
        
        SELECTION_TRESHOLD = self.VAL_FREE + 15
        
        # --- 1. Candidate Selection ---
        mask_unknown = (self.grid >= SELECTION_TRESHOLD) & (self.grid <= self.VAL_OCCUPIED)

        # --- 2. Wall Skin Protection ---

        # If we run out of available voxel we skip this wall skin filter
        if allow_risky_skin:

            mask_final_candidates = mask_unknown
            self.get_logger().info("⚠️ Searching inside WALL SKINS (Risky Mode).")
        
        else:
            
            # Create the mask for the occupied voxels
            mask_obstacles = (self.grid > self.VAL_OCCUPIED)

            if np.any(mask_obstacles):

                # Create a safety skin around the occupied voxels (iteration is the thickness)
                mask_skin = binary_dilation(mask_obstacles, iterations=1)

                # Valid candidates for being a target are the unknown voxels that are not in the wall skin
                mask_final_candidates = mask_unknown & (~mask_skin)
            else:

                # If there are no occupied voxels
                mask_final_candidates = mask_unknown

        # Extract indexes
        unknown_indices = np.argwhere(mask_final_candidates)


        # --- Fallback: Skin too much aggressive? ---

        if len(unknown_indices) == 0: 

            # If the Skin filter removed everything, let's try using the raw candidates.
            # Better to risk getting close to a wall than stopping completely.
            unknown_indices = np.argwhere(mask_unknown)

            if len(unknown_indices) == 0:
                return None, None # Truly nothing to see


        # --- 3. Pre-Filtering ---

        # Performance Filter
        # Check if there are too many unknown voxels: if so, select 5000 random ones
        temp_indices = unknown_indices
        if len(temp_indices) > 2000:
             rand_idx = np.random.choice(len(temp_indices), 2000, replace=False)
             temp_indices = temp_indices[rand_idx]

        # Ignored List Filter
        candidates = []     # List of candidates
        for idx in temp_indices:
            t_idx = tuple(idx)
            if t_idx not in self.ignored_targets:
                candidates.append(idx)
                
        if not candidates: 
            return None, None
        
        candidates = np.array(candidates)


        # --- 4. Deadlock Logic: Safe VS Risky ---

        final_pool = []
        is_risky = False

        if self.last_deadlock_pos is not None:

            # Vettorizzazione distanze per il deadlock
            ix, iy, iz = candidates[:, 0], candidates[:, 1], candidates[:, 2]
            pts_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
            pts_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
            pts_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
            positions = np.column_stack((pts_x, pts_y, pts_z))
            
            dists_from_deadlock = np.linalg.norm(positions - self.last_deadlock_pos, axis=1)
            safe_mask = dists_from_deadlock > self.DEADLOCK_MIN_DIST
            
            safe_candidates = candidates[safe_mask]
            
            if len(safe_candidates) > 0:
                final_pool = safe_candidates
            else:
                self.get_logger().warn("⚠️ No Safe Target Found! Choosing in the Risky Pool.")
                final_pool = candidates
                is_risky = True
        
        # If we are not in deadlock
        else:
            final_pool = candidates

        # Downsampling su final_pool
        if len(final_pool) > 500:
            rand_idx = np.random.choice(len(final_pool), 500, replace=False)
            final_pool = final_pool[rand_idx]

        # --- 5. Scoring Loop (Completamente Vettorizzato) ---

        if len(final_pool) == 0:
            return None, None

        # Garantisce la forma matriciale a 2 Dimensioni (evita crash di unpacking)
        final_pool = np.atleast_2d(final_pool)

        ix, iy, iz = final_pool[:, 0], final_pool[:, 1], final_pool[:, 2]
        pts_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        pts_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        pts_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        positions = np.column_stack((pts_x, pts_y, pts_z))

        # A. Frontality
        y_min, y_max = self.WS_BOUNDS['y']
        y_range = max(y_max - y_min, 1.0)
        y_ratios = (positions[:, 1] - y_min) / y_range
        norm_frontality = 1.0 - np.clip(y_ratios, 0.0, 1.0)

        # B. Distance
        max_dist_norm = 1.5
        real_dists = np.linalg.norm(positions - robot_pos, axis=1)
        dist_ratios = np.clip(real_dists / max_dist_norm, 0.0, 1.0)
        norm_dist_scores = 1.0 - dist_ratios

        # C. Relevance
        norm_relevance = np.array([self.get_relevance(x, y, z) for x, y, z in final_pool])

        # Scores Finale
        scores = (self.SCORE_FRONTALITY * norm_frontality) + (self.SCORE_RELEVANCE * norm_relevance) + (self.SCORE_DISTANCE * norm_dist_scores)
        
        best_idx_flat = np.argmax(scores)
        
        best_scores = {
            'total': float(scores[best_idx_flat]),
            'frontality': float(norm_frontality[best_idx_flat]),
            'relevance': float(norm_relevance[best_idx_flat]),
            'distance': float(norm_dist_scores[best_idx_flat])
        }
        
        
        # --- FORMATTAZIONE A PROVA DI BOMBA ---
        # Estrae i valori appiattendo eventuali numpy array annidati e forzando gli int nativi
        best_row = np.array(final_pool[best_idx_flat]).flatten()
        best_idx = (int(best_row[0]), int(best_row[1]), int(best_row[2]))
        
        print("\n🎯 Target Selected:", best_idx)
        
        return best_idx, best_scores
    
    
    """
    Callback to check the deadlock condition and trigger the flag to start the deadlock avoidance routine
    """
    def cb_deadlock(self, msg):

        if self.node_start_time is None:
            self.node_start_time = time.time()
            self.last_moving_time = time.time() 
            return

        elapsed_time = time.time() - self.node_start_time

        if elapsed_time < self.TIME_TRESHOLD_DEADLOCK:
            self.last_moving_time = time.time()
            return
   
        if len(msg.data) < 2:       # That means that one of the two values is missing
            return
        
        # Extract the max values
        current_vel = msg.data[0]
        current_tau = msg.data[1]
        
        # Flag to determine if the deadlock is occurring (torques probably are considering also the gravity compensation contribute, then avoid using torque control for now)
        is_static = (current_vel < self.DEADLOCK_VEL_EPS) #=and (current_tau < self.DEADLOCK_TORQUE_EPS)=#
        
        # We are still moving
        if not is_static:
            self.last_moving_time = time.time()                 # Reset timer to check if the deadlock is taking too much timer
            self.is_deadlocked = False
        
        # We are not moving
        else:
            time_stuck = time.time() - self.last_moving_time    # How much time we are still
            
            # If we exceed the possible time to stay in deadlock without triggering the deadlock routine 
            if time_stuck > self.DEADLOCK_TIME_THRESHOLD:
                if not self.is_deadlocked and self.current_target_idx is not None:
                    self.get_logger().warn(f"WARNING DEADLOCK DETECTED! Blocked for {time_stuck:.1f}s. Resetting target.")
                    self.is_deadlocked = True

    
    """
    Timer callback: Pubblica la lista degli ostacoli correnti (Flattened).
    Format: [x1, y1, z1, x2, y2, z2, ...]
    """
    def publish_obstacles_worker(self):
        
        # 1. Ottieni gli ostacoli ottimizzati (Spatial Hashing + Sorting)
        optimized_obstacles = self.get_downsampled_obstacles()
        
        msg = VmcObstacles()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0" 
        
        # Insert in the message the obstacle list of repulsors
        for obs_pos in optimized_obstacles:
            p = Point()
            p.x = obs_pos[0]
            p.y = obs_pos[1]
            p.z = obs_pos[2]
            msg.obstacles.append(p)

        self.pub_obstacles.publish(msg)
        
        # --- Repulsor Visualization ---
        
        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "active_repulsors"
        marker.id = 999
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.scale = Vector3(x=0.05, y=0.05, z=0.05) 
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.7) 

        for p in msg.obstacles:
            marker.points.append(p) 

        self.pub_repulsors_viz.publish(marker)
    
    
    """
    Finds a FREE voxel at a specific distance from the target.
    OTTIMIZZATA: Estrazione matriciale pura (Zero cicli For)
    """
    def find_recovery_voxel(self, target_idx, min_dist, max_dist):
        cx, cy, cz = target_idx
        target_pos = self.grid_to_world(cx, cy, cz)

        # 1. Limiti del Bounding Box (Cubo in indici)
        r_vox_max = int(np.ceil(max_dist / self.res_x))
        
        ix_min = max(0, cx - r_vox_max)
        ix_max = min(self.N_VOXELS_X - 1, cx + r_vox_max)
        iy_min = max(0, cy - r_vox_max)
        iy_max = min(self.N_VOXELS_Y - 1, cy + r_vox_max)
        iz_min = max(0, cz - r_vox_max)
        iz_max = min(self.N_VOXELS_Z - 1, cz + r_vox_max)

        if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
            return None

        # =========================================================
        # 2. ESTRAZIONE VETTORIZZATA
        # =========================================================
        # Ritagliamo il blocco 3D di interesse dalla griglia principale
        # (+1 perché lo slicing in Python esclude l'estremo superiore)
        sub_grid = self.grid[ix_min:ix_max+1, iy_min:iy_max+1, iz_min:iz_max+1]
        
        # Troviamo TUTTI gli indici dei voxel LIBERI in una frazione di millisecondo
        local_free_indices = np.argwhere(sub_grid <= self.VAL_FREE)
        
        if len(local_free_indices) == 0:
            self.get_logger().warn("Nessun voxel libero nel raggio di recovery.")
            return None

        # Convertiamo gli indici locali in indici globali della mappa
        global_indices = local_free_indices + np.array([ix_min, iy_min, iz_min])

        # =========================================================
        # 3. FILTRO DISTANZA A CIAMBELLA (EXACT DISTANCE)
        # =========================================================
        # Calcoliamo le coordinate mondo per tutti i candidati in parallelo
        pts_x = self.WS_BOUNDS['x'][0] + (global_indices[:, 0] + 0.5) * self.res_x
        pts_y = self.WS_BOUNDS['y'][0] + (global_indices[:, 1] + 0.5) * self.res_y
        pts_z = self.WS_BOUNDS['z'][0] + (global_indices[:, 2] + 0.5) * self.res_z
        
        # Distanza al quadrato (evita np.linalg.norm che è lenta)
        dists_sq = (pts_x - target_pos[0])**2 + (pts_y - target_pos[1])**2 + (pts_z - target_pos[2])**2
        
        # Maschera: teniamo solo i voxel tra min_dist e max_dist
        valid_mask = (dists_sq >= min_dist**2) & (dists_sq <= max_dist**2)
        
        final_candidates = global_indices[valid_mask]

        if len(final_candidates) == 0:
            self.get_logger().warn("Nessun voxel rispetta i limiti di distanza min/max.")
            return None

        # =========================================================
        # 4. SCELTA FINALE
        # =========================================================
        # Peschiamo casualmente UN solo voxel dalla lista dei sopravvissuti perfetti
        choice_idx = np.random.choice(len(final_candidates))
        best_voxel = final_candidates[choice_idx]
        
        return (int(best_voxel[0]), int(best_voxel[1]), int(best_voxel[2]))
    
    
    
    """
    Finds all the voxels around the target voxel that are in a sphere of radius R to
    create an IgnoredZone. It returns the list of all the indeces found.
    """
    def create_ignored_zone(self, center_idx):
        
        cx, cy, cz = center_idx                         # Center of the ignored target voxel
        
        r_vox = int(np.ceil(                            # Radius of the zone computed in number of voxels
            self.DEADLOCK_ZONE_RADIUS / self.res_x
        ))
        
        # Bounding box to identify for the ignored zone (so we don't need to use all the grid)
        ix_min, ix_max = max(0, cx - r_vox), min(self.N_VOXELS_X, cx + r_vox + 1)
        iy_min, iy_max = max(0, cy - r_vox), min(self.N_VOXELS_Y, cy + r_vox + 1)
        iz_min, iz_max = max(0, cz - r_vox), min(self.N_VOXELS_Z, cz + r_vox + 1)
        
        # Create a grid for local coordinates in the BB
        x_range = np.arange(ix_min, ix_max)
        y_range = np.arange(iy_min, iy_max)
        z_range = np.arange(iz_min, iz_max)
        
        # Meshgrid 3D: grid of 3D coordinates composed by 1D vectors. X, Y, Z are the coordinates for the points in the BB
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Compute the distance from the center for each voxel
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        
        # Create the mask for the voxels inside the BB
        mask = dist_sq <= r_vox**2
        
        # Extract the valid indexes 
        valid_x = X[mask]
        valid_y = Y[mask]
        valid_z = Z[mask]
        
        # Create list of tuples for the set of ignored voxels
        indices_to_ignore = []

        # Filter for unknown voxels only
        for x, y, z in zip(valid_x, valid_y, valid_z):
            
            val = self.grid[x, y, z]
            
            # If this is unknown, then it will be ignored, otherwise we skip it
            if val > self.VAL_FREE and val < self.VAL_OCCUPIED:
                indices_to_ignore.append((int(x), int(y), int(z)))

        # Create the zone and append it to the ignore_zones array
        if len(indices_to_ignore) > 0:
            new_zone = IgnoredZone(indices_to_ignore)
            self.ignored_zones.append(new_zone)
            self.get_logger().warn(f"🚫 Created IGNOREZONE with {len(indices_to_ignore)} voxels (R={self.DEADLOCK_ZONE_RADIUS}m)")
            
        return indices_to_ignore


   
    """
    Uses the analogy to the water buckets to update the empty space in the grid.
    Do not iterate over all the voxels in the grid, skip the voxels where the water in it 
    is zero, this means that they are more likely to be surely empty and then there's no 
    need to check there.
    Z-BUFFER OCCLUSION: Voxels hidden behind obstacles will not be cleared!
    """
    def update_free_space_in_fov(self, T_world_cam, points_cam):
                
        # --- 1. Pre-Filter ---
        ix_min, ix_max, iy_min, iy_max, iz_min, iz_max = self.get_frustum_aabb(T_world_cam)

        sub_grid = self.grid[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]
        if sub_grid.size == 0: return
        
        local_indices = np.argwhere((sub_grid > self.VAL_MIN) & (sub_grid < self.VAL_LOCK))
        if len(local_indices) == 0: return

        # Compute World Coordinates 
        ix = local_indices[:, 0] + ix_min
        iy = local_indices[:, 1] + iy_min
        iz = local_indices[:, 2] + iz_min
        
        # --- 2. Geometric Check ---
        pts_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        pts_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        pts_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z

        pts_world = np.column_stack((pts_x, pts_y, pts_z))
        
        T_cam_world = np.linalg.inv(T_world_cam)
        pts_cam = pts_world @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        x_c, y_c, z_c = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]


        # --- 3. Z-BUFFER OCCLUSION CULLING (NEW) ---
        
        # Parametri Z-Buffer (risoluzione "schermo")
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)

        # Inizializziamo lo schermo virtuale con la distanza massima
        z_buffer = np.full((self.GRID_W, self.GRID_H), self.MAX_DEPTH)

        # Proiettiamo la PointCloud (gli Ostacoli fisici) nello Z-Buffer
        if points_cam is not None and len(points_cam) > 0:
            pc_z = points_cam[:, 2]
            
            # Consideriamo solo i punti davanti alla camera
            valid_z = pc_z > 0.01 
            pc_x = points_cam[valid_z, 0]
            pc_y = points_cam[valid_z, 1]
            pc_z = pc_z[valid_z]

            # Coordinate Proiettive (U, V) -> da -1.0 a +1.0
            pc_u = pc_x / (pc_z * tan_h)
            pc_v = pc_y / (pc_z * tan_v)

            # Mappatura in "pixel" dello schermo virtuale
            pc_bin_u = ((pc_u + 1.0) / 2.0 * self.GRID_W).astype(int)
            pc_bin_v = ((pc_v + 1.0) / 2.0 * self.GRID_H).astype(int)

            # Filtro per i pixel dentro lo schermo
            valid_bins = (pc_bin_u >= 0) & (pc_bin_u < self.GRID_W) & (pc_bin_v >= 0) & (pc_bin_v < self.GRID_H)
            
            # Registriamo la profondità MINIMA dell'ostacolo per ogni pixel
            np.minimum.at(z_buffer, (pc_bin_u[valid_bins], pc_bin_v[valid_bins]), pc_z[valid_bins])

        # --- 4. Frustum & Occlusion Check for Voxels ---
        
        # Check base (il voxel è geometricamente dentro la piramide visiva?)
        mask_fov_geom = (z_c > 0.05) & (z_c < self.MAX_DEPTH)
        
        valid_indices_mask = np.zeros(len(x_c), dtype=bool)

        if np.any(mask_fov_geom):
            v_x = x_c[mask_fov_geom]
            v_y = y_c[mask_fov_geom]
            v_z = z_c[mask_fov_geom]

            # Proiettiamo i Voxel nello stesso schermo virtuale
            v_u = v_x / (v_z * tan_h)
            v_v = v_y / (v_z * tan_v)

            v_bin_u = ((v_u + 1.0) / 2.0 * self.GRID_W).astype(int)
            v_bin_v = ((v_v + 1.0) / 2.0 * self.GRID_H).astype(int)

            # Il voxel è visibile nello schermo?
            in_screen = (v_bin_u >= 0) & (v_bin_u < self.GRID_W) & (v_bin_v >= 0) & (v_bin_v < self.GRID_H)

            # Tolleranza: essendo i voxel dei cubi, aggiungiamo la diagonale del voxel al controllo di occlusione
            margin = self.TARGET_VOXEL_SIZE * 1.5 

            # Maschera temporanea per i voxel non occlusi
            visible = np.zeros(len(v_z), dtype=bool)
            
            u_in = v_bin_u[in_screen]
            v_in = v_bin_v[in_screen]
            z_in = v_z[in_screen]

            # IL CUORE DELL'OCCLUSIONE: La profondità del Voxel deve essere <= della profondità dell'ostacolo registrata nel pixel!
            visible[in_screen] = z_in <= (z_buffer[u_in, v_in] + margin)

            # Assegniamo i risultati alla maschera globale
            valid_indices_mask[mask_fov_geom] = visible
            
            
        # --- 5. Water Bucket Analogy Application ---
        if np.any(valid_indices_mask):
            final_ix = ix[valid_indices_mask]
            final_iy = iy[valid_indices_mask]
            final_iz = iz[valid_indices_mask]
            
            current_vals = self.grid[final_ix, final_iy, final_iz]
            new_vals = np.clip(current_vals - self.MISS_DEC, self.VAL_MIN, self.VAL_MAX)
            self.grid[final_ix, final_iy, final_iz] = new_vals


    """
    Check all the ignored zones and if the 10% of them has been seen then free the entire zone
    """
    def check_and_liberate_zones(self):
        
        if not self.ignored_zones:
            return

        zones_to_remove = []        # List of zones to remove from the ignored zone list
        
        # Iterate over all the ignored zones to check
        for zone in self.ignored_zones:
            
            indices = np.array(zone.member_indices)
            if len(indices) == 0: continue
            
            # Estrae tutti i valori della grid per questi indici in UN SOLO PASSAGGIO C
            vals = self.grid[indices[:, 0], indices[:, 1], indices[:, 2]]
            
            # Crea la maschera per i risolti e li somma istantaneamente
            mask_resolved = (vals < self.VAL_FREE) | (vals > self.VAL_OCCUPIED)
            resolved_count = np.sum(mask_resolved)
            
            # Percentage Computation
            percentage = resolved_count / zone.total_count
            
            # Trigger 
            if percentage >= 0.10: # 10% Free Threshold
                self.get_logger().info(f"🔓 Zone free! (Seen {resolved_count}/{zone.total_count} voxels).")
                
                # Use set.difference_update for bulk removal (molto più veloce)
                self.ignored_targets.difference_update(zone.member_indices)
                
                zones_to_remove.append(zone)

        # Clean List of Zones
        for z in zones_to_remove:
            try:
                self.ignored_zones.remove(z)
            except ValueError:
                pass 



    """
    Get the relevance for the given voxel. The more are the neighbors unknown voxels the higher is the relevance.
    """
    def get_relevance(self, ix, iy, iz):
        
        # Get the 3x3x3 cube around the voxel
        x_min, x_max = max(0, ix - 1), min(self.N_VOXELS_X, ix + 2)
        y_min, y_max = max(0, iy - 1), min(self.N_VOXELS_Y, iy + 2)
        z_min, z_max = max(0, iz - 1), min(self.N_VOXELS_Z, iz + 2)
        
        neighborhood = self.grid[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Count how many unknowns there are and remove the central voxel
        mask_unknown_neighbors = (neighborhood >= self.VAL_FREE) & (neighborhood <= self.VAL_OCCUPIED)
        unknown_neighbors = np.sum(mask_unknown_neighbors)
        if (self.grid[ix, iy, iz] >= self.VAL_FREE) and (self.grid[ix, iy, iz] <= self.VAL_OCCUPIED):
             unknown_neighbors -= 1
             
        # Normalize the relevance
        return max(0, unknown_neighbors) / 26.0


    """
    Extracts occupied voxels from the persistent grid and filters them
    to maintain a minimum distance (e.g., 10cm) between them.
    Uses 'Spatial Hashing' for ultra-fast processing:
        We divide the space into cells of size 'min_distance_rep'.
        Points falling into the same cell will have the same integer key.
    """
    def get_downsampled_obstacles(self):
        
        # Find all the indexes of occupied voxels
        idx_occupied = np.argwhere(self.grid >= self.VAL_OCCUPIED)
        
        if len(idx_occupied) == 0:
            return []

        # Vectorization and transform in world coordinates
        ix = idx_occupied[:, 0]
        iy = idx_occupied[:, 1]
        iz = idx_occupied[:, 2]

        points_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        points_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        points_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        
        # Occupied voxels matrix
        points = np.column_stack((points_x, points_y, points_z))
        
        
        # --- 1. Spatial Hashing ---
        
        keys = np.floor(points / self.min_distance_rep).astype(int)
        
        # # We use a dictionary to keep only one point for each spatial key
        # unique_obstacles = {}
        
        # # Iterate and fill the dictionary.
        # for i in range(points.shape[0]):
            
        #     # The key is a tuple (k_x, k_y, k_z)
        #     k = tuple(keys[i])
            
        #     # If this key is not represented then do it now
        #     if k not in unique_obstacles:
        #         unique_obstacles[k] = points[i]
        
        # # List of spatially distributed obstacles
        # sparse_obstacles = list(unique_obstacles.values())
        _, unique_indices = np.unique(keys, axis=0, return_index=True)

        # List of spatially distributed obstacles
        sparse_obstacles = points[unique_indices].tolist()

        # --- 2. Distance Sorting & Limiting ---

        # If we have more obstacles than allowed, we prioritize the closest ones
        if len(sparse_obstacles) > self.MAX_OBSTACLES:
            
            # We need the robot position to compute distances
            if self.latest_cam_pose is not None:
                robot_pos = self.latest_cam_pose['pos']
                
                # Convert to numpy for fast vectorized math
                obs_array = np.array(sparse_obstacles)
                
                # Compute Euclidean distances from robot/camera
                dists = np.linalg.norm(obs_array - robot_pos, axis=1)
                
                # Get the indices that would sort the array by distance
                sorted_indices = np.argsort(dists)
                
                # Take only the top MAX_OBSTACLES indices
                top_indices = sorted_indices[:self.MAX_OBSTACLES]
                
                # Return the subset
                return obs_array[top_indices].tolist()
            
            else:
                # Fallback if pose is unknown: just cut the list arbitrarily
                return sparse_obstacles[:self.MAX_OBSTACLES]

        return sparse_obstacles


    def grid_to_world(self, ix, iy, iz):
        x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        return np.array([x, y, z])
        

    """
    Compute the Axis-Aligned Bounding Box (AABB) of the frustum in grid coordinates.
    This is used to limit the voxel search only to the area the camera is looking at,
    instead of checking the entire workspace.
    """
    def get_frustum_aabb(self, T_world_cam):
        
        # --- 1. Define frustum lenght [Camera Frame]
        z_near = 0.07
        z_far = self.MAX_DEPTH
        
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)

        # Frustum box (Near plane + Far plane)
        corners_cam = [
            # Origin (useful approximation)
            [0, 0, 0], 
            # Near Plane
            [-z_near * tan_h, -z_near * tan_v, z_near],
            [ z_near * tan_h, -z_near * tan_v, z_near],
            [ z_near * tan_h,  z_near * tan_v, z_near],
            [-z_near * tan_h,  z_near * tan_v, z_near],
            # Far Plane
            [-z_far * tan_h, -z_far * tan_v, z_far],
            [ z_far * tan_h, -z_far * tan_v, z_far],
            [ z_far * tan_h,  z_far * tan_v, z_far],
            [-z_far * tan_h,  z_far * tan_v, z_far],
        ]
        corners_cam = np.array(corners_cam)
        
        # --- 2. Transform [World Frame]
        # Add column of 1s for homogeneous coordinates
        ones = np.ones((corners_cam.shape[0], 1))
        corners_cam_h = np.hstack([corners_cam, ones])
        
        # Apply T_world_cam transformation to points
        corners_world = (T_world_cam @ corners_cam_h.T).T[:, :3]
        
        # --- 3. Convert to Grid Indices
        ix = ((corners_world[:, 0] - self.WS_BOUNDS['x'][0]) / self.res_x).astype(int)
        iy = ((corners_world[:, 1] - self.WS_BOUNDS['y'][0]) / self.res_y).astype(int)
        iz = ((corners_world[:, 2] - self.WS_BOUNDS['z'][0]) / self.res_z).astype(int)
        
        # 4. Find Min and Max (with clipping to stay within grid bounds)
        min_ix, max_ix = np.min(ix), np.max(ix)
        min_iy, max_iy = np.min(iy), np.max(iy)
        min_iz, max_iz = np.min(iz), np.max(iz)
        
        return (
            max(0, min_ix), min(self.N_VOXELS_X, max_ix + 1),
            max(0, min_iy), min(self.N_VOXELS_Y, max_iy + 1),
            max(0, min_iz), min(self.N_VOXELS_Z, max_iz + 1)
        )
        
     
    """
    Pure geometric check function.
    Returns True if a volume (sphere enclosing the voxel/node) touches the frustum.
    """
    def check_frustum_overlap(self, center_in_cam_frame, radius):
        
        x_c, y_c, z_c = center_in_cam_frame
        
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)

        # Check Depth
        if (z_c - radius) > self.MAX_DEPTH: return False
        if (z_c + radius) < self.MIN_DEPTH: return False

        # Check Lateral sides (adding radius as margin)
        if np.abs(x_c) > (z_c * tan_h + radius): return False
        if np.abs(y_c) > (z_c * tan_v + radius): return False
        
        return True


    
    # ============================================================================================================================================
    # --- Visualization Functions ---
    # ============================================================================================================================================

    def publish_experiment_config(self):
        msg = VmcMapConfig()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.workspace_bounds_x = [float(self.WS_BOUNDS['x'][0]), float(self.WS_BOUNDS['x'][1])]
        msg.workspace_bounds_y = [float(self.WS_BOUNDS['y'][0]), float(self.WS_BOUNDS['y'][1])]
        msg.workspace_bounds_z = [float(self.WS_BOUNDS['z'][0]), float(self.WS_BOUNDS['z'][1])]
        
        msg.voxel_size = float(self.TARGET_VOXEL_SIZE)
        msg.hit_increment = float(self.HIT_INC)
        msg.miss_decrement = float(self.MISS_DEC)
        msg.val_occupied_thresh = int(self.VAL_OCCUPIED)
        msg.val_free_thresh = int(self.VAL_FREE)

        msg.score_relevance = float(self.SCORE_RELEVANCE)
        msg.score_distance = float(self.SCORE_DISTANCE)
        msg.score_frontality = float(self.SCORE_FRONTALITY)
        
        self.pub_log_config.publish(msg)
        self.get_logger().info("💾 Configurazione Log pubblicata.")


    """ 
    Draw the frustum attached to the camera and moving with it.
    """
    def publish_frustum(self):
        
        marker = Marker()
        marker.header.frame_id = "fr3_link0" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "camera_frustum"
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale = Vector3(x=0.005, y=0.0, z=0.0)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8) # Cyan

        # Setup Trasformation Matrix
        r_cam = R.from_quat(self.latest_cam_pose['quat'])
        t_cam = self.latest_cam_pose['pos']
        R_wc = r_cam.as_matrix() 
        
        # Geometry of the Frustum
        z = self.MAX_DEPTH 
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)
        x = z * tan_h
        y = z * tan_v
        
        # Points in Camera Frame
        p_c_origin = np.array([0, 0, 0])
        p_c_tl = np.array([-x, -y, z]) 
        p_c_tr = np.array([ x, -y, z]) 
        p_c_br = np.array([ x,  y, z]) 
        p_c_bl = np.array([-x,  y, z]) 
        
        corners_cam = [p_c_origin, p_c_tl, p_c_tr, p_c_br, p_c_bl]
        
        # Trasformation
        pts_w = []
        for p in corners_cam:
            # Rotation + Traslation
            p_rot = R_wc @ p 
            p_final = p_rot + t_cam
            pts_w.append(p_final)

        origin, tl, tr, br, bl = pts_w
        
        lines = [
            origin, tl, origin, tr, origin, br, origin, bl, # Rays
            tl, tr, tr, br, br, bl, bl, tl                  # Rectangular Base
        ]
        
        for p in lines:
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            
        self.pub_frustum.publish(marker)


    """ 
    Callback to launch the publishers for the grid.
    It is slower because we don't want to stess the PC and then we use a slower timer because we don't need
    to be so fast with this.
    """
    def publish_grid_worker(self):
        threading.Thread(target=self._heavy_viz_task).start()


    """
    Callback that publishes:
    - Grid
    - Workspace boundaries as lines
    - Exploration status text  
    """
    def _heavy_viz_task(self):
        self.publish_voxel_grid()
        self.publish_workspace_boundary()
        self.publish_exploration_status()

    
    """ 
    Callback to publish the furstum.
    It is faster because we need to have the frustum move with the robot realtime.
    """
    def publish_fast_viz(self):
        if self.latest_cam_pose is not None:
            self.publish_frustum()
    
    

    """
    Publish the voxel grid.
    Visualization:
    - Unknown: Gray
    - Occupied: Red
    - Target: Yellow
    - Ignored: Blue
    """
    def publish_voxel_grid(self):
        
        # Marker Setup
        marker = Marker()
        marker.header.frame_id = "fr3_link0" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "voxel_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        
        # Voxel Scale
        marker.scale = Vector3(x=self.res_x, y=self.res_y, z=self.res_z)
        
        
        # --- Colors Definition ---
        
        # Unknown: Gray
        c_unknown = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.2)

        # Free: Transparent
        # c_free = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.05) 
        
        # Occupied: Red
        c_occupied = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) 
        
        # Target: Yellow
        c_target   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        # Ignored: Blue
        c_ignored = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.1)


        # --- 1. Unknown Space---
        
        # Unknown: VAL_FREE < Water Level < VAL_OCCUPIED
        idx_unknown = np.argwhere((self.grid >= self.VAL_FREE) & (self.grid <= self.VAL_OCCUPIED))

        # Downsampling 
        step_unk = 1
        if len(idx_unknown) > 10000:
            step_unk = 1 
        elif len(idx_unknown) > 5000:
            step_unk = 1 

        idx_unknown_viz = idx_unknown[::step_unk]

        for ix, iy, iz in idx_unknown_viz:

            # Skip so we don't substitute blue markers
            if (int(ix), int(iy), int(iz)) in self.ignored_targets:
                continue

            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_unknown)
        

        # --- 2. Free Space (Removed/Transparent) ---

        """
        # Free: VAL_MIN < Water Level < VAL_FREE
        idx_free = np.argwhere((self.grid < self.VAL_FREE) & (self.grid > 0))

        # Downsampling
        step_free = 1
        if len(idx_unknown) > 10000:
            step_free = 1 
        elif len(idx_unknown) > 5000:
            step_free = 1 

        idx_free_viz = idx_free[::2] 

        for ix, iy, iz in idx_free_viz:
            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_free)
        """

        # --- 3. Occupied Space ---
        
        # Occupied: VAL_OCCUPIED < Water Level
        idx_occupied = np.argwhere(self.grid > self.VAL_OCCUPIED)
        
        # Downsampling (if there are too many voxels to check)
        step_occ = 1
        if len(idx_occupied) > 5000:
             step_occ = 2
        
        idx_occupied_viz = idx_occupied[::step_occ]

        for ix, iy, iz in idx_occupied_viz:

            # Skip so we don't substitute blue markers
            if (ix, iy, iz) in self.ignored_targets:
                continue

            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_occupied)


        # --- 4. Target ---
        
        if self.current_target_idx is not None:
            
            # Take target indexes
            t_ix, t_iy, t_iz = self.current_target_idx
            
            # Take target coordinates
            p_target = self.grid_to_world(t_ix, t_iy, t_iz)
            
            # Update the target voxel color
            marker.points.append(Point(x=p_target[0], y=p_target[1], z=p_target[2]))
            marker.colors.append(c_target)
            

        # --- 5. Ignored ---

        if len(self.ignored_targets) > 0:
            for (ix, iy, iz) in self.ignored_targets:
                p = self.grid_to_world(ix, iy, iz)
                marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
                marker.colors.append(c_ignored)

        self.pub_viz.publish(marker)


    """
    Publish the Exploration Satus Text.
    Computes the percentage of explored voxels (Free + Occupied) and publish a 3D text.
    """
    def publish_exploration_status(self):
     
        # --- 1. Compute Statistics ---
        
        total_voxels = self.N_VOXELS_X * self.N_VOXELS_Y * self.N_VOXELS_Z                  # Total number of voxels
        
        mask_explored = (self.grid < self.VAL_FREE) | (self.grid > self.VAL_OCCUPIED)       # Set to 1 for each non-empty voxel
        
        explored_count = np.count_nonzero(mask_explored)                                    # Total number of non-empty voxels
        
        # Percentage Computation
        percentage = (explored_count / total_voxels) * 100.0


        # --- 2. Creation Text Marker ---
        
        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "status_text"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        x_max = self.WS_BOUNDS['x'][1]
        y_center = (self.WS_BOUNDS['y'][0] + self.WS_BOUNDS['y'][1]) / 2.0
        z_min = self.WS_BOUNDS['z'][0]

        marker.pose.position.x = x_max + 0.1
        marker.pose.position.y = y_center
        marker.pose.position.z = z_min 
        
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.scale.x = 0.15
        marker.scale.y = 0.0
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.text = f"Found {explored_count} / {total_voxels} Voxels"
        
        if percentage >= self.MINIMUM_EXPLORED_PERCENTAGE:
            self.get_logger().info("Reached 95%", " of the exploration at t: %s", self.get_clock().now().to_string)
            self.pub_completion_time.publish(self.get_clock().now().to_msg())


        self.pub_status_text.publish(marker)


    """
    Publish the workspace boundaries as lines.
    """
    def publish_workspace_boundary(self):

        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace_bounds"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        
        marker.scale.x = 0.005                                      # Line Width
        
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.4)        # Color

        # Coordinates Limits
        x_min, x_max = self.WS_BOUNDS['x']
        y_min, y_max = self.WS_BOUNDS['y']
        z_min, z_max = self.WS_BOUNDS['z']

        # Box Vertices
        p0 = Point(x=x_min, y=y_min, z=z_min)
        p1 = Point(x=x_max, y=y_min, z=z_min)
        p2 = Point(x=x_max, y=y_max, z=z_min)
        p3 = Point(x=x_min, y=y_max, z=z_min)
        
        p4 = Point(x=x_min, y=y_min, z=z_max)
        p5 = Point(x=x_max, y=y_min, z=z_max)
        p6 = Point(x=x_max, y=y_max, z=z_max)
        p7 = Point(x=x_min, y=y_max, z=z_max)

        # Define the 12 lines
        lines = [
            # Base 
            p0, p1,  p1, p2,  p2, p3,  p3, p0,
            # Roof
            p4, p5,  p5, p6,  p6, p7,  p7, p4,
            # Vertical Columns
            p0, p4,  p1, p5,  p2, p6,  p3, p7
        ]
        
        marker.points = lines
        self.pub_ws_bounds.publish(marker)


# ======================================================================
# --- Main Function ---
# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = MapLogicNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
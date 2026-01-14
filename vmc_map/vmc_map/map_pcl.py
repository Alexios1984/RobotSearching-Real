import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import message_filters 
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

# ROS Messages
from vmc_interfaces.msg import ObjectDetection3DArray, VmcRobotState, VmcControlTarget
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray

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
        
        super().__init__('map_pcl')

        self.node_start_time = None                 # Used not to go in deadlock at time 0.0

        self.current_target_idx = None              # Current target voxel index (ix, iy, iz)
        
        self.decision_lock = threading.Lock()       # Lock for decision loop
        
        
        # ============================== 
        # --- Workspace Configuration ---
        # ==============================
        
        self.WS_BOUNDS = {                  # Definition of the global limits (World Frame)
            'x': [-0.2, 0.5],   
            'y': [ 0.45, 0.9],
            'z': [ 0.2, 0.85]
        }

        self.TARGET_VOXEL_SIZE = 0.06             # Voxel size in meters
        
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
        self.VAL_MAX = 100                  # Full bucket
        self.VAL_UNKNOWN = 50               # Default uncertainty level
        self.VAL_OCCUPIED = 80              # Min Threshold to say "It's occupied!"
        self.VAL_FREE = 20                  # Max Threshold to say "It's free for sure"
        self.VAL_LOCK = self.VAL_MAX        # Once full, we don't decrease it anymore (for stability)
        
        
        # --- Bucket Quantities ---
        
        self.HIT_INC = 20                   # How much water we add if we see an obstacle (Rapid Increment)
        self.MISS_DEC = 10                  # How much water we lose if we see free space (Slow Decay)


        # --- Voxel Grid Initialization --- 
       
        self.grid = np.full(
            (self.N_VOXELS_X, self.N_VOXELS_Y, self.N_VOXELS_Z), 
            self.VAL_UNKNOWN, 
            dtype=np.int16
        )        
        
        
        # --- Camera Configuration ---
        
        # Field of View from the Realsense D405 specs
        self.FOV_H = np.deg2rad(87.0) 
        self.FOV_V = np.deg2rad(58.0)
        self.MAX_DEPTH = 0.40               # Max depth to consider (meters). Less than the sensor max for performance (for safety) 


        # --- Logic Configuration ---
        
        self.MAX_OBSTACLES = 100                        # Max number of repulsors to consider (for performance). Selected among the points of the point cloud.

        self.latest_cam_pose = None                     # Latest Camera Pose (from Robot State)
        
        self.current_visible_obstacles = []             # Current visible obstacles (updated from PCL callback)

        self.cb_group = ReentrantCallbackGroup()        # Lock for the group where all can access at the same time (for parallelization). The default group is the MutuallyExsclusiveCallbackGroup that means that just one per time can be executed

        self.min_distance_rep = 0.10                    # Minimum distance between repulsors 
        
        # --- Deadlock Configuration ---
        
        self.TIME_TRESHOLD_DEADLOCK = 5.0               # Time to wait before starting checking for deadlocks (to avoid initial ignored zone)

        self.DEADLOCK_VEL_EPS = 0.02                    # Min treshold for the links velocity
        self.DEADLOCK_TORQUE_EPS = 0.20                 # Min torque for the joints
        self.DEADLOCK_TIME_THRESHOLD = 1.0              # Time below the tresholds needed to trigger the deadlock
        self.DEADLOCK_ZONE_RADIUS = 0.10                # Radius of the ignored zone to consider from the deadlock target

        self.DEADLOCK_MIN_DIST = 0.40                   # Minimum distance from last deadlock to pick a new target (for avoiding stucking again in the same area)
        self.last_deadlock_pos = None                   # Last deadlock target coordinates
        
        self.last_moving_time = time.time()             # Last time we moved (used to compute the time spent in a configuration)
        self.is_deadlocked = False                      # Flag for the deadlock
        
        self.ignored_targets = set()                    # Set for the ignored target voxels
        self.ignored_targets.clear()

        self.ignored_zones = []                         # List of ignored zones


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
            slop=0.05               # slop=0.05: tollerance time
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
            VmcControlTarget, '/vmc/target_obstacles', 10)

        self.pub_debug_pcl = self.create_publisher(                 # Pointcloud filtered (within the workspace)
            PointCloud2, '/vmc/debug_obstacles', 10)

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


        # ============================
        # --- Timers ---
        # ============================
        
        # Timer Visualization 
        self.timer_grid_viz = self.create_timer(0.5, self.publish_grid_worker, callback_group=self.cb_group)
        self.timer_fast_viz = self.create_timer(0.01, self.publish_fast_viz, callback_group=self.cb_group)

        # Logic Timer 
        self.timer = self.create_timer(0.001, self.decision_loop, callback_group=self.cb_group)

    
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
        
        
        # --- 2. Clean Empty Space ---
        
        # "Empty" the cone in front of the camera before filling it with new obstacles.
        self.update_free_space_in_fov(T_world_cam)

        # --- 3. Analyze Pointcloud ---
        
        # Read the points. skip_nans=True removes invalid points (NaN/Inf)
        cloud_data = pc2.read_points_numpy(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        if cloud_data.shape[0] == 0:
            return

        # Handle structured vs. unstructured array types
        if cloud_data.dtype.names:
            points_cam = np.column_stack([cloud_data['x'], cloud_data['y'], cloud_data['z']])   # Structured
        else:
            points_cam = cloud_data                                                             # Unstructured


        # --- 4. Tranformation Points in World Frame ---
        
        # Transform (considering homogeneous vectors) and take just the vector of 3 coordinates
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_h = np.hstack([points_cam, ones])
        
        points_world = (T_world_cam @ points_cam_h.T).T[:, :3]


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
        
        radius = 0.02          # If there is a point in a sphere of 2 cm radius with few neighbors, then it is just noise
        min_neighbors = 6      # Minimum number of neighbors to be considered a real point

        # Point is noise
        if len(valid_points) >= min_neighbors:
            
            # Create a KD-tree for efficient nearest neighbor searches
            tree = cKDTree(valid_points)                

            # Query the KD-tree for the k-th nearest neighbor within the specified radius
            dists, _ = tree.query(                      
                valid_points,   
                k=min_neighbors, 
                distance_upper_bound=radius
            ) 
            
            mask_clean = dists[:, -1] != float('inf')   # Create a mask to identify points that have at least 'min_neighbors' within 'radius'

            """
            `dists` is an array of distances to the `k` nearest neighbors for each point in `valid_points`. If a point has fewer than `k` 
            neighbors within `distance_upper_bound`, the remaining distances for that point will be `inf`.

            In the context of the provided code, `dists[:, -1]` is used to check the distance to the `k`-th nearest neighbor. 
            If this distance is `inf`, it means the point does not have `k` neighbors within the specified `radius`, and thus it's considered 
            an isolated point (noise).
            """
                   
            valid_points = valid_points[mask_clean]

        if valid_points.shape[0] == 0:
            return

        # --- 7. Debug Visualization ---
        
        debug_header = Header()
        debug_header.frame_id = "fr3_link0" 
        debug_header.stamp = self.get_clock().now().to_msg()
        
        # Create the pointcloud with all the filtered points
        pc2_msg = pc2.create_cloud_xyz32(debug_header, valid_points)
        
        # Publish the filtered pointcloud
        self.pub_debug_pcl.publish(pc2_msg)
        
        
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
        
        # Ensure values stay within the defined range [VAL_MIN, VAL_LOCK]
        new_vals = np.clip(new_vals, self.VAL_MIN, self.VAL_LOCK)
        
        # Update the grid
        self.grid[uix, uiy, uiz] = new_vals


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
        r_sq = r_vox**2
        
        # Create the mask for the voxels inside the BB
        mask = dist_sq <= r_sq
        
        # Extract the valid indexes 
        valid_x = X[mask]
        valid_y = Y[mask]
        valid_z = Z[mask]
        
        # Create list of tuples for the set of ignored voxels
        indices = [(int(x), int(y), int(z)) for x, y, z in zip(valid_x, valid_y, valid_z)]
        
        # Create the zone and append it to the ignore_zones array
        if len(indices) > 0:
            new_zone = IgnoredZone(indices)
            self.ignored_zones.append(new_zone)
            self.get_logger().warn(f"🚫 Created IGNOREZONE with {len(indices)} voxels (R={self.DEADLOCK_ZONE_RADIUS}m)")
            
        return indices

    """
    Uses the analogy to the water buckets to update the empty space in the grid.
    Do not iterate over all the voxels in the grid, skip the voxels where the water in it 
    is zero, this means that they are more likely to be surely empty and then there's no 
    need to check there.
    """
    def update_free_space_in_fov(self, T_world_cam):
                
        # --- 1. Candidates Selection (Optimization) ---
        
        # Check the voxels that are surely empty or full
        indices_to_check = np.argwhere((self.grid > self.VAL_MIN) & (self.grid < self.VAL_LOCK))

        if len(indices_to_check) == 0:
            return


        # --- 2. Compute World Coordinates ---
        
        ix = indices_to_check[:, 0]
        iy = indices_to_check[:, 1]
        iz = indices_to_check[:, 2]

        pts_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        pts_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        pts_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z

        # Create matrix (N, 4) for homogeneous product
        pts_world = np.column_stack((pts_x, pts_y, pts_z, np.ones_like(pts_x)))


        # --- 3. Projection to Camera Frame ---
        
        # Take the transformation matrix from the input arguments
        T_cam_world = np.linalg.inv(T_world_cam)
        
        # Position in camera frame
        pts_cam = (T_cam_world @ pts_world.T).T 
        x_c = pts_cam[:, 0]
        y_c = pts_cam[:, 1]
        z_c = pts_cam[:, 2]


        # --- 4. Frustum Geometry ---
        
        # Values for the frustum
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)

        # Geometric Mask (in distance range and FOV range)
        mask_fov = (z_c > 0.05) & (z_c < self.MAX_DEPTH) & \
                   (np.abs(x_c) < (z_c * tan_h)) & \
                   (np.abs(y_c) < (z_c * tan_v))


        # --- 5. Water Bucket Analogy Application ---
        
        # Filter the indexes with the mask
        valid_indices = indices_to_check[mask_fov]

        if len(valid_indices) > 0:
            
            # Take single valid indexes coordinates
            v_ix = valid_indices[:, 0]
            v_iy = valid_indices[:, 1]
            v_iz = valid_indices[:, 2]

            # Take current water values for each voxel
            current_vals = self.grid[v_ix, v_iy, v_iz]

            # Subtraction: make water evaporate if voxel is seen empty
            new_vals = current_vals - self.MISS_DEC

            # Clamp to VAL_MIN (0)
            self.grid[v_ix, v_iy, v_iz] = np.clip(new_vals, self.VAL_MIN, self.VAL_MAX)
            

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
        c_occupied = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9) 
        
        # Target: Yellow
        c_target   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        # Ignored: Blue
        c_ignored = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.6)


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
        marker.text = f"Exploration:{percentage:.1f}%"

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
            
            
            # --- Deadlock Management ---
            
            if self.is_deadlocked:
                
                if curr_idx is not None:
                    
                    self.get_logger().warn(f"🚫 Target {curr_idx} in DEADLOCK. I will ignore the entire zone!")
                    
                    # 1. Call the function to return the indexes of the voxels in the ignored zone
                    zone_indices = self.create_ignored_zone(curr_idx)
                    
                    # 2. Adds these voxels to the ignored zone
                    for idx in zone_indices:
                        self.ignored_targets.add(idx)
                    
                    # 3. Save the ignored target for the Safe\Risky Search
                    self.last_deadlock_pos = self.grid_to_world(*curr_idx)
                    
                    # 4. Reset
                    self.current_target_idx = None
                    
                else:
                    
                    self.get_logger().warn("WARNING Deadlock Phantom. Reset.")
                    
                # Reset flag deadlock 
                self.is_deadlocked = False
                self.last_moving_time = time.time()
                return 
                
                
            # --- Check Present Target ---
            
            # Check if the current target voxel has been "resolved" (seen as free or occupied). If so, search for a new target
            if curr_idx is not None:
                
                current_val = self.grid[curr_idx]       # Present Target 
                
                # Flag for a voxel with a specific attribute
                is_resolved = (current_val < self.VAL_FREE) or (current_val > self.VAL_OCCUPIED)
                
                if is_resolved:
                    
                    self.get_logger().info(f"✅ TARGET {curr_idx} Solved! (Val: {current_val}). Finding a new one.")
                    
                    # Reset Taret
                    self.current_target_idx = None      
                else:
                    target_pos = self.grid_to_world(*curr_idx)
                    self.publish_command(target_pos)
                    return
            
            
            # --- Select New Target ---
            
            # If we get here means that we do not have a target and we need to find one.
            # Search now a new one calling the ad-hoc function 
            best_idx = self.find_best_unknown_target(cam_pos)

            # Whenever we cannot find a target means that we should search in the ignored list
            if best_idx is None:
                
                if len(self.ignored_targets) > 0:           # Check if there is any ignored voxel
                    
                    self.get_logger().warn(f"♻️ FINITI TARGET. Reset Totale (ignore_list + Deadlock Pos).")
                    
                    # Reset the ignored voxels list and the target 
                    self.ignored_targets.clear()
                    self.last_deadlock_pos = None  
                    
                    # Now that we have done a reset we can search for a new target 
                    best_idx = self.find_best_unknown_target(cam_pos)
                    
                else:
                    
                    self.get_logger().info("🎉 Completed Exploration!", throttle_duration_sec=5.0)

            if best_idx:
                self.current_target_idx = best_idx
                target_pos = self.grid_to_world(*best_idx)
                self.publish_command(target_pos)
            else:
                # Se è ANCORA None anche dopo il reset, allora abbiamo davvero finito tutto.
                pass

        finally:
            self.decision_lock.release()


    """
    Check all the ignored zones and if the 10% of them has been seen then free the entire zone
    """
    def check_and_liberate_zones(self):
        
        if not self.ignored_zones:
            return

        zones_to_remove = []        # List of zones to remove from the ignored zone list
        
        # Iterate over all the ignored zones to check
        for zone in self.ignored_zones:
            
            resolved_count = 0      # Number of seen voxels in the zone
            
            # Iterate over the voxels in the zone
            for idx in zone.member_indices:
                
                # Take the value of the voxel
                val = self.grid[idx]

                # If solved
                if val < self.VAL_FREE or val > self.VAL_OCCUPIED:
                    resolved_count += 1
            
            # Percentage Computation
            percentage = resolved_count / zone.total_count
            
            # Trigger 
            if percentage >= 0.10: # 10% Free Treshold
                
                self.get_logger().info(f"🔓 Zone free! (Seen {resolved_count}/{zone.total_count} voxels).")
                
                # Remove members of the global ignore list  
                for idx in zone.member_indices:
                    self.ignored_targets.discard(idx)
                
                zones_to_remove.append(zone)

        # Clean List of Zones
        for z in zones_to_remove:
            try:
                self.ignored_zones.remove(z)
            except ValueError:
                pass 


    """
    Select the best best unknown voxel to go to.
    Includes: 
    - Hysteresis: ignore voxels that have been just seen, they have to be above a safe treshold to be interesting (this ensure not to fall into the deadlock loop: Seen - Noise - New Target - Seen).
    - Wall Skin Filter: we don't want target attached to the walls (occupied voxels) because often these voxels are just noise or walls.
    - Ignore List Check
    - Safe/Risky Logic for Deadlock: divide the candidates in two groups, the safe ones are far from the deadlock target and the risky ones are close.
    - Score Logic: the target is chosen as the voxel with the highest score, which is given by distance from camera position, relevance and frontality.
    """
    def find_best_unknown_target(self, robot_pos):
        
        SELECTION_TRESHOLD = self.VAL_FREE + 15
        
        
        # --- 1. Candidate Selection (with hysteresis) ---
       
        mask_unknown = (self.grid >= SELECTION_TRESHOLD) & (self.grid <= self.VAL_OCCUPIED)


        # --- 2. Wall Skin Protection (Walls Filter) ---
        
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


        # --- Fallback: Skin tto much aggressive? ---
        
        if len(unknown_indices) == 0: 
            
            # If the Skin filter removed everything, let's try using the raw candidates.
            # Better to risk getting close to a wall than stopping completely.
            unknown_indices = np.argwhere(mask_unknown)
            
            if len(unknown_indices) == 0:
                return None # Truly nothing to see


        # --- 3. Pre-Filtering (Performance & ignore_list) ---
        
        candidates = []                     # List of candidates
        
        # Performance Filter
        # Check if there are too many unknown voxels: if so, select 5000 random ones
        temp_indices = unknown_indices
        if len(temp_indices) > 5000:
             rand_idx = np.random.choice(len(temp_indices), 5000, replace=False)
             temp_indices = temp_indices[rand_idx]

        # Ignored List Filter
        for idx in temp_indices:
            t_idx = tuple(idx)
            if t_idx not in self.ignored_targets:
                candidates.append(idx)
                
        if not candidates: 
            return None
        
        candidates = np.array(candidates)   # Returns an array numpy (better)


        # --- 4. Deadlock Logic: Safe VS Risky ---
        
        safe_candidates = []            # Far from deadlock voxel
        risky_candidates = []           # Close from deadlock voxel
        
        final_pool = []                 # Pool on which to iterate for the score

        # If we are ine deadlock
        if self.last_deadlock_pos is not None:
            
            for idx in candidates:
                
                pos = self.grid_to_world(*idx)                                          # World Position
                dist_from_deadlock = np.linalg.norm(pos - self.last_deadlock_pos)       # Distance from deadlock
                
                # Pool division 
                if dist_from_deadlock > self.DEADLOCK_MIN_DIST:
                    safe_candidates.append(idx)
                else:
                    risky_candidates.append(idx)
            
            # Select a Pool
            if len(safe_candidates) > 0:        # There are safe candidates
                
                final_pool = safe_candidates
                
                # Downsampling
                if len(final_pool) > 500:
                     rand_idx = np.random.choice(len(final_pool), 500, replace=False)
                     final_pool = [final_pool[i] for i in rand_idx]
                
            else:                               # There are no safe candidates

                # final_pool = risky_candidates
                
                self.get_logger().warn("⚠️ No Safe Target Found! Choosing in the Risky Pool.")
        
        # If we are not in deadlock
        else:

            final_pool = candidates             # All the candidates are good
            
            # Downsampling 
            if len(final_pool) > 500:
                rand_idx = np.random.choice(len(final_pool), 500, replace=False)
                final_pool = final_pool[rand_idx]


        # --- 5. Scoring Loop ---
        
        # Initialization
        best_score = -np.inf
        best_idx = None
        
        # Normalization Parameters (the relevenca one is in the get_relevance function)
        y_min, y_max = self.WS_BOUNDS['y']          
        y_range = max(y_max - y_min, 1.0)           # Frontality Normalization Parameter
        max_dist_norm = 1.5                         # Distance Normalization Parameter
        

        for idx in final_pool:
            
            # Convert index in world coordinates
            pos = self.grid_to_world(*idx)

            # A. Frontality (0.0 - 1.0): we prefer to choose voxels near the robot's base
            y_ratio = (pos[1] - y_min) / y_range
            norm_frontality = 1.0 - np.clip(y_ratio, 0.0, 1.0)

            # B. Distance (0.0 - 1.0): we prefer to choose voxels close to the camera
            real_dist = np.linalg.norm(pos - robot_pos)
            dist_ratio = min(real_dist / max_dist_norm, 1.0)
            norm_dist_score = 1.0 - dist_ratio

            # C. Relevance (0.0 - 1.0): we prefer to choose voxels that are in a dense cloud of unknown voxels
            norm_relevance = self.get_relevance(idx[0], idx[1], idx[2])

            # Final Score Formula
            # Weights:
            # 3.0 Frontality
            # 2.0 Relevance  
            # 1.0 Distance   
            score = (3.0 * norm_frontality) + (2.0 * norm_relevance) + (1.0 * norm_dist_score)
            
            if score > best_score:
                best_score = score
                best_idx = tuple(idx)

        return best_idx

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
        
        
        # --- Spatial Hashing ---
        
        keys = np.floor(points / self.min_distance_rep).astype(int)
        
        # We use a dictionary to keep only one point for each spatial key
        unique_obstacles = {}
        
        # Iterate and fill the dictionary.
        for i in range(points.shape[0]):
            
            # The key is a tuple (k_x, k_y, k_z)
            k = tuple(keys[i])
            
            # If this key is not represented then do it now
            if k not in unique_obstacles:
                unique_obstacles[k] = points[i]
        
        # Restituisce solo i punti filtrati
        return list(unique_obstacles.values())


    """
    Publishes a VmcControlTarget message to guide the robot.
    This message includes:
    - The target attractor point for the robot's end-effector.
    - A list of active obstacles (repulsors) derived from the voxel grid,
        downsampled to maintain a minimum distance between them.
    - A visualization marker for the active repulsors.
    """
    def publish_command(self, target_pos):
        
        # Create the message for julia script
        msg = VmcControlTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Insert in the message the target 
        msg.target_attractor = Point(x=target_pos[0], y=target_pos[1], z=target_pos[2])

        # Filter the obstacle pointcloud 
        optimized_obstacles = self.get_downsampled_obstacles()

        # Insert in the message the obstacle list of repulsors
        for obs_pos in optimized_obstacles:
            msg.active_obstacles.append(Point(x=obs_pos[0], y=obs_pos[1], z=obs_pos[2]))
            
        # Publish the message to julia
        self.pub_target.publish(msg)


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

        for p in msg.active_obstacles:
            marker.points.append(p) 

        self.pub_repulsors_viz.publish(marker)

    def grid_to_world(self, ix, iy, iz):
        x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        return np.array([x, y, z])
        

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
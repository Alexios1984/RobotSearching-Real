# Log message with all the map node parameters

std_msgs/Header header

# Workspace Configuration
string type_experiment
float64 ws_x_min
float64 ws_x_max
float64 ws_y_min
float64 ws_y_max
float64 ws_z_min
float64 ws_z_max
float64 target_voxel_size
int32 n_voxels_x
int32 n_voxels_y
int32 n_voxels_z

# Probabilistic (Buckets)
int32 val_min
int32 val_max
int32 val_unknown
int32 val_occupied
int32 val_free
int32 val_lock
int32 hit_inc
int32 miss_dec

# Camera & FOV
float64 fov_h
float64 fov_v
float64 max_depth
float64 min_depth

# Logic & Scores
int32 max_obstacles
float64 min_distance_rep
int32 min_points_per_voxel
float64 score_frontality
float64 score_relevance
float64 score_distance

# Deadlock Configuration
float64 deadlock_vel_eps
float64 deadlock_torque_eps
float64 deadlock_time_threshold
float64 deadlock_zone_radius
float64 deadlock_min_dist
float64 recovery_min_dist
float64 recovery_max_dist
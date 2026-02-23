# Message containing target positions 

std_msgs/Header header

geometry_msgs/Point target_position

# Target Coordinates
int32 grid_x
int32 grid_y
int32 grid_z

# Score associated to the selected target (0.0 if the target is old)
float64 score_total
float64 score_frontality
float64 score_relevance
float64 score_distance
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import json

# ROS 2 Messages
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, Int32, Float32, String

class DepthVisionNode(Node):
    
    """
    Initializes a ROS2 node that processes depth images to generate a PointCloud2 message.
    """
    def __init__(self):
        
        super().__init__('vision_pcl')
        
        self.bridge = CvBridge()        # OpenCV Bridge for image conversion

        self.prev_time = time.time()    # Needed for FPS computation

        # ==========================
        # --- CAMERA CONFIGURATION ---
        # ==========================
        
        self.min_dist = 0.07            # Minimum distance to consider for detected obstacles (minimum distance Realsense)
        self.max_dist = 0.50            # Maximum distance to consider for detected obstacles (maximum distances Realsense)
        self.depth_scale = 0.001        # RealSense default (mm -> m)
        self.decimation = 5             # Skip factor for trivially downsampling the depth image
        self.camera_info = None

        # ==========================
        # --- COMMUNICATION TOPICS ---
        # ==========================
        
        # -- Subscribers --
        
        self.sub_info = self.create_subscription(       # Subscriber to the colored image
            CameraInfo, 
            'camera/camera/aligned_depth_to_color/camera_info', 
            self.cb_info, 
            10
        )
        
        self.sub_depth = self.create_subscription(      # Subscriber to the depth info
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw', 
            self.cb_depth, 
            10
        )
        
        # -- Publishers --
        
        self.pub_pcl = self.create_publisher(           # Publisher of the Pointcloud
            PointCloud2, 
            '/vision/obstacle_cloud', 
            10
        )
        
        # -- Log Publishers --
        
        self.pub_config = self.create_publisher(String, '/vision/node_config', 10)
        self.pub_count = self.create_publisher(Int32, '/vision/point_count', 10)
        
        
        # Logging
        self.get_logger().info(f"Depth Node Ready! Range: {self.min_dist}-{self.max_dist}m")
        
        self.publish_config_once()    # Timer for the one shot camera configuration parameters publish


    """
    Callback for CameraInfo messages to store intrinsic parameters.
    """
    def cb_info(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera Info Acquired!")


    """
    Callback for depth image messages to process and publish PointCloud2.
    """
    def cb_depth(self, msg):
        
        if self.camera_info is None:
            return

        # Convert depth image from the camera to OpenCV format for processing
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        
        # Downsampling (take a pixel every n pixel)
        if self.decimation > 1:
            depth_image = depth_image[::self.decimation, ::self.decimation]

        # Obtain depth values in meters to create point cloud filtered by distance
        z_vals = depth_image.astype(np.float32) * self.depth_scale

        # Validity Mask (Range 7cm - 60cm)
        mask = (z_vals > self.min_dist) & (z_vals < self.max_dist)
        
        # Without valid points, skip processing
        if not np.any(mask):
            return

        # Reprojection 3D using Pinhole Camera Model 
        fx = self.camera_info.k[0] / self.decimation
        fy = self.camera_info.k[4] / self.decimation
        cx = self.camera_info.k[2] / self.decimation
        cy = self.camera_info.k[5] / self.decimation

        # Pixels coordinates of valid points to reproject
        v_idx, u_idx = np.where(mask)
        
        # Take z info from image using the valid mask
        z_valid = z_vals[v_idx, u_idx]

        # Formulas Pinhole Camera Model
        x_valid = (u_idx - cx) * z_valid / fx
        y_valid = (v_idx - cy) * z_valid / fy

        # PointCloud2 Creation from valid points
        points_3d = np.vstack((x_valid, y_valid, z_valid)).T

        # Header for PointCloud2
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id 

        # Create the message of PointCloud2
        pcl_msg = pc2.create_cloud_xyz32(header, points_3d)
        
        # Publish the cleaned pointcloud
        self.pub_pcl.publish(pcl_msg)

        # Publish the count of the points sent
        self.pub_count.publish(Int32(data=points_3d.shape[0])) 
        
        
    def publish_config_once(self):
        
        config_data = {
            "min_dist_m": self.min_dist,
            "max_dist_m": self.max_dist,
            "decimation_factor": self.decimation,
            "depth_scale": self.depth_scale,
            "node_name": self.get_name()
        }
        
        msg = String()
        msg.data = json.dumps(config_data)
        self.pub_config.publish(msg)
        self.get_logger().info(f"Config published to /vision/node_config: {msg.data}")
        # Cancelliamo il timer, basta farlo una volta
        return

def main(args=None):
    rclpy.init(args=args)
    node = DepthVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
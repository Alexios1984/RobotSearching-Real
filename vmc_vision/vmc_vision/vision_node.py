import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

# ROS 2 Messages
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point, Vector3
from vmc_interfaces.msg import ObjectDetection3D, ObjectDetection3DArray
from sensor_msgs_py import point_cloud2 as pc2
import struct

# YOLO
from ultralytics import YOLO

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # --- CONFIGURATION ---
        self.target_objects = ['bottle', 'cup'] # Objects to find
        self.conf_threshold = 0.2 # Confidence threshold (means that detections with lower confidence will be ignored)
        self.depth_scale = 0.001 # For RealSense (mm -> m)
        self.prev_time = 0

        # Load Model YOLO (Segmentazione)
        self.get_logger().info("Caricamento modello YOLO-Seg...")
        try:
            self.model = YOLO('yolov8n-seg.pt') 
        except Exception as e:
            self.get_logger().error(f"Error YOLO: {e}")
            return

        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_depth = None

        # Subscribers
        self.sub_info = self.create_subscription(
            CameraInfo, 'camera/camera/aligned_depth_to_color/camera_info', self.cb_info, 10)
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_rect_raw', self.cb_rgb, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.cb_depth, 10)
        
        # Publisher Immagine Debug (Bounding Box + Maschere)
        self.pub_debug_image = self.create_publisher(
            Image, '/vision/debug_image', 10)

        # Publisher (Custom Message)
        self.pub_detections = self.create_publisher(
            ObjectDetection3DArray, '/vision/detections', 10)
        
        # 1. Pointcloud Original (Full)
        self.pub_pcl_raw = self.create_publisher(PointCloud2, '/vision/pointcloud/raw', 10)
        
        # 2. Pointcloud Filtred (< 50cm)
        self.pub_pcl_limit = self.create_publisher(PointCloud2, '/vision/pointcloud/limit_50cm', 10)
        
        # 3. Pointcloud only Detected objects (All)
        self.pub_pcl_objects_all = self.create_publisher(PointCloud2, '/vision/pointcloud/objects_all', 10)
        
        # 4. Pointcloud for each Detected Object (Individual)
        self.pub_pcl_individual = {}
        for obj_name in self.target_objects:
            topic_name = f'/vision/pointcloud/class/{obj_name}'
            self.pub_pcl_individual[obj_name] = self.create_publisher(PointCloud2, topic_name, 10)

        self.get_logger().info("Vision Node Ready! Waiting Images...")

    def cb_info(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera Info Read!")

    def cb_depth(self, msg):
        # Convert depth in array numpy (mm)
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

    def cb_rgb(self, msg):
        if self.latest_depth is None or self.camera_info is None:
            return

        # --- CALCOLO FPS ---
        curr_time = time.time()
        fps = 0
        if self.prev_time != 0:
            delta = curr_time - self.prev_time
            if delta > 0:
                fps = 1.0 / delta
        self.prev_time = curr_time
        
        # Convert RGB
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # --- 1. INFERENCE YOLO ---
        results = self.model(cv_image, verbose=False, conf=self.conf_threshold)
        
        # --- DEBUG VISUALIZATION (MASKS + BOXES + FPS) ---
        debug_frame = cv_image.copy()
        H, W = debug_frame.shape[:2]
        
        # Layer per le maschere (trasparenza)
        mask_overlay = np.zeros_like(debug_frame, dtype=np.uint8)
        
        # Colori
        color = (0, 255, 0) # Verde
        fps_color = (255, 255, 255) # Bianco

        for r in results:
            if r.masks is None: continue

            masks_data = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                # FILTRO OGGETTI TARGET
                if cls_name not in self.target_objects:
                    continue

                # 1. DISEGNO MASCHERA (Sul layer overlay)
                raw_mask = masks_data[i]
                mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_overlay[mask] = color

                # 2. DISEGNO BOX E NOME
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name} {conf:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(debug_frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. FUSIONE (BLEND)
        debug_frame = cv2.addWeighted(debug_frame, 1.0, mask_overlay, 0.3, 0)

        # 4. DISEGNO FPS (Bianco, in alto a destra)
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (W - 130, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        # Pubblichiamo
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.pub_debug_image.publish(debug_msg)
        # ---------------------------------------------------
        
        detections_msg = ObjectDetection3DArray()
        detections_msg.header = msg.header # Keep the same header

        H, W = cv_image.shape[:2]
        
        # Camera intrinsic Parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        for r in results:
            if r.masks is None: continue

            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                score = float(box.conf[0])

                # Filters only target object
                if cls_name not in self.target_objects:
                    continue

                # --- 2. 3D Points Extraction ---
                # Size the mask to the image original dimension
                mask_raw = masks[i]
                mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                # Find pixel indices where mask is True
                # Optimization: Use NumPy masking instead of loop for
                v_idx, u_idx = np.where(mask)
                
                # If the mask is too small, skip
                if len(v_idx) < 50: continue

                # Take depth values at these pixel locations
                z_vals = self.latest_depth[v_idx, u_idx] * self.depth_scale
                
                # Filter valid depth values (0.05m < z < 2.0m)
                valid_z_mask = (z_vals > 0.05) & (z_vals < 0.7)
                z_vals = z_vals[valid_z_mask]
                u_valid = u_idx[valid_z_mask]
                v_valid = v_idx[valid_z_mask]

                if len(z_vals) < 10: continue

                # Reprojection 2D -> 3D (Vectorial)
                x_vals = (u_valid - cx) * z_vals / fx
                y_vals = (v_valid - cy) * z_vals / fy

                # Stack of points (N, 3)
                points_3d = np.vstack((x_vals, y_vals, z_vals)).T

                # --- 3. COMPUTE CENTROID AND BBOX ---
                # Centroid (Mean)
                centroid = np.mean(points_3d, axis=0)

                # BBox (Max - Min)
                min_pt = np.min(points_3d, axis=0)
                max_pt = np.max(points_3d, axis=0)
                dims = max_pt - min_pt

                # --- 4. CREATE THE MESSAGE ---
                det = ObjectDetection3D()
                det.class_name = cls_name
                det.score = score
                
                det.center.x = float(centroid[0])
                det.center.y = float(centroid[1])
                det.center.z = float(centroid[2])

                det.bbox_size.x = float(dims[0])
                det.bbox_size.y = float(dims[1])
                det.bbox_size.z = float(dims[2])

                detections_msg.detections.append(det)

                # Debug Log (optional)
                self.get_logger().info(f"Found {cls_name} at {centroid} dim {dims}")

        # Publish array
        self.pub_detections.publish(detections_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
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

# Kalman Filter
from .KalmanFilter import StabilizedKalmanFilter

class ObjectTracker:
    def __init__(self, class_name):
        self.class_name = class_name
        
        # 1. Kalman per la POSIZIONE (x, y, z)
        # dt=0.1 (circa 10Hz), dim=3
        self.kf = StabilizedKalmanFilter(dt=0.1, dim=3, stabilization_factor=0.9, max_missing=10)
        
        # 2. Low-Pass Filter per le DIMENSIONI (w, h, d)
        self.bbox_smooth = None 
        self.alpha_dim = 0.05 # 5% misura nuova, 95% memoria (MOLTO stabile)
        

        self.active = True

    def update(self, measured_pos, measured_dim):
        """
        measured_pos: np.array [x, y, z] o None
        measured_dim: np.array [w, h, d] o None
        """
        # --- A. Aggiornamento Posizione (Kalman) ---
        if measured_pos is not None:
            self.kf.predict()
            self.kf.update(measured_pos)
        else:
            # Se non vedo l'oggetto, predico e basta
            self.kf.predict()
            self.kf.update(None) # Gestisce il missing_count internamente

        # --- B. Aggiornamento Dimensioni (Low Pass) ---
        if measured_dim is not None:
            if self.bbox_smooth is None:
                self.bbox_smooth = measured_dim
            else:
                # Formula: Old * 0.95 + New * 0.05
                self.bbox_smooth = (self.bbox_smooth * (1 - self.alpha_dim)) + (measured_dim * self.alpha_dim)
        
        # Controllo se il tracker è "morto" (perso da troppi frame)
        if self.kf.missing_count > self.kf.max_missing:
            self.active = False

    def get_state(self):
        pos = self.kf.get_position() # Ritorna [x, y, z]
        vel = self.kf.get_velocity()
        
        # Se non ho ancora una dimensione valida, ritorno zeri
        dims = self.bbox_smooth if self.bbox_smooth is not None else np.zeros(3)
        
        return pos, dims, vel

class YoloVisionNode(Node):
    def __init__(self):
        super().__init__('vision_kalman')

        # --- CONFIGURATION ---
        self.target_objects = ['bottle', 'cup'] # Objects to find
        self.conf_threshold = 0.2 # Confidence threshold (means that detections with lower confidence will be ignored)
        self.depth_scale = 0.001 # For RealSense (mm -> m)
        self.prev_time = 0
        
        # --- TRACKING MEMORY ---
        # Dizionario: chiave = class_name (es. 'bottle'), valore = ObjectTracker
        self.trackers = {}


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
        
        # # 1. Pointcloud Original (Full)
        # self.pub_pcl_raw = self.create_publisher(PointCloud2, '/vision/pointcloud/raw', 10)
        
        # # 2. Pointcloud Filtred (< 50cm)
        # self.pub_pcl_limit = self.create_publisher(PointCloud2, '/vision/pointcloud/limit_50cm', 10)
        
        # # 3. Pointcloud only Detected objects (All)
        # self.pub_pcl_objects_all = self.create_publisher(PointCloud2, '/vision/pointcloud/objects_all', 10)
        
        # # 4. Pointcloud for each Detected Object (Individual)
        # self.pub_pcl_individual = {}
        # for obj_name in self.target_objects:
        #     topic_name = f'/vision/pointcloud/class/{obj_name}'
        #     self.pub_pcl_individual[obj_name] = self.create_publisher(PointCloud2, topic_name, 10)

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
        
        # --- INIZIO AGGIUNTA CALCOLO FPS ---
        curr_time = time.time()
        fps = 0
        if self.prev_time != 0:
            delta = curr_time - self.prev_time
            if delta > 0:
                fps = 1.0 / delta
        self.prev_time = curr_time
        # --- FINE AGGIUNTA ---

        # Convert RGB
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # --- 1. INFERENCE YOLO ---
        results = self.model(cv_image, verbose=False, conf=self.conf_threshold)
        
        # --- DEBUG VISUALIZATION (MASKS + BOXES + FPS) ---
        debug_frame = cv_image.copy()
        H, W = debug_frame.shape[:2]

        # Layer per le maschere (trasparenza)
        mask_overlay = np.zeros_like(debug_frame, dtype=np.uint8)
        
        # Colore Verde
        color = (0, 255, 0) 
        text_color = (0, 255, 0)
        fps_color = (255, 255, 255) # Bianco

        for r in results:
            # Se non ci sono maschere, saltiamo
            if r.masks is None:
                continue

            masks_data = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                # --- FILTRO OGGETTI TARGET ---
                if cls_name not in self.target_objects:
                    continue

                # 1. DISEGNO MASCHERA (Sul layer overlay)
                raw_mask = masks_data[i]
                mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_overlay[mask] = color

                # 2. DISEGNO BOX E NOME (Sul frame principale)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name} {conf:.2f}"
                text_y = max(y1 - 10, 20)
                cv2.putText(debug_frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 3. FUSIONE (BLEND)
        # Sovrapponiamo le maschere verdi con il 30% di opacità
        debug_frame = cv2.addWeighted(debug_frame, 1.0, mask_overlay, 0.3, 0)

        # 4. DISEGNO FPS (Dopo il blend, così rimane bianco brillante)
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (W - 130, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        # Pubblichiamo
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.pub_debug_image.publish(debug_msg)
        
        # --- INIZIO AGGIUNTA TESTO ---
        H, W = cv_image.shape[:2]
        # Scritta bianca (255,255,255) in alto a destra
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (W - 130, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # --- FINE AGGIUNTA ---

        # Colore per le box (BGR): Verde brillante
        box_color = (0, 255, 0) 
        text_color = (0, 255, 0)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                # --- IL FILTRO MAGICO ---
                # Se non è nella nostra lista target, SALTA il disegno
                if cls_name not in self.target_objects:
                    continue
                
                # Estraiamo coordinate BBox 2D (interi per OpenCV)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Disegna Rettangolo
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), box_color, 2)

                # Disegna Testo (Nome + Confidenza)
                label = f"{cls_name} {conf:.2f}"
                # Calcola posizione testo (un po' sopra la box, ma non fuori schermo)
                text_y = max(y1 - 10, 20)
                cv2.putText(debug_frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Pubblica l'immagine disegnata a mano
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.pub_debug_image.publish(debug_msg)

        # --- 1. ESTRAZIONE DATI 3D (Per il resto del codice) ---
        current_frame_detections = {}

        H, W = cv_image.shape[:2]
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        for r in results:
           
            if r.masks is None: 
                continue

            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                if cls_name not in self.target_objects: continue
                
                # --- ESTRAZIONE 3D SEMPLIFICATA (Usa BBox invece di Maschere) ---
                # Se non usi la segmentazione, questo è un modo più rapido 
                # per prendere il punto centrale 3D.
                
                mask_raw = masks[i]
                mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                v_idx, u_idx = np.where(mask)
                
                # Se la maschera è troppo piccola (rumore), ignoriamo
                if len(v_idx) < 50: 
                    continue

                # 3. Prendiamo la profondità SOLO in quei pixel
                z_vals = self.latest_depth[v_idx, u_idx] * self.depth_scale
                
                # 4. Filtriamo valori di profondità validi (rimuoviamo zeri e outliers)
                valid_mask = (z_vals > 0.05) & (z_vals < 0.7)
                z_vals = z_vals[valid_mask]
                
                if len(z_vals) < 10: 
                    continue # Troppi pochi punti validi

                # Recuperiamo le coordinate u, v corrispondenti ai punti validi
                u_valid = u_idx[valid_mask]
                v_valid = v_idx[valid_mask]

                # 5. Riproiezione 3D (Vettoriale per velocità)
                # x = (u - cx) * z / fx
                x_vals = (u_valid - cx) * z_vals / fx
                y_vals = (v_valid - cy) * z_vals / fy

                # Creiamo la nuvola di punti dell'oggetto (N, 3)
                points_3d = np.vstack((x_vals, y_vals, z_vals)).T

                # --- STATISTICHE OGGETTO ---
                # Centroide: Media di tutti i punti (Molto più stabile del singolo pixel centrale!)
                centroid = np.mean(points_3d, axis=0)

                # Dimensioni: Max - Min
                min_pt = np.min(points_3d, axis=0)
                max_pt = np.max(points_3d, axis=0)
                dims = max_pt - min_pt

                # Salviamo per il Tracker
                current_frame_detections[cls_name] = (centroid, dims, float(box.conf[0]))

        # --- 2. AGGIORNAMENTO TRACKERS E PUBBLICAZIONE ---
        
        out_msg = ObjectDetection3DArray()
        out_msg.header = msg.header

        all_known_classes = set(self.trackers.keys()) | set(current_frame_detections.keys())

        for cls_name in all_known_classes:
            
            if cls_name not in self.trackers:
                self.trackers[cls_name] = ObjectTracker(cls_name)

            tracker = self.trackers[cls_name]
            measurement = current_frame_detections.get(cls_name)
            
            if measurement is not None:
                raw_pos, raw_dim, raw_score = measurement
                tracker.update(raw_pos, raw_dim)
                score_to_pub = raw_score
            else:
                tracker.update(None, None)
                score_to_pub = 0.0

            if tracker.active:
                smooth_pos, smooth_dim, _ = tracker.get_state()
                
                det = ObjectDetection3D()
                det.class_name = cls_name
                det.score = score_to_pub
                det.center.x = float(smooth_pos[0])
                det.center.y = float(smooth_pos[1])
                det.center.z = float(smooth_pos[2])
                # det.bbox_size.x = float(smooth_dim[0])
                # det.bbox_size.y = float(smooth_dim[1])
                # det.bbox_size.z = float(smooth_dim[2]) #####################################################################################################################################################################################################################
                # det.bbox_size.z = float(smooth_dim[1])
                det.bbox_size.x = 0.03
                det.bbox_size.y = 0.03
                det.bbox_size.z = 0.03

                # self.get_logger().info(f"Found {cls_name} at {smooth_pos} dim {smooth_dim}")

                out_msg.detections.append(det)
            else:
                pass

        self.trackers = {k: v for k, v in self.trackers.items() if v.active}
        self.pub_detections.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
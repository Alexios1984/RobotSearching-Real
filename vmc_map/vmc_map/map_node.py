import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS Messages
from vmc_interfaces.msg import ObjectDetection3DArray, VmcRobotState, VmcControlTarget
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker

class MapLogicNode(Node):
    def __init__(self):
        super().__init__('map_node')

        # --- 1. CONFIGURAZIONE WORKSPACE ---
        # Definizione dei limiti globali (World Frame)
        self.WS_BOUNDS = {
            'x': [-0.2, 0.5],   
            'y': [ 0.45, 0.9],
            'z': [ 0.2, 0.85]
        }

        # --- CONFIGURAZIONE RISOLUZIONE (SCELTA TUA) ---
        # Decidi tu quanti cubetti vuoi su ogni asse
        self.N_VOXELS_X = 20 
        self.N_VOXELS_Y = 20
        self.N_VOXELS_Z = 20

        # Calcolo dimensioni dinamiche del singolo Voxel (dx, dy, dz)
        self.dim_x = self.WS_BOUNDS['x'][1] - self.WS_BOUNDS['x'][0]
        self.dim_y = self.WS_BOUNDS['y'][1] - self.WS_BOUNDS['y'][0]
        self.dim_z = self.WS_BOUNDS['z'][1] - self.WS_BOUNDS['z'][0]

        self.res_x = self.dim_x / self.N_VOXELS_X
        self.res_y = self.dim_y / self.N_VOXELS_Y
        self.res_z = self.dim_z / self.N_VOXELS_Z

        self.get_logger().info(f"Workspace: {self.dim_x:.2f}x{self.dim_y:.2f}x{self.dim_z:.2f} m")
        self.get_logger().info(f"Voxel Res: {self.res_x:.3f}x{self.res_y:.3f}x{self.res_z:.3f} m")
        self.get_logger().info(f"Grid Size: {self.N_VOXELS_X}x{self.N_VOXELS_Y}x{self.N_VOXELS_Z}")

        # 0=Unknown, 1=Free, 2=Occupied
        self.grid = np.zeros((self.N_VOXELS_X, self.N_VOXELS_Y, self.N_VOXELS_Z), dtype=np.int8)

        # --- CAMERA CONFIG ---
        self.FOV_H = np.deg2rad(87.0) 
        self.FOV_V = np.deg2rad(58.0)
        self.MAX_DEPTH = 0.40 # <--- IMPOSTATO A 40 per sicurezza

        self.latest_cam_pose = None 
        self.current_visible_obstacles = [] 

        self.locked_objects = set()

        # --- COMUNICAZIONE ---
        self.sub_vision = self.create_subscription(
            ObjectDetection3DArray, '/vision/detections', self.cb_vision, 10)
            
        self.sub_robot = self.create_subscription(
            VmcRobotState, '/vmc/robot_state', self.cb_robot_state, 10)

        self.pub_target = self.create_publisher(VmcControlTarget, '/vmc/target', 10)
        
        self.pub_viz = self.create_publisher(Marker, '/vmc/voxel_grid_viz', 10)
        self.pub_frustum = self.create_publisher(Marker, '/vmc/camera_frustum', 10)

        # Timer Visualization (2 Hz)
        self.timer_viz = self.create_timer(0.01, self.publish_visualization_loop)

        # Logic Timer (10Hz)
        self.timer = self.create_timer(0.01, self.decision_loop)
        self.current_target_idx = None 

    def cb_robot_state(self, msg):
        # NOTA: Usiamo cam_pose perché il msg ha quel nome, ma dentro c'è la CAMERA
        self.latest_cam_pose = {
            'pos': np.array([msg.cam_pose.position.x, msg.cam_pose.position.y, msg.cam_pose.position.z]),
            'quat': np.array([msg.cam_pose.orientation.x, msg.cam_pose.orientation.y, msg.cam_pose.orientation.z, msg.cam_pose.orientation.w])
        }

        q = self.latest_cam_pose['quat']
        if np.linalg.norm(q - np.array([0,0,0,1])) < 0.01:
            self.get_logger().warn(f"⚠️ QUATERNIONE IDENTITÀ RILEVATO: {q}")

    def cb_vision(self, msg):
        if self.latest_cam_pose is None: return

        # 1. Transform Setup
        r_cam = R.from_quat(self.latest_cam_pose['quat'])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = r_cam.as_matrix()
        T_world_cam[:3, 3] = self.latest_cam_pose['pos']
        
        # 2. ANTI-RUMORE (Pulizia nel cono)
        self.update_free_space_in_fov(T_world_cam)

        # self.current_visible_obstacles = []

        # 3. Aggiorna Occupied
        for det in msg.detections:

            # --- IL BLOCCO LOGICO ---
            if det.class_name in self.locked_objects:
                # L'abbiamo già mappato! Ignoriamo i nuovi dati per questa classe.
                continue 
            
            self.get_logger().info(f"🔒 LOCKING position for: {det.class_name}")
            self.locked_objects.add(det.class_name)
            # ------------------------

            # --- VECCHIO CODICE (DA RIMUOVERE/COMMENTARE) ---
            # p_cam_center = np.array([det.center.x, det.center.y, det.center.z, 1.0])
            # p_world_center = (T_world_cam @ p_cam_center)[:3]
            # self.current_visible_obstacles.append(p_world_center)
            # 
            # half_size = np.array([det.bbox_size.x, det.bbox_size.y, det.bbox_size.z]) / 2.0
            # min_w = p_world_center - half_size
            # max_w = p_world_center + half_size
            # -----------------------------------------------

            # --- NUOVO CODICE (METODO 8 VERTICI) ---
            
            # A. Recuperiamo dimensioni e centro in Camera Frame
            dx, dy, dz = det.bbox_size.x / 2.0, det.bbox_size.y / 2.0, det.bbox_size.z / 2.0
            cx, cy, cz = det.center.x, det.center.y, det.center.z

            # B. Creiamo gli 8 vertici della box (nel frame Camera)
            # Nota: La box è allineata agli assi della camera
            corners_cam = np.array([
                [cx - dx, cy - dy, cz - dz, 1.0],
                [cx - dx, cy - dy, cz + dz, 1.0],
                [cx - dx, cy + dy, cz - dz, 1.0],
                [cx - dx, cy + dy, cz + dz, 1.0],
                [cx + dx, cy - dy, cz - dz, 1.0],
                [cx + dx, cy - dy, cz + dz, 1.0],
                [cx + dx, cy + dy, cz - dz, 1.0],
                [cx + dx, cy + dy, cz + dz, 1.0]
            ])

            # C. Trasformiamo tutti gli 8 punti nel World Frame
            # T_world_cam è (4x4), corners_cam.T è (4x8) -> Risultato (4x8)
            corners_world = (T_world_cam @ corners_cam.T).T 
            
            # Prendiamo solo x,y,z (togliamo la w=1 omogenea)
            corners_world = corners_world[:, :3]

            # D. Calcoliamo il centro nel mondo (Media degli 8 punti o trasformazione diretta)
            p_world_center = np.mean(corners_world, axis=0)
            self.current_visible_obstacles.append(p_world_center)

            # E. Troviamo il Bounding Box che avvolge tutto (AABB nel World Frame)
            min_w = np.min(corners_world, axis=0)
            max_w = np.max(corners_world, axis=0)

            # --- FINE NUOVO CODICE ---
            
            ix_min, iy_min, iz_min = self.world_to_grid(min_w)
            ix_max, iy_max, iz_max = self.world_to_grid(max_w)
            
            # Clipping indici
            ix_min, ix_max = max(0, ix_min), min(self.N_VOXELS_X, ix_max + 1)
            iy_min, iy_max = max(0, iy_min), min(self.N_VOXELS_Y, iy_max + 1)
            iz_min, iz_max = max(0, iz_min), min(self.N_VOXELS_Z, iz_max + 1)
            
            self.grid[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] = 2

    def update_free_space_in_fov(self, T_world_cam):
        # Check solo sui voxel non liberi
        indices_to_check = np.argwhere(self.grid == 0)
        if len(indices_to_check) == 0: return

        # Calcola centro dei voxel in World
        # ATTENZIONE: Ora usiamo res_x, res_y, res_z separati
        pts_world = np.zeros((len(indices_to_check), 4))
        pts_world[:, 0] = self.WS_BOUNDS['x'][0] + (indices_to_check[:, 0] + 0.5) * self.res_x
        pts_world[:, 1] = self.WS_BOUNDS['y'][0] + (indices_to_check[:, 1] + 0.5) * self.res_y
        pts_world[:, 2] = self.WS_BOUNDS['z'][0] + (indices_to_check[:, 2] + 0.5) * self.res_z
        pts_world[:, 3] = 1.0

        # World -> Camera
        # Inversa rigida: R.T * (P - t)
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]
        
        # P_cam = R^T * (P_world - T)
        pts_cam = (pts_world[:, :3] - t_wc) @ R_wc 
        
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)
        
        # Check Piramide (Z-forward convention)
        mask = (z > 0.05) & (z < self.MAX_DEPTH) & \
               (np.abs(x) < (z * tan_h)) & \
               (np.abs(y) < (z * tan_v))
        
        free_indices = indices_to_check[mask]
        if len(free_indices) > 0:
            self.grid[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] = 1

    def publish_visualization_loop(self):
        self.publish_voxel_grid()
        if self.latest_cam_pose is not None:
            self.publish_frustum()
    
    def publish_voxel_grid(self):
        marker = Marker()
        marker.header.frame_id = "fr3_link0" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "voxel_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        
        # Scale dei voxel
        marker.scale = Vector3(x=self.res_x*0.95, y=self.res_y*0.95, z=self.res_z*0.95)
        
        # --- DEFINIZIONE COLORI ---
        c_unknown = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.6) # Grigio trasparente
        c_occupied = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # Rosso pieno
        c_target   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # Giallo pieno (TARGET)
        
        # 1. Unknown (Grigio)
        idx_unknown = np.argwhere(self.grid == 0)
        if len(idx_unknown) > 15000: # Downsampling
             idx_unknown = idx_unknown[np.random.choice(len(idx_unknown), 15000, replace=False)]

        for ix, iy, iz in idx_unknown:
            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_unknown)
            
        # 2. Occupied (Rosso)
        idx_occupied = np.argwhere(self.grid == 2)
        for ix, iy, iz in idx_occupied:
            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_occupied)

        # --- 3. TARGET (GIALLO) ---
        # Disegniamo il target sopra a tutto il resto
        if self.current_target_idx is not None:
            # Spacchettiamo la tupla dell'indice
            t_ix, t_iy, t_iz = self.current_target_idx
            
            # Calcoliamo la posizione nel mondo
            p_target = self.grid_to_world(t_ix, t_iy, t_iz)
            
            # Aggiungiamo il punto e il colore giallo alla lista
            marker.points.append(Point(x=p_target[0], y=p_target[1], z=p_target[2]))
            marker.colors.append(c_target)
            
        self.pub_viz.publish(marker)

    def publish_frustum(self):
        """ Disegna la piramide orientata correttamente nel world frame """
        marker = Marker()
        marker.header.frame_id = "fr3_link0" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "camera_frustum"
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale = Vector3(x=0.005, y=0.0, z=0.0)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.8) # Cyan

        # Setup Trasformazione
        r_cam = R.from_quat(self.latest_cam_pose['quat'])
        t_cam = self.latest_cam_pose['pos']
        R_wc = r_cam.as_matrix() # Matrice di rotazione (3x3)
        
        # Geometria (Z è la profondità ottica)
        z = self.MAX_DEPTH # 50 cm
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)
        x = z * tan_h
        y = z * tan_v
        
        # Punti nel FRAME CAMERA (Optical: Z-forward)
        p_c_origin = np.array([0, 0, 0])
        p_c_tl = np.array([-x, -y, z]) 
        p_c_tr = np.array([ x, -y, z]) 
        p_c_br = np.array([ x,  y, z]) 
        p_c_bl = np.array([-x,  y, z]) 
        
        corners_cam = [p_c_origin, p_c_tl, p_c_tr, p_c_br, p_c_bl]
        
        # TRASFORMAZIONE MANUALE: P_world = R * P_cam + T
        # Se questo passaggio è fatto bene, la piramide DEVE ruotare.
        pts_w = []
        for p in corners_cam:
            # Rotazione + Traslazione
            p_rot = R_wc @ p 
            p_final = p_rot + t_cam
            pts_w.append(p_final)

        origin, tl, tr, br, bl = pts_w
        
        lines = [
            origin, tl, origin, tr, origin, br, origin, bl, # Raggi
            tl, tr, tr, br, br, bl, bl, tl                  # Base rettangolare
        ]
        
        for p in lines:
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            
        self.pub_frustum.publish(marker)

    def decision_loop(self):
        if self.latest_cam_pose is None: return
        cam_pos = self.latest_cam_pose['pos']
        
        if self.current_target_idx:
            if self.grid[self.current_target_idx] != 0:
                self.current_target_idx = None
            else:
                target_voxel_world = self.grid_to_world(*self.current_target_idx)
                self.publish_command(target_voxel_world)
                return

        best_idx = self.find_best_unknown_target(cam_pos)
        if best_idx:
            self.current_target_idx = best_idx
            target_pos = self.grid_to_world(*best_idx)
            self.publish_command(target_pos)

    def find_best_unknown_target(self, robot_pos):
        unknown_indices = np.argwhere(self.grid == 0)
        if len(unknown_indices) == 0: return None

        candidates = unknown_indices
        # Campionamento se ce ne sono troppi (per non rallentare)
        if len(candidates) > 2000000000:###############################################################################
            indices = np.random.choice(len(candidates), 200, replace=False)
            candidates = candidates[indices]

        best_score = -np.inf
        best_idx = None

        # --- PARAMETRI PER NORMALIZZAZIONE ---
        # 1. Limiti Y per Frontality
        y_min = self.WS_BOUNDS['y'][0]
        y_max = self.WS_BOUNDS['y'][1]
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
        
        # 2. Distanza Massima Teorica (Diagonale del Workspace)
        # Usiamo 1.5m come stima sicura per normalizzare
        max_dist_norm = 1.5 

        for idx in candidates:
            pos = self.grid_to_world(*idx)

            # --- A. FRONTALITY (0.0 a 1.0) ---
            # Vogliamo priorità alla BASE (Y minore).
            # Se pos[1] == y_min (0.4), ratio è 0 -> score 1.0 (OTTIMO)
            # Se pos[1] == y_max (1.0), ratio è 1 -> score 0.0 (PESSIMO)
            y_ratio = (pos[1] - y_min) / y_range
            norm_frontality = 1.0 - np.clip(y_ratio, 0.0, 1.0)

            # --- B. DISTANZA (0.0 a 1.0) ---
            # Più vicino = Punteggio più alto
            real_dist = np.linalg.norm(pos - robot_pos)
            dist_ratio = min(real_dist / max_dist_norm, 1.0)
            norm_dist_score = 1.0 - dist_ratio

            # --- C. RELEVANCE (0.0 a 1.0) ---
            # Già normalizzata dalla funzione get_relevance (0=isolato, 1=tanti unknown vicini)
            norm_relevance = self.get_relevance(idx[0], idx[1], idx[2])

            # --- SCORE FINALE ---
            # Pesi (Tunabili):
            # 3.0 * Frontality -> Priorità assoluta a scansionare dal basso verso l'alto
            # 2.0 * Relevance  -> Preferiamo zone "succose" con tanti ignoti
            # 1.0 * Distanza   -> Preferiamo non muoverci troppo se possibile
            score = (3.0 * norm_frontality) + (1.5 * norm_relevance) + (2.0 * norm_dist_score)
            
            if score > best_score:
                best_score = score
                best_idx = tuple(idx)

        return best_idx

    def get_relevance(self, ix, iy, iz):
        x_min, x_max = max(0, ix - 1), min(self.N_VOXELS_X, ix + 2)
        y_min, y_max = max(0, iy - 1), min(self.N_VOXELS_Y, iy + 2)
        z_min, z_max = max(0, iz - 1), min(self.N_VOXELS_Z, iz + 2)
        neighborhood = self.grid[x_min:x_max, y_min:y_max, z_min:z_max]
        unknown_neighbors = np.sum(neighborhood == 0) - 1
        return max(0, unknown_neighbors) / 26.0

    def publish_command(self, target_pos):
        msg = VmcControlTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_attractor = Point(x=target_pos[0], y=target_pos[1], z=target_pos[2])
        for obs_pos in self.current_visible_obstacles:
            msg.active_obstacles.append(Point(x=obs_pos[0], y=obs_pos[1], z=obs_pos[2]))
        self.pub_target.publish(msg)

    def grid_to_world(self, ix, iy, iz):
        x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        return np.array([x, y, z])
        
    def world_to_grid(self, p):
        # Protezione divisione per zero e clamping
        ix = int((p[0] - self.WS_BOUNDS['x'][0]) / self.res_x)
        iy = int((p[1] - self.WS_BOUNDS['y'][0]) / self.res_y)
        iz = int((p[2] - self.WS_BOUNDS['z'][0]) / self.res_z)
        return ix, iy, iz

def main(args=None):
    rclpy.init(args=args)
    node = MapLogicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
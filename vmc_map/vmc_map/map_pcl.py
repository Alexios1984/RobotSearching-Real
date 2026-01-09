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


class IgnoredZone:
    def __init__(self, indices):
        self.member_indices = indices  # Lista di tuple (x,y,z) o array
        self.total_count = len(indices)

class MapLogicNode(Node):
    def __init__(self):
        super().__init__('map_pcl')

        # LOCK PER IL LOOP DECISIONALE
        self.decision_lock = threading.Lock()

        # --- WORKSPACE CONFIGURATION ---
        # Definition of the global limits (World Frame)
        self.WS_BOUNDS = {
            'x': [-0.2, 0.5],   
            'y': [ 0.45, 0.9],
            'z': [ 0.2, 0.85]
        }

        # Dimensions of the workspace
        self.dim_x = self.WS_BOUNDS['x'][1] - self.WS_BOUNDS['x'][0]
        self.dim_y = self.WS_BOUNDS['y'][1] - self.WS_BOUNDS['y'][0]
        self.dim_z = self.WS_BOUNDS['z'][1] - self.WS_BOUNDS['z'][0]
        
        # --- VOXELS RESOLUTION ---
        self.VOXEL_SIZE = 0.035  
        # self.N_VOXELS_X = 20 
        # self.N_VOXELS_Y = 20
        # self.N_VOXELS_Z = 20
        self.N_VOXELS_X = int(np.ceil(self.dim_x / self.VOXEL_SIZE))
        self.N_VOXELS_Y = int(np.ceil(self.dim_y / self.VOXEL_SIZE))
        self.N_VOXELS_Z = int(np.ceil(self.dim_z / self.VOXEL_SIZE))

        # Voxels' dimensions
        self.res_x = self.dim_x / self.N_VOXELS_X
        self.res_y = self.dim_y / self.N_VOXELS_Y
        self.res_z = self.dim_z / self.N_VOXELS_Z

        self.get_logger().info(f"Workspace: {self.dim_x:.2f}x{self.dim_y:.2f}x{self.dim_z:.2f} m")
        self.get_logger().info(f"Voxel Res: {self.res_x:.3f}x{self.res_y:.3f}x{self.res_z:.3f} m")
        self.get_logger().info(f"Grid Size: {self.N_VOXELS_X}x{self.N_VOXELS_Y}x{self.N_VOXELS_Z}")

        # --- CONFIGURAZIONE PROBABILISTICA (I SECCHI) ---
        self.VAL_MIN = 0        # Secchio vuoto
        self.VAL_MAX = 100      # Secchio pieno
        self.VAL_UNKNOWN = 50   # NUOVO CENTRO
        self.VAL_OCCUPIED = 80  # Soglia per dire "È un muro!"
        self.VAL_FREE = 20      # Soglia per dire "È libero sicurissimo"
        self.VAL_LOCK = self.VAL_MAX
        
        # Pesi
        self.HIT_INC = 20      # Quanta acqua verso se vedo un ostacolo (Fast Attack)
        self.MISS_DEC = 10    # Quanta acqua perdo se vedo libero (Slow Decay)

        # Griglia: Usiamo float o int. Int8 va bene (0-100).
        # Inizializziamo a 50 (Incertezza totale) o 0 (Vuoto). 
        # Partiamo da 0 per pulizia.
        self.grid = np.full(
            (self.N_VOXELS_X, self.N_VOXELS_Y, self.N_VOXELS_Z), 
            self.VAL_UNKNOWN, 
            dtype=np.int16
        )        # --- CAMERA CONFIG ---
        self.FOV_H = np.deg2rad(87.0) 
        self.FOV_V = np.deg2rad(58.0)
        self.MAX_DEPTH = 0.40 # <--- IMPOSTATO A 40 per sicurezza

        self.current_visible_obstacles = [] 

        self.locked_objects = set()
        self.latest_cam_pose = None

        self.cb_group = ReentrantCallbackGroup()

        # --- DEADLOCK CONFIG ---
        self.DEADLOCK_VEL_EPS = 0.02      # 2 cm/s (Soglia velocità)
        self.DEADLOCK_TORQUE_EPS = 0.20   # 0.2 Nm (Soglia coppia)
        self.DEADLOCK_TIME_THRESHOLD = 1.0 

        self.DEADLOCK_MIN_DIST = 0.40  # 50 cm di raggio di fuga
        self.last_deadlock_pos = None  # Coordinate (x,y,z) dell'ultimo deadlock
        
        self.last_moving_time = time.time() # Ultima volta che ci siamo mossi
        self.is_deadlocked = False
        self.ignored_targets = set()      # La "Blacklist" dei target falliti

        self.ignored_zones = [] # Lista di oggetti IgnoredZone
        self.DEADLOCK_ZONE_RADIUS = 0.30 # 30cm di raggio attorno al voxel fallito

        # --- Subscribers SINCRONIZZATI ---
        # Invece di create_subscription, usiamo message_filters.Subscriber
        self.sub_vision_filter = message_filters.Subscriber(self, PointCloud2, '/vision/obstacle_cloud',callback_group=self.cb_group)
        self.sub_robot_filter = message_filters.Subscriber(self, VmcRobotState, '/vmc/robot_state', callback_group=self.cb_group)
        self.sub_deadlock = self.create_subscription(
            Float64MultiArray, '/vmc/deadlock_data', self.cb_deadlock, 10, callback_group=self.cb_group
        )

        # Creiamo il Sincronizzatore
        # queue_size=10: Quanti messaggi tenere in memoria per cercare il match
        # slop=0.05: Tolleranza di 50ms (se i tempi differiscono meno di così, li accoppia)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_vision_filter, self.sub_robot_filter], 
            queue_size=25, 
            slop=0.05
        )
        self.ts.registerCallback(self.cb_synced_data)

        # --- Publisher --- 
        self.pub_repulsors_viz = self.create_publisher(Marker, '/vmc/repulsors_viz', 10)
        self.pub_target = self.create_publisher(VmcControlTarget, '/vmc/target', 10)

        self.pub_debug_pcl = self.create_publisher(PointCloud2, '/vmc/debug_obstacles', 10)
        
        self.pub_viz = self.create_publisher(Marker, '/vmc/voxel_grid_viz', 10)
        self.pub_frustum = self.create_publisher(Marker, '/vmc/camera_frustum', 10)

        self.pub_ws_bounds = self.create_publisher(Marker, '/vmc/workspace_bounds', 10)

        self.pub_status_text = self.create_publisher(Marker, '/vmc/status_text', 10)

        # Timer Visualization (2 Hz)
        self.timer_grid_viz = self.create_timer(0.5, self.publish_grid_worker, callback_group=self.cb_group)
        self.timer_fast_viz = self.create_timer(0.01, self.publish_fast_viz, callback_group=self.cb_group)

        # Logic Timer (10Hz)
        self.timer = self.create_timer(0.001, self.decision_loop, callback_group=self.cb_group)
        self.current_target_idx = None 


    def cb_deadlock(self, msg):
        # msg.data = [max_link_vel, max_joint_torque]
        if len(msg.data) < 2: return
        
        current_vel = msg.data[0]
        current_tau = msg.data[1]
        
        # Logica: Se siamo sotto entrambe le soglie, il timer parte.
        # Se ci muoviamo (sopra soglia), resettiamo il timer.
        
        is_static = (current_vel < self.DEADLOCK_VEL_EPS) #=and (current_tau < self.DEADLOCK_TORQUE_EPS)=#
        
        if not is_static:
            self.last_moving_time = time.time() # Reset timer
            self.is_deadlocked = False
        else:
            # Siamo fermi. Da quanto tempo?
            time_stuck = time.time() - self.last_moving_time
            
            if time_stuck > self.DEADLOCK_TIME_THRESHOLD:
                if not self.is_deadlocked and self.current_target_idx is not None:
                    self.get_logger().warn(f"💀 DEADLOCK DETECTED! Stuck for {time_stuck:.1f}s. Resetting target.")
                    self.is_deadlocked = True

    def cb_synced_data(self, pcl_msg, state_msg):
        """
        Callback Sincronizzata: Riceve PointCloud e Stato Robot con timestamp allineati.
        Esegue filtraggio, trasformazione e aggiornamento mappa.
        """
        
        # --- 1. ESTRAZIONE POSA SINCRONIZZATA ---
        # Usiamo ESPLICITAMENTE i dati di QUESTO messaggio per trasformare QUESTA nuvola.
        # Questo elimina l'effetto "swimming" (scia/ritardo).
        current_cam_pose = {
            'pos': np.array([state_msg.cam_pose.position.x, state_msg.cam_pose.position.y, state_msg.cam_pose.position.z]),
            'quat': np.array([state_msg.cam_pose.orientation.x, state_msg.cam_pose.orientation.y, state_msg.cam_pose.orientation.z, state_msg.cam_pose.orientation.w])
        }

        # Aggiorniamo la variabile globale per gli altri thread (Visualizzazione Frustum, Logica Target)
        # Nota: La logica di decisione userà l'ultima posa disponibile, che va bene,
        # ma la MAPPATURA deve usare quella sincrona.
        self.latest_cam_pose = current_cam_pose 

        # --- 2. SETUP MATRICE TRASFORMAZIONE (Camera -> World) ---
        r_cam = R.from_quat(current_cam_pose['quat'])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = r_cam.as_matrix()
        T_world_cam[:3, 3] = current_cam_pose['pos']

        # --- 3. PULIZIA SPAZIO LIBERO (Raycasting approssimato) ---
        # "Svuotiamo" il cono davanti alla camera prima di riempirlo con i nuovi ostacoli.
        self.update_free_space_in_fov(T_world_cam)

        # --- 4. PARSING POINTCLOUD ---
        # Leggiamo i punti. skip_nans=True rimuove i punti invalidi (NaN/Inf)
        cloud_data = pc2.read_points_numpy(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        if cloud_data.shape[0] == 0:
            return

        # Gestione robusta del tipo di array (Strutturato vs Matrice)
        if cloud_data.dtype.names:
            points_cam = np.column_stack([cloud_data['x'], cloud_data['y'], cloud_data['z']])
        else:
            points_cam = cloud_data

        # --- 5. TRASFORMAZIONE NEL WORLD FRAME ---
        # P_world = T * P_cam
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_h = np.hstack([points_cam, ones])
        
        # Trasformiamo e prendiamo solo x,y,z
        points_world = (T_world_cam @ points_cam_h.T).T[:, :3]

        # --- 6. FILTRO WORKSPACE (Crop Box) ---
        x_min, x_max = self.WS_BOUNDS['x']
        y_min, y_max = self.WS_BOUNDS['y']
        z_min, z_max = self.WS_BOUNDS['z']

        mask_ws = (
            (points_world[:, 0] >= x_min) & (points_world[:, 0] <= x_max) &
            (points_world[:, 1] >= y_min) & (points_world[:, 1] <= y_max) &
            (points_world[:, 2] >= z_min) & (points_world[:, 2] <= z_max)
        )

        valid_points = points_world[mask_ws]

        if valid_points.shape[0] == 0:
            return
        
        # --- FILTRO SPAZIALE (Radius Outlier Removal) ---
        # TUNING DEL BRO: Stringiamo i parametri per uccidere i punti isolati!
        radius = 0.05          # Riduciamo a 5cm (prima era 15). Se non hai amici a 5cm, sei rumore.
        min_neighbors = 6      # Aumentiamo a 6. Devi essere in un gruppo solido.

        # --- 7. IL BRO FILTER (Radius Outlier Removal) 🧹 ---
        # Rimuove i "flying pixels" (punti isolati/rumore).
        # Un punto sopravvive solo se ha almeno 'min_neighbors' entro 'radius'.
        
        radius = 0.15          # 15 cm
        min_neighbors = 4      # Densità minima
        
        if len(valid_points) >= min_neighbors:
            # Creiamo l'albero KDTree (molto veloce)
            tree = cKDTree(valid_points)
            
            # Calcoliamo le distanze dei vicini
            # k=min_neighbors significa "cerca fino al 4° vicino"
            # distance_upper_bound=radius taglia la ricerca se sono troppo lontani
            dists, _ = tree.query(valid_points, k=min_neighbors, distance_upper_bound=radius)
            
            # Se la distanza del k-esimo vicino è infinita, vuol dire che non esiste entro il raggio.
            # Teniamo solo quelli che hanno vicini reali.
            mask_clean = dists[:, -1] != float('inf')
            
            valid_points = valid_points[mask_clean]

        if valid_points.shape[0] == 0:
            return

        # --- 8. VISUALIZZAZIONE DEBUG (Opzionale ma utile) ---
        # Pubblichiamo su RViz esattamente ciò che stiamo per mettere nella griglia.
        # Usiamo il frame "fr3_link0" perché i punti sono già in coordinate World.
        debug_header = Header()
        debug_header.frame_id = "fr3_link0" 
        debug_header.stamp = self.get_clock().now().to_msg()
        
        pc2_msg = pc2.create_cloud_xyz32(debug_header, valid_points)
        self.pub_debug_pcl.publish(pc2_msg)
        
        # --- 9. DOWNSAMPLING ---
        # Prendiamo un punto ogni N per non appesantire il mapping, 
        # tanto i voxel sono grandi.
        step = 1 # Puoi aumentarlo a 2 o 3 se la CPU soffre
        final_points = valid_points[::step]
        
        # Aggiorniamo la lista per i repulsori (usata poi in publish_command)
        # Nota: Qui potremmo usare get_downsampled_obstacles, ma per visualizzazione raw teniamo questo
        self.current_visible_obstacles = final_points 

        # --- 10. MAPPING (Aggiornamento Griglia) ---
        # Convertiamo coordinate metriche -> indici griglia
        ix = ((final_points[:, 0] - x_min) / self.res_x).astype(int)
        iy = ((final_points[:, 1] - y_min) / self.res_y).astype(int)
        iz = ((final_points[:, 2] - z_min) / self.res_z).astype(int)

        # Clipping di sicurezza
        ix = np.clip(ix, 0, self.N_VOXELS_X - 1)
        iy = np.clip(iy, 0, self.N_VOXELS_Y - 1)
        iz = np.clip(iz, 0, self.N_VOXELS_Z - 1)

        # 1. Recupera indici unici
        unique_indices = np.unique(np.column_stack((ix, iy, iz)), axis=0)
        uix, uiy, uiz = unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]
        
        # 2. Incrementa (Fast Attack)
        current_vals = self.grid[uix, uiy, uiz]
        new_vals = current_vals + self.HIT_INC
        
        # 3. Clamp a VAL_LOCK (100)
        # Una volta che tocca 100, la funzione update_free_space non lo toccherà più.
        new_vals = np.clip(new_vals, self.VAL_MIN, self.VAL_LOCK)
        
        self.grid[uix, uiy, uiz] = new_vals

    def publish_grid_worker(self):
        """ Worker Lento: Disegna i cubetti (pesante) """
        # Eseguiamo in un thread separato per non bloccare il timer veloce
        threading.Thread(target=self._heavy_viz_task).start()

    def _heavy_viz_task(self):
        self.publish_voxel_grid()
        self.publish_workspace_boundary()
        self.publish_exploration_status()

    def publish_fast_viz(self):
        """ Worker Veloce: Disegna solo il Frustum (leggerissimo) """
        if self.latest_cam_pose is not None:
            self.publish_frustum()
    
    def create_ignored_zone(self, center_idx):
        """
        Trova tutti i voxel entro un raggio R dal centro e crea una IgnoredZone.
        Restituisce la lista degli indici trovati.
        """
        cx, cy, cz = center_idx
        
        # Calcoliamo il raggio in unità voxel
        # Es. 30cm / 3.5cm = 8.5 voxel
        r_vox = int(np.ceil(self.DEADLOCK_ZONE_RADIUS / self.res_x))
        
        # Bounding box per non iterare su tutta la griglia
        ix_min, ix_max = max(0, cx - r_vox), min(self.N_VOXELS_X, cx + r_vox + 1)
        iy_min, iy_max = max(0, cy - r_vox), min(self.N_VOXELS_Y, cy + r_vox + 1)
        iz_min, iz_max = max(0, cz - r_vox), min(self.N_VOXELS_Z, cz + r_vox + 1)
        
        # Creiamo griglia coordinate locale
        x_range = np.arange(ix_min, ix_max)
        y_range = np.arange(iy_min, iy_max)
        z_range = np.arange(iz_min, iz_max)
        
        # Meshgrid 3D
        # Nota: 'ij' indexing per coerenza con le matrici
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Calcolo distanza al quadrato (più veloce della radice)
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        r_sq = r_vox**2
        
        # Maschera sferica
        mask = dist_sq <= r_sq
        
        # Estraiamo gli indici validi
        valid_x = X[mask]
        valid_y = Y[mask]
        valid_z = Z[mask]
        
        # Creiamo lista di tuple per il set
        indices = list(zip(valid_x, valid_y, valid_z))
        
        if len(indices) > 0:
            # Creiamo l'oggetto zona e lo salviamo
            new_zone = IgnoredZone(indices)
            self.ignored_zones.append(new_zone)
            self.get_logger().warn(f"🚫 Creata BLACKZONE con {len(indices)} voxel (R={self.DEADLOCK_ZONE_RADIUS}m)")
            
        return indices

    def update_free_space_in_fov(self, T_world_cam):
        """
        Versione "Full Grid & Sync": 
        1. Usa la matrice T_world_cam SINCRONIZZATA passata come argomento.
        2. Non usa subgrid (lavora su tutto il workspace).
        3. Usa la logica probabilistica (Secchi d'acqua).
        """
        
        # --- 1. SELEZIONE CANDIDATI (Ottimizzazione) ---
        # Invece di controllare 8000 voxel geometricamente, controlliamo PRIMA
        # chi ha bisogno di essere aggiornato.
        # Se un voxel è già al minimo (0), è inutile sottrarre valore. 
        # Risparmiamo calcoli escludendo i voxel già vuoti.
        
        # Cerchiamo indici dove il valore è > VAL_MIN (0)
        # Questo sostituisce il vecchio "argwhere != 1" e rimuove la necessità della subgrid.
        indices_to_check = np.argwhere((self.grid > self.VAL_MIN) & (self.grid < self.VAL_LOCK))

        if len(indices_to_check) == 0:
            return

        # --- 2. CALCOLO COORDINATE WORLD ---
        # Vettorizzazione pura: calcoliamo le posizioni nel mondo di tutti i candidati
        ix = indices_to_check[:, 0]
        iy = indices_to_check[:, 1]
        iz = indices_to_check[:, 2]

        pts_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        pts_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        pts_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z

        # Creiamo matrice (N, 4) per la moltiplicazione omogenea
        pts_world = np.column_stack((pts_x, pts_y, pts_z, np.ones_like(pts_x)))

        # --- 3. PROIEZIONE NEL CAMERA FRAME (SYNC FIX) ---
        # Qui sta la correzione fondamentale: Usiamo T_world_cam passata come argomento!
        # Non tocchiamo self.latest_cam_pose.
        T_cam_world = np.linalg.inv(T_world_cam)
        
        # Moltiplicazione: Trasformiamo i punti dal Mondo alla Camera
        pts_cam = (T_cam_world @ pts_world.T).T # Risultato (N, 4)

        x_c = pts_cam[:, 0]
        y_c = pts_cam[:, 1]
        z_c = pts_cam[:, 2]

        # --- 4. GEOMETRIA FRUSTUM (Idrante) ---
        tan_h = np.tan(self.FOV_H / 2.0)
        tan_v = np.tan(self.FOV_V / 2.0)

        # Maschera Geometrica
        # z > 0.05: Near clip (non pulire i primi 5cm per sicurezza braccio)
        # z < MAX_DEPTH: Fino a dove arriva l'idrante
        mask_fov = (z_c > 0.05) & (z_c < self.MAX_DEPTH) & \
                   (np.abs(x_c) < (z_c * tan_h)) & \
                   (np.abs(y_c) < (z_c * tan_v))

        # --- 5. APPLICAZIONE DECAY (Secchi d'Acqua) ---
        # Filtriamo gli indici originali usando la maschera geometrica
        valid_indices = indices_to_check[mask_fov]

        if len(valid_indices) > 0:
            # Estraiamo le coordinate grid dei voxel colpiti
            v_ix = valid_indices[:, 0]
            v_iy = valid_indices[:, 1]
            v_iz = valid_indices[:, 2]

            # Lettura valore attuale
            current_vals = self.grid[v_ix, v_iy, v_iz]

            # Sottrazione (Evaporazione fiducia)
            new_vals = current_vals - self.MISS_DEC

            # Clamp a VAL_MIN (0) e Scrittura
            self.grid[v_ix, v_iy, v_iz] = np.clip(new_vals, self.VAL_MIN, self.VAL_MAX)
            
            # Opzionale: Debug
            # self.get_logger().info(f"🧹 Pulizia Sync: {len(valid_indices)} voxel aggiornati.")

    def publish_voxel_grid(self):
        marker = Marker()
        marker.header.frame_id = "fr3_link0" 
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "voxel_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        
        # Scale dei voxel
        # Nota: Li facciamo leggermente più piccoli per vedere "attraverso"
        marker.scale = Vector3(x=self.res_x, y=self.res_y, z=self.res_z)
        
        # --- DEFINIZIONE COLORI ---
        # Free: Verde/Ciano, molto trasparente (Alpha basso)
        c_free = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.2) 
        
        # Occupied: Rosso pieno, opaco
        c_occupied = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9) 
        
        # Target: Giallo
        c_target   = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        # --- 1. FREE SPACE (Il trucco della "Nebbia") ---
        # Invece di Unknown, mostriamo dove abbiamo già guardato!
        idx_free = np.argwhere((self.grid < self.VAL_FREE) & (self.grid > 0))
        
        # DOWNSAMPLING AGGRESSIVO:
        # Ne prendiamo solo 1 ogni 8 (o 10). 
        # Questo riduce il carico dell'80-90% ma ti fa vedere comunque il volume.
        # step_free = 8 
        # if len(idx_free) > 0:
        #     idx_free_viz = idx_free[::step_free] 
            
        #     for ix, iy, iz in idx_free_viz:
        #         p = self.grid_to_world(ix, iy, iz)
        #         marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
        #         marker.colors.append(c_free)
            
        # --- 2. OCCUPIED (Rosso) ---
        # Questi sono pericoli, li vogliamo vedere bene.
        idx_occupied = np.argwhere(self.grid > self.VAL_OCCUPIED)
        
        # Downsampling leggero solo se sono troppi (es. un muro enorme)
        step_occ = 1
        if len(idx_occupied) > 5000:
             step_occ = 2
        
        idx_occupied_viz = idx_occupied[::step_occ]

        for ix, iy, iz in idx_occupied_viz:
            p = self.grid_to_world(ix, iy, iz)
            marker.points.append(Point(x=p[0], y=p[1], z=p[2]))
            marker.colors.append(c_occupied)

        # --- 3. TARGET (GIALLO) ---
        if self.current_target_idx is not None:
            t_ix, t_iy, t_iz = self.current_target_idx
            p_target = self.grid_to_world(t_ix, t_iy, t_iz)
            
            # Target lo facciamo un po' più grosso magari (sovrascrivendo scale se fosse un marker diverso, 
            # ma qui è una list, quindi si becca la scale globale. Va bene uguale).
            marker.points.append(Point(x=p_target[0], y=p_target[1], z=p_target[2]))
            marker.colors.append(c_target)
            
        self.pub_viz.publish(marker)

    def publish_exploration_status(self):
        """
        Calcola la % di voxel esplorati (Free + Occupied) e pubblica un testo 3D.
        """
        # 1. Calcolo Statistiche
        total_voxels = self.N_VOXELS_X * self.N_VOXELS_Y * self.N_VOXELS_Z
        
        # Conta tutto ciò che NON è 0 (Unknown). 
        # Quindi conta sia 1 (Free) che 2 (Occupied).
        mask_explored = (self.grid < self.VAL_FREE) | (self.grid > self.VAL_OCCUPIED)
        
        # Contiamo quanti 'True' ci sono nella maschera
        explored_count = np.count_nonzero(mask_explored)
        
        # Calcolo percentuale
        percentage = (explored_count / total_voxels) * 100.0

        # 2. Creazione Marker Testo
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

        # Lo spostiamo leggermente più avanti (+0.05 su X) per non compenetrare il wireframe
        marker.pose.position.x = x_max + 0.1
        marker.pose.position.y = y_center
        # Lo mettiamo 10cm sopra il tetto del box per leggibilità
        marker.pose.position.z = z_min 
        
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.scale.x = 0.15
        marker.scale.y = 0.0
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.text = f"Exploration:{percentage:.1f}%"

        self.pub_status_text.publish(marker)

    def publish_workspace_boundary(self):
        """
        Disegna un 'cubo' (wireframe) che rappresenta i limiti del workspace.
        """
        marker = Marker()
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workspace_bounds"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Spessore della linea (1cm va bene)
        marker.scale.x = 0.005 
        
        # Colore: Bianco Semitrasparente (elegante e non intrusivo)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.4)

        # Coordinate dei limiti
        x_min, x_max = self.WS_BOUNDS['x']
        y_min, y_max = self.WS_BOUNDS['y']
        z_min, z_max = self.WS_BOUNDS['z']

        # Gli 8 vertici del box
        p0 = Point(x=x_min, y=y_min, z=z_min)
        p1 = Point(x=x_max, y=y_min, z=z_min)
        p2 = Point(x=x_max, y=y_max, z=z_min)
        p3 = Point(x=x_min, y=y_max, z=z_min)
        
        p4 = Point(x=x_min, y=y_min, z=z_max)
        p5 = Point(x=x_max, y=y_min, z=z_max)
        p6 = Point(x=x_max, y=y_max, z=z_max)
        p7 = Point(x=x_min, y=y_max, z=z_max)

        # Definiamo le 12 linee (ogni coppia è un segmento)
        lines = [
            # Base (Z min)
            p0, p1,  p1, p2,  p2, p3,  p3, p0,
            # Tetto (Z max)
            p4, p5,  p5, p6,  p6, p7,  p7, p4,
            # Colonne verticali
            p0, p4,  p1, p5,  p2, p6,  p3, p7
        ]
        
        marker.points = lines
        self.pub_ws_bounds.publish(marker)

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

        if not self.decision_lock.acquire(blocking=False):
            return

        try:
            if self.latest_cam_pose is None: return
            cam_pos = self.latest_cam_pose['pos']
            
            # Snapshot locale per evitare race conditions (MultiThreading safety)
            # Copiamo il valore attuale in una variabile locale e usiamo quella.
            curr_idx = self.current_target_idx 

            self.check_and_liberate_zones()
            
            # --- GESTIONE DEADLOCK ---
            if self.is_deadlocked:
                # Controllo di sicurezza: Abbiamo davvero un target da bannare?
                if curr_idx is not None:
                    self.get_logger().warn(f"🚫 Target {curr_idx} in DEADLOCK. Banno l'intera zona!")
                    
                    # 1. Calcola e registra la zona
                    zone_indices = self.create_ignored_zone(curr_idx)
                    
                    # 2. Aggiungi TUTTI i vicini alla blacklist globale
                    for idx in zone_indices:
                        self.ignored_targets.add(idx)
                    
                    # 3. Salva la posizione per la logica Safe/Risky
                    self.last_deadlock_pos = self.grid_to_world(*curr_idx)
                    
                    # 4. Reset
                    self.current_target_idx = None
                else:
                    self.get_logger().warn("💀 Deadlock Phantom. Reset.")
                # Reset flag deadlock in ogni caso
                self.is_deadlocked = False
                self.last_moving_time = time.time()
                return # Esci per questo ciclo
                
            # ------------------------------

            # --- CONTROLLO TARGET CORRENTE (Commitment) ---
            # Nota: Usiamo curr_idx (la copia locale) per coerenza
            if curr_idx is not None:
                # Leggiamo il valore attuale del voxel target
                current_val = self.grid[curr_idx]
                
                # DEFINIZIONE DI "VISTO":
                is_resolved = (current_val < self.VAL_FREE) or (current_val > self.VAL_OCCUPIED)
                
                if is_resolved:
                    self.get_logger().info(f"✅ TARGET {curr_idx} RISOLTO! (Val: {current_val}). Ne cerco un altro.")
                    self.current_target_idx = None # Scriviamo sulla variabile globale per liberarla
                else:
                    target_pos = self.grid_to_world(*curr_idx)
                    self.publish_command(target_pos)
                    return
            
            # =========================================================
            # SE SIAMO QUI, SIGNIFICA CHE NON ABBIAMO UN TARGET ATTIVO.
            # SOLO ORA POSSIAMO CERCARNE UNO NUOVO.
            # =========================================================

            best_idx = self.find_best_unknown_target(cam_pos)

            # IL BRO FIX: Logica "Raschiare il Barile"
            if best_idx is None:
                # Se non trovo target validi, MA ho dei target nella blacklist...
                if len(self.ignored_targets) > 0:
                    self.get_logger().warn(f"♻️ FINITI TARGET. Reset Totale (Blacklist + Deadlock Pos).")
                    self.ignored_targets.clear()
                    self.last_deadlock_pos = None  # <--- RESET ANCHE QUESTO
                    
                    best_idx = self.find_best_unknown_target(cam_pos)
                else:
                    self.get_logger().info("🎉 ESPLORAZIONE COMPLETATA (O nessun target raggiungibile)!", throttle_duration_sec=5.0)

            if best_idx:
                self.current_target_idx = best_idx
                target_pos = self.grid_to_world(*best_idx)
                self.publish_command(target_pos)
            else:
                # Se è ANCORA None anche dopo il reset, allora abbiamo davvero finito tutto.
                pass
        finally:
            self.decision_lock.release()

    def check_and_liberate_zones(self):
        """
        Controlla tutte le zone ignorate.
        Se > 10% dei voxel di una zona sono stati 'Risolti' (Visti come Free o Occupied),
        libera l'intera zona dalla blacklist.
        """
        if not self.ignored_zones:
            return

        zones_to_remove = []
        
        for zone in self.ignored_zones:
            # Contiamo quanti membri sono stati risolti
            resolved_count = 0
            
            for idx in zone.member_indices:
                val = self.grid[idx]
                # È risolto se NON è ignoto (cioè se è Free o Occupied)
                # Nota: qui usiamo le soglie probabilistiche
                if val < self.VAL_FREE or val > self.VAL_OCCUPIED:
                    resolved_count += 1
            
            # Calcolo percentuale
            percentage = resolved_count / zone.total_count
            
            if percentage >= 0.10: # 10% SOGLIA LIBERAZIONE
                self.get_logger().info(f"🔓 LIBERATA ZONA (Visti {resolved_count}/{zone.total_count} voxel).")
                
                # Rimuoviamo i membri dalla blacklist globale
                for idx in zone.member_indices:
                    # Usiamo discard per non avere errori se non c'è
                    self.ignored_targets.discard(idx)
                
                zones_to_remove.append(zone)

        # Pulizia lista zone
        for z in zones_to_remove:
            try:
                self.ignored_zones.remove(z)
            except ValueError:
                pass # Era già stato rimosso, pazienza. Niente crash.

    def find_best_unknown_target(self, robot_pos):
        """
        Seleziona il miglior voxel ignoto.
        Include: Isteresi, Wall Skin Filter, Blacklist check e Logica Safe/Risky per Deadlock.
        """
        
        # --- 1. SELEZIONE CANDIDATI (Con Isteresi) ---
        # Ignoriamo i voxel che sono "appena" diventati liberi (es. 30).
        # Devono essere sopra una soglia di sicurezza (es. 30 + 15 = 45) per essere interessanti.
        # Questo impedisce il loop "Risolto -> Rumore -> Nuovo Target -> Risolto".
        SOGLIA_SELEZIONE = self.VAL_FREE + 15
        mask_unknown = (self.grid >= SOGLIA_SELEZIONE) & (self.grid <= self.VAL_OCCUPIED)

        # --- 2. WALL SKIN PROTECTION (Filtro Muri) ---
        # Non vogliamo target appiccicati ai muri (spesso sono solo rumore o irraggiungibili).
        mask_obstacles = (self.grid > self.VAL_OCCUPIED)

        if np.any(mask_obstacles):
            # Dilatazione di 2 iterazioni (~7cm): Creiamo un cuscinetto di sicurezza.
            mask_skin = binary_dilation(mask_obstacles, iterations=2)
            
            # I candidati validi sono: IGNOTI ma NON PELLE DEL MURO.
            mask_final_candidates = mask_unknown & (~mask_skin)
        else:
            mask_final_candidates = mask_unknown

        # Estraiamo gli indici
        unknown_indices = np.argwhere(mask_final_candidates)

        # --- FALLBACK 1: Skin troppo aggressiva? ---
        if len(unknown_indices) == 0: 
            # Se il filtro Skin ha rimosso tutto, proviamo a usare i candidati grezzi.
            # Meglio rischiare di avvicinarsi a un muro che fermarsi del tutto.
            unknown_indices = np.argwhere(mask_unknown)
            if len(unknown_indices) == 0:
                return None # Davvero nulla da vedere

        # --- 3. PRE-FILTRAGGIO (Performance & Blacklist) ---
        candidates = []
        
        # Se sono troppi (es. > 5000), ne prendiamo un campione casuale subito per velocità
        temp_indices = unknown_indices
        if len(temp_indices) > 5000:
             rand_idx = np.random.choice(len(temp_indices), 5000, replace=False)
             temp_indices = temp_indices[rand_idx]

        # Filtro Blacklist (Ignored Targets)
        for idx in temp_indices:
            t_idx = tuple(idx)
            if t_idx not in self.ignored_targets:
                candidates.append(idx)
                
        if not candidates: return None
        candidates = np.array(candidates) # Ritorna a numpy array per comodità

        # --- 4. LOGICA DEADLOCK: SAFE vs RISKY ---
        # Se siamo appena usciti da un deadlock, dividiamo i candidati in due gruppi.
        
        safe_candidates = []   # Lontani dal punto di blocco
        risky_candidates = []  # Vicini al punto di blocco
        
        # Pool su cui iterare per lo scoring
        final_pool = [] 

        if self.last_deadlock_pos is not None:
            # Iteriamo per separare i gruppi
            for idx in candidates:
                pos = self.grid_to_world(*idx)
                dist_from_deadlock = np.linalg.norm(pos - self.last_deadlock_pos)
                
                if dist_from_deadlock > self.DEADLOCK_MIN_DIST:
                    safe_candidates.append(idx)
                else:
                    risky_candidates.append(idx)
            
            # DECISIONE PRIORITÀ
            if len(safe_candidates) > 0:
                final_pool = safe_candidates
                # (Opzionale) Downsampling se i safe sono troppi
                if len(final_pool) > 500:
                     rand_idx = np.random.choice(len(final_pool), 500, replace=False)
                     final_pool = [final_pool[i] for i in rand_idx]
                
                # self.get_logger().info(f"🔎 Priorità SAFE: Analizzo {len(final_pool)} target lontani dal deadlock.")
            else:
                # Se non c'è nulla di lontano, dobbiamo rischiare vicino
                final_pool = risky_candidates
                self.get_logger().warn("⚠️ Nessun target Safe! Costretto a usare fallback RISKY.")
        else:
            # Nessun deadlock attivo, tutti i candidati sono buoni
            final_pool = candidates
            # Downsampling standard se sono troppi
            if len(final_pool) > 500:
                rand_idx = np.random.choice(len(final_pool), 500, replace=False)
                # Nota: candidates è np.array, final_pool qui pure
                final_pool = final_pool[rand_idx]

        # --- 5. SCORING LOOP ---
        best_score = -np.inf
        best_idx = None
        
        # Parametri normalizzazione
        y_min, y_max = self.WS_BOUNDS['y']
        y_range = max(y_max - y_min, 1.0)
        max_dist_norm = 1.5 

        for idx in final_pool:
            # Convertiamo indice in coordinate mondo
            pos = self.grid_to_world(*idx)

            # A. FRONTALITY (0.0 - 1.0): Preferiamo spazzare lungo Y in ordine
            y_ratio = (pos[1] - y_min) / y_range
            norm_frontality = 1.0 - np.clip(y_ratio, 0.0, 1.0)

            # B. DISTANCE (0.0 - 1.0): Preferiamo target vicini al robot
            real_dist = np.linalg.norm(pos - robot_pos)
            dist_ratio = min(real_dist / max_dist_norm, 1.0)
            norm_dist_score = 1.0 - dist_ratio

            # C. RELEVANCE (0.0 - 1.0): Preferiamo zone dense di ignoto
            norm_relevance = self.get_relevance(idx[0], idx[1], idx[2])

            # FORMULA SCORE FINALE
            # Pesi:
            # 3.0 Frontality -> Ordine di pulizia
            # 2.0 Relevance  -> Efficienza (scoprire grossi buchi)
            # 1.0 Distanza   -> Efficienza energetica
            score = (3.0 * norm_frontality) + (2.0 * norm_relevance) + (1.0 * norm_dist_score)
            
            if score > best_score:
                best_score = score
                best_idx = tuple(idx)

        return best_idx

    def get_relevance(self, ix, iy, iz):
        # Estrai il cubo 3x3x3 intorno al voxel
        x_min, x_max = max(0, ix - 1), min(self.N_VOXELS_X, ix + 2)
        y_min, y_max = max(0, iy - 1), min(self.N_VOXELS_Y, iy + 2)
        z_min, z_max = max(0, iz - 1), min(self.N_VOXELS_Z, iz + 2)
        
        neighborhood = self.grid[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Vecchio: np.sum(neighborhood == 0)
        # NUOVO: Contiamo quanti vicini sono ancora nel "Limbo" (tra FREE e OCCUPIED)
        # Ovvero quanti sono ancora "Grigi" (~50)
        
        mask_unknown_neighbors = (neighborhood >= self.VAL_FREE) & (neighborhood <= self.VAL_OCCUPIED)
        unknown_neighbors = np.sum(mask_unknown_neighbors)
        
        # Rimuoviamo il voxel centrale dal conteggio (se era unknown anche lui)
        # (Opzionale, ma matematicamente corretto)
        if (self.grid[ix, iy, iz] >= self.VAL_FREE) and (self.grid[ix, iy, iz] <= self.VAL_OCCUPIED):
             unknown_neighbors -= 1
             
        # Normalizziamo su 26 vicini possibili
        return max(0, unknown_neighbors) / 26.0

    def get_downsampled_obstacles(self, min_dist=0.07):
        """
        Estrae i voxel occupati dalla griglia persistente e li filtra
        per mantenere una distanza minima (es. 10cm) tra loro.
        Usa 'Spatial Hashing' per essere ultra-veloce.
        """
        # 1. Trova tutti gli indici dei voxel occupati (2)
        idx_occupied = np.argwhere(self.grid >= self.VAL_OCCUPIED)
        
        if len(idx_occupied) == 0:
            return []

        # 2. Vettorizzazione: Converti tutti gli indici in coordinate World in un colpo solo
        # (Molto più veloce di chiamare grid_to_world in un ciclo for)
        ix = idx_occupied[:, 0]
        iy = idx_occupied[:, 1]
        iz = idx_occupied[:, 2]

        points_x = self.WS_BOUNDS['x'][0] + (ix + 0.5) * self.res_x
        points_y = self.WS_BOUNDS['y'][0] + (iy + 0.5) * self.res_y
        points_z = self.WS_BOUNDS['z'][0] + (iz + 0.5) * self.res_z
        
        # Matrice Nx3 dei punti occupati
        points = np.column_stack((points_x, points_y, points_z))

        # 3. Spatial Hashing (Il trucco del Bro)
        # Dividiamo lo spazio in celle grandi 'min_dist'. 
        # Punti che cadono nella stessa cella avranno la stessa chiave intera.
        # floor(1.25 / 0.07) -> chiave 12. 
        keys = np.floor(points / min_dist).astype(int)

        # Usiamo un dizionario per tenere UN SOLO punto per ogni chiave spaziale
        unique_obstacles = {}
        
        # Iteriamo e riempiamo il dizionario. 
        # Essendo in Python, questo loop è il collo di bottiglia, ma con griglie normali è rapido.
        for i in range(points.shape[0]):
            # La chiave è una tupla (k_x, k_y, k_z)
            k = tuple(keys[i])
            # Se questa cella spaziale non è ancora rappresentata, aggiungiamo il punto
            if k not in unique_obstacles:
                unique_obstacles[k] = points[i]
        
        # Restituisce solo i punti filtrati
        return list(unique_obstacles.values())

    def publish_command(self, target_pos):
        msg = VmcControlTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_attractor = Point(x=target_pos[0], y=target_pos[1], z=target_pos[2])

        # --- MODIFICA IL BRO: USIAMO I VOXEL FILTRATI ---
        # Invece di self.current_visible_obstacles, calcoliamo i repulsori dalla griglia
        # con una spaziatura di 10 cm (0.10).
        optimized_obstacles = self.get_downsampled_obstacles(min_dist=0.10)

        for obs_pos in optimized_obstacles:
            msg.active_obstacles.append(Point(x=obs_pos[0], y=obs_pos[1], z=obs_pos[2]))
            
        self.pub_target.publish(msg)

        # --- VISUALIZZAZIONE REPULSORI ---
        marker = Marker()
        # ... (il resto del codice del marker rimane uguale, userà msg.active_obstacles che ora è corretto)
        marker.header.frame_id = "fr3_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "active_repulsors"
        marker.id = 999
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        
        # Magari aumentiamo leggermente la dimensione visiva per far capire che coprono 10cm
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
        
    def world_to_grid(self, p):
        # Protezione divisione per zero e clamping
        ix = int((p[0] - self.WS_BOUNDS['x'][0]) / self.res_x)
        iy = int((p[1] - self.WS_BOUNDS['y'][0]) / self.res_y)
        iz = int((p[2] - self.WS_BOUNDS['z'][0]) / self.res_z)
        return ix, iy, iz

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
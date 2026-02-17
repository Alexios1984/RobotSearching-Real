import argparse
import os
import re
from pathlib import Path
import pandas as pd

# ROS 2 imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def process_bag(bag_path):
    print(f"📂 Apertura ROS bag: {bag_path}")
    
    # 1. Capire il formato del database (sqlite3 o mcap)
    storage_id = 'sqlite3'
    actual_uri = bag_path
    p = Path(bag_path)
    
    if p.suffix == '.mcap':
        storage_id = 'mcap'
    elif p.is_dir():
        mcap_files = list(p.glob('*.mcap'))
        if mcap_files:
            storage_id = 'mcap'
            actual_uri = str(mcap_files[0])  # Punta dritto al file .mcap!
            
    # 2. Inizializzare il lettore della bag
    reader = SequentialReader()
    # Usiamo actual_uri invece di bag_path
    storage_options = StorageOptions(uri=actual_uri, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"❌ Errore nell'apertura della bag: {e}")
        return

    # 3. Mappare i nomi dei topic ai loro tipi di messaggio
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    # Liste per raccogliere i dati
    target_data = []
    telemetry_data = []
    deadlock_data = []
    obstacles_data = []
    voxel_viz_data = []
    status_text_data = []
    map_stats_data = []
    link_poses_data = []
    map_config_data = []

    # Lista dei topic da estrarre
    target_topics = [
        '/vmc/target_point', 
        '/vmc/telemetry', 
        '/vmc/deadlock_data',
        '/vmc/active_obstacles',
        '/vmc/voxel_grid_viz',
        '/vmc/status_text',
        '/vmc/map_stats',
        '/vmc/link_poses',
        '/vmc/log/config'
    ]

    print("⏳ Estrazione dei messaggi in corso...")
    
    # 4. Ciclo di lettura dei messaggi
    msg_count = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        
        # Ignora i topic che non ci interessano
        if topic not in target_topics:
            continue
            
        # Deserializza il messaggio
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        
        # Converte il timestamp in secondi
        t_sec = timestamp / 1e9
        
        # --- Estrazione Dati per Topic ---
        
        if topic == '/vmc/target_point':
            target_data.append({
                'time_sec': t_sec,
                'target_x': msg.target_position.x,
                'target_y': msg.target_position.y,
                'target_z': msg.target_position.z,
                'score_total': getattr(msg, 'score_total', 0.0),
                'score_frontality': getattr(msg, 'score_frontality', 0.0),
                'score_relevance': getattr(msg, 'score_relevance', 0.0),
                'score_distance': getattr(msg, 'score_distance', 0.0),
            })
            
        elif topic == '/vmc/telemetry':
            row = {'time_sec': t_sec}
            for i, v in enumerate(msg.link_velocities):
                row[f'link_{i+1}_vel'] = v
            for i, tq in enumerate(msg.base_torques):
                row[f'J{i+1}_base_torque'] = tq
            for i, tq in enumerate(msg.limit_torques):
                row[f'J{i+1}_limit_torque'] = tq
            for i, tq in enumerate(msg.total_torques):
                row[f'J{i+1}_total_torque'] = tq
            telemetry_data.append(row)
            
        elif topic == '/vmc/deadlock_data':
            deadlock_data.append({
                'time_sec': t_sec,
                'max_link_vel': msg.data[0] if len(msg.data) > 0 else 0.0,
                'max_joint_torque': msg.data[1] if len(msg.data) > 1 else 0.0,
            })
            
        elif topic == '/vmc/active_obstacles':
            obstacles_data.append({
                'time_sec': t_sec,
                'num_obstacles': len(msg.obstacles),
                # Salviamo la lista di (x,y,z) come stringa nel caso servisse analizzarli
                'obstacles_positions': str([(round(p.x, 3), round(p.y, 3), round(p.z, 3)) for p in msg.obstacles])
            })
            
        elif topic == '/vmc/voxel_grid_viz':
            # Il marker CUBE_LIST contiene i cubetti nell'array 'points'
            voxel_viz_data.append({
                'time_sec': t_sec,
                'num_voxels_rendered': len(msg.points)
            })
            
        elif topic == '/vmc/status_text':
            row = {
                'time_sec': t_sec,
                'raw_text': msg.text
            }
            # Estrazione intelligente dei numeri dal testo "Found X / Y Voxels"
            match = re.search(r'Found (\d+)\s*/\s*(\d+)', msg.text)
            if match:
                explored = int(match.group(1))
                total = int(match.group(2))
                row['explored_count'] = explored
                row['total_count'] = total
                row['completion_pct'] = (explored / total) * 100.0 if total > 0 else 0.0
            
            status_text_data.append(row)
        
        elif topic == '/vmc/map_stats':
            map_stats_data.append({
                'time_sec': t_sec,
                'total_voxels': msg.total_voxels,
                'unknown_voxels': msg.unknown_voxels,
                'free_voxels': msg.free_voxels,
                'occupied_voxels': msg.occupied_voxels,
                # Percentuale di completamento bella e pronta
                'explored_pct': ((msg.free_voxels + msg.occupied_voxels) / msg.total_voxels) * 100.0 if msg.total_voxels > 0 else 0.0
            })
            
        elif topic == '/vmc/link_poses':
            row = {'time_sec': t_sec}
            # Iteriamo sui nomi e sulle posizioni in parallelo
            for name, pos in zip(msg.link_names, msg.link_positions):
                row[f'{name}_x'] = pos.x
                row[f'{name}_y'] = pos.y
                row[f'{name}_z'] = pos.z
            link_poses_data.append(row)

        elif topic == '/vmc/log/config':
            # Salviamo solo la prima istanza della configurazione per evitare duplicati
            if not map_config_data:
                map_config_data.append({
                    'time_sec': t_sec,
                    # Workspace Configuration 
                    'exp_type': msg.type_experiment,
                    'ws_x_min': msg.ws_x_min,
                    'ws_x_max': msg.ws_x_max,
                    'ws_y_min': msg.ws_y_min,
                    'ws_y_max': msg.ws_y_max,
                    'ws_z_min': msg.ws_z_min,
                    'ws_z_max': msg.ws_z_max,
                    'target_voxel_size': msg.target_voxel_size,
                    'n_voxels_x': msg.n_voxels_x,
                    'n_voxels_y': msg.n_voxels_y,
                    'n_voxels_z': msg.n_voxels_z,

                    # Probabilistic (Buckets) 
                    'val_min': msg.val_min,
                    'val_max': msg.val_max,
                    'val_unknown': msg.val_unknown,
                    'val_occupied': msg.val_occupied,
                    'val_free': msg.val_free,
                    'val_lock': msg.val_lock,
                    'hit_inc': msg.hit_inc,
                    'miss_dec': msg.miss_dec,

                    # Camera & FOV 
                    'fov_h_rad': msg.fov_h,
                    'fov_v_rad': msg.fov_v,
                    'max_depth': msg.max_depth,
                    'min_depth': msg.min_depth,

                    # Logic & Scores 
                    'max_obstacles': msg.max_obstacles,
                    'min_dist_rep': msg.min_distance_rep,
                    'min_pts_voxel': msg.min_points_per_voxel,
                    'score_frontality': msg.score_frontality,
                    'score_relevance': msg.score_relevance,
                    'score_distance': msg.score_distance,

                    # Deadlock Configuration 
                    'deadlock_vel_eps': msg.deadlock_vel_eps,
                    'deadlock_torque_eps': msg.deadlock_torque_eps,
                    'deadlock_time_thresh': msg.deadlock_time_threshold,
                    'deadlock_zone_radius': msg.deadlock_zone_radius,
                    'deadlock_min_dist': msg.deadlock_min_dist,
                    'recovery_min_dist': msg.recovery_min_dist,
                    'recovery_max_dist': msg.recovery_max_dist
                })
            
        msg_count += 1

    print(f"✅ Letti {msg_count} messaggi rilevanti.")

    # 5. Salvataggio in CSV
    output_dir = Path(bag_path).name + "_csv_results"
    os.makedirs(output_dir, exist_ok=True)

    # Funzione helper per salvare i dataframe
    def save_df(data_list, filename):
        if data_list:
            df = pd.DataFrame(data_list)
            df.to_csv(f"{output_dir}/{filename}", index=False)
            print(f"📊 Salvato {output_dir}/{filename}")

    save_df(target_data, "target_data.csv")
    save_df(telemetry_data, "telemetry_data.csv")
    save_df(deadlock_data, "deadlock_data.csv")
    save_df(obstacles_data, "active_obstacles.csv")
    save_df(voxel_viz_data, "voxel_grid_viz.csv")
    save_df(status_text_data, "status_text.csv")
    save_df(map_stats_data, "map_stats.csv")
    save_df(link_poses_data, "link_poses.csv")
    save_df(map_config_data, "experiment_metadata.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estrai dati VMC e mappa da una ROS 2 bag in CSV.')
    parser.add_argument('bag_path', type=str, help='Il percorso alla cartella della ROS bag')
    
    args = parser.parse_args()
    process_bag(args.bag_path)
from config.settings import settings
from config.s3 import s3_client
import uuid
from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
import random
from sklearn.cluster import KMeans
from collections import defaultdict
from sqlalchemy import insert, update, func
from config.db import engine, videos, video_results

# --- GLOBAL MODEL INITIALIZATION ---
model = YOLO("modelv5.pt")

# --- (Fungsi helper Anda seperti get_dominant_color, run_worker, dll. tidak berubah) ---
def get_dominant_color(image, k=1):
    if image is None or image.size == 0: return (128, 128, 128)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (20, 20))
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        return tuple(map(int, kmeans.cluster_centers_[0]))
    except Exception:
        return (128, 128, 128)

def get_random_frame(cap, frame_count):
    if frame_count <= 0: return None
    random_position = random.randint(int(frame_count * 0.1), int(frame_count * 0.9))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_position)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count / 2))
        ret, frame = cap.read()
    return frame

def analyze_video(job):
    """Fungsi utama untuk memproses dan menganalisis video."""
    print(f"Processing job: {job}")
    # ... (Bagian awal fungsi Anda untuk setup, deteksi warna, dll. tidak berubah) ...
    # ...
    # --- FASE 1: PENYIAPAN PRA-ANALISIS ---
    video_url = job.get('video_url')
    os.makedirs("tmp", exist_ok=True)
    local_video_path = f"tmp/{uuid.uuid4()}.mp4"
    s3_client.download_file(settings.S3_BUCKET, video_url, local_video_path)

    cap_setup = cv2.VideoCapture(local_video_path)
    if not cap_setup.isOpened():
        return json.dumps({"error": f"Could not open video at {local_video_path}"})
        
    width = int(cap_setup.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_setup.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_setup.get(cv2.CAP_PROP_FPS) if cap_setup.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = int(cap_setup.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap_setup.release()
        return json.dumps({"error": "Video has no frames"})

    print(f"Video info: {width}x{height}, {fps:.2f} fps, {frame_count} frames")

    thumbnail_frame = get_random_frame(cap_setup, frame_count)
    thumbnail_path = f"tmp/{uuid.uuid4()}.jpg"
    if thumbnail_frame is not None:
        cv2.imwrite(thumbnail_path, thumbnail_frame)

    output_video_path = f"tmp/{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter.fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # ... (Fallback logic for VideoWriter tidak berubah)

    # --- Identifikasi Warna Tim ---
    print("Extracting dominant colors for team detection...")
    all_colors = []
    player_class_id = 2 
    cap_setup.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(50, frame_count)):
        ret, frame = cap_setup.read()
        if not ret: break
        results = model(frame, verbose=False)
        for result in results:
            player_boxes = result.boxes[result.boxes.cls == player_class_id] 
            for box in player_boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                if roi.size > 0:
                    all_colors.append(get_dominant_color(roi))
    
    cap_setup.release()
    print("--- Pre-analysis complete. ---")

    team_colors = np.array([[0, 0, 255], [255, 0, 0]]) # Default
    if len(all_colors) >= 2:
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(all_colors)
        team_colors = kmeans.cluster_centers_
    print(f"Team RGB colors identified: {team_colors}")

    # --- FASE 2: LOOP PEMROSESAN UTAMA ---
    cap_main = cv2.VideoCapture(local_video_path)
    if not cap_main.isOpened():
        return json.dumps({"error": "Failed to open video for processing."})
        
    id_to_team = {}
    player_data, shot_data = [], []
    pixel_to_meter, frame_idx, last_progress = None, 0, -5
    ball_state = {'center': None, 'velocity': (0,0), 'frames_unseen': 0}
    player_shot_cooldowns = defaultdict(int)
    ball_carrier_state = {'track_id': None, 'frame_idx': 0}
    
    print("Starting analysis...")
    while cap_main.isOpened():
        # ... (Loop while Anda yang berisi logika tracking dan deteksi tidak berubah) ...
        ret, frame = cap_main.read()
        if not ret: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False, classes=[0, 1, 2], imgsz=1088)
        result = results[0]
        annotated_frame = frame.copy()
        
        player_detections = result.boxes[result.boxes.cls == player_class_id]
        ball_detections = result.boxes[result.boxes.cls == 0] 

        if len(ball_detections) > 0:
            b_x1, b_y1, b_x2, b_y2 = map(int, ball_detections[0].xyxy[0])
            current_center = (int((b_x1 + b_x2) / 2), int((b_y1 + b_y2) / 2))
            if ball_state['center']:
                ball_state['velocity'] = (current_center[0] - ball_state['center'][0], current_center[1] - ball_state['center'][1])
            ball_state['center'] = current_center
            ball_state['frames_unseen'] = 0
        elif ball_state['center'] and ball_state['frames_unseen'] < (fps / 2):
            ball_state['center'] = (ball_state['center'][0] + ball_state['velocity'][0], ball_state['center'][1] + ball_state['velocity'][1])
            ball_state['frames_unseen'] += 1
            cv2.circle(annotated_frame, ball_state['center'], 10, (0, 0, 255), 1)
        else:
            ball_state['center'] = None
            ball_state['velocity'] = (0, 0)
        
        if ball_state['center']:
            cv2.circle(annotated_frame, ball_state['center'], 7, (0, 165, 255), -1)

        player_data_this_frame = []
        if result.boxes.id is not None:
            tracked_players = result.boxes[result.boxes.cls == player_class_id]
            for box in tracked_players:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                if track_id not in id_to_team:
                    roi = frame[int(y1 + 0.3*(y2-y1)):int(y1 + 0.7*(y2-y1)), x1:x2]
                    if roi.size > 0:
                        player_color = get_dominant_color(roi)
                        dist1 = np.linalg.norm(np.array(player_color) - team_colors[0])
                        dist2 = np.linalg.norm(np.array(player_color) - team_colors[1])
                        team_id = 0 if dist1 < dist2 else 1
                        id_to_team[track_id] = team_id
                
                team_id = id_to_team.get(track_id, 0)
                player_info = {"frame": frame_idx, "player_id": track_id, "team_id": team_id, "bbox": [x1, y1, x2, y2]}
                player_data.append(player_info)
                player_data_this_frame.append(player_info)
                
                color = tuple(map(int, team_colors[team_id]))
                draw_color = (color[2], color[1], color[0]) 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(annotated_frame, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        
        current_ball_carrier_id = None
        if ball_state['center'] and player_data_this_frame:
            closest_player = min(player_data_this_frame, key=lambda p: np.hypot(((p['bbox'][0] + p['bbox'][2])/2 - ball_state['center'][0]), ((p['bbox'][1] + p['bbox'][3])/2 - ball_state['center'][1])))
            dist_to_ball = np.hypot(((closest_player['bbox'][0] + closest_player['bbox'][2])/2 - ball_state['center'][0]), ((closest_player['bbox'][1] + closest_player['bbox'][3])/2 - ball_state['center'][1]))
            if dist_to_ball < 50:
                current_ball_carrier_id = closest_player['player_id']
                ball_carrier_state['track_id'] = current_ball_carrier_id
                ball_carrier_state['frame_idx'] = frame_idx
        
                # --- Blok Deteksi Tembakan yang Diperbarui ---
        if ball_carrier_state['track_id'] is not None and current_ball_carrier_id is None and (frame_idx - ball_carrier_state['frame_idx'] < 5): # Sedikit menambah jendela waktu
            shooter_id = ball_carrier_state['track_id']
            if player_shot_cooldowns[shooter_id] == 0:
                ball_vy = ball_state['velocity'][1]
                
                # Kondisi utama: bola bergerak ke atas
                if ball_vy < -2:
                    
                    # Cari data pemain yang melakukan tembakan di frame saat ini
                    shooter_data = next((p for p in player_data_this_frame if p['player_id'] == shooter_id), None)
                    
                    if shooter_data:
                        print(f"\n--- SHOT ATTEMPT detected by Player #{shooter_id} at frame {frame_idx}! ---")
                        
                        # Ambil bbox dari data shooter
                        x1, y1, x2, y2 = shooter_data['bbox']
                        # Hitung koordinat tengah pemain
                        shooter_x = int((x1 + x2) / 2)
                        shooter_y = int((y1 + y2) / 2)
                        
                        # --- PERUBAHAN UTAMA: Tambahkan koordinat pemain ke shot_data ---
                        shot_data.append({
                            "frame": frame_idx, 
                            "player_id": shooter_id,
                            "team_id": shooter_data['team_id'], # Tambahkan juga team_id
                            "result": "attempt",
                            "player_coords": {"x": shooter_x, "y": shooter_y}, # Koordinat pemain
                            "ball_coords": {
                                "x": ball_state['center'][0], 
                                "y": ball_state['center'][1]
                            }
                        })
                        
                        # Mulai cooldown untuk pemain ini
                        player_shot_cooldowns[shooter_id] = int(fps * 2)

        for pid in list(player_shot_cooldowns.keys()):
            if player_shot_cooldowns[pid] > 0:
                player_shot_cooldowns[pid] -= 1

        out.write(annotated_frame)
        frame_idx += 1
        progress = (frame_idx / frame_count) * 100
        print(f"🚀 Processing Frame: {frame_idx}/{frame_count} ({progress:.2f}%)", end='\r')

        if int(progress) % 5 == 0 and int(progress) != last_progress:
            last_progress = int(progress)

    print("\nProcessing finished.")
    cap_main.release()
    out.release()

    # --- BAGIAN YANG DIUBAH: DARI SINI SAMPAI AKHIR ---

    # --- 1. Persiapan Data untuk Database ---
    converted_players = []
    for p in player_data:
        x1, y1, x2, y2 = p["bbox"]
        converted_players.append({
            "frame": p["frame"], "player_id": p["player_id"], "team_id": p["team_id"],
            "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)
        })

    # --- 2. Unggah Aset Media ke S3 ---
    s3_video_key = f"results/{job.get('id')}_{uuid.uuid4()}.mp4"
    s3_thumbnail_key = f"thumbnails/{job.get('id')}_{uuid.uuid4()}.jpg"

    print("Uploading video and thumbnail to S3...")
    try:
        s3_client.upload_file(output_video_path, settings.S3_BUCKET, s3_video_key, ExtraArgs={'ContentType': 'video/mp4'})
        if thumbnail_path and os.path.exists(thumbnail_path):
            s3_client.upload_file(thumbnail_path, settings.S3_BUCKET, s3_thumbnail_key, ExtraArgs={'ContentType': 'image/jpeg'})
        print("Media assets uploaded successfully.")
    except Exception as e:
        print(f"FATAL: S3 upload for media failed: {e}")
        raise RuntimeError('ERR_MEDIA_UPLOAD')
    finally:
        # 4. Bersihkan file sementara di akhir
        print("Cleaning up temporary files...")
        temp_files = [local_video_path, output_video_path, thumbnail_path]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try: 
                    os.remove(temp_file)
                    print('x')
                except Exception as e: 
                    print(f"Error cleaning up {temp_file}: {e}")

    # --- 3. Simpan Hasil Analisis ke Database ---
    try:
        print("Connecting to database to save results...")
        with engine.begin() as conn:
            # a. Update tabel 'videos' dengan URL thumbnail
            print(f"Updating video record ID: {job.get('id')}")
            video_stmt = update(videos).where(
                videos.c.id == job.get('id')
            ).values(
                thumbnail_url=s3_thumbnail_key,
                
            )
            conn.execute(video_stmt)
            
            # b. Sisipkan semua data hasil analisis ke tabel 'video_results'
            print("Inserting analysis results into database...")
            # Menghitung pixel_to_meter jika memungkinkan
            if result.boxes and len(result.boxes[result.boxes.cls == 1]) > 0:
                last_net_box = result.boxes[result.boxes.cls == 1][-1].xyxy[0]
                pixel_to_meter = np.mean([last_net_box[2] - last_net_box[0], last_net_box[3] - last_net_box[1]]) / 0.45
                court_length_px, court_width_px = (int(28 / pixel_to_meter), int(15 / pixel_to_meter)) if pixel_to_meter else (None, None)
            else:
                court_length_px, court_width_px = None, None

            result_stmt = insert(video_results).values(
                video_id=job.get('id'),
                court_length_px=court_length_px if court_length_px is not None else 0,
                court_width_px=court_width_px if court_width_px is not None else 0,
                video_url=s3_video_key,
                tracking=converted_players, # Konversi ke string JSON
                shot=shot_data,             # Konversi ke string JSON
            )
            conn.execute(result_stmt)
            print("Database records saved successfully.")
    except Exception as e:
         print(f"FATAL ERROR during database operation: {e}")
         raise RuntimeError('ERR_DATABASE_SAVE')

    print("Job completed and all data saved!")
import asyncio
from bullmq import Job
from config.settings import settings
from config.s3 import s3_client
import uuid
from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from sort import Sort

model = YOLO("bestv2.pt")

def get_dominant_color(image, k=1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (20, 20))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return tuple(map(int, kmeans.cluster_centers_[0]))


# Fungsi untuk mendeteksi ring dan mengukur diameter dalam pixel
def detect_ring_and_measure_diameter(frame, model):
    results = model(frame)
    ring_bbox = None
    
    # Deteksi objek dalam frame
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == 1:  # Ring basket
                x1, y1, x2, y2 = map(int, box[:4])
                ring_bbox = [x1, y1, x2, y2]
                
                # Menghitung diameter dalam pixel (jarak horizontal antara x1 dan x2)
                ring_diameter_pixel = x2 - x1
                return ring_bbox, ring_diameter_pixel
    
    return None, 0  # Mengembalikan None dan 0 jika tidak terdeteksi


# Fungsi untuk mengukur lapangan berdasarkan diameter ring dalam pixel
def measure_court_in_pixels(ring_diameter_real_world, ring_diameter_pixel, court_length_real, court_width_real):
    # Menghitung skala (pixel per meter)
    scale = ring_diameter_pixel / ring_diameter_real_world
    
    # Mengonversi panjang dan lebar lapangan dari meter ke pixel
    court_length_pixel = scale * court_length_real  # Panjang lapangan dalam pixel
    court_width_pixel = scale * court_width_real    # Lebar lapangan dalam pixel
    
    return court_length_pixel, court_width_pixel


async def run_worker(job: Job, token=None):
    loop = asyncio.get_running_loop()
    result = await asyncio.to_thread(analyze_video, job, loop)
    print('selesai')
    return result


def analyze_video(job: Job, loop: asyncio.AbstractEventLoop):
    print(job.data)
    video_url = job.data.get('video_url')
    os.makedirs("tmp", exist_ok=True)

    local_video_path = f"tmp/{uuid.uuid4()}.mp4"
    s3_client.download_file(settings.S3_BUCKET, video_url, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = f"tmp/{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Tentukan diameter ring dalam dunia nyata (dalam meter)
    ring_diameter_real_world = 0.45  # Diameter ring basket dalam meter
    
    # Tentukan panjang dan lebar lapangan (dalam meter)
    court_length_real = 28  # Panjang lapangan dalam meter
    court_width_real = 15   # Lebar lapangan dalam meter

    # Deteksi ring dan hitung diameter ring dalam pixel
    ring_bbox, ring_diameter_pixel = detect_ring_and_measure_diameter(frame, model)
    if ring_bbox is None:
        print("Ring tidak terdeteksi")
        return

    print(f"Diameter ring dalam pixel: {ring_diameter_pixel}")

    # Mengukur lapangan berdasarkan diameter ring
    court_length_pixel, court_width_pixel = measure_court_in_pixels(ring_diameter_real_world, ring_diameter_pixel, court_length_real, court_width_real)

    print(f"Panjang lapangan dalam pixel: {court_length_pixel}")
    print(f"Lebar lapangan dalam pixel: {court_width_pixel}")

    # Lanjutkan dengan logika pelacakan dan deteksi pemain...
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    id_to_team = {}
    team_counts = defaultdict(int)
    player_data = []
    shot_data = []
    ball_history = []
    ring_bbox = None

    i = 0
    last_progress = -5 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = []
        ball_coords = None
        results = model(frame)
        ring_found = False

        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(result.boxes.conf[0]) if hasattr(result.boxes, 'conf') else 0.9

                if int(cls) == 0:  # Ball
                    ball_coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                elif int(cls) == 1 and not ring_found:  # Ring
                    ring_bbox = [x1, y1, x2, y2]
                    ring_found = True

                elif int(cls) == 2:  # Player
                    frame_dets.append([x1, y1, x2, y2, conf])

        # Update tracker
        tracks = tracker.update(np.array(frame_dets))

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
            if track_id not in id_to_team and roi.size > 0:
                dominant = get_dominant_color(roi)
                dists = [np.linalg.norm(np.array(dominant) - np.array(tc)) for tc in team_colors]
                team_id = int(np.argmin(dists))
                id_to_team[track_id] = team_id

            team_id = id_to_team.get(track_id, -1)
            color = tuple(map(int, team_colors[team_id])) if team_id != -1 else (255, 255, 255)
            team_counts[f"Team {team_id+1}"] += 1

            player_data.append({
                "frame": i,
                "player_id": int(track_id),
                "team_id": team_id,
                "bbox": [x1, y1, x2, y2]
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'#{track_id} | T{team_id+1}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if ball_coords and ring_bbox:
            ball_history.append(ball_coords)
            rx1, ry1, rx2, ry2 = ring_bbox
            ring_center = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
            dx = abs(ball_coords[0] - ring_center[0])
            dy = abs(ball_coords[1] - ring_center[1])
            if dx < 50 and dy < 50:
                nearest_player = min(player_data[-len(tracks):], key=lambda p: np.hypot(p['bbox'][0]-ball_coords[0], p['bbox'][1]-ball_coords[1]))
                shot_data.append({
                    "x": ball_coords[0],
                    "y": ball_coords[1],
                    "team_id": nearest_player["team_id"],
                    "result": "made"
                })

        out.write(frame)
        i += 1
        progress = int((i / frame_count) * 100)
        print(progress)

        if progress % 5 == 0 and progress != last_progress:
            asyncio.run_coroutine_threadsafe(job.updateProgress(progress),loop)
            last_progress = progress

    cap.release()
    out.release()

    player_json_path = f"tmp/{uuid.uuid4()}.json"
    shot_json_path = f"tmp/{uuid.uuid4()}.json"

    with open(player_json_path, "w") as f:
        json.dump({"players": player_data}, f)

    with open(shot_json_path, "w") as f:
        json.dump({"shots": shot_data}, f)

    # Upload JSON and video
    s3_json_key = f"uploads/{uuid.uuid4()}.json"
    s3_shot_key = f"uploads/{uuid.uuid4()}.json"
    s3_video_key = f"uploads/{uuid.uuid4()}.mp4"

    s3_client.upload_file(player_json_path, settings.S3_BUCKET, s3_json_key)
    s3_client.upload_file(shot_json_path, settings.S3_BUCKET, s3_shot_key)
    s3_client.upload_file(output_video_path, settings.S3_BUCKET, s3_video_key)

    print("selesai")

    return json.dumps({
        "json_result": s3_json_key,
        "shot_result": s3_shot_key,
        "video_result": s3_video_key
    })

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

async def analyze_video(job: Job, token=None):
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

    # Step 1: Kumpulkan warna dominan pemain untuk cluster tim
    dominant_colors = []
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 2:  # Player
                    x1, y1, x2, y2 = map(int, box[:4])
                    roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                    if roi.size > 0:
                        dominant_colors.append(get_dominant_color(roi))

    kmeans = KMeans(n_clusters=2, n_init=10).fit(dominant_colors)
    team_colors = kmeans.cluster_centers_

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker = Sort()
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

        # Hitung pixel_to_meter jika ring terdeteksi
        pixel_to_meter = None
        if ring_bbox:
            ring_px_diameter = np.mean([ring_bbox[2] - ring_bbox[0], ring_bbox[3] - ring_bbox[1]])
            pixel_to_meter = ring_px_diameter / 0.45  # diameter ring 0.45 m

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
                shot_data.append({
                    "x": ball_coords[0],
                    "y": ball_coords[1],
                    "team_id": team_id,  
                    "result": "made"
                })

        out.write(frame)
        i += 1
        progress = int((i / frame_count) * 100)
        print(progress)

        if progress % 5 == 0 and progress != last_progress:
            await job.updateProgress(progress)
            
            last_progress = progress
        

    cap.release()
    out.release()

    if pixel_to_meter:
        court_length_px = int(28 / pixel_to_meter)
        court_width_px = int(15 / pixel_to_meter)
    else:
        court_length_px = court_width_px = None

    # Ubah bbox player menjadi koordinat tengah
    converted_players = []
    for p in player_data:
        x1, y1, x2, y2 = p["bbox"]
        converted_players.append({
            "frame": p["frame"],
            "player_id": p["player_id"],
            "team_id": p["team_id"],
            "x": int((x1 + x2) / 2),
            "y": int((y1 + y2) / 2)
        })

    player_json = {
        "court_length_px": court_length_px,
        "court_width_px": court_width_px,
        "players": converted_players
    }

    shot_json = {
        "court_length_px": court_length_px,
        "court_width_px": court_width_px,
        "shots": shot_data
    }

    player_json_path = f"tmp/{uuid.uuid4()}.json"
    shot_json_path = f"tmp/{uuid.uuid4()}.json"

    with open(player_json_path, "w") as f:
        json.dump(player_json, f)

    with open(shot_json_path, "w") as f:
        json.dump(shot_json, f)

    # Upload ke S3
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

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

def analyze_shot(ball_coords, ring_coords, tracker, frame, team_colors, width, height):
    # Tentukan jarak tembakan (misalnya 50 piksel dari ring)
    shot_radius = 50
    dx = abs(ball_coords[0] - ring_coords[0])
    dy = abs(ball_coords[1] - ring_coords[1])

    # Cek apakah bola cukup dekat dengan ring untuk dianggap sebagai tembakan
    if dx < shot_radius and dy < shot_radius:
        # Cari pemain terdekat dengan bola
        nearest_player = None
        min_distance = float('inf')
        for track in tracker:
            player_x, player_y, track_id = map(int, track[:3])
            distance = np.linalg.norm(np.array([player_x, player_y]) - np.array(ball_coords))
            if distance < min_distance:
                min_distance = distance
                nearest_player = track_id

        # Tandai pemain yang melakukan tembakan dan timnya
        if nearest_player is not None:
            team_id = id_to_team.get(nearest_player, -1)
            color = tuple(map(int, team_colors[team_id])) if team_id != -1 else (255, 255, 255)
            # Gambar bounding box dan ID tembakan
            cv2.putText(frame, f"Shot by T{team_id+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return nearest_player, team_id, ball_coords, "Shot Detected"
    return None, None, None, "No Shot"

def analyze_video():
    video_url = "testvideo.mp4"
    os.makedirs("tmp", exist_ok=True)

    local_video_path = f"tmp/{video_url}"
    s3_client.download_file(settings.S3_BUCKET, video_url, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = f"tmp/{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Dominant color untuk clustering tim
    dominant_colors = []
    i = 0
    while i < 50 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 2:  # class player
                    x1, y1, x2, y2 = map(int, box[:4])
                    roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                    if roi.size > 0:
                        dominant_colors.append(get_dominant_color(roi))
        i += 1

    kmeans = KMeans(n_clusters=2, n_init=10).fit(dominant_colors)
    team_colors = kmeans.cluster_centers_

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video
    tracker = Sort()
    id_to_team = {}
    team_counts = defaultdict(int)
    detections = []
    shot_data = []
    ball_history = []

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = []
        ball_coords = None
        ring_coords = None

        results = model(frame)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(result.boxes.conf[0]) if hasattr(result.boxes, 'conf') else 0.9

                if int(cls) == 0:  # bola
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    ball_coords = (cx, cy)
                    print(f"[Frame {i}] Bola terdeteksi di {ball_coords}")

                elif int(cls) == 1:  # ring
                    ring_coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    print(f"[Frame {i}] Ring terdeteksi di {ring_coords}")

                elif int(cls) == 2:  # player
                    frame_dets.append([x1, y1, x2, y2, conf])

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'#{track_id} | T{team_id+1}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections.append({
                "frame": i,
                "track_id": int(track_id),
                "team_id": team_id,
                "bbox": [x1, y1, x2, y2]
            })

        if ball_coords:
            ball_history.append(ball_coords)
            if ring_coords:
                # Analisis tembakan berdasarkan posisi bola dan ring
                nearest_player, team_id, shot_coords, status = analyze_shot(ball_coords, ring_coords, tracks, frame, team_colors, width, height)
                if status == "Shot Detected":
                    shot_data.append({
                        "x": round(shot_coords[0] / width * 8, 2),
                        "y": round(shot_coords[1] / height * 7, 2),
                        "team_id": team_id,
                        "result": "made"
                    })

        out.write(frame)
        i += 1
        progress = int((i / frame_count) * 100)
        print(f"Job progress: {progress}%")

    cap.release()
    out.release()

    # Generate file paths
    json_tracking_path = f"tmp/{uuid.uuid4()}.json"
    json_shot_path = f"tmp/{uuid.uuid4()}.json"
    s3_tracking_key = f"uploads/{uuid.uuid4()}.json"
    s3_shot_key = f"uploads/{uuid.uuid4()}.json"
    s3_video_key = f"uploads/{uuid.uuid4()}.mp4"

    # Save JSON files
    with open(json_tracking_path, "w") as f:
        json.dump(detections, f)
    with open(json_shot_path, "w") as f:
        json.dump(shot_data, f)

    # Upload to S3
    s3_client.upload_file(json_tracking_path, settings.S3_BUCKET, s3_tracking_key)
    s3_client.upload_file(json_shot_path, settings.S3_BUCKET, s3_shot_key)
    s3_client.upload_file(output_video_path, settings.S3_BUCKET, s3_video_key)

    print("Tracking selesai. Diupload ke S3.")

    result = {
        "json_result": s3_tracking_key,
        "video_result": s3_video_key,
        "shot_result": s3_shot_key
    }

    return result

a = analyze_video()
print(a)

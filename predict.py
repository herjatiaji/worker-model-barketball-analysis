import cv2
import os
import uuid
import json
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
from config.s3 import s3_client
from config.settings import settings

model = YOLO("bestv2.pt") 

def get_dominant_color(image, k=1):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (20, 20))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def analyze_video():
    video_url = "testvideo.mp4"
    os.makedirs("tmp", exist_ok=True)
    local_video_path = f"tmp/{video_url}"
    s3_client.download_file(settings.S3_BUCKET, video_url, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = f"tmp/{uuid.uuid4()}.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #get dominant color
    dominant_colors = []
    i = 0
    while i < 50 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 2:
                    x1, y1, x2, y2 = map(int, box)
                    roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                    if roi.size > 0:
                        dominant_colors.append(get_dominant_color(roi))
        i += 1

    kmeans = KMeans(n_clusters=2, n_init=10).fit(dominant_colors)
    team_colors = kmeans.cluster_centers_

    #track
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracker = ByteTrack()
    id_to_team = {}
    team_counts = defaultdict(int)
    detections_log = []
    ball_tracks = []
    ring_boxes = []
    shot_count = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        dets = []
        classes = []
        confidences = []

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            dets.append([x1, y1, x2, y2])
            classes.append(int(cls))
            confidences.append(float(conf))

        detection = Detections(
            xyxy=np.array(dets),
            confidence=np.array(confidences),
            class_id=np.array(classes)
        )

        tracks = tracker.update_with_detections(detection)

        for track in tracks:
            if track.track_id is None:
                continue

            x1, y1, x2, y2 = map(int, track.tlbr)
            track_id = int(track.track_id)
            cls_id = int(track.class_id) if track.class_id is not None else -1


            if cls_id == 2:  # Player
                roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                if track_id not in id_to_team and roi.size > 0:
                    dominant = get_dominant_color(roi)
                    dists = [np.linalg.norm(np.array(dominant) - np.array(tc)) for tc in team_colors]
                    team_id = int(np.argmin(dists))
                    id_to_team[track_id] = team_id

                team_id = id_to_team.get(track_id, -1)
                color = tuple(map(int, team_colors[team_id])) if team_id != -1 else (255, 255, 255)
                label = f"Player #{track_id} T{team_id+1}"
                team_counts[f"Team {team_id+1}"] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            elif cls_id == 0:  # Ball
                label = f"Ball #{track_id}"
                color = (0, 255, 255)
                ball_tracks.append(((x1 + x2) // 2, (y1 + y2) // 2))

            elif cls_id == 1:  # Ring
                label = "Ring"
                color = (255, 0, 255)
                ring_boxes.append((x1, y1, x2, y2))

            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections_log.append({
                "frame": frame_index,
                "track_id": track_id,
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2]
            })

        # Shot detection
        for bx, by in ball_tracks[-5:]:
            for rx1, ry1, rx2, ry2 in ring_boxes:
                if rx1 <= bx <= rx2 and ry1 <= by <= ry2:
                    shot_count += 1
                    ball_tracks = []  # reseta track
                    break

        out.write(frame)
        frame_index += 1
        print(f"Progress: {int((frame_index / frame_count) * 100)}%")

    cap.release()
    out.release()

    # Save output
    json_output = {
        "detections": detections_log,
        "shot_count": shot_count
    }

    json_path = f"tmp/{uuid.uuid4()}.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f)

    s3_client.upload_file(json_path, settings.S3_BUCKET, f"outputs/{uuid.uuid4()}.json")
    s3_client.upload_file(output_video_path, settings.S3_BUCKET, f"outputs/{uuid.uuid4()}.mp4")

    print("Analisis selesai. File diupload ke S3.")

if __name__ == "__main__":
    analyze_video()

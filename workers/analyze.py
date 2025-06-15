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
import random
import time
from sklearn.cluster import KMeans
from collections import defaultdict

# --- GLOBAL MODEL INITIALIZATION ---
model = YOLO("bestv2.pt")

def get_dominant_color(image, k=1):
    """Extracts the dominant color from an image ROI."""
    if image is None or image.size == 0:
        return (128, 128, 128)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (20, 20))
        pixels = image.reshape(-1, 3)
        if pixels.shape[0] < k:
             return tuple(map(int, pixels[0])) if pixels.shape[0] > 0 else (128, 128, 128)
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        return tuple(map(int, kmeans.cluster_centers_[0]))
    except Exception:
        return (128, 128, 128)


async def run_worker(job: Job, token=None):
    """Asynchronously runs the video analysis task."""
    loop = asyncio.get_running_loop()
    result = await asyncio.to_thread(analyze_video, job, loop)
    print('Selesai')
    return result


def get_random_frame(cap, frame_count):
    """Gets a random frame from the video for thumbnail generation."""
    if frame_count <= 0: return None
    random_position = random.randint(int(frame_count * 0.1), int(frame_count * 0.9))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_position)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count / 2))
        ret, frame = cap.read()
    return frame


def analyze_video(job: Job, loop: asyncio.AbstractEventLoop):
    """The main function to process and analyze the video."""
    print(f"Processing job: {job.data}")
    video_url = job.data.get('video_url')
    os.makedirs("tmp", exist_ok=True)

    local_video_path = f"tmp/{uuid.uuid4()}.mp4"
    s3_client.download_file(settings.S3_BUCKET, video_url, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {local_video_path}")
        return json.dumps({"error": "Could not open video"})
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("Error: Video has no frames.")
        return json.dumps({"error": "Video has no frames"})

    print(f"Video info: {width}x{height}, {fps:.2f} fps, {frame_count} frames")

    thumbnail_frame = get_random_frame(cap, frame_count)
    thumbnail_path = f"tmp/{uuid.uuid4()}.jpg"
    if thumbnail_frame is not None:
        cv2.imwrite(thumbnail_path, thumbnail_frame)

    output_video_path = f"tmp/{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Warning: Could not open video writer with 'avc1' codec. Falling back to 'mp4v'.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("FATAL: Could not open video writer with any codec.")
            return json.dumps({"error": "Failed to create output video file."})

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Extracting dominant colors for team detection...")
    dominant_colors = []
    for _ in range(min(50, frame_count)):
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, verbose=False)
        for result in results:
            player_boxes = result.boxes[result.boxes.cls == 2]
            for box in player_boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                if roi.size > 0:
                    dominant_colors.append(get_dominant_color(roi))

    if len(dominant_colors) >= 2:
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(dominant_colors)
        team_colors = kmeans.cluster_centers_
    else:
        team_colors = np.array([[255, 0, 0], [0, 0, 255]])
    print("Team colors identified.")

    cap.release()
    cap = cv2.VideoCapture(local_video_path)
    
    id_to_team = {}
    player_data = []
    shot_data = []
    pixel_to_meter = None

    frame_idx = 0
    last_progress = -5
    shot_cooldown = 0
    
    print("Starting tracking with robust shot detection...")
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False)
        result = results[0]
        
        annotated_frame = frame.copy()
        
        # --- Detect Ball and Ring (Frame-by-Frame) ---
        ball_boxes = result.boxes[result.boxes.cls == 0]
        ring_boxes = result.boxes[result.boxes.cls == 1]
        
        ball_coords = None
        ring_bbox = None
        
        if len(ball_boxes) > 0:
            x1, y1, x2, y2 = map(int, ball_boxes[0].xyxy[0])
            ball_coords = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(annotated_frame, ball_coords, 7, (0, 255, 255), -1)

        if len(ring_boxes) > 0:
            rx1, ry1, rx2, ry2 = map(int, ring_boxes[0].xyxy[0])
            ring_bbox = [rx1, ry1, rx2, ry2]
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            pixel_to_meter = np.mean([rx2 - rx1, ry2 - ry1]) / 0.45

        # --- Track Players ---
        player_data_this_frame = []
        if result.boxes.id is not None:
            tracked_players = result.boxes[result.boxes.cls == 2]
            for box in tracked_players:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                if track_id not in id_to_team:
                    roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                    team_id = int(np.argmin([np.linalg.norm(np.array(get_dominant_color(roi)) - tc) for tc in team_colors]))
                    id_to_team[track_id] = team_id

                team_id = id_to_team.get(track_id, 0)
                player_info = {"frame": frame_idx, "player_id": track_id, "team_id": team_id, "bbox": [x1, y1, x2, y2]}
                player_data.append(player_info)
                player_data_this_frame.append(player_info)

                color = tuple(map(int, team_colors[team_id]))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- ROBUST SHOT DETECTION LOGIC ---
        if shot_cooldown > 0:
            shot_cooldown -= 1

        if ball_coords and ring_bbox:
            ring_center_x = (ring_bbox[0] + ring_bbox[2]) // 2
            ring_center_y = (ring_bbox[1] + ring_bbox[3]) // 2
            ring_width = ring_bbox[2] - ring_bbox[0]

            # Define a dynamic shot zone based on the ring's size
            zone_x1 = ring_center_x - ring_width // 2
            zone_x2 = ring_center_x + ring_width // 2
            zone_y1 = ring_center_y - 30 # A fixed vertical tolerance
            zone_y2 = ring_center_y + 30

            # --- DEBUGGING VISUAL: Draw the "Shot Zone" on the video ---
            cv2.rectangle(annotated_frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 1)
            
            # Check if ball is in the zone and cooldown is over
            if shot_cooldown == 0 and (zone_x1 < ball_coords[0] < zone_x2) and (zone_y1 < ball_coords[1] < zone_y2):
                print(f"\n--- Shot event detected at frame {frame_idx}! ---")
                
                if player_data_this_frame:
                    nearest_player = min(player_data_this_frame, key=lambda p: np.hypot(((p['bbox'][0] + p['bbox'][2])/2 - ball_coords[0]), ((p['bbox'][1] + p['bbox'][3])/2 - ball_coords[1])))
                    shot_data.append({
                        "frame": frame_idx, "x": ball_coords[0], "y": ball_coords[1], 
                        "team_id": nearest_player["team_id"], "player_id": nearest_player["player_id"], "result": "made"
                    })
                
                shot_cooldown = int(fps) # Start 1-second cooldown
                
        out.write(annotated_frame)
        frame_idx += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps_speed = 1 / processing_time if processing_time > 0 else float('inf')
        progress = (frame_idx / frame_count) * 100
        print(f"🚀 Processing Frame: {frame_idx}/{frame_count} ({progress:.2f}%) | Speed: {fps_speed:.2f} FPS", end='\r')

        if int(progress) % 5 == 0 and int(progress) != last_progress:
            asyncio.run_coroutine_threadsafe(job.updateProgress(int(progress)), loop)
            last_progress = int(progress)

    print("\nProcessing finished.")
    cap.release()
    out.release()
    
    if pixel_to_meter:
        court_length_px, court_width_px = int(28 / pixel_to_meter), int(15 / pixel_to_meter)
    else:
        court_length_px, court_width_px = None, None

    converted_players = []
    for p in player_data:
        x1, y1, x2, y2 = p["bbox"]
        converted_players.append({
            "frame": p["frame"], "player_id": p["player_id"], "team_id": p["team_id"],
            "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)
        })

    player_json = {"court_length_px": court_length_px, "court_width_px": court_width_px, "players": converted_players}
    shot_json = {"court_length_px": court_length_px, "court_width_px": court_width_px, "shots": shot_data}

    player_json_path, shot_json_path = f"tmp/{uuid.uuid4()}.json", f"tmp/{uuid.uuid4()}.json"

    with open(player_json_path, "w") as f: json.dump(player_json, f, indent=2)
    with open(shot_json_path, "w") as f: json.dump(shot_json, f, indent=2)

    s3_tracking_key, s3_shot_key = f"uploads/{uuid.uuid4()}.json", f"uploads/{uuid.uuid4()}.json"
    s3_video_key, s3_thumbnail_key = f"uploads/{uuid.uuid4()}.mp4", f"uploads/thumbnails/{uuid.uuid4()}.jpg"

    s3_client.upload_file(player_json_path, settings.S3_BUCKET, s3_tracking_key, ExtraArgs={'ContentType': 'application/json'})
    s3_client.upload_file(shot_json_path, settings.S3_BUCKET, s3_shot_key, ExtraArgs={'ContentType': 'application/json'})
    s3_client.upload_file(output_video_path, settings.S3_BUCKET, s3_video_key, ExtraArgs={'ContentType': 'video/mp4'})
    if os.path.exists(thumbnail_path):
        s3_client.upload_file(thumbnail_path, settings.S3_BUCKET, s3_thumbnail_key, ExtraArgs={'ContentType': 'image/jpeg'})

    temp_files = [local_video_path, output_video_path, player_json_path, shot_json_path, thumbnail_path]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {e}")

    print("S3 Upload complete. Job finished.")

    return json.dumps({
        "tracking_result": s3_tracking_key,
        "shot_result": s3_shot_key,
        "video_result": s3_video_key,
        "thumbnail_url": s3_thumbnail_key
    })
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
# IMPORTANT: Make sure this points to your latest and best trained model.
model = YOLO('modelv5.pt')

def get_dominant_hue(image, k=1):
    """
    Extracts the dominant hue from an image ROI using the HSV color space.
    The Hue channel is more robust to lighting changes.
    """
    if image is None or image.size == 0:
        return -1  # Return an invalid hue

    try:
        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # We only need the Hue channel (hsv_image[:, :, 0]) for color classification
        hue_channel = hsv_image[:, :, 0]
        
        # Reshape for KMeans
        pixels = hue_channel.reshape(-1, 1)

        if pixels.shape[0] < k:
            return int(pixels[0][0]) if pixels.shape[0] > 0 else -1

        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        
        # Return the single dominant hue value (0-179 in OpenCV)
        return int(kmeans.cluster_centers_[0][0])
    except Exception:
        return -1


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
        cv2.imwrite(thumbnail_path, thumbnail_path)

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

    # --- NEW: Team Hue Identification ---
    print("Extracting dominant hues for team detection...")
    dominant_hues = []
    for _ in range(min(50, frame_count)):
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, verbose=False)
        for result in results:
            player_boxes = result.boxes[result.boxes.cls == 2] # Player is class 2
            for box in player_boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                if roi.size > 0:
                    hue = get_dominant_hue(roi)
                    if hue != -1:
                        dominant_hues.append(hue)

    team_hues = []
    if len(dominant_hues) >= 2:
        hues_for_clustering = np.array(dominant_hues).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(hues_for_clustering)
        team_hues = kmeans.cluster_centers_.flatten()
        print(f"Team hues identified: {team_hues}")
    else:
        team_hues = np.array([5, 110]) # Default hues (e.g., Red and Blue)
        print("Using default team hues.")

    # We classify using Hue, but draw with fixed BGR colors for visibility.
    # Team 0 will be Blue, Team 1 will be Red.
    drawing_colors = [(255, 0, 0), (0, 0, 255)] 

    # Reset video capture for the main processing loop
    cap.release()
    cap = cv2.VideoCapture(local_video_path)
    
    id_to_team = {}
    player_data = []
    shot_data = []
    pixel_to_meter = None

    frame_idx = 0
    last_progress = -5
    shot_cooldown = 0
    
    print("Starting tracking with robust team detection...")
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False)
        result = results[0]
        
        annotated_frame = frame.copy()
        player_data_this_frame = []
        
        if result.boxes.id is not None:
            tracked_players = result.boxes[result.boxes.cls == 2]
            for box in tracked_players:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])

                # --- NEW: Team Assignment using Hue distance ---
                if track_id not in id_to_team:
                    roi = frame[int(y1 + 0.3 * (y2 - y1)):int(y1 + 0.7 * (y2 - y1)), x1:x2]
                    player_hue = get_dominant_hue(roi)
                    if player_hue != -1:
                        # Calculate circular distance for Hues (0-179 range)
                        dist1 = min(abs(player_hue - team_hues[0]), 180 - abs(player_hue - team_hues[0]))
                        dist2 = min(abs(player_hue - team_hues[1]), 180 - abs(player_hue - team_hues[1]))
                        team_id = 0 if dist1 < dist2 else 1
                        id_to_team[track_id] = team_id

                team_id = id_to_team.get(track_id, 0)
                player_info = {"frame": frame_idx, "player_id": track_id, "team_id": team_id, "bbox": [x1, y1, x2, y2]}
                player_data.append(player_info)
                player_data_this_frame.append(player_info)

                color = drawing_colors[team_id]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- AI-POWERED SHOT DETECTION (Class ID 1) ---
        if shot_cooldown > 0:
            shot_cooldown -= 1
        made_shot_boxes = result.boxes[result.boxes.cls == 1]
        if len(made_shot_boxes) > 0 and shot_cooldown == 0:
            print(f"\n--- AI detected a 'made' shot at frame {frame_idx}! ---")
            shot_box = made_shot_boxes[0].xyxy[0]
            shot_x, shot_y = int((shot_box[0] + shot_box[2])/2), int((shot_box[1] + shot_box[3])/2)
            cv2.circle(annotated_frame, (shot_x, shot_y), 15, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "GOAL!", (shot_x + 10, shot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if player_data_this_frame:
                nearest_player = min(player_data_this_frame, key=lambda p: np.hypot(((p['bbox'][0] + p['bbox'][2])/2 - shot_x), ((p['bbox'][1] + p['bbox'][3])/2 - shot_y)))
                shot_data.append({"frame": frame_idx, "x": shot_x, "y": shot_y, "team_id": nearest_player["team_id"], "player_id": nearest_player["player_id"], "result": "made"})
            shot_cooldown = int(fps)

        # --- Draw other objects using correct NEW class IDs ---
        other_objects = result.boxes[~np.isin(result.boxes.cls.cpu(), [1, 2])]
        for box in other_objects:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls == 0: # basketball
                cv2.circle(annotated_frame, (int((x1+x2)/2), int((y1+y2)/2)), 7, (0, 165, 255), -1)
            elif cls == 3: # rim
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            elif cls == 4: # shoot
                cv2.putText(annotated_frame, "SHOOT", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
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
    
    if len(result.boxes[result.boxes.cls == 3]) > 0: # If rim was detected
        # A simple scaling based on the last known rim detection
        last_rim_box = result.boxes[result.boxes.cls == 3][-1].xyxy[0]
        pixel_to_meter = np.mean([last_rim_box[2] - last_rim_box[0], last_rim_box[3] - last_rim_box[1]]) / 0.45
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
        "tracking_result": s3_tracking_key, # Replace with actual key
        "shot_result": s3_shot_key,
        "video_result": s3_video_key,
        "thumbnail_url": s3_thumbnail_key
    })
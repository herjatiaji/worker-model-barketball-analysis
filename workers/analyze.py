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
model = YOLO("modelv5.pt")


def get_jersey_histogram(roi):
    """
    Calculates a normalized color histogram in the HSV space for a jersey ROI.
    This creates a "fingerprint" of the jersey's color profile.
    """
    if roi is None or roi.size == 0:
        return None
    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Calculate histogram for Hue (0-179) and Saturation (0-255). We ignore Value (brightness).
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [18, 3], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()
    except Exception as e:
        print(f"Error creating histogram: {e}")
        return None


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

    # --- PHASE 1: PRE-ANALYSIS SETUP ---
    print("--- Starting pre-analysis phase ---")
    cap_setup = cv2.VideoCapture(local_video_path)
    if not cap_setup.isOpened():
        print(f"Error: Could not open video at {local_video_path}")
        return json.dumps({"error": "Could not open video"})
        
    width = int(cap_setup.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_setup.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_setup.get(cv2.CAP_PROP_FPS) if cap_setup.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = int(cap_setup.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("Error: Video has no frames.")
        cap_setup.release()
        return json.dumps({"error": "Video has no frames"})

    print(f"Video info: {width}x{height}, {fps:.2f} fps, {frame_count} frames")

    thumbnail_frame = get_random_frame(cap_setup, frame_count)
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
            cap_setup.release()
            return json.dumps({"error": "Failed to create output video file."})

    print("Extracting jersey histograms for team detection...")
    all_histograms = []
    player_class_id = 2 # Your class ID for 'player'
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
                hist = get_jersey_histogram(roi)
                if hist is not None:
                    all_histograms.append(hist)
    
    cap_setup.release()
    print("--- Pre-analysis complete. ---")

    team_signature_histograms = []
    if len(all_histograms) >= 2:
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42).fit(all_histograms)
        team_signature_histograms = kmeans.cluster_centers_
        print("Team signature histograms identified.")
    else:
        print("Warning: Not enough players found to determine team colors automatically.")
    
    drawing_colors = [(255, 0, 0), (0, 0, 255)]

    # --- PHASE 2: MAIN PROCESSING LOOP ---
    print("Opening a fresh video capture for the main tracking loop...")
    cap_main = cv2.VideoCapture(local_video_path)
    if not cap_main.isOpened():
        return json.dumps({"error": "Failed to open video for processing."})

    id_to_team = {}
    player_data = []
    shot_data = []
    pixel_to_meter = None
    frame_idx = 0
    last_progress = -5
    
    ball_state = {'center': None, 'velocity': (0,0), 'frames_unseen': 0}
    player_shot_cooldowns = defaultdict(int)
    ball_carrier_state = {'track_id': None, 'frame_idx': 0}

    print("Starting analysis with Shot Attempt detection...")
    while cap_main.isOpened():
        ret, frame = cap_main.read()
        if not ret: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()
        
        ball_detections = result.boxes[result.boxes.cls == 0]
        player_detections = result.boxes[result.boxes.cls == 2]

        if ball_detections:
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
            tracked_players = result.boxes[result.boxes.cls == 2]
            for box in tracked_players:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                if track_id not in id_to_team:
                    roi = frame[int(y1 + 0.3*(y2-y1)):int(y1 + 0.7*(y2-y1)), x1:x2]
                    player_hist = get_jersey_histogram(roi)
                    if player_hist is not None and len(team_signature_histograms) == 2:
                        sim1 = cv2.compareHist(np.float32(player_hist), np.float32(team_signature_histograms[0]), cv2.HISTCMP_CORREL)
                        sim2 = cv2.compareHist(np.float32(player_hist), np.float32(team_signature_histograms[1]), cv2.HISTCMP_CORREL)
                        id_to_team[track_id] = 0 if sim1 > sim2 else 1
                
                team_id = id_to_team.get(track_id, 0)
                player_info = {"frame": frame_idx, "player_id": track_id, "team_id": team_id, "bbox": [x1, y1, x2, y2]}
                player_data.append(player_info)
                player_data_this_frame.append(player_info)
                color = drawing_colors[team_id]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"#{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        current_ball_carrier_id = None
        if ball_state['center'] and player_data_this_frame:
            closest_player = min(player_data_this_frame, key=lambda p: np.hypot(((p['bbox'][0] + p['bbox'][2])/2 - ball_state['center'][0]), ((p['bbox'][1] + p['bbox'][3])/2 - ball_state['center'][1])))
            dist_to_ball = np.hypot(((closest_player['bbox'][0] + closest_player['bbox'][2])/2 - ball_state['center'][0]), ((closest_player['bbox'][1] + closest_player['bbox'][3])/2 - ball_state['center'][1]))
            if dist_to_ball < 50:
                current_ball_carrier_id = closest_player['player_id']
                ball_carrier_state['track_id'] = current_ball_carrier_id
                ball_carrier_state['frame_idx'] = frame_idx
        
        if ball_carrier_state['track_id'] is not None and current_ball_carrier_id is None and (frame_idx - ball_carrier_state['frame_idx'] < 3):
            if player_shot_cooldowns[ball_carrier_state['track_id']] == 0:
                ball_vy = ball_state['velocity'][1]
                if ball_vy < -2:
                    print(f"\n--- SHOT ATTEMPT detected by Player #{ball_carrier_state['track_id']} at frame {frame_idx}! ---")
                    shot_data.append({"frame": frame_idx, "player_id": ball_carrier_state['track_id'], "result": "attempt"})
                    player_shot_cooldowns[ball_carrier_state['track_id']] = int(fps * 2)

        for pid in list(player_shot_cooldowns.keys()):
            if player_shot_cooldowns[pid] > 0:
                player_shot_cooldowns[pid] -= 1

        out.write(annotated_frame)
        frame_idx += 1
        progress = (frame_idx / frame_count) * 100
        print(f"🚀 Processing Frame: {frame_idx}/{frame_count} ({progress:.2f}%)", end='\r')

        if int(progress) % 5 == 0 and int(progress) != last_progress:
            asyncio.run_coroutine_threadsafe(job.updateProgress(int(progress)), loop)
            last_progress = int(progress)

    print("\nProcessing finished.")
    cap_main.release()
    out.release()
    
    if pixel_to_meter is None and len(result.boxes[result.boxes.cls == 1]) > 0:
        last_net_box = result.boxes[result.boxes.cls == 1][-1].xyxy[0]
        pixel_to_meter = np.mean([last_net_box[2] - last_net_box[0], last_net_box[3] - last_net_box[1]]) / 0.45
    
    court_length_px, court_width_px = (int(28 / pixel_to_meter), int(15 / pixel_to_meter)) if pixel_to_meter else (None, None)

    converted_players = []
    for p in player_data:
        x1, y1, x2, y2 = p["bbox"]
        converted_players.append({"frame": p["frame"], "player_id": p["player_id"], "team_id": p["team_id"], "x": int((x1 + x2) / 2), "y": int((y1 + y2) / 2)})

    player_json = {"court_length_px": court_length_px, "court_width_px": court_width_px, "players": converted_players}
    shot_json = {"court_length_px": court_length_px, "court_width_px": court_width_px, "shots": shot_data}

    player_json_path, shot_json_path = f"tmp/{uuid.uuid4()}.json", f"tmp/{uuid.uuid4()}.json"
    with open(player_json_path, "w") as f: json.dump(player_json, f, indent=2)
    with open(shot_json_path, "w") as f: json.dump(shot_json, f, indent=2)

    s3_tracking_key = f"uploads/{uuid.uuid4()}.json"
    s3_shot_key = f"uploads/{uuid.uuid4()}.json"
    s3_video_key = f"uploads/{uuid.uuid4()}.mp4"
    s3_thumbnail_key = f"uploads/thumbnails/{uuid.uuid4()}.jpg"

    try:
        s3_client.upload_file(player_json_path, settings.S3_BUCKET, s3_tracking_key, ExtraArgs={'ContentType': 'application/json'})
        s3_client.upload_file(shot_json_path, settings.S3_BUCKET, s3_shot_key, ExtraArgs={'ContentType': 'application/json'})
        s3_client.upload_file(output_video_path, settings.S3_BUCKET, s3_video_key, ExtraArgs={'ContentType': 'video/mp4'})
        if thumbnail_path and os.path.exists(thumbnail_path):
            s3_client.upload_file(thumbnail_path, settings.S3_BUCKET, s3_thumbnail_key, ExtraArgs={'ContentType': 'image/jpeg'})
    except Exception as e:
        print(f"FATAL: S3 upload failed: {e}")
    finally:
        temp_files = [local_video_path, output_video_path, player_json_path, shot_json_path, thumbnail_path]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception as e: print(f"Error cleaning up {temp_file}: {e}")

    print("S3 Upload complete. Job finished.")
    return json.dumps({"tracking_result": s3_tracking_key, "shot_result": s3_shot_key, "video_result": s3_video_key, "thumbnail_url": s3_thumbnail_key})
import asyncio
from bullmq import Job
from config.settings import settings
from config.s3 import s3_client
import uuid
from ultralytics import YOLO
import cv2
import os
import json


model = YOLO("src/analyze_worker/bestv2.pt")


async def analyze_video(job: Job, token=None):
    video_url = job.data.video_url
    # download to tmp folder
    s3_client.download_file(settings.S3_BUCKET, video_url, f"tmp/{video_url}")

    cap = cv2.VideoCapture(f"tmp/{video_url}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []
    
    # Buat video writer untuk simpan hasil prediksi (opsional)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"tmp/{uuid.uuid4()}.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            annotated_frame = result.plot()  # Gambarkan bounding box di frame

            # Simpan frame ke video output
            out.write(annotated_frame)

            # Tambahkan ke list deteksi (optional)
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                detections.append({
                    "frame": i,
                    "class_id": int(cls),
                    "bbox": [x1, y1, x2, y2]
                })

        i += 1
        progress = int((i / frame_count) * 100)
        await job.updateProgress(progress)
        print(f"Job progress: {progress}%")

    cap.release()
    out.release()  # jangan lupa release writer


    # Simpan hasil sebagai JSON
    output_filename = f"tmp/{uuid.uuid4()}.json"
    with open(output_filename, "w") as f:
        json.dump(detections, f)

    # Upload ke S3
    s3_key = f"uploads/{uuid.uuid4()}.json"
    s3_client.upload_file(output_filename, settings.S3_BUCKET, s3_key)
    video_key = f"uploads/{uuid.uuid4()}.mp4"
    s3_client.upload_file(output_path, settings.S3_BUCKET, video_key)
    print(f"Predicted video uploaded to: {video_key}")


    print("Job completed.")


print(uuid.uuid4());
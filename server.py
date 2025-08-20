from fastapi import FastAPI
from pydantic import BaseModel
import time
from workers.analyze_sync import analyze_video

app = FastAPI()

class Video(BaseModel):
    id: int
    url: str

@app.post("/analyze")
def analyze(video: Video):
    job = {
        'id': video.id,
        'video_url': video.url
    }

    analyze_video(job)

    return {
        "statusCode": 200,
        "message": "Completed",
        "data": job,
    }
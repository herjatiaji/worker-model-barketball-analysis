import asyncio
import signal
from bullmq import Worker
from config.settings import settings
from workers.analyze import analyze_video


async def main():
    shutdown_event = asyncio.Event()

    def signal_handler(signal, frame):
        print("Signal received, shutting down.")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = Worker("videoQueue", analyze_video, {
        "connection": settings.REDIS_URL,
        "lockDuration": 600000
    })

    await shutdown_event.wait()

    print("Cleaning up worker...")
    await worker.close()
    print("Worker shut down successfully.")

if __name__ == "__main__":
    asyncio.run(main())
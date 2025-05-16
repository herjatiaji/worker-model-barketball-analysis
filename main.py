import asyncio
import signal
from bullmq import Worker
from config.settings import settings
from workers.analyze import analyze_video, run_worker


async def main():
    shutdown_event = asyncio.Event()

    def signal_handler(signal, frame):
        print("Signal received, shutting down.")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = Worker("videoQueue", run_worker, {
        "connection": settings.REDIS_URL,
    })

    await shutdown_event.wait()

    print("Cleaning up worker...")
    await worker.close()
    print("Worker shut down successfully.")

if __name__ == "__main__":
    asyncio.run(main())
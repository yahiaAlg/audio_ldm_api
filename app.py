from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import multiprocessing as mp
from diffusers import AudioLDMPipeline
import numpy as np
import base64
import io
import scipy.io.wavfile
from typing import Optional
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from contextlib import contextmanager
import threading
import queue
import os

# Set number of threads for PyTorch
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

# Initialize FastAPI app
app = FastAPI(
    title="AudioLDM API",
    description="Text to Audio Generation CPU Version",
    version="1.0.0",
)

# Initialize rate limiter
limiter = Limiter(key_func=lambda _: "global")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
REPO_ID = "cvssp/audioldm-s-full-v2"
pipe = None
request_queue = queue.Queue()
model_lock = threading.Lock()


class AudioRequest(BaseModel):
    prompt: str
    audio_length: Optional[float] = 5.0
    num_inference_steps: Optional[int] = 10
    guidance_scale: Optional[float] = 2.5
    negative_prompt: Optional[str] = None


def initialize_model():
    global pipe
    if pipe is None:
        with model_lock:
            if pipe is None:  # Double-check pattern
                pipe = AudioLDMPipeline.from_pretrained(
                    REPO_ID, torch_dtype=torch.float32  # Use float32 for CPU
                )


@contextmanager
def get_model():
    initialize_model()
    try:
        with model_lock:
            yield pipe
    finally:
        # Clear any cached memory
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/generate-audio")
async def generate_audio(request: Request, audio_request: AudioRequest):
    try:
        with get_model() as pipe:
            # Input validation
            if not audio_request.prompt:
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            if audio_request.audio_length <= 0 or audio_request.audio_length > 30:
                raise HTTPException(
                    status_code=400,
                    detail="Audio length must be between 0 and 30 seconds",
                )
            if (
                audio_request.num_inference_steps <= 0
                or audio_request.num_inference_steps > 50
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Number of inference steps must be between 0 and 50",
                )

            # Generate audio
            audio = pipe(
                prompt=audio_request.prompt,
                audio_length_in_s=audio_request.audio_length,
                num_inference_steps=audio_request.num_inference_steps,
                guidance_scale=audio_request.guidance_scale,
                negative_prompt=audio_request.negative_prompt,
            ).audios[0]

            # Convert to WAV format
            buffer = io.BytesIO()
            scipy.io.wavfile.write(buffer, rate=16000, data=audio)
            buffer.seek(0)

            # Convert to base64
            audio_base64 = base64.b64encode(buffer.read()).decode()

            return {
                "status": "success",
                "audio_base64": audio_base64,
                "sample_rate": 16000,
                "duration": audio_request.audio_length,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cpu_threads": torch.get_num_threads(),
        "processor_count": mp.cpu_count(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

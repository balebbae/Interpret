"""
FastAPI service for bilingual audio separation
Deploy this to Google Cloud Run with GPU support

Uses pyannote.audio for speaker diarization to separate bilingual sermons.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import os
import uuid
from typing import Optional
import logging
import numpy as np
import torch
from collections import defaultdict
from pydub import AudioSegment
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face token for pyannote.audio authentication
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    logger.info("Hugging Face token configured")
else:
    logger.warning("No Hugging Face token found - pyannote.audio may fail to load models")

app = FastAPI(title="Audio Separation Service")

# Global pipeline - loaded once at startup
_diarization_pipeline = None

def get_diarization_pipeline():
    """Get or load the diarization pipeline (singleton pattern)"""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        logger.info("Loading pyannote diarization pipeline...")
        try:
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=HF_TOKEN
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _diarization_pipeline.to(device)
            logger.info(f"Pipeline loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pyannote pipeline: {e}")
    return _diarization_pipeline

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Firestore in production)
jobs_store = {}


class ProcessRequest(BaseModel):
    jobId: str
    inputFile: str
    inputBucket: str
    outputBucket: str


class JobStatus(BaseModel):
    jobId: str
    status: str
    progress: Optional[int] = None
    outputFiles: Optional[dict] = None
    error: Optional[str] = None


def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Download file from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} from {bucket_name}")


def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Upload file to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"Uploaded {destination_blob_name} to {bucket_name}")


# ============================================================================
# Audio Processing Functions (from Colab)
# ============================================================================

def ensure_wav_16k_mono(input_path, target_sr=16000):
    """
    Convert input audio to 16kHz mono WAV once and reuse it.
    Returns path to the WAV file.
    """
    base, _ = os.path.splitext(input_path)
    wav_path = base + "_16k_mono.wav"

    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        # Simple peak normalization
        change_dB = -audio.max_dBFS
        audio = audio.apply_gain(change_dB)
        audio.export(wav_path, format="wav")
        logger.info(f"Created {wav_path}")
    else:
        logger.info(f"Using cached {wav_path}")

    return wav_path


def load_audio_to_mono_16k(path, target_sr=16000):
    """
    Load an audio file with pydub, convert to mono float32 [-1, 1], resample to 16 kHz.
    Returns (waveform, sample_rate).
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sr)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / (2 ** (8 * audio.sample_width - 1))

    return samples, target_sr


def run_diarization(wav_path):
    """
    Run speaker diarization on a 16k mono WAV using GPU if available.
    Returns list of (start_sec, end_sec, speaker_id), sorted by time.
    """
    logger.info(f"Running diarization on {wav_path}")

    pipeline = get_diarization_pipeline()
    diarization = pipeline(wav_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), speaker))

    segments.sort(key=lambda x: x[0])
    return segments


def assign_speakers_to_tracks(diar_segments):
    """
    Decide for each speaker whether they are 'lang1' or 'lang2'.

    Strategy:
      - Compute speaking time per speaker.
      - Sort speakers by total duration.
      - Longest speaker -> 'lang1'
      - Second longest speaker -> 'lang2'
      - Any remaining speakers (if any) -> 'lang1' by default.
    """
    durations = defaultdict(float)
    for start, end, spk in diar_segments:
        durations[spk] += (end - start)

    speakers = sorted(durations, key=durations.get, reverse=True)
    if not speakers:
        raise RuntimeError("No speakers found in diarization.")

    speaker_track = {}

    if len(speakers) >= 1:
        speaker_track[speakers[0]] = "lang1"
    if len(speakers) >= 2:
        speaker_track[speakers[1]] = "lang2"

    for spk in speakers[2:]:
        # Fallback: group minor speakers with lang1
        speaker_track[spk] = "lang1"

    for spk in speakers:
        logger.info(f"Speaker {spk}: duration={durations[spk]:.1f}s -> {speaker_track[spk]}")

    return speaker_track


def build_tracks_from_diarization(y, sr, diar_segments, speaker_track):
    """
    Concatenate segments for speakers assigned to lang1 vs lang2.
    """
    chunks_1 = []
    chunks_2 = []

    for start, end, spk in diar_segments:
        s_idx = int(start * sr)
        e_idx = int(end * sr)
        seg_audio = y[s_idx:e_idx]
        if seg_audio.size == 0:
            continue

        lab = speaker_track.get(spk, "lang1")
        if lab == "lang1":
            chunks_1.append(seg_audio)
        else:
            chunks_2.append(seg_audio)

    track_1 = np.concatenate(chunks_1) if chunks_1 else np.array([], dtype=np.float32)
    track_2 = np.concatenate(chunks_2) if chunks_2 else np.array([], dtype=np.float32)

    return track_1, track_2


def array_to_audiosegment(y, sr=16000):
    """
    Convert 1D float32 numpy array in [-1,1] to a pydub AudioSegment.
    If empty, return 1s of silence.
    """
    if y.size == 0:
        return AudioSegment.silent(duration=1000, frame_rate=sr)

    y = np.clip(y, -1.0, 1.0)
    int_samples = (y * 32767).astype(np.int16)
    audio = AudioSegment(
        int_samples.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    return audio


def export_tracks(track_1, track_2, sr, output1_path, output2_path):
    """
    Export two mono MP3 files to specified paths.
    """
    audio1 = array_to_audiosegment(track_1, sr)
    audio2 = array_to_audiosegment(track_2, sr)

    audio1.export(output1_path, format="mp3", bitrate="128k")
    audio2.export(output2_path, format="mp3", bitrate="128k")

    return output1_path, output2_path


def process_audio(input_file: str, job_id: str) -> tuple[str, str]:
    """
    Main audio processing pipeline - separates bilingual sermon into two tracks.

    Uses pyannote.audio for speaker diarization to identify speakers,
    then splits audio by speaker into language1 and language2 tracks.

    Args:
        input_file: Path to input MP3 file
        job_id: Unique job identifier

    Returns:
        tuple: (path_to_language1_file, path_to_language2_file)
    """
    logger.info(f"Starting audio processing for job {job_id}")

    # 1. Convert to 16k mono WAV once (used for both diarization and slicing)
    wav_path = ensure_wav_16k_mono(input_file, target_sr=16000)

    # 2. Load audio as numpy array for slicing
    y, sr = load_audio_to_mono_16k(wav_path, target_sr=16000)
    logger.info(f"Loaded audio: {len(y)} samples at {sr} Hz")

    # 3. Diarization (who speaks when)
    diar_segments = run_diarization(wav_path)
    num_speakers = len({spk for _, _, spk in diar_segments})
    logger.info(f"Diarization segments: {len(diar_segments)}, speakers: {num_speakers}")

    if not diar_segments:
        raise RuntimeError("No segments from diarization – check audio/pyannote.")

    # 4. Assign each speaker to track 'lang1' or 'lang2'
    speaker_track = assign_speakers_to_tracks(diar_segments)

    # 5. Build tracks
    track_1, track_2 = build_tracks_from_diarization(y, sr, diar_segments, speaker_track)
    logger.info(f"Built tracks: lang1={len(track_1)} samples, lang2={len(track_2)} samples")

    # 6. Export
    output1 = f"/tmp/{job_id}_language1.mp3"
    output2 = f"/tmp/{job_id}_language2.mp3"
    export_tracks(track_1, track_2, sr, output1, output2)

    # 7. Cleanup intermediate WAV file
    if os.path.exists(wav_path):
        os.remove(wav_path)
        logger.info(f"Cleaned up {wav_path}")

    logger.info(f"Saved {output1} and {output2}")
    return output1, output2


async def process_job_async(job_id: str, input_bucket: str, input_file: str, output_bucket: str):
    """Background task to process audio separation"""
    try:
        logger.info(f"Starting job {job_id}")

        # Update status to processing
        jobs_store[job_id] = {
            "status": "processing",
            "progress": 10
        }

        # Download input file
        input_path = f"/tmp/{job_id}_input.mp3"
        download_from_gcs(input_bucket, input_file, input_path)

        jobs_store[job_id]["progress"] = 30

        # Process audio (this is where your model runs)
        logger.info(f"Processing audio for job {job_id}")
        output1_path, output2_path = process_audio(input_path, job_id)

        jobs_store[job_id]["progress"] = 70

        # Upload output files
        output1_name = f"{job_id}_language1.mp3"
        output2_name = f"{job_id}_language2.mp3"

        upload_to_gcs(output_bucket, output1_path, output1_name)
        upload_to_gcs(output_bucket, output2_path, output2_name)

        # Update job status
        jobs_store[job_id] = {
            "status": "completed",
            "progress": 100,
            "outputFiles": {
                "language1": output1_name,
                "language2": output2_name
            }
        }

        # Cleanup temp files
        os.remove(input_path)
        os.remove(output1_path)
        os.remove(output2_path)

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        jobs_store[job_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.post("/process")
async def process_audio_endpoint(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Endpoint to start audio processing job"""

    # Store initial job status
    jobs_store[request.jobId] = {
        "status": "pending",
        "progress": 0
    }

    # Start background processing
    background_tasks.add_task(
        process_job_async,
        request.jobId,
        request.inputBucket,
        request.inputFile,
        request.outputBucket
    )

    return {"message": "Processing started", "jobId": request.jobId}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Endpoint to check job status"""

    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs_store[job_id]
    return {
        "jobId": job_id,
        **job_data
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

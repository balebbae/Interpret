"""
Modal serverless GPU application for bilingual audio separation.
Optimized for maximum performance with A10G GPU.

Uses pyannote.audio 3.1 for speaker diarization to separate bilingual sermons
into two separate audio tracks (language1 and language2).
"""

import modal
import os
import tempfile
import base64
from collections import defaultdict

# Define the Modal app
app = modal.App("audio-separator")


def download_models():
    """Pre-download pyannote models during image build."""
    from pyannote.audio import Pipeline
    import torch

    print("Pre-downloading pyannote models...")
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")

    # Download the pipeline (this caches models)
    # Using v2.x for better bilingual speaker separation
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    print("Models downloaded successfully!")


# Build optimized container image with pre-downloaded models
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04")
    .apt_install(
        "python3.10",
        "python3-pip",
        "ffmpeg",
        "libsndfile1",
        "git"
    )
    .run_commands("ln -sf /usr/bin/python3.10 /usr/bin/python")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "pyannote.audio==3.1.1",
        "pydub==0.25.1",
        "numpy==1.26.4",
        "huggingface_hub==0.23.5",
        "fastapi",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface")]
    )
)


@app.cls(
    gpu="L4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,  # 30 minute timeout for very long audio files
    scaledown_window=180,  # Keep warm for 3 minutes
    memory=8192,  # 8GB RAM for long audio processing
)
class AudioSeparator:
    """
    Stateful class that pre-loads the pyannote diarization model on startup
    and processes audio files on demand.
    """

    @modal.enter()
    def load_model(self):
        """Load the pyannote pipeline once when container starts."""
        import torch
        from pyannote.audio import Pipeline

        print("Loading pyannote diarization pipeline v2.x...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            print("WARNING: No HUGGING_FACE_TOKEN found!")
        else:
            print(f"HuggingFace token found: {hf_token[:10]}...")

        # Load pipeline v2.x for better bilingual separation
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

        # Optimize batch sizes for faster processing
        if hasattr(self.pipeline, '_segmentation'):
            self.pipeline._segmentation.batch_size = 64  # default is 32
            print("Segmentation batch size set to 64")
        if hasattr(self.pipeline, '_embedding'):
            self.pipeline._embedding.batch_size = 64  # default is 32
            print("Embedding batch size set to 64")

        print(f"Pipeline loaded on {self.device}")

    @modal.fastapi_endpoint(method="POST")
    async def separate(self, request: dict):
        """
        Process audio file and return separated language tracks.

        Expects JSON with:
        - audio_base64: Base64 encoded MP3 file

        Returns JSON with:
        - language1: Base64 encoded MP3 of first language track
        - language2: Base64 encoded MP3 of second language track
        """
        import torch
        import torchaudio
        import numpy as np
        import time

        audio_base64 = request.get("audio_base64")
        if not audio_base64:
            return {"error": "No audio_base64 provided"}, 400

        # Decode the audio file
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input file temporarily for torchaudio
            input_path = os.path.join(tmpdir, "input.mp3")
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            print("Starting optimized audio processing...")
            total_start = time.time()

            # 1. Load and preprocess with torchaudio (GPU-accelerated)
            preprocess_start = time.time()
            waveform, sample_rate = self._load_and_preprocess_audio(input_path)
            print(f"Preprocessing completed in {time.time() - preprocess_start:.1f}s")
            print(f"Loaded audio: {waveform.shape[1]} samples at {sample_rate} Hz")
            print(f"Audio duration: {waveform.shape[1] / sample_rate / 60:.1f} minutes")

            # 2. Run diarization with FP16 mixed precision
            diar_start = time.time()
            diar_segments = self._run_diarization_optimized(waveform, sample_rate)
            print(f"Diarization completed in {time.time() - diar_start:.1f}s")

            num_speakers = len({spk for _, _, spk in diar_segments})
            print(f"Found {len(diar_segments)} segments, {num_speakers} speakers")

            if not diar_segments:
                return {"error": "No segments from diarization"}, 500

            # 3. Assign speakers to tracks
            speaker_track = self._assign_speakers_to_tracks(diar_segments)

            # 4. Build tracks (convert waveform to numpy for slicing)
            build_start = time.time()
            y = waveform.squeeze(0).cpu().numpy()  # Convert to numpy
            track_1, track_2 = self._build_tracks_from_diarization(
                y, sample_rate, diar_segments, speaker_track
            )
            print(f"Track building completed in {time.time() - build_start:.1f}s")
            print(f"Built tracks: lang1={len(track_1)} samples, lang2={len(track_2)} samples")

            # 5. Export to MP3
            export_start = time.time()
            output1_path = os.path.join(tmpdir, "language1.mp3")
            output2_path = os.path.join(tmpdir, "language2.mp3")
            self._export_tracks(track_1, track_2, sample_rate, output1_path, output2_path)
            print(f"Export completed in {time.time() - export_start:.1f}s")

            # 6. Read and encode output files
            with open(output1_path, "rb") as f:
                lang1_bytes = f.read()
            with open(output2_path, "rb") as f:
                lang2_bytes = f.read()

            total_time = time.time() - total_start
            print(f"Total processing completed in {total_time:.1f}s")

            return {
                "language1": base64.b64encode(lang1_bytes).decode("utf-8"),
                "language2": base64.b64encode(lang2_bytes).decode("utf-8"),
            }

    def _load_and_preprocess_audio(self, input_path: str, target_sr: int = 16000):
        """Load and preprocess audio using torchaudio (faster than pydub)."""
        import torchaudio
        import torch

        print(f"Loading audio with torchaudio...")

        # Load audio file
        waveform, sr = torchaudio.load(input_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print("Converted to mono")

        # Resample to target sample rate if needed
        if sr != target_sr:
            print(f"Resampling from {sr}Hz to {target_sr}Hz...")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Normalize audio (peak normalization)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, target_sr

    def _run_diarization_optimized(self, waveform, sample_rate: int):
        """Run speaker diarization with FP16 mixed precision."""
        import torch
        import time

        print(f"Running diarization on {self.device}")
        print("Starting diarization pipeline with FP16 mixed precision...")

        start_time = time.time()

        # Use automatic mixed precision for faster GPU computation
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Pass waveform directly to pipeline (no disk I/O)
            # Force exactly 2 speakers for bilingual audio separation
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=2
            )

        elapsed = time.time() - start_time
        print(f"Diarization inference completed in {elapsed:.1f} seconds")

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((float(turn.start), float(turn.end), speaker))

        segments.sort(key=lambda x: x[0])
        print(f"Found {len(segments)} segments")
        return segments

    def _assign_speakers_to_tracks(self, diar_segments):
        """Assign speakers to lang1/lang2 based on speaking duration."""
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

        # Fallback: group minor speakers with lang1
        for spk in speakers[2:]:
            speaker_track[spk] = "lang1"

        for spk in speakers:
            print(f"Speaker {spk}: duration={durations[spk]:.1f}s -> {speaker_track[spk]}")

        return speaker_track

    def _build_tracks_from_diarization(self, y, sr, diar_segments, speaker_track):
        """Concatenate segments for speakers assigned to lang1 vs lang2."""
        import numpy as np

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

    def _export_tracks(self, track_1, track_2, sr, output1_path, output2_path):
        """Export two mono MP3 files."""
        import numpy as np
        from pydub import AudioSegment

        def array_to_audiosegment(y, sr=16000):
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

        audio1 = array_to_audiosegment(track_1, sr)
        audio2 = array_to_audiosegment(track_2, sr)

        audio1.export(output1_path, format="mp3", bitrate="128k")
        audio2.export(output2_path, format="mp3", bitrate="128k")


# Optional: Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the audio separator locally."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_app.py -- <path_to_mp3>")
        return

    input_file = sys.argv[1]

    with open(input_file, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    separator = AudioSeparator()
    result = separator.separate.remote({"audio_base64": audio_base64})

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save output files
    with open("language1.mp3", "wb") as f:
        f.write(base64.b64decode(result["language1"]))
    with open("language2.mp3", "wb") as f:
        f.write(base64.b64decode(result["language2"]))

    print("Saved language1.mp3 and language2.mp3")

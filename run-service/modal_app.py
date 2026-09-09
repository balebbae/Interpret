"""
Modal serverless GPU application for bilingual audio separation.

Takes an uploaded MP3, runs pyannote speaker diarization on a 16 kHz mono copy,
then cuts two single-language tracks out of the native-rate audio and encodes
them to MP3 with ffmpeg. Progress is streamed to the browser over SSE.
"""

import modal
import os
import tempfile
import base64
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

app = modal.App("audio-separator")

DIARIZATION_SR = 16000
MP3_BITRATE = "128k"

# Timeline clean-up applied to the raw diarization output
MIN_SEGMENT_SEC = 0.25   # drop shorter segments (back-channels, glitches)
MERGE_GAP_SEC = 0.5      # join same-speaker segments separated by less than this
SEGMENT_PAD_SEC = 0.15   # extend each segment so word edges are not clipped
FADE_SEC = 0.015         # ramp applied at every cut to avoid clicks


def download_models():
    """Pre-download pyannote models during image build."""
    from pyannote.audio import Pipeline

    print("Pre-downloading pyannote models...")
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.environ.get("HUGGING_FACE_TOKEN"),
    )
    print("Models downloaded successfully!")


image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04")
    .apt_install("python3.10", "python3-pip", "ffmpeg", "libsndfile1", "git")
    .run_commands("ln -sf /usr/bin/python3.10 /usr/bin/python")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "pyannote.audio==3.1.1",
        "speechbrain==1.0.3",  # 1.1+ dropped use_auth_token, which pyannote 3.1.1 still passes
        "numpy==1.26.4",
        "huggingface_hub==0.23.5",
        "fastapi",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_function(download_models, secrets=[modal.Secret.from_name("huggingface")])
)


# --------------------------------------------------------------------------- #
# Audio helpers (plain functions so they can be unit-tested without Modal)
# --------------------------------------------------------------------------- #

def probe_sample_rate(path: str) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate", "-of", "default=nw=1:nk=1", path,
        ],
        capture_output=True, text=True,
    )
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines:
        raise ValueError("Could not read audio file: no audio stream found")
    return int(lines[0])


def decode_audio(path: str, sample_rate: int, dtype: str):
    """Decode any ffmpeg-readable file to mono PCM at `sample_rate`. dtype: 'float32' or 'int16'."""
    import numpy as np

    fmt = {"float32": "f32le", "int16": "s16le"}[dtype]
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", path,
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", fmt, "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(f"Could not decode audio file: {detail[-1] if detail else 'unknown ffmpeg error'}")
    return np.frombuffer(proc.stdout, dtype=dtype)


def encode_mp3(pcm_int16, sample_rate: int, output_path: str, bitrate: str = MP3_BITRATE):
    """Encode mono int16 PCM to MP3 via ffmpeg/libmp3lame."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
            "-codec:a", "libmp3lame", "-b:a", bitrate, output_path,
        ],
        input=pcm_int16.tobytes(), capture_output=True, check=True,
    )


def clean_segments(
    segments,
    total_duration: float,
    min_segment: float = MIN_SEGMENT_SEC,
    merge_gap: float = MERGE_GAP_SEC,
    pad: float = SEGMENT_PAD_SEC,
):
    """
    Tidy raw diarization output: drop micro-segments, pad every segment a little,
    and merge same-speaker segments that are separated by a short pause so the
    speaker's natural rhythm survives concatenation.

    segments: iterable of (start, end, speaker). Returns the same shape, sorted by start.
    """
    by_speaker = defaultdict(list)
    for start, end, speaker in segments:
        if end - start >= min_segment:
            by_speaker[speaker].append((start, end))

    cleaned = []
    for speaker, spans in by_speaker.items():
        spans.sort()
        merged = []
        for start, end in spans:
            start = max(0.0, start - pad)
            end = min(total_duration, end + pad)
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        cleaned.extend((s, e, speaker) for s, e in merged)

    cleaned.sort(key=lambda x: x[0])
    return cleaned


def assign_speakers_to_tracks(segments):
    """Rank speakers by total speaking time: longest -> lang1, second -> lang2, rest -> lang1."""
    durations = defaultdict(float)
    for start, end, speaker in segments:
        durations[speaker] += end - start

    ranked = sorted(durations, key=durations.get, reverse=True)
    if not ranked:
        raise RuntimeError("No speakers found in diarization.")

    speaker_track = {speaker: "lang1" for speaker in ranked}
    if len(ranked) >= 2:
        speaker_track[ranked[1]] = "lang2"

    for speaker in ranked:
        print(f"Speaker {speaker}: duration={durations[speaker]:.1f}s -> {speaker_track[speaker]}")
    return speaker_track


def build_track(audio_int16, sample_rate: int, spans, fade_sec: float = FADE_SEC):
    """Concatenate [start, end) spans of `audio_int16` with a short fade at every edge."""
    import numpy as np

    fade_len = int(fade_sec * sample_rate)
    chunks = []
    for start, end in spans:
        chunk = audio_int16[int(start * sample_rate):int(end * sample_rate)]
        if chunk.size == 0:
            continue
        chunk = chunk.copy()
        n = min(fade_len, chunk.size // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            chunk[:n] = (chunk[:n] * ramp).astype(np.int16)
            chunk[-n:] = (chunk[-n:] * ramp[::-1]).astype(np.int16)
        chunks.append(chunk)

    if not chunks:
        return np.zeros(sample_rate, dtype=np.int16)  # 1 s of silence
    return np.concatenate(chunks)


# --------------------------------------------------------------------------- #
# Modal service
# --------------------------------------------------------------------------- #

@app.cls(
    gpu="L4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,  # 30 minute timeout for very long audio files
    scaledown_window=360,  # Keep warm for 6 minutes
    memory=8192,
)
class AudioSeparator:
    """Pre-loads the pyannote diarization pipeline once per container and processes uploads on demand."""

    @modal.enter()
    def load_model(self):
        import torch
        from pyannote.audio import Pipeline

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            print("WARNING: No HUGGING_FACE_TOKEN found!")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)

        # Embedding extraction dominates GPU time; both stages default to 32.
        self.pipeline.segmentation_batch_size = 64
        self.pipeline.embedding_batch_size = 64

        print(f"Pipeline loaded on {self.device}")

    # ------------------------------------------------------------------ #
    # Core (blocking) pipeline shared by the web endpoint and local runs
    # ------------------------------------------------------------------ #

    def _run_pipeline(self, input_path: str, workdir: str, progress) -> dict:
        """
        Run the full separation on `input_path`. `progress(stage, message, percent)` is
        called from the worker thread. Returns paths of the two tracks plus timings.
        """
        import numpy as np
        import torch

        timings = {}
        total_start = time.time()

        # 1. Decode a 16 kHz mono copy for diarization
        progress("preprocess", "Decoding audio...", 30)
        t0 = time.time()
        native_sr = probe_sample_rate(input_path)
        diar_audio = decode_audio(input_path, DIARIZATION_SR, "float32")
        peak = np.abs(diar_audio).max()
        if peak > 0:
            diar_audio = diar_audio / peak
        waveform = torch.from_numpy(np.ascontiguousarray(diar_audio)).unsqueeze(0)
        duration = waveform.shape[1] / DIARIZATION_SR
        timings["decode"] = time.time() - t0
        progress("preprocess", f"Audio loaded: {duration / 60:.1f} minutes ({native_sr} Hz)", 40)

        # 2. Diarize on the GPU while ffmpeg decodes the native-rate audio on the CPU
        progress("diarization", "Running AI speaker separation...", 45)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as pool:
            native_future = pool.submit(decode_audio, input_path, native_sr, "int16")
            raw_segments = self._diarize(waveform)
            timings["diarization"] = time.time() - t0
            t0 = time.time()
            native_audio = native_future.result()
        timings["native_decode_wait"] = time.time() - t0

        segments = clean_segments(raw_segments, duration)
        num_speakers = len({spk for _, _, spk in segments})
        print(f"{len(raw_segments)} raw segments -> {len(segments)} cleaned, {num_speakers} speakers")
        progress("diarization", f"Found {num_speakers} speakers in {len(segments)} segments", 75)
        if not segments:
            raise RuntimeError("No speech segments found in the audio")

        # 3. Cut tracks from the native-rate audio
        progress("build", "Building separate language tracks...", 80)
        t0 = time.time()
        speaker_track = assign_speakers_to_tracks(segments)
        spans = {"lang1": [], "lang2": []}
        for start, end, speaker in segments:
            spans[speaker_track[speaker]].append((start, end))
        track_1 = build_track(native_audio, native_sr, spans["lang1"])
        track_2 = build_track(native_audio, native_sr, spans["lang2"])
        del native_audio
        timings["build"] = time.time() - t0

        # 4. Encode both MP3s in parallel
        progress("export", "Exporting MP3 files...", 90)
        t0 = time.time()
        output1 = os.path.join(workdir, "language1.mp3")
        output2 = os.path.join(workdir, "language2.mp3")
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(encode_mp3, track_1, native_sr, output1)
            f2 = pool.submit(encode_mp3, track_2, native_sr, output2)
            f1.result()
            f2.result()
        timings["export"] = time.time() - t0
        timings["total"] = time.time() - total_start

        print("Timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))
        return {
            "tracks": [output1, output2],
            "duration_seconds": duration,
            "sample_rate": native_sr,
            "num_speakers": num_speakers,
            "num_segments": len(segments),
            "segments": [(round(s, 3), round(e, 3), speaker_track[spk]) for s, e, spk in segments],
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }

    def _diarize(self, waveform):
        """Run the pyannote pipeline on a (1, samples) 16 kHz tensor. Returns [(start, end, speaker)]."""
        import torch

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": DIARIZATION_SR},
                num_speakers=2,
            )

        segments = [
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda x: x[0])
        return segments

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    @modal.method()
    def separate_bytes(self, audio_bytes: bytes) -> dict:
        """Synchronous variant used by `modal run` for local testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp3")
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            def log_progress(stage, message, percent):
                print(f"[{percent:3d}%] {stage}: {message}")

            result = self._run_pipeline(input_path, tmpdir, log_progress)
            with open(result["tracks"][0], "rb") as f:
                result["language1"] = f.read()
            with open(result["tracks"][1], "rb") as f:
                result["language2"] = f.read()
            del result["tracks"]
            return result

    @modal.fastapi_endpoint(method="POST")
    async def separate(self, item: dict):
        """
        Process an uploaded MP3 and stream results back as Server-Sent Events.

        Expects JSON: {"audio_base64": "<base64 MP3>"}

        Events:
        - progress: {stage, message, progress}
        - complete: {language1, language2, duration_seconds, num_speakers, num_segments,
                     segments: [[start, end, "lang1"|"lang2"], ...], timings, progress: 100}
        - error:    {message}
        """
        from fastapi.responses import StreamingResponse
        import asyncio
        import json
        import traceback

        async def event_generator():
            def send_event(event_type: str, data: dict) -> str:
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            try:
                audio_base64 = item.get("audio_base64")
                if not audio_base64:
                    yield send_event("error", {"message": "audio_base64 is required"})
                    return

                loop = asyncio.get_running_loop()

                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = os.path.join(tmpdir, "input.mp3")

                    yield send_event("progress", {
                        "stage": "upload", "message": "Upload received, decoding audio...", "progress": 5,
                    })
                    await asyncio.sleep(0)

                    try:
                        size_mb = await loop.run_in_executor(
                            None, self._write_uploaded_audio, audio_base64, input_path
                        )
                    except ValueError as e:
                        yield send_event("error", {"message": str(e)})
                        return

                    yield send_event("progress", {
                        "stage": "upload", "message": f"Received {size_mb:.1f} MB of audio", "progress": 25,
                    })
                    await asyncio.sleep(0)

                    # Run the blocking pipeline in a worker thread; it reports progress
                    # through a queue so we can keep streaming SSE events meanwhile.
                    queue: asyncio.Queue = asyncio.Queue()

                    def progress(stage, message, percent):
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("progress", {"stage": stage, "message": message, "progress": percent}),
                        )

                    future = loop.run_in_executor(None, self._run_pipeline, input_path, tmpdir, progress)
                    future.add_done_callback(lambda f: queue.put_nowait(("done", f)))

                    while True:
                        kind, payload = await queue.get()
                        if kind == "progress":
                            yield send_event("progress", payload)
                        else:
                            result = payload.result()  # re-raises pipeline errors
                            break

                    yield send_event("progress", {
                        "stage": "encode", "message": "Encoding results...", "progress": 95,
                    })
                    await asyncio.sleep(0)

                    with open(result["tracks"][0], "rb") as f:
                        lang1_bytes = f.read()
                    with open(result["tracks"][1], "rb") as f:
                        lang2_bytes = f.read()

                    yield send_event("complete", {
                        "language1": base64.b64encode(lang1_bytes).decode("utf-8"),
                        "language2": base64.b64encode(lang2_bytes).decode("utf-8"),
                        "duration_seconds": round(result["duration_seconds"], 1),
                        "num_speakers": result["num_speakers"],
                        "num_segments": result["num_segments"],
                        "segments": result["segments"],
                        "timings": result["timings"],
                        "progress": 100,
                    })

            except Exception as e:
                print(f"Error in event_generator: {e}")
                print(traceback.format_exc())
                yield send_event("error", {"message": str(e)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            },
        )

    def _write_uploaded_audio(self, audio_base64: str, output_path: str) -> float:
        """Decode a base64 MP3 upload to disk. Returns size in MB."""
        import binascii

        # Tolerate a data URL prefix ("data:audio/mpeg;base64,...")
        if audio_base64.startswith("data:"):
            audio_base64 = audio_base64.split(",", 1)[-1]

        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("audio_base64 is not valid base64 data")

        if not audio_bytes:
            raise ValueError("Uploaded audio file is empty")

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        size_mb = len(audio_bytes) / (1024 * 1024)
        print(f"Wrote uploaded audio: {size_mb:.1f} MB -> {output_path}")
        return size_mb


@app.local_entrypoint()
def main(path: str, out_dir: str = "."):
    """Test the separator: modal run modal_app.py --path ./sermon.mp3 [--out-dir ./out]"""
    import json

    with open(path, "rb") as f:
        audio_bytes = f.read()

    result = AudioSeparator().separate_bytes.remote(audio_bytes)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "language1.mp3"), "wb") as f:
        f.write(result.pop("language1"))
    with open(os.path.join(out_dir, "language2.mp3"), "wb") as f:
        f.write(result.pop("language2"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Audio: {result['duration_seconds'] / 60:.1f} min, {result['num_speakers']} speakers, "
          f"{result['num_segments']} segments")
    print("Timings: " + ", ".join(f"{k}={v}s" for k, v in result["timings"].items()))
    print(f"Saved language1.mp3, language2.mp3 and result.json to {out_dir}")

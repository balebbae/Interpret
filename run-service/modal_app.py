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


class DownloadProgress:
    """Thread-safe progress tracker for yt-dlp downloads."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.percent = 0.0
        self.status = "starting"
        self.speed = None
        self.eta = None

    def update(self, d):
        """Called by yt-dlp progress hook with download info."""
        with self._lock:
            if d['status'] == 'downloading':
                self.status = 'downloading'

                # Try to get percentage from various yt-dlp fields
                if 'downloaded_bytes' in d and 'total_bytes' in d and d['total_bytes'] > 0:
                    self.percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                elif 'downloaded_bytes' in d and 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                    self.percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                elif '_percent_str' in d:
                    # Parse "X.Y%" string
                    try:
                        self.percent = float(d['_percent_str'].replace('%', '').strip())
                    except:
                        pass

                self.speed = d.get('speed')
                self.eta = d.get('eta')

                # Debug: print progress updates
                print(f"[DownloadProgress] {self.percent:.1f}% - speed: {self.speed}")

            elif d['status'] == 'finished':
                self.percent = 100.0
                self.status = 'finished'
                print(f"[DownloadProgress] Download finished!")

    def get_progress(self):
        """Thread-safe getter for progress."""
        with self._lock:
            return self.percent, self.speed


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
        "git",
        "aria2"  # PHASE 2: Multi-connection downloader for 5-10x faster YouTube downloads
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
        "yt-dlp>=2024.12.23",  # YouTube downloader with bot detection handling
        force_build=True,
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface"
    })
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface")]
    )
)


@app.cls(
    gpu="L4",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("youtube-cookies"),
    ],
    timeout=1800,  # 30 minute timeout for very long audio files
    scaledown_window=360,  # Keep warm for 3 minutes
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

    def _download_youtube_audio(self, youtube_url: str, output_path: str, progress_tracker=None):
        """
        Download audio from YouTube URL and convert to MP3.

        Uses yt-dlp with authenticated cookies to bypass YouTube's bot detection.
        Cookies must be provided via Modal secret 'youtube-cookies'.

        Args:
            youtube_url: YouTube video URL
            output_path: Path to save MP3 file
            progress_tracker: Optional DownloadProgress object to track download progress

        Raises:
            RuntimeError: If download fails or cookies are missing/invalid
        """
        import yt_dlp
        import time
        import traceback
        import tempfile
        import os
        from datetime import datetime

        print(f"Downloading audio from YouTube: {youtube_url}")

        # === Cookie Validation and Setup ===
        cookie_file_path = None
        youtube_cookies = os.getenv("YOUTUBE_COOKIES")

        if youtube_cookies:
            # Debug: Show what we received
            print(f"  → Cookie content length: {len(youtube_cookies)} characters")
            print(f"  → Cookie content preview: {youtube_cookies[:100]}...")

            # Write cookies to temp file
            cookie_fd, cookie_file_path = tempfile.mkstemp(suffix=".txt", text=True)
            try:
                os.write(cookie_fd, youtube_cookies.encode('utf-8'))
                os.close(cookie_fd)
                print(f"✓ YouTube cookies loaded from secret (saved to {cookie_file_path})")

                # Validate cookie format
                cookie_lines = youtube_cookies.strip().split('\n')
                auth_cookies = []
                cookie_count = 0

                for line in cookie_lines:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 7:  # Valid Netscape cookie format
                        cookie_count += 1
                        cookie_name = parts[5]
                        # Check for important auth cookies
                        if cookie_name in ['SID', '__Secure-1PSID', 'SAPISID', '__Secure-3PAPISID']:
                            expiry = parts[4]
                            auth_cookies.append(cookie_name)
                            # Check if cookie is expired (basic check)
                            try:
                                expiry_timestamp = int(expiry)
                                current_timestamp = int(datetime.now().timestamp())
                                if expiry_timestamp < current_timestamp:
                                    print(f"  ⚠ Warning: Cookie {cookie_name} appears expired")
                            except:
                                pass
                    elif len(parts) > 0:
                        # Debug: show malformed lines
                        print(f"  ⚠ Skipping malformed cookie line (has {len(parts)} parts): {line[:80]}")

                print(f"  → Parsed {cookie_count} cookies from file")
                if auth_cookies:
                    print(f"  → Found auth cookies: {', '.join(auth_cookies)}")
                else:
                    print(f"  ⚠ Warning: No YouTube auth cookies found (SID, SAPISID, etc.)")
                    print(f"    This may cause authentication failures. Export fresh cookies from browser.")

            except Exception as e:
                print(f"✗ Failed to process cookie file: {e}")
                import traceback
                print(traceback.format_exc())
                cookie_file_path = None
        else:
            print("⚠ WARNING: No YouTube cookies found!")
            print("  Downloads will likely fail due to YouTube's bot detection.")
            print("  To fix: Export cookies from browser and create Modal secret:")
            print("  modal secret create youtube-cookies YOUTUBE_COOKIES=\"$(cat cookies.txt)\"")

        # === Configure Download Options ===
        base_opts = {
            # Format selection - flexible with good fallbacks
            'format': 'bestaudio/best',  # Simple and reliable

            # === PHASE 1 OPTIMIZATIONS: Concurrent downloads & HTTP tuning ===
            'concurrent_fragment_downloads': 8,  # Download 8 fragments simultaneously (default: 1)
            'http_chunk_size': 10485760,         # 10MB chunks (faster than default small chunks)
            'retries': 10,                       # Retry failed downloads
            'fragment_retries': 10,              # Retry failed fragments
            'socket_timeout': 30,                # Increase timeout for stability

            # === PHASE 2: aria2c external downloader (TEMPORARILY DISABLED FOR TESTING) ===
            # 'external_downloader': 'aria2c',
            # 'external_downloader_args': {
            #     'aria2c': [
            #         '--max-connection-per-server=16',
            #         '--min-split-size=1M',
            #         '--split=16',
            #         '--max-concurrent-downloads=16',
            #         '--file-allocation=none',
            #         '--continue=true',
            #         '--max-overall-download-limit=0',
            #         '--disk-cache=64M',
            #     ]
            # },

            # YouTube-specific optimizations (TEMPORARILY DISABLED FOR TESTING)
            # 'extractor_args': {
            #     'youtube': {
            #         'player_client': ['android', 'web'],
            #     }
            # },

            # Audio processing
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],

            # Output settings
            'outtmpl': output_path.replace('.mp3', ''),
            'quiet': False,
            'no_warnings': False,

            # Headers - mobile user agent may help avoid throttling
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
            },

            # IPv4 preference (sometimes faster than IPv6)
            'source_address': '0.0.0.0',
        }

        # Add cookies to base options (REQUIRED)
        if not cookie_file_path:
            raise RuntimeError(
                "Cannot download: No cookies provided. "
                "YouTube requires authentication cookies to bypass bot detection. "
                "Export cookies from your browser and add them as a Modal secret."
            )

        base_opts['cookiefile'] = cookie_file_path

        # Add progress hook if tracker is provided
        if progress_tracker:
            base_opts['progress_hooks'] = [progress_tracker.update]

        # === Download with Basic Strategy (Only one that works) ===
        print("Attempting download with basic strategy + cookies...")

        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                ydl.download([youtube_url])
            print("✓ SUCCESS! YouTube audio downloaded")

        except Exception as e:
            # Comprehensive error capture
            error_type = type(e).__name__
            error_msg = str(e).strip()

            if not error_msg:
                error_msg = repr(e) if repr(e) != 'Exception()' else f"{error_type} (no message)"

            print(f"✗ Download failed: {error_msg[:300]}")

            # Provide actionable advice
            error_details = [f"Failed to download YouTube audio: {error_msg}"]

            if 'bot' in error_msg.lower() or 'sign in' in error_msg.lower():
                error_details.append("\n  SOLUTION: Refresh your cookies!")
                error_details.append("  1. Sign in to YouTube in your browser")
                error_details.append("  2. Export fresh cookies (use browser extension)")
                error_details.append("  3. Run: python run-service/fix_secret.py")
            elif '403' in error_msg or 'Forbidden' in error_msg:
                error_details.append("\n  ISSUE: 403 Forbidden - authentication or geo-restriction")
                error_details.append("  Try refreshing your cookies")
            elif '429' in error_msg:
                error_details.append("\n  ISSUE: Rate limited - too many requests")
                error_details.append("  Wait a few minutes before trying again")
            elif 'format' in error_msg.lower():
                error_details.append("\n  ISSUE: Format not available")
                error_details.append("  The video may be region-locked or live stream ended")

            raise RuntimeError('\n'.join(error_details))

        finally:
            # Cleanup: Remove cookie file if it was created
            if cookie_file_path and os.path.exists(cookie_file_path):
                try:
                    os.remove(cookie_file_path)
                    print(f"✓ Cleaned up cookie file: {cookie_file_path}")
                except Exception as e:
                    print(f"⚠ Failed to cleanup cookie file: {e}")

    @modal.fastapi_endpoint(method="POST")
    async def separate(self, item: dict):
        """
        Process audio file and return separated language tracks using Server-Sent Events.

        Expects JSON with:
        - youtube_url: YouTube video URL to download and process

        Returns SSE stream with:
        - progress events: Real-time progress updates (0-100%)
        - complete event: Final result with base64-encoded MP3 tracks
        - error events: Error messages if processing fails
        """
        from fastapi.responses import StreamingResponse
        import torch
        import torchaudio
        import numpy as np
        import time
        import json
        import traceback

        async def event_generator():
            """Generate Server-Sent Events for progress updates and final result."""
            import asyncio

            def send_event(event_type: str, data: dict) -> str:
                """Format data as SSE event."""
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            try:
                # Validate input
                youtube_url = item.get("youtube_url")
                if not youtube_url:
                    yield send_event("error", {"message": "No youtube_url provided"})
                    return

                with tempfile.TemporaryDirectory() as tmpdir:
                    # 1. Download YouTube audio with real-time progress
                    yield send_event("progress", {
                        "stage": "download",
                        "message": "Starting download...",
                        "progress": 0
                    })
                    await asyncio.sleep(0)  # Flush event

                    input_path = os.path.join(tmpdir, "input.mp3")
                    try:
                        # Create progress tracker
                        progress_tracker = DownloadProgress()

                        # Run blocking download in thread pool to prevent blocking event loop
                        loop = asyncio.get_event_loop()
                        download_future = loop.run_in_executor(
                            None,
                            self._download_youtube_audio,
                            youtube_url,
                            input_path,
                            progress_tracker
                        )

                        # Poll progress while downloading
                        print("[EventGenerator] Starting progress polling loop...")
                        last_percent = 0
                        poll_count = 0
                        while not download_future.done():
                            await asyncio.sleep(0.5)  # Check every 500ms
                            poll_count += 1

                            # Thread-safe read of progress
                            current_percent, speed = progress_tracker.get_progress()
                            if poll_count % 10 == 0:  # Every 5 seconds
                                print(f"[EventGenerator] Poll #{poll_count}: progress={current_percent:.1f}%, task_done={download_future.done()}")
                            if current_percent > last_percent:
                                last_percent = current_percent
                                # Map 0-100% download progress to 0-25% overall progress
                                overall_progress = int(current_percent * 0.25)

                                # Format speed if available
                                speed_str = ""
                                if speed:
                                    speed_mb = speed / (1024 * 1024)
                                    speed_str = f" at {speed_mb:.1f}MB/s"

                                print(f"[EventGenerator] Yielding progress: {current_percent:.1f}% (overall: {overall_progress}%)")
                                yield send_event("progress", {
                                    "stage": "download",
                                    "message": f"Downloading: {current_percent:.0f}%{speed_str}",
                                    "progress": overall_progress
                                })

                        # Wait for completion (in case it finished between checks)
                        await download_future

                        yield send_event("progress", {
                            "stage": "download",
                            "message": "Download complete!",
                            "progress": 25
                        })
                        await asyncio.sleep(0)  # Flush event
                    except RuntimeError as e:
                        yield send_event("error", {"message": str(e)})
                        return

                    print("Starting optimized audio processing...")
                    total_start = time.time()

                    # 2. Load and preprocess
                    yield send_event("progress", {
                        "stage": "preprocess",
                        "message": "Loading and preprocessing audio...",
                        "progress": 30
                    })
                    await asyncio.sleep(0)  # Flush event

                    preprocess_start = time.time()
                    waveform, sample_rate = await loop.run_in_executor(
                        None,
                        self._load_and_preprocess_audio,
                        input_path
                    )
                    print(f"Preprocessing completed in {time.time() - preprocess_start:.1f}s")

                    duration_minutes = waveform.shape[1] / sample_rate / 60
                    yield send_event("progress", {
                        "stage": "preprocess",
                        "message": f"Audio loaded: {duration_minutes:.1f} minutes",
                        "progress": 40
                    })
                    await asyncio.sleep(0)  # Flush event

                    # 3. Run diarization (longest step)
                    yield send_event("progress", {
                        "stage": "diarization",
                        "message": "Running AI speaker separation (this takes longest)...",
                        "progress": 45
                    })
                    await asyncio.sleep(0)  # Flush event

                    diar_start = time.time()
                    diar_segments = await loop.run_in_executor(
                        None,
                        self._run_diarization_optimized,
                        waveform,
                        sample_rate
                    )
                    print(f"Diarization completed in {time.time() - diar_start:.1f}s")

                    num_speakers = len({spk for _, _, spk in diar_segments})
                    print(f"Found {len(diar_segments)} segments, {num_speakers} speakers")

                    yield send_event("progress", {
                        "stage": "diarization",
                        "message": f"Found {num_speakers} speakers in {len(diar_segments)} segments",
                        "progress": 75
                    })
                    await asyncio.sleep(0)  # Flush event

                    if not diar_segments:
                        yield send_event("error", {"message": "No segments from diarization"})
                        return

                    # 4. Build tracks
                    yield send_event("progress", {
                        "stage": "build",
                        "message": "Building separate language tracks...",
                        "progress": 80
                    })
                    await asyncio.sleep(0)  # Flush event

                    speaker_track = self._assign_speakers_to_tracks(diar_segments)

                    build_start = time.time()
                    y = waveform.squeeze(0).cpu().numpy()
                    track_1, track_2 = await loop.run_in_executor(
                        None,
                        self._build_tracks_from_diarization,
                        y,
                        sample_rate,
                        diar_segments,
                        speaker_track
                    )
                    print(f"Track building completed in {time.time() - build_start:.1f}s")

                    # 5. Export to MP3
                    yield send_event("progress", {
                        "stage": "export",
                        "message": "Exporting MP3 files...",
                        "progress": 90
                    })
                    await asyncio.sleep(0)  # Flush event

                    export_start = time.time()
                    output1_path = os.path.join(tmpdir, "language1.mp3")
                    output2_path = os.path.join(tmpdir, "language2.mp3")
                    await loop.run_in_executor(
                        None,
                        self._export_tracks,
                        track_1,
                        track_2,
                        sample_rate,
                        output1_path,
                        output2_path
                    )
                    print(f"Export completed in {time.time() - export_start:.1f}s")

                    # 6. Read and encode output files
                    yield send_event("progress", {
                        "stage": "encode",
                        "message": "Encoding results...",
                        "progress": 95
                    })
                    await asyncio.sleep(0)  # Flush event

                    with open(output1_path, "rb") as f:
                        lang1_bytes = f.read()
                    with open(output2_path, "rb") as f:
                        lang2_bytes = f.read()

                    total_time = time.time() - total_start
                    print(f"Total processing completed in {total_time:.1f}s")

                    # 7. Send final result
                    yield send_event("complete", {
                        "language1": base64.b64encode(lang1_bytes).decode("utf-8"),
                        "language2": base64.b64encode(lang2_bytes).decode("utf-8"),
                        "progress": 100
                    })

            except Exception as e:
                print(f"Error in event_generator: {e}")
                print(traceback.format_exc())
                yield send_event("error", {
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })

        # Return streaming response with SSE
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

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
    """Test the audio separator locally with a YouTube URL."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_app.py -- <youtube_url>")
        print("Example: modal run modal_app.py -- https://www.youtube.com/watch?v=VIDEO_ID")
        return

    youtube_url = sys.argv[1]

    separator = AudioSeparator()
    result = separator.separate.remote({"youtube_url": youtube_url})

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save output files
    with open("language1.mp3", "wb") as f:
        f.write(base64.b64decode(result["language1"]))
    with open("language2.mp3", "wb") as f:
        f.write(base64.b64decode(result["language2"]))

    print("Saved language1.mp3 and language2.mp3")

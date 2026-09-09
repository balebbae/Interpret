"""
Modal serverless GPU application for bilingual audio separation.

Takes an uploaded MP3 and splits it into one track per language:

1. pyannote speaker diarization (16 kHz mono copy) finds *when* someone speaks and
   where the turns change.
2. Whisper's language-identification head labels every turn with *which language*
   is spoken, restricted to the two languages requested (or auto-detected).
3. The turns of each language are cut from the native-rate audio and encoded to MP3.

Routing by language rather than by speaker means any number of preachers,
interpreters or announcers is handled correctly. Progress is streamed over SSE.
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

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# Ratio of the segmentation window (10 s) to slide between inferences (pyannote default 0.1).
SEGMENTATION_STEP = 0.1
# min_duration_off fills short intra-speaker pauses before clustering.
PIPELINE_OVERRIDES = {"segmentation": {"min_duration_off": 0.1}}

# Spoken-language identification. Whisper "small" restricted to two languages agrees
# with large-v3 on 99.8 % of speech time while being ~8x cheaper.
LID_MODEL = "small"
LID_MAX_UNIT_SEC = 20.0   # split long turns so every unit fits Whisper's 30 s window
LID_MIN_UNIT_SEC = 0.4    # shorter turns carry no usable phonetic evidence
LID_MIN_CONF = 0.8        # below this, fall back to the voice's local majority language
LID_CONTEXT_SEC = 120.0   # window (each side) for that local majority vote
LID_BATCH = 32

# Timeline clean-up applied to the per-language turns
MIN_SEGMENT_SEC = 0.25   # drop shorter segments (back-channels, glitches)
MERGE_GAP_SEC = 0.5      # join same-language segments separated by less than this
SEGMENT_PAD_SEC = 0.15   # extend each segment so word edges are not clipped
FADE_SEC = 0.015         # ramp applied at every cut to avoid clicks


def download_models():
    """Pre-download models during image build."""
    import whisper
    from pyannote.audio import Pipeline

    Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=os.environ.get("HUGGING_FACE_TOKEN"))
    whisper.load_model(LID_MODEL, device="cpu", download_root="/root/.cache/whisper")
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
        "openai-whisper==20231117",
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


def make_units(segments, max_unit: float = LID_MAX_UNIT_SEC, min_unit: float = LID_MIN_UNIT_SEC):
    """
    Turn raw diarization turns into language-ID units: drop turns too short to
    identify, split long ones into <= max_unit pieces. Returns [(start, end, speaker)].
    """
    units = []
    for start, end, speaker in sorted(segments):
        if end - start < min_unit:
            continue
        n = max(1, int(-(-(end - start) // max_unit)))  # ceil
        step = (end - start) / n
        for i in range(n):
            units.append((round(start + i * step, 3), round(start + (i + 1) * step, 3), speaker))
    return units


def pick_languages(units, probs, requested):
    """
    Decide the two output languages. `probs[i]` maps language code -> Whisper probability
    for units[i]. If `requested` has two codes they win; otherwise the two languages with
    the most (probability-weighted) speech time are used.
    """
    if requested and len(requested) == 2:
        return list(requested)
    weight = defaultdict(float)
    for (start, end, _), p in zip(units, probs):
        for lang, prob in p.items():
            weight[lang] += (end - start) * prob
    ranked = sorted(weight, key=weight.get, reverse=True)
    if requested:
        ranked = list(requested) + [lang for lang in ranked if lang not in requested]
    if len(ranked) < 2:
        raise RuntimeError("Could not detect two languages in the audio")
    return ranked[:2]


def assign_languages(
    units,
    probs,
    languages,
    min_conf: float = LID_MIN_CONF,
    context_sec: float = LID_CONTEXT_SEC,
):
    """
    Label each unit with one of `languages`.

    The label is the arg-max of the Whisper probabilities restricted to the two
    languages. When that decision is not confident (very short fragments, an
    interpreter echoing a name...), the unit takes the language its voice cluster
    speaks most around that point in time, since a given voice rarely switches.
    Returns [(start, end, lang, confidence)].
    """
    restricted = []
    for p in probs:
        scores = {lang: p.get(lang, 0.0) for lang in languages}
        z = sum(scores.values())
        if z <= 0:
            restricted.append({lang: 1.0 / len(languages) for lang in languages})
        else:
            restricted.append({lang: v / z for lang, v in scores.items()})

    confident = []
    for (start, end, speaker), r in zip(units, restricted):
        lang = max(r, key=r.get)
        if r[lang] >= min_conf:
            confident.append((start, end, speaker, lang))

    def local_majority(start, end, speaker):
        votes = defaultdict(float)
        for s, e, spk, lang in confident:
            if spk == speaker and s < end + context_sec and e > start - context_sec:
                votes[lang] += e - s
        if not votes:  # fall back to the whole recording, any voice
            for s, e, _, lang in confident:
                votes[lang] += e - s
        return max(votes, key=votes.get) if votes else None

    labelled = []
    for (start, end, speaker), r in zip(units, restricted):
        lang = max(r, key=r.get)
        conf = r[lang]
        if conf < min_conf:
            lang = local_majority(start, end, speaker) or lang
        labelled.append((start, end, lang, round(conf, 3)))
    return labelled


def clean_segments(
    segments,
    total_duration: float,
    min_segment: float = MIN_SEGMENT_SEC,
    merge_gap: float = MERGE_GAP_SEC,
    pad: float = SEGMENT_PAD_SEC,
):
    """
    Tidy a labelled timeline: drop micro-segments, pad every segment a little,
    and merge same-label segments that are separated by a short pause so the
    natural rhythm survives concatenation.

    segments: iterable of (start, end, label). Returns the same shape, sorted by start.
    """
    by_label = defaultdict(list)
    for start, end, label in segments:
        if end - start >= min_segment:
            by_label[label].append((start, end))

    cleaned = []
    for label, spans in by_label.items():
        spans.sort()
        merged = []
        for start, end in spans:
            start = max(0.0, start - pad)
            end = min(total_duration, end + pad)
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        cleaned.extend((s, e, label) for s, e in merged)

    cleaned.sort(key=lambda x: x[0])
    return cleaned


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


def validate_languages(requested):
    """Normalise the optional `languages` request field to a list of Whisper language codes."""
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    if requested is None:
        return []
    if not isinstance(requested, list) or len(requested) > 2:
        raise ValueError("languages must be a list of at most two language codes, e.g. [\"en\", \"zh\"]")
    codes = []
    for lang in requested:
        if not isinstance(lang, str):
            raise ValueError("languages must be strings")
        code = lang.strip().lower()
        code = TO_LANGUAGE_CODE.get(code, code)
        if code not in LANGUAGES:
            raise ValueError(f"Unknown language {lang!r}")
        if code not in codes:
            codes.append(code)
    return codes


def language_name(code: str) -> str:
    from whisper.tokenizer import LANGUAGES

    return LANGUAGES.get(code, code).title()


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
    """Pre-loads the diarization and language-ID models once per container and processes uploads on demand."""

    @modal.enter()
    def load_model(self):
        import torch
        import whisper
        from pyannote.audio import Pipeline

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            print("WARNING: No HUGGING_FACE_TOKEN found!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.time()
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
        pipeline.to(self.device)
        # Embedding extraction dominates GPU time; both stages default to 32.
        pipeline.segmentation_batch_size = 64
        pipeline.embedding_batch_size = 64
        pipeline._segmentation.step = SEGMENTATION_STEP * pipeline._segmentation.duration
        params = pipeline.parameters(instantiated=True)
        for section, values in PIPELINE_OVERRIDES.items():
            params[section].update(values)
        pipeline.instantiate(params)
        self.pipeline = pipeline
        print(f"Pipeline {DIARIZATION_MODEL} loaded on {self.device} in {time.time() - t0:.1f}s")

        t0 = time.time()
        self.lid = whisper.load_model(LID_MODEL, device=self.device, download_root="/root/.cache/whisper")
        print(f"Whisper {LID_MODEL} loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # Core (blocking) pipeline shared by the web endpoint and local runs
    # ------------------------------------------------------------------ #

    def _run_pipeline(self, input_path: str, workdir: str, progress, languages=None) -> dict:
        """
        Run the full separation on `input_path`. `progress(stage, message, percent)` is
        called from the worker thread. `languages` is an optional list of up to two
        language codes. Returns paths of the two tracks plus metadata and timings.
        """
        import numpy as np
        import torch

        requested = validate_languages(languages)
        timings = {}
        total_start = time.time()

        # 1. Decode a 16 kHz mono copy for diarization / language ID
        progress("preprocess", "Decoding audio...", 30)
        t0 = time.time()
        native_sr = probe_sample_rate(input_path)
        diar_audio = decode_audio(input_path, DIARIZATION_SR, "float32")
        peak = np.abs(diar_audio).max()
        if peak > 0:
            diar_audio = diar_audio / peak
        diar_audio = np.ascontiguousarray(diar_audio)
        # Upload once; pyannote slices windows straight off the GPU tensor.
        waveform = torch.from_numpy(diar_audio).unsqueeze(0).to(self.device)
        duration = waveform.shape[1] / DIARIZATION_SR
        timings["decode"] = time.time() - t0
        progress("preprocess", f"Audio loaded: {duration / 60:.1f} minutes ({native_sr} Hz)", 40)

        # 2. Diarize on the GPU while ffmpeg decodes the native-rate audio on the CPU
        progress("diarization", "Finding speech turns...", 45)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=1) as pool:
            native_future = pool.submit(decode_audio, input_path, native_sr, "int16")
            raw_segments = self._diarize(waveform)
            del waveform
            timings["diarization"] = time.time() - t0

            # 3. Language ID on every turn
            num_speakers = len({spk for _, _, spk in raw_segments})
            units = make_units(raw_segments)
            if not units:
                raise RuntimeError("No speech segments found in the audio")
            progress("language_id", f"Identifying the language of {len(units)} speech turns...", 60)
            t0 = time.time()
            probs = self._identify_languages(diar_audio, units)
            timings["language_id"] = time.time() - t0

            t0 = time.time()
            native_audio = native_future.result()
        timings["native_decode_wait"] = time.time() - t0

        # 4. Route every turn to one of the two languages
        picked = pick_languages(units, probs, requested)
        labelled = assign_languages(units, probs, picked)
        segments = clean_segments([(s, e, lang) for s, e, lang, _ in labelled], duration)
        track_of = {picked[0]: "lang1", picked[1]: "lang2"}
        uncertain = sum(e - s for s, e, _, conf in labelled if conf < LID_MIN_CONF)
        per_lang = defaultdict(float)
        for s, e, lang in segments:
            per_lang[lang] += e - s
        print(
            f"{len(raw_segments)} turns / {num_speakers} voices -> {len(units)} units -> "
            f"{len(segments)} segments; " + ", ".join(f"{l}={per_lang[l]:.0f}s" for l in picked)
            + f"; uncertain={uncertain:.1f}s"
        )
        progress(
            "language_id",
            f"{language_name(picked[0])}: {per_lang[picked[0]] / 60:.1f} min, "
            f"{language_name(picked[1])}: {per_lang[picked[1]] / 60:.1f} min",
            75,
        )

        # 5. Cut tracks from the native-rate audio
        progress("build", "Building separate language tracks...", 80)
        t0 = time.time()
        spans = {"lang1": [], "lang2": []}
        for start, end, lang in segments:
            spans[track_of[lang]].append((start, end))
        track_1 = build_track(native_audio, native_sr, spans["lang1"])
        track_2 = build_track(native_audio, native_sr, spans["lang2"])
        del native_audio
        timings["build"] = time.time() - t0

        # 6. Encode both MP3s in parallel
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
            "model": f"{DIARIZATION_MODEL} + whisper-{LID_MODEL} language ID",
            "duration_seconds": duration,
            "sample_rate": native_sr,
            "languages": {
                track_of[lang]: {
                    "code": lang,
                    "name": language_name(lang),
                    "seconds": round(per_lang[lang], 1),
                }
                for lang in picked
            },
            "languages_requested": requested,
            "num_speakers": num_speakers,
            "num_segments": len(segments),
            "uncertain_seconds": round(uncertain, 1),
            "segments": [(round(s, 3), round(e, 3), track_of[lang]) for s, e, lang in segments],
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }

    def _diarize(self, waveform):
        """
        Run pyannote on a (1, samples) 16 kHz tensor with no speaker-count constraint.
        FP32 on purpose: autocast corrupts the speaker embeddings. Returns [(start, end, speaker)].
        """
        import torch

        with torch.inference_mode():
            diarization = self.pipeline({"waveform": waveform, "sample_rate": DIARIZATION_SR})

        segments = [
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda x: x[0])
        return segments

    def _identify_languages(self, audio, units):
        """Whisper language probabilities for every unit. Returns a list of {code: prob} dicts."""
        import torch
        import whisper

        probs = []
        n_mels = self.lid.dims.n_mels
        for i in range(0, len(units), LID_BATCH):
            mels = []
            for start, end, _ in units[i:i + LID_BATCH]:
                clip = audio[int(start * DIARIZATION_SR):int(end * DIARIZATION_SR)]
                mels.append(whisper.log_mel_spectrogram(whisper.pad_or_trim(clip), n_mels=n_mels))
            mel = torch.stack(mels).to(self.device)
            if self.device.type == "cuda":
                mel = mel.half()  # Whisper casts its weights to the input dtype
            with torch.inference_mode():
                _, batch_probs = self.lid.detect_language(self.lid.encoder(mel))
            probs.extend(batch_probs)
        return probs

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    @modal.method()
    def separate_bytes(self, audio_bytes: bytes, languages=None) -> dict:
        """Synchronous variant used by `modal run` for local testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp3")
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            def log_progress(stage, message, percent):
                print(f"[{percent:3d}%] {stage}: {message}")

            result = self._run_pipeline(input_path, tmpdir, log_progress, languages)
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

        Expects JSON: {"audio_base64": "<base64 MP3>", "languages": ["en", "zh"] (optional)}
        `languages` holds 0-2 Whisper language codes; missing ones are auto-detected.

        Events:
        - progress: {stage, message, progress}
        - complete: {language1, language2, model, duration_seconds, languages: {lang1: {code, name,
                     seconds}, lang2: {...}}, languages_requested, num_speakers, num_segments,
                     uncertain_seconds, segments: [[start, end, "lang1"|"lang2"], ...], timings,
                     progress: 100}
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
                try:
                    languages = validate_languages(item.get("languages"))
                except ValueError as e:
                    yield send_event("error", {"message": str(e)})
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

                    future = loop.run_in_executor(
                        None, self._run_pipeline, input_path, tmpdir, progress, languages
                    )
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
                        "model": result["model"],
                        "duration_seconds": round(result["duration_seconds"], 1),
                        "languages": result["languages"],
                        "languages_requested": result["languages_requested"],
                        "num_speakers": result["num_speakers"],
                        "num_segments": result["num_segments"],
                        "uncertain_seconds": result["uncertain_seconds"],
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
def main(path: str, out_dir: str = ".", languages: str = ""):
    """Test the separator: modal run modal_app.py --path ./sermon.mp3 [--out-dir ./out] [--languages en,zh]"""
    import json

    with open(path, "rb") as f:
        audio_bytes = f.read()

    codes = [c for c in languages.split(",") if c.strip()] or None
    result = AudioSeparator().separate_bytes.remote(audio_bytes, codes)

    os.makedirs(out_dir, exist_ok=True)
    for track in ("lang1", "lang2"):
        info = result["languages"][track]
        name = f"{info['name'].lower()}.mp3"
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(result.pop("language1" if track == "lang1" else "language2"))
        print(f"{track}: {info['name']} ({info['code']}), {info['seconds'] / 60:.1f} min -> {name}")
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Audio: {result['duration_seconds'] / 60:.1f} min, {result['num_speakers']} voices, "
          f"{result['num_segments']} segments, {result['uncertain_seconds']}s uncertain ({result['model']})")
    print("Timings: " + ", ".join(f"{k}={v}s" for k, v in result["timings"].items()))
    print(f"Saved tracks and result.json to {out_dir}")

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpret is a bilingual audio separator web app that splits sermons with interpretation into two separate language tracks. Users upload an MP3; the app finds every speech turn (pyannote speaker diarization), labels each turn with its spoken language (Whisper language ID) and returns one downloadable MP3 per language.

## Common Commands

```bash
# Development
npm run dev      # Start Next.js dev server at localhost:3000
npm run build    # Build for production
npm run lint     # Run ESLint
npx tsc --noEmit # Type-check

# Modal GPU Service
cd run-service
python -m py_compile modal_app.py                                       # Syntax check
modal run modal_app.py --path ./sermon.mp3 --out-dir ./out --languages en,zh  # Test on a file
modal deploy modal_app.py                                               # Deploy (CI does this on push to main)
```

## Architecture

**Frontend (Next.js 16 + React 19):**
- Single-page app: MP3 drag/drop plus two language selectors (Auto-detect or a Whisper language code)
- Talks directly to the Modal web API (`NEXT_PUBLIC_MODAL_ENDPOINT`): chunked upload, then SSE for progress
- Real-time progress tracking (upload → queue → preprocess → diarization → language_id → build → export → publish)
- Downloads are plain links to `GET /download/{job_id}/{lang1|lang2}`; nothing is base64-encoded anywhere

**Backend Processing (Modal Serverless GPU, `run-service/modal_app.py`):**
- CPU web function `api` (FastAPI, `web_image`): upload/separate/download routes, shares the `audio-separator-jobs` Volume mounted at `/jobs`
- Stateful Modal class `AudioSeparator` on an L4 GPU; pyannote 3.1 and Whisper `small` are pre-loaded once per container; `separate_job(job_id, languages)` is a generator invoked with `remote_gen.aio`
- Routes audio by **language**, not by speaker; any number of voices is fine
- Jobs live at `/jobs/<32-hex uuid>/` (`input.mp3` → `language1.mp3`, `language2.mp3`, `result.json`) and are purged after 24 h

**Data Flow:**
1. Browser `POST /upload` → `{job_id, chunk_bytes}`; `PUT /upload/{job_id}/{index}` raw 8 MB chunks (3 in flight); `POST /upload/{job_id}/complete {chunks}` assembles them (200 MB max)
2. Browser `POST /separate {job_id, languages?: ["en", "zh"]}` (0–2 codes; missing ones are auto-detected); the API streams the GPU generator's events as SSE
3. ffmpeg decodes a 16 kHz mono copy (models) and, in parallel, a native-rate int16 copy (output)
4. pyannote/speaker-diarization-3.1 in **FP32** with no speaker-count constraint → speech turns
5. Turns become ≤ 20 s units; Whisper's language head scores each unit, restricted to the two languages; low-confidence units take the language their voice speaks most nearby
6. Same-language segments are padded/merged, cut from the native-rate audio with 15 ms fades, encoded to MP3 in parallel
7. Tracks are written into the job directory; the `complete` SSE event carries `downloads: {lang1, lang2}` (paths relative to the API base), `languages: {lang1: {code, name, seconds}, lang2}`, `num_speakers`, `uncertain_seconds`, `segments`, `timings`
8. `GET /download/{job_id}/{lang1|lang2}` serves `audio/mpeg` with `Content-Disposition: attachment; filename="<language>.mp3"`

## Key Technical Details

- **Do not enable autocast / FP16 for pyannote**: it corrupts the speaker embeddings (collapses to one speaker). Whisper runs in FP16.
- `speechbrain==1.0.3` is pinned: 1.1+ dropped `use_auth_token`, which pyannote 3.1.1 still passes.
- Pure helper functions (`make_units`, `pick_languages`, `assign_languages`, `clean_segments`, `build_track`, `decode_audio`, `encode_mp3`, `job_path`, `assemble_chunks`, `purge_old_jobs`) have no Modal dependency and can be tested locally with NumPy + ffmpeg.
- `job_path` only accepts 32-hex ids, so every filesystem path derived from a request stays under `/jobs`.
- The web image has no Whisper/torch: `check_languages_shape` runs on the CPU side, full `validate_languages` runs on the GPU.
- Use the async Modal Volume calls (`JOBS_VOLUME.commit.aio()` / `reload.aio()`) inside FastAPI handlers; never hold a file open across an `await` (a concurrent reload would fail).
- Inside `separate_job` the pipeline runs in a worker thread and progress events are relayed through a queue so the generator keeps yielding while the GPU works.
- Deployment: `.github/workflows/modal-deploy.yml` runs `modal deploy` on push to `main` touching `run-service/**` (needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repo secrets).

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://<workspace>--audio-separator-api.modal.run  # Frontend (the `api` web function)
HUGGING_FACE_TOKEN=hf_xxx                                         # Modal secret "huggingface" (pyannote access)
```

## Important Files

- `app/page.tsx` - Main UI: MP3 upload, language selectors, SSE handling, progress display, downloads
- `run-service/modal_app.py` - GPU processing service with AudioSeparator class and SSE streaming
- `lib/types.ts` - TypeScript interfaces for requests/responses and the language option list
- `components/ui/simple-growth-tree.tsx` - Animated tree visualization (decorative)

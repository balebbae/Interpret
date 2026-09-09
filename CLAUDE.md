# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpret is a bilingual audio separator web app that splits sermons with interpretation into two separate language tracks using AI-powered speaker diarization. Users upload an MP3, and the app separates speakers and provides two downloadable MP3 files.

## Common Commands

```bash
# Development
npm run dev      # Start Next.js dev server at localhost:3000
npm run build    # Build for production
npm run lint     # Run ESLint

# Modal GPU Service
cd run-service
modal deploy modal_app.py                                   # Deploy to Modal
modal serve modal_app.py                                    # Ephemeral dev endpoint (hot reload)
modal run modal_app.py --path ./sermon.mp3 --out-dir ./out  # Run on a local MP3, writes language{1,2}.mp3 + result.json
```

## Architecture

**Frontend (Next.js 16 + React 19):**
- Single-page app with an MP3 drop zone (`react-dropzone`, 200 MB limit)
- Direct communication with Modal GPU endpoint via Server-Sent Events (SSE)
- Real-time progress tracking (upload → decode → diarization → build → export)
- Results are base64-decoded client-side for download

**Backend Processing (Modal Serverless GPU):**
- `run-service/modal_app.py` - Stateful Modal class with pyannote.audio pipeline
- Uses L4 GPU with pre-loaded `pyannote/speaker-diarization` (the v2.1 pipeline; library is pyannote.audio 3.1.1)
- Forces exactly 2 speakers for bilingual separation
- Assigns speakers to tracks based on total speaking duration (longer = track 1)

**Data Flow:**
1. User drops an MP3 → browser reads it as base64
2. POST request sent to Modal endpoint with `{audio_base64: string}`
3. ffmpeg decodes a 16 kHz mono float32 copy (peak-normalised) for the model
4. Speaker diarization with pyannote.audio (FP16 mixed precision, batch size 64); in parallel, ffmpeg decodes a native-sample-rate int16 copy for output
5. Timeline clean-up: drop <0.25 s segments, pad 0.15 s, merge same-speaker gaps <0.5 s
6. Build two tracks from the native-rate audio with 15 ms fades at every cut
7. Encode both tracks in parallel with ffmpeg/libmp3lame (128 kbps) and base64 encode
8. Stream progress updates via SSE throughout; `complete` event includes per-stage `timings`
9. Browser receives base64 MP3s and offers downloads

## Key Technical Details

### Real-Time Progress (Server-Sent Events)
- Frontend uses `fetch()` with `Accept: text/event-stream` header and parses events incrementally
- Modal streams SSE events: `progress`, `complete`, `error`
- Progress stages: upload (5-25%), preprocess (30-40%), diarization (45-75%), build (80%), export (90%), encode (95-100%)
- The blocking pipeline runs in a worker thread (`_run_pipeline`) and reports progress through an `asyncio.Queue`

### Speaker Diarization Pipeline
- Pyannote models pre-downloaded during Modal image build (cached at `/root/.cache/huggingface`)
- Diarization runs with `num_speakers=2` constraint for bilingual audio
- Uses FP16 automatic mixed precision (`torch.amp.autocast`) on GPU
- Batch sizes set via `pipeline.segmentation_batch_size` / `pipeline.embedding_batch_size` (the public knobs in pyannote 3.1.1)
- Audio passed as in-memory tensor (no disk I/O during inference)
- `speechbrain` is pinned to 1.0.3: 1.1+ removed `use_auth_token`, which pyannote 3.1.1 still passes, so the image build fails otherwise

### Audio helpers
- `probe_sample_rate`, `decode_audio`, `encode_mp3`, `clean_segments`, `assign_speakers_to_tracks`, `build_track` are plain functions at module level so they can be unit-tested without Modal or a GPU
- Tunables live at the top of `modal_app.py`: `MIN_SEGMENT_SEC`, `MERGE_GAP_SEC`, `SEGMENT_PAD_SEC`, `FADE_SEC`, `MP3_BITRATE`

### Performance Notes
- 30-minute timeout for very long audio files
- 8GB RAM allocation for processing
- 6-minute scaledown window keeps container warm
- Output tracks are cut from the native-rate audio, not the 16 kHz model input, so output bandwidth matches the source

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-endpoint.modal.run  # Frontend
HUGGING_FACE_TOKEN=hf_xxx                                         # Modal secret `huggingface` (pyannote access)
```

## Important Files

- `app/page.tsx` - Main UI with MP3 drop zone, SSE handling, progress display, download logic
- `run-service/modal_app.py` - GPU processing service with AudioSeparator class and SSE streaming
- `lib/types.ts` - TypeScript interfaces for requests/responses
- `components/ui/simple-growth-tree.tsx` - Animated tree visualization (decorative)

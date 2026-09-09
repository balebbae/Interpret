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
- Direct communication with Modal GPU endpoint via Server-Sent Events (SSE)
- Real-time progress tracking (upload → preprocess → diarization → language_id → build → export)
- Results are base64-decoded client-side for download; tracks are named by language

**Backend Processing (Modal Serverless GPU, `run-service/modal_app.py`):**
- Stateful Modal class `AudioSeparator` on an L4 GPU; pyannote 3.1 and Whisper `small` are pre-loaded once per container
- Routes audio by **language**, not by speaker; any number of voices is fine

**Data Flow:**
1. Browser POSTs `{audio_base64, languages?: ["en", "zh"]}` (0–2 codes; missing ones are auto-detected)
2. ffmpeg decodes a 16 kHz mono copy (models) and, in parallel, a native-rate int16 copy (output)
3. pyannote/speaker-diarization-3.1 in **FP32** with no speaker-count constraint → speech turns
4. Turns become ≤ 20 s units; Whisper's language head scores each unit, restricted to the two languages; low-confidence units take the language their voice speaks most nearby
5. Same-language segments are padded/merged, cut from the native-rate audio with 15 ms fades, encoded to MP3 in parallel
6. `complete` SSE event carries both base64 MP3s, `languages: {lang1: {code, name, seconds}, lang2}`, `num_speakers`, `uncertain_seconds`, `segments`, `timings`

## Key Technical Details

- **Do not enable autocast / FP16 for pyannote**: it corrupts the speaker embeddings (collapses to one speaker). Whisper runs in FP16.
- `speechbrain==1.0.3` is pinned: 1.1+ dropped `use_auth_token`, which pyannote 3.1.1 still passes.
- Pure helper functions (`make_units`, `pick_languages`, `assign_languages`, `clean_segments`, `build_track`, `decode_audio`, `encode_mp3`) have no Modal dependency and can be tested locally with NumPy + ffmpeg.
- All blocking work runs in a thread pool via `loop.run_in_executor()` so the SSE stream keeps flowing.
- Deployment: `.github/workflows/modal-deploy.yml` runs `modal deploy` on push to `main` touching `run-service/**` (needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repo secrets).

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-endpoint.modal.run  # Frontend
HUGGING_FACE_TOKEN=hf_xxx                                         # Modal secret "huggingface" (pyannote access)
```

## Important Files

- `app/page.tsx` - Main UI: MP3 upload, language selectors, SSE handling, progress display, downloads
- `run-service/modal_app.py` - GPU processing service with AudioSeparator class and SSE streaming
- `lib/types.ts` - TypeScript interfaces for requests/responses and the language option list
- `components/ui/simple-growth-tree.tsx` - Animated tree visualization (decorative)

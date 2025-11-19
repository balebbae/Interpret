# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpret is a bilingual audio separator web app that splits sermons with interpretation into two separate language tracks using AI-powered speaker diarization. Users provide a YouTube URL, and the app downloads the audio, separates speakers, and provides two downloadable MP3 files.

## Common Commands

```bash
# Development
npm run dev      # Start Next.js dev server at localhost:3000
npm run build    # Build for production
npm run lint     # Run ESLint

# Modal GPU Service
cd run-service
modal deploy modal_app.py                              # Deploy to Modal
modal run modal_app.py -- <youtube_url>                # Test locally with YouTube URL
modal secret create youtube-cookies YOUTUBE_COOKIES="$(cat cookies.txt)"  # Required for YouTube downloads
```

## Architecture

**Frontend (Next.js 16 + React 19):**
- Single-page app with YouTube URL input
- Direct communication with Modal GPU endpoint via Server-Sent Events (SSE)
- Real-time progress tracking (download → preprocessing → diarization → export)
- Results are base64-decoded client-side for download

**Backend Processing (Modal Serverless GPU):**
- `run-service/modal_app.py` - Stateful Modal class with pyannote.audio pipeline
- Uses L4 GPU with pre-loaded speaker diarization model (pyannote 3.1)
- Forces exactly 2 speakers for bilingual separation
- Assigns speakers to tracks based on total speaking duration (longer = track 1)

**Data Flow:**
1. User submits YouTube URL → validated client-side
2. POST request sent to Modal endpoint with `{youtube_url: string}`
3. Modal downloads audio via yt-dlp (requires YouTube cookies for authentication)
4. Audio preprocessing: convert to mono, resample to 16kHz, normalize
5. Speaker diarization with pyannote.audio (FP16 mixed precision, batch size 64)
6. Build two tracks by concatenating segments per speaker
7. Export to MP3 (128kbps) and base64 encode
8. Stream progress updates via SSE throughout entire process
9. Browser receives base64 MP3s and offers downloads

## Key Technical Details

### YouTube Download Requirements
- **Cookies Required**: YouTube blocks unauthenticated downloads. Must export browser cookies and create Modal secret `youtube-cookies`
- Uses `yt-dlp` with concurrent fragment downloads (8 parallel), 10MB chunks, mobile user agent
- Downloads audio-only format, converts to MP3 via FFmpeg
- See README.md for detailed cookie export instructions

### Real-Time Progress (Server-Sent Events)
- Frontend uses `fetch()` with `Accept: text/event-stream` header
- Modal streams SSE events: `progress`, `complete`, `error`
- Progress stages: download (0-25%), preprocess (30-40%), diarization (45-75%), build (80%), export (90-95%), encode (95-100%)
- `DownloadProgress` class provides thread-safe progress tracking during YouTube download

### Speaker Diarization Pipeline
- Pyannote models pre-downloaded during Modal image build (cached at `/root/.cache/huggingface`)
- Diarization runs with `num_speakers=2` constraint for bilingual audio
- Uses FP16 automatic mixed precision (`torch.cuda.amp.autocast`) for 2x speedup on GPU
- Batch sizes increased from 32 to 64 for segmentation and embedding
- Audio passed as in-memory tensor (no disk I/O during inference)

### Performance Optimizations
- 30-minute timeout for very long audio files
- 8GB RAM allocation for processing
- 3-minute scaledown window keeps container warm
- All blocking operations run in thread pool via `loop.run_in_executor()` to prevent blocking SSE stream

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-endpoint.modal.run  # Frontend
HUGGING_FACE_TOKEN=hf_xxx                                         # Modal secret (pyannote access)
YOUTUBE_COOKIES=<netscape_cookie_format>                          # Modal secret (YouTube auth)
```

## Important Files

- `app/page.tsx` - Main UI with YouTube URL input, SSE handling, progress display, download logic
- `run-service/modal_app.py` - GPU processing service with AudioSeparator class and SSE streaming
- `lib/types.ts` - TypeScript interfaces for requests/responses
- `components/ui/simple-growth-tree.tsx` - Animated tree visualization (decorative)

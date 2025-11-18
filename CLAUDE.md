# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpret is a bilingual audio separator web app that splits sermons with interpretation into two separate language tracks using AI-powered speaker diarization.

## Common Commands

```bash
# Development
npm run dev      # Start Next.js dev server at localhost:3000
npm run build    # Build for production
npm run lint     # Run ESLint

# Modal GPU Service
cd run-service
modal deploy modal_app.py           # Deploy to Modal
modal run modal_app.py -- <file>    # Test locally with an MP3 file
```

## Architecture

**Frontend (Next.js 16 + React 19):**
- Single-page app with client-side file handling
- Direct communication with Modal GPU endpoint (bypasses Vercel size limits)
- Files are base64 encoded client-side and sent directly to Modal
- Results are base64 decoded client-side for download

**Backend Processing (Modal Serverless GPU):**
- `run-service/modal_app.py` - Stateful Modal class with pyannote.audio pipeline
- Uses L4 GPU with pre-loaded speaker diarization model
- Forces exactly 2 speakers for bilingual separation
- Assigns speakers to tracks based on total speaking duration (longer = track 1)

**Data Flow:**
1. User uploads MP3 → converted to base64 in browser
2. Base64 sent directly to Modal endpoint (NEXT_PUBLIC_MODAL_ENDPOINT)
3. Modal processes with pyannote speaker diarization
4. Two base64-encoded MP3s returned
5. Browser decodes and offers downloads

## Key Technical Details

- No API routes in Next.js - all processing happens client-side + Modal
- Modal container pre-downloads pyannote models during image build for fast cold starts
- Uses FP16 mixed precision and increased batch sizes (64) for faster GPU processing
- 30-minute timeout for long audio files, 8GB RAM allocation
- HuggingFace token required for pyannote model access (stored as Modal secret)

## Environment Variables

```
NEXT_PUBLIC_MODAL_ENDPOINT=<modal_web_endpoint_url>
HUGGING_FACE_TOKEN=hf_xxx  # Required for Modal deployment
```

## Important Files

- `app/page.tsx` - Main UI with file upload, processing logic, and download handling
- `run-service/modal_app.py` - GPU processing service (AudioSeparator class)
- `lib/types.ts` - TypeScript interfaces for API responses
- `components/ui/file-upload.tsx` - Drag-and-drop upload component (uses react-dropzone)
- `components/ui/simple-growth-tree.tsx` - Decorative animated tree visualization

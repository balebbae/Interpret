# Interpret - Bilingual Audio Separator

A web application for splitting bilingual sermons into two clean language-only audio tracks.

## Overview

Interpret allows users to upload an MP3 file containing bilingual audio (e.g., sermons with interpretation) and automatically separates it into two clean audio tracks - one for each language. It uses AI-powered speaker diarization via pyannote.audio to identify and separate speakers.

## How It Works

### High-Level Flow

1. **Input**: User drops an MP3 file (browser converts it to base64)
2. **Process**: Request sent directly to Modal GPU endpoint as `{audio_base64}`
3. **Diarize**: pyannote.audio identifies 2 speakers via diarization
4. **Separate**: Audio segments grouped by speaker (longer speaker = Track 1)
5. **Return**: Two base64-encoded MP3s plus stage timings returned to browser
6. **Download**: Browser decodes and offers file downloads

### Audio Processing Pipeline (Modal GPU)

The core separation happens in `run-service/modal_app.py`:

1. **Decode** (ffmpeg)
   - One 16 kHz mono float32 copy, peak-normalised, for the diarization model
   - One mono int16 copy at the file's **native sample rate** for the output tracks (decoded in parallel with diarization)

2. **Speaker Diarization** (pyannote.audio)
   - Neural network identifies "who spoke when"
   - Forces exactly 2 speaker clusters (`num_speakers=2`)
   - Outputs timestamped segments: `[(0.5s, 3.2s, SPEAKER_00), (3.2s, 8.1s, SPEAKER_01), ...]`
   - FP16 mixed precision, segmentation and embedding batch size 64

3. **Timeline Clean-up**
   - Drop segments shorter than 0.25 s (back-channels, glitches)
   - Pad every segment by 0.15 s so word edges are not clipped
   - Merge same-speaker segments separated by less than 0.5 s

4. **Speaker-to-Track Assignment**
   - Calculate total speaking duration per speaker
   - Speaker with **more total time** becomes Track 1
   - Assumes both languages have roughly equal content

5. **Track Building**
   - Slice the native-rate audio for each segment and apply a 15 ms fade at every cut
   - Concatenate all segments per speaker into continuous tracks

6. **MP3 Export**
   - Both tracks encoded in parallel with ffmpeg/libmp3lame at 128 kbps, at the native sample rate

**Important**: The pipeline separates by **voice identity**, not by language detection. It assumes the two speakers are speaking different languages (e.g., original speaker + interpreter).

## Architecture

- **Frontend**: Next.js 16 with React 19, Tailwind CSS v4
- **GPU Processing**: Modal serverless GPU (L4) with pyannote.audio speaker diarization
- **Communication**: Direct client-to-Modal API

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- [Modal](https://modal.com) account (for GPU processing)
- HuggingFace account with access to pyannote models

### Local Development

1. **Install frontend dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env.local
   ```

   Fill in your Modal endpoint URL after deployment.

3. **Deploy Modal service**:
   ```bash
   cd run-service
   modal secret create huggingface HUGGING_FACE_TOKEN=hf_your_token
   modal deploy modal_app.py
   ```

   Copy the web endpoint URL to your `.env.local`.

   **Automatic deploys:** `.github/workflows/modal-deploy.yml` runs `modal deploy` whenever a
   push to `main` touches `run-service/` (or manually via *Actions → Deploy Modal service → Run
   workflow*). Add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as repository Actions secrets
   (create a token at https://modal.com/settings/tokens).

   To test the service without the frontend:
   ```bash
   modal run modal_app.py --path ./sermon.mp3 --out-dir ./out
   ```

4. **Run development server**:
   ```bash
   npm run dev
   ```

5. **Open browser**:
   Visit [http://localhost:3000](http://localhost:3000)

## Project Structure

```
interpret/
├── app/                          # Next.js app directory
│   ├── page.tsx                  # Main page: MP3 drop zone, progress, downloads
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles (Tailwind v4)
├── components/                   # React components
│   └── ui/
│       ├── input.tsx             # Input component
│       └── simple-growth-tree.tsx # Animated tree visualization
├── lib/                          # Utility functions
│   ├── types.ts                  # TypeScript interfaces
│   └── utils.ts                  # General utilities (cn helper)
├── run-service/                  # Modal GPU service
│   ├── modal_app.py              # AudioSeparator class with pyannote pipeline
│   └── requirements.txt          # Python dependencies
└── .env.local                    # Local environment variables
```
## Technology Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS v4** - Utility-first CSS
- **Framer Motion** - Animation library
- **React Dropzone** - File upload handling
- **TypeScript** - Type safety

### GPU Processing Service
- **Modal** - Serverless GPU platform
- **Python 3.10** - Programming language
- **pyannote.audio 3.1** - Speaker diarization
- **PyTorch + CUDA** - GPU acceleration
- **ffmpeg** - Audio decoding and MP3 export

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-endpoint.modal.run
HUGGING_FACE_TOKEN=hf_your_token  # For Modal secret
```

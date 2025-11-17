# Interpret - Bilingual Audio Separator

A web application for splitting bilingual sermons into two clean language-only audio tracks.

## Overview

Interpret allows users to upload MP3 files containing bilingual audio (e.g., sermons with interpretation) and automatically separates them into two clean audio tracks - one for each language. It uses AI-powered speaker diarization via pyannote.audio to identify and separate speakers.

## How It Works

### High-Level Flow

1. **Upload**: User drops MP3 file, browser converts to base64
2. **Process**: Base64 sent directly to Modal GPU endpoint
3. **Diarize**: pyannote.audio identifies 2 speakers via diarization
4. **Separate**: Audio segments grouped by speaker (longer speaker = Track 1)
5. **Return**: Two base64-encoded MP3s returned to browser
6. **Download**: Browser decodes and offers file downloads

### Audio Processing Pipeline (Modal GPU)

The core separation happens in `run-service/modal_app.py`:

1. **Audio Preprocessing**
   - Load MP3 with torchaudio
   - Convert stereo to mono (average channels)
   - Resample to 16kHz (required by pyannote)
   - Peak normalize to [-1, 1] range

2. **Speaker Diarization** (pyannote.audio)
   - Neural network identifies "who spoke when"
   - Forces exactly 2 speaker clusters (`num_speakers=2`)
   - Outputs timestamped segments: `[(0.5s, 3.2s, SPEAKER_00), (3.2s, 8.1s, SPEAKER_01), ...]`
   - Uses FP16 mixed precision and batch size 64 for GPU optimization

3. **Speaker-to-Track Assignment**
   - Calculate total speaking duration per speaker
   - Speaker with **more total time** becomes Track 1
   - Assumes both languages have roughly equal content

4. **Track Building**
   - Iterate through diarization segments chronologically
   - Slice audio array for each segment: `audio[start:end]`
   - Concatenate all segments per speaker into continuous tracks

5. **MP3 Export**
   - Convert float32 samples to int16
   - Export via pydub at 128kbps

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
│   ├── page.tsx                  # Main page with upload/download logic
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles (Tailwind v4)
├── components/                   # React components
│   └── ui/
│       ├── file-upload.tsx       # Drag-and-drop upload (react-dropzone)
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
- **torchaudio** - Audio loading/preprocessing
- **pydub** - MP3 export

## Environment Variables

```bash
NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-endpoint.modal.run
HUGGING_FACE_TOKEN=hf_your_token  # For Modal secret
```

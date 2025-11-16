# Interpret - Bilingual Audio Separator

A web application for splitting bilingual sermons into two clean language-only audio tracks.

## Overview

Interpret allows users to upload MP3 files containing bilingual audio (e.g., sermons with interpretation) and automatically separates them into two clean audio tracks - one for each language.

## Architecture

- **Frontend**: Next.js 16 with React 19, Tailwind CSS v4
- **Backend API**: Next.js API Routes
- **Processing Service**: Python FastAPI on Google Cloud Run (GPU-enabled)
- **Storage**: Google Cloud Storage
- **Job Tracking**: In-memory store (can be upgraded to Firestore)

## Features

- Drag-and-drop MP3 file upload
- Real-time processing status updates
- Progress tracking with visual progress bar
- Automatic download of separated audio tracks
- Beautiful animated tree visualization
- Responsive design with dark mode support

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Google Cloud account (for deployment)
- Python 3.10+ (for processing service)

### Local Development

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env.local` and fill in your values:
   ```bash
   cp .env.example .env.local
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   Visit [http://localhost:3000](http://localhost:3000)

## Project Structure

```
interpret/
├── app/                          # Next.js app directory
│   ├── api/                      # API routes
│   │   ├── upload/               # File upload endpoint
│   │   ├── status/[jobId]/       # Job status endpoint
│   │   └── download/[jobId]/     # Download endpoint
│   ├── page.tsx                  # Main page
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles
├── components/                   # React components
│   └── ui/
│       ├── file-upload.tsx       # File upload component
│       ├── input.tsx             # Input component
│       └── simple-growth-tree.tsx # Animated tree
├── lib/                          # Utility functions
│   ├── storage.ts                # Google Cloud Storage helpers
│   ├── job-store.ts              # Job state management
│   ├── types.ts                  # TypeScript types
│   └── utils.ts                  # General utilities
├── cloud-run-service/            # Processing service
│   ├── main.py                   # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Container configuration
├── .env.local                    # Local environment variables
└── DEPLOYMENT.md                 # Deployment instructions
```

## API Endpoints

### Frontend API Routes

- `POST /api/upload` - Upload MP3 file and start processing
- `GET /api/status/[jobId]` - Get job processing status
- `GET /api/download/[jobId]` - Get signed URLs for downloads

### Processing Service API

- `POST /process` - Start audio separation job
- `GET /status/{job_id}` - Check job status
- `GET /health` - Health check

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for comprehensive deployment instructions covering:

1. Google Cloud setup (Cloud Run, Storage, Service Accounts)
2. Adapting your Colab notebook
3. Deploying the processing service
4. Configuring the frontend
5. Production deployment options

## Technology Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS v4** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **React Dropzone** - File upload handling

### Backend
- **Next.js API Routes** - Serverless functions
- **Google Cloud Storage** - File storage
- **TypeScript** - Type safety

### Processing Service
- **Python 3.10+** - Programming language
- **FastAPI** - Web framework
- **Google Cloud Run** - Serverless container platform
- **CUDA/PyTorch** - GPU acceleration (optional)

## Development Notes

### Current Limitations

1. **Job Storage**: Uses in-memory storage (resets on server restart)
   - For production, migrate to Firestore or PostgreSQL

2. **Authentication**: No auth implemented
   - Add API keys or OAuth for production

3. **File Size Limits**: Default limits apply
   - Adjust for larger files if needed

4. **Rate Limiting**: Not implemented
   - Add rate limiting for production

### Future Enhancements

- [ ] Persistent job storage with Firestore
- [ ] User authentication and accounts
- [ ] Job history and management
- [ ] Email notifications when processing completes
- [ ] Webhook support instead of polling
- [ ] Support for other audio formats
- [ ] Batch processing multiple files
- [ ] Advanced audio separation parameters
- [ ] YouTube URL support (download and process)

## Support

For deployment help, see [DEPLOYMENT.md](./DEPLOYMENT.md)

For issues and questions, please open a GitHub issue.

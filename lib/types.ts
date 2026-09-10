// Whisper language codes the UI offers; the backend accepts any Whisper code.
export const LANGUAGE_OPTIONS = [
  { code: "en", name: "English" },
  { code: "zh", name: "Chinese" },
  { code: "ko", name: "Korean" },
  { code: "ja", name: "Japanese" },
  { code: "es", name: "Spanish" },
  { code: "pt", name: "Portuguese" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "ru", name: "Russian" },
  { code: "vi", name: "Vietnamese" },
  { code: "tl", name: "Tagalog" },
  { code: "id", name: "Indonesian" },
  { code: "hi", name: "Hindi" },
  { code: "ar", name: "Arabic" },
  { code: "sw", name: "Swahili" },
] as const;

export type LanguageCode = (typeof LANGUAGE_OPTIONS)[number]["code"];

// POST /upload -> new job to receive chunks
export interface UploadStart {
  job_id: string;
  chunk_bytes: number;
  max_bytes: number;
}

// POST /separate: run the separation on an uploaded job
export interface SeparationRequest {
  job_id: string;
  languages?: string[]; // 0-2 language codes; missing ones are auto-detected
}

export interface TrackLanguage {
  code: string;
  name: string;
  seconds: number; // speech routed to this track
}

// `complete` SSE event from POST /separate
export interface SeparationResult {
  job_id: string;
  downloads: { lang1: string; lang2: string }; // paths relative to the API base URL
  model: string;
  duration_seconds: number;
  languages: { lang1: TrackLanguage; lang2: TrackLanguage };
  languages_requested: string[];
  num_speakers: number; // distinct voices found by diarization
  num_segments: number;
  uncertain_seconds: number; // speech whose language the model was not confident about
  segments: [start: number, end: number, track: "lang1" | "lang2"][];
  timings: { total: number } & Record<string, number>; // seconds per stage
}

// Error response
export interface ErrorResponse {
  error: string;
}

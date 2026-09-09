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

// Request to Modal audio separation endpoint
export interface SeparationRequest {
  audio_base64: string; // Base64 encoded MP3 upload
  languages?: string[]; // 0-2 language codes; missing ones are auto-detected
}

export interface TrackLanguage {
  code: string;
  name: string;
  seconds: number; // speech routed to this track
}

// `complete` SSE event from the Modal audio separation endpoint
export interface SeparationResult {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
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

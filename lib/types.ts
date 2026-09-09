// Request to Modal audio separation endpoint
export interface SeparationRequest {
  audio_base64: string; // Base64 encoded MP3 upload
}

// `complete` SSE event from the Modal audio separation endpoint
export interface SeparationResult {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
  duration_seconds: number;
  num_speakers: number;
  num_segments: number;
  segments: [start: number, end: number, track: "lang1" | "lang2"][];
  timings: { total: number } & Record<string, number>; // seconds per stage
}

// Error response
export interface ErrorResponse {
  error: string;
}

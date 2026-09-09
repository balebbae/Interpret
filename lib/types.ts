// Request to Modal audio separation endpoint (exactly one source)
export type SeparationRequest =
  | { youtube_url: string; audio_base64?: never } // YouTube video URL
  | { audio_base64: string; youtube_url?: never }; // Base64 encoded MP3 upload

// Response from Modal audio separation endpoint
export interface SeparationResult {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
}

// Error response
export interface ErrorResponse {
  error: string;
}

// Request to Modal audio separation endpoint
export interface SeparationRequest {
  youtube_url: string; // YouTube video URL
}

// Response from Modal audio separation endpoint
export interface SeparationResult {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
}

// Error response
export interface ErrorResponse {
  error: string;
}

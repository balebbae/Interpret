// Response from Modal audio separation endpoint
export interface SeparationResult {
  success: boolean;
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
  message: string;
}

// Error response
export interface ErrorResponse {
  error: string;
}

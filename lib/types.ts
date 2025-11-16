export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface ProcessingJob {
  jobId: string;
  status: JobStatus;
  inputFile: string;
  outputFiles?: {
    language1: string;
    language2: string;
  };
  createdAt: string;
  updatedAt: string;
  error?: string;
  progress?: number;
}

export interface UploadResponse {
  jobId: string;
  status: JobStatus;
  message: string;
}

export interface StatusResponse {
  job: ProcessingJob;
}

export interface DownloadResponse {
  urls: {
    language1: string;
    language2: string;
  };
}

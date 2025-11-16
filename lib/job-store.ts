import { ProcessingJob, JobStatus } from './types';

// In-memory job store (replace with Firestore/database in production)
const jobs = new Map<string, ProcessingJob>();

export function createJob(jobId: string, inputFile: string): ProcessingJob {
  const job: ProcessingJob = {
    jobId,
    status: 'pending',
    inputFile,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  jobs.set(jobId, job);
  return job;
}

export function getJob(jobId: string): ProcessingJob | undefined {
  return jobs.get(jobId);
}

export function updateJobStatus(
  jobId: string,
  status: JobStatus,
  data?: Partial<ProcessingJob>
): ProcessingJob | undefined {
  const job = jobs.get(jobId);
  if (!job) return undefined;

  const updatedJob: ProcessingJob = {
    ...job,
    ...data,
    status,
    updatedAt: new Date().toISOString(),
  };
  jobs.set(jobId, updatedJob);
  return updatedJob;
}

export function deleteJob(jobId: string): boolean {
  return jobs.delete(jobId);
}

export function getAllJobs(): ProcessingJob[] {
  return Array.from(jobs.values());
}

// NOTE: This is an in-memory store and will be cleared on server restart.
// For production, replace with Firestore or a persistent database:
//
// import { Firestore } from '@google-cloud/firestore';
// const db = new Firestore();
// const collection = db.collection('processing-jobs');
//
// export async function createJob(...) {
//   const doc = await collection.add(job);
//   return { ...job, jobId: doc.id };
// }

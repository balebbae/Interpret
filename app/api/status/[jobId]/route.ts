import { NextRequest, NextResponse } from 'next/server';
import { getJob } from '@/lib/job-store';
import { StatusResponse } from '@/lib/types';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;

    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }

    const job = getJob(jobId);

    if (!job) {
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 }
      );
    }

    // Optionally: Poll Cloud Run service for latest status
    const cloudRunEndpoint = process.env.CLOUD_RUN_ENDPOINT;
    if (cloudRunEndpoint && (job.status === 'pending' || job.status === 'processing')) {
      try {
        const response = await fetch(`${cloudRunEndpoint}/status/${jobId}`);
        if (response.ok) {
          const cloudRunStatus = await response.json();
          // Update local job store with Cloud Run status
          // (In production, this would be done via webhook or Pub/Sub)
          if (cloudRunStatus.status) {
            job.status = cloudRunStatus.status;
            job.progress = cloudRunStatus.progress;
            if (cloudRunStatus.outputFiles) {
              job.outputFiles = cloudRunStatus.outputFiles;
            }
            if (cloudRunStatus.error) {
              job.error = cloudRunStatus.error;
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch status from Cloud Run:', error);
        // Continue with local status
      }
    }

    const response: StatusResponse = {
      job,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Status check error:', error);
    return NextResponse.json(
      { error: 'Failed to check job status' },
      { status: 500 }
    );
  }
}

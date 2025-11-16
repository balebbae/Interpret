import { NextRequest, NextResponse } from 'next/server';
import { getJob } from '@/lib/job-store';
import { getSignedUrl } from '@/lib/storage';
import { DownloadResponse } from '@/lib/types';

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

    if (job.status !== 'completed') {
      return NextResponse.json(
        { error: 'Job is not completed yet', status: job.status },
        { status: 400 }
      );
    }

    if (!job.outputFiles) {
      return NextResponse.json(
        { error: 'Output files not available' },
        { status: 404 }
      );
    }

    const bucketName = process.env.GOOGLE_CLOUD_BUCKET_OUTPUT;
    if (!bucketName) {
      return NextResponse.json(
        { error: 'Server configuration error: output bucket not configured' },
        { status: 500 }
      );
    }

    // Generate signed URLs for both output files
    const [language1Url, language2Url] = await Promise.all([
      getSignedUrl(bucketName, job.outputFiles.language1, 60),
      getSignedUrl(bucketName, job.outputFiles.language2, 60),
    ]);

    const response: DownloadResponse = {
      urls: {
        language1: language1Url,
        language2: language2Url,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Download error:', error);
    return NextResponse.json(
      { error: 'Failed to generate download URLs' },
      { status: 500 }
    );
  }
}

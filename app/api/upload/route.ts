import { NextRequest, NextResponse } from 'next/server';
import { uploadToStorage } from '@/lib/storage';
import { createJob } from '@/lib/job-store';
import { UploadResponse } from '@/lib/types';
import { randomUUID } from 'crypto';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.includes('audio/mpeg') && !file.name.endsWith('.mp3')) {
      return NextResponse.json(
        { error: 'Invalid file type. Only MP3 files are allowed.' },
        { status: 400 }
      );
    }

    // Generate unique job ID
    const jobId = randomUUID();
    const fileName = `${jobId}-${file.name}`;

    // Convert File to Buffer
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Upload to Google Cloud Storage
    const bucketName = process.env.GOOGLE_CLOUD_BUCKET_INPUT;
    if (!bucketName) {
      return NextResponse.json(
        { error: 'Server configuration error: bucket not configured' },
        { status: 500 }
      );
    }

    const gcsPath = await uploadToStorage(
      buffer,
      fileName,
      bucketName,
      file.type
    );

    // Create job in store
    const job = createJob(jobId, fileName);

    // Submit job to Cloud Run processing service (non-blocking with timeout)
    const cloudRunEndpoint = process.env.CLOUD_RUN_ENDPOINT;
    if (cloudRunEndpoint) {
      // Fire and forget - don't wait for Cloud Run response
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      fetch(`${cloudRunEndpoint}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId,
          inputFile: fileName,
          inputBucket: bucketName,
          outputBucket: process.env.GOOGLE_CLOUD_BUCKET_OUTPUT,
        }),
        signal: controller.signal,
      })
        .then(() => clearTimeout(timeoutId))
        .catch((error) => {
          clearTimeout(timeoutId);
          console.error('Failed to submit job to Cloud Run:', error);
        });
      // Don't await - return immediately to user
    }

    const response: UploadResponse = {
      jobId,
      status: job.status,
      message: 'File uploaded successfully. Processing started.',
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Failed to upload file' },
      { status: 500 }
    );
  }
}

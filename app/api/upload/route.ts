import { NextRequest, NextResponse } from 'next/server';

export interface SeparationResponse {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
}

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

    // Convert File to base64
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const audioBase64 = buffer.toString('base64');

    // Get Modal endpoint
    const modalEndpoint = process.env.MODAL_ENDPOINT;
    if (!modalEndpoint) {
      return NextResponse.json(
        { error: 'Server configuration error: Modal endpoint not configured' },
        { status: 500 }
      );
    }

    // Send to Modal for processing (synchronous - wait for result)
    console.log('Sending audio to Modal for processing...');
    const modalResponse = await fetch(modalEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        audio_base64: audioBase64,
      }),
    });

    if (!modalResponse.ok) {
      const errorData = await modalResponse.json().catch(() => ({}));
      console.error('Modal processing failed:', errorData);
      return NextResponse.json(
        { error: errorData.error || 'Audio processing failed' },
        { status: 500 }
      );
    }

    const result: SeparationResponse = await modalResponse.json();
    console.log('Modal processing complete');

    // Return the separated audio files
    return NextResponse.json({
      success: true,
      language1: result.language1,
      language2: result.language2,
      message: 'Audio separation completed successfully',
    });
  } catch (error) {
    console.error('Upload/processing error:', error);
    return NextResponse.json(
      { error: 'Failed to process audio file' },
      { status: 500 }
    );
  }
}

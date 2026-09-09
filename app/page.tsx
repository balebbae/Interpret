"use client";

import Image from "next/image";
import { SimpleTree } from "@/components/ui/simple-growth-tree";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Upload, X } from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import type { SeparationRequest, SeparationResult } from "@/lib/types";
import { cn } from "@/lib/utils";

const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

const formatFileSize = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

// Reads via data URL so large files are encoded natively instead of byte-by-byte in JS
const fileToBase64 = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      resolve(dataUrl.slice(dataUrl.indexOf(',') + 1));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });

const base64ToBlob = (base64: string, type: string) =>
  new Blob([Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))], { type });

const formatDuration = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m} min ${s} s` : `${s} s`;
};

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SeparationResult | null>(null);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [audioFile, setAudioFile] = useState<File | null>(null);

  const onDrop = useCallback((accepted: File[], rejections: FileRejection[]) => {
    if (rejections.length > 0) {
      const code = rejections[0].errors[0]?.code;
      setError(
        code === 'file-too-large'
          ? `File is too large. Maximum size is ${formatFileSize(MAX_UPLOAD_BYTES)}.`
          : 'Only MP3 files are supported.'
      );
      return;
    }
    if (accepted[0]) {
      setAudioFile(accepted[0]);
      setError(null);
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: { 'audio/mpeg': ['.mp3'] },
    maxSize: MAX_UPLOAD_BYTES,
    disabled: isProcessing,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!audioFile) {
      setError('Please select an MP3 file');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setProcessingStatus("Initializing...");

    try {
      setProcessingStatus("Reading audio file...");
      const requestBody: SeparationRequest = { audio_base64: await fileToBase64(audioFile) };
      setProcessingStatus(`Uploading ${formatFileSize(audioFile.size)}...`);

      const modalEndpoint = process.env.NEXT_PUBLIC_MODAL_ENDPOINT;
      if (!modalEndpoint) {
        throw new Error('Modal endpoint not configured');
      }

      // Use fetch with streaming for Server-Sent Events
      const response = await fetch(modalEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to start processing');
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      // Read the SSE stream. The final `complete` event carries both MP3s and can be
      // hundreds of MB, so only scan newly received bytes for the event delimiter.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let scanFrom = 0;
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let delimiter: number;
        while ((delimiter = buffer.indexOf('\n\n', scanFrom)) !== -1) {
          const message = buffer.slice(0, delimiter);
          buffer = buffer.slice(delimiter + 2);
          scanFrom = 0;

          const dataStart = message.indexOf('\ndata: ');
          if (!message.startsWith('event: ') || dataStart === -1) continue;

          const eventType = message.slice(7, dataStart);
          const data = JSON.parse(message.slice(dataStart + 7));

          if (eventType === 'progress') {
            setProgress(data.progress);
            setProcessingStatus(data.message);
          } else if (eventType === 'complete') {
            setResult(data as SeparationResult);
            setProgress(100);
            setProcessingStatus("Processing complete!");
            finished = true;
            break;
          } else if (eventType === 'error') {
            throw new Error(data.message);
          }
        }
        scanFrom = Math.max(0, buffer.length - 1);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process audio');
      setProcessingStatus("");
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = (language: 'language1' | 'language2', trackName: string) => {
    if (!result) return;

    try {
      const blob = base64ToBlob(result[language], 'audio/mpeg');

      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${trackName}.mp3`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download file:', err);
      setError('Failed to download file');
    }
  };

  return (
    <div
      className="flex flex-col h-screen font-sans overflow-hidden"
      style={{ backgroundColor: 'rgb(248, 245, 236)' }}
    >
      {/* Header */}
      <header className="flex-shrink-0 pt-6 pb-10 px-6">
        <div className="flex items-center gap-2">
          <Image
            src="/logo.png"
            alt="Interpret Logo"
            width={42}
            height={42}
            className="flex-shrink-0"
          />
          <h1 className="text-2xl font-bold text-neutral-700 dark:text-neutral-300">
            Interpret
          </h1>
        </div>
        <p className="text-sm text-neutral-500 dark:text-neutral-900 mt-2">
          Split bilingual sermons into two clean one language audio tracks
        </p>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col px-6 py-4 overflow-hidden">
        {/* Input Form */}
        <div className="flex-shrink-0">
          <form onSubmit={handleSubmit} className="max-w-xl mx-auto">
            <div className="flex gap-2">
              <div
                {...getRootProps({
                  className: cn(
                    "flex-1 min-w-0 flex items-center gap-2 h-9 px-3 rounded-md border border-dashed text-sm transition-colors",
                    isProcessing ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:border-neutral-400",
                    isDragActive ? "border-blue-500 bg-blue-50" : "border-input bg-transparent"
                  ),
                })}
              >
                <input {...getInputProps()} />
                <Upload className="h-4 w-4 flex-shrink-0 text-neutral-500" />
                {audioFile ? (
                  <>
                    <span className="truncate text-neutral-700">{audioFile.name}</span>
                    <span className="flex-shrink-0 text-xs text-neutral-500">
                      {formatFileSize(audioFile.size)}
                    </span>
                    {!isProcessing && (
                      <button
                        type="button"
                        aria-label="Remove file"
                        className="ml-auto flex-shrink-0 cursor-pointer text-neutral-400 hover:text-neutral-700"
                        onClick={(e) => {
                          e.stopPropagation();
                          setAudioFile(null);
                        }}
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </>
                ) : (
                  <span className="truncate text-neutral-500">
                    {isDragActive ? 'Drop MP3 here...' : 'Drop an MP3 here or click to browse'}
                  </span>
                )}
              </div>
              <Button
                type="submit"
                className="cursor-pointer rounded-em"
                disabled={isProcessing || !audioFile}
              >
                {isProcessing ? 'Processing...' : 'Separate'}
              </Button>
            </div>
            <p className="text-xs text-neutral-500 mt-2">
              MP3 files up to {formatFileSize(MAX_UPLOAD_BYTES)}
            </p>
          </form>
        </div>

        {/* Processing Status */}
        {(isProcessing || processingStatus) && (
          <div className="flex-shrink-0 mt-4 max-w-xl mx-auto w-full px-6">
            {/* Progress Bar */}
            {isProcessing && (
              <div className="mb-3">
                <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-xs text-neutral-500">
                  <span>{progress}%</span>
                  <span>{progress < 25 ? 'Uploading' : progress < 45 ? 'Loading' : progress < 80 ? 'Processing' : progress < 95 ? 'Building' : 'Finalizing'}</span>
                </div>
              </div>
            )}
            {/* Status Message */}
            <p className="text-sm text-neutral-600 text-center">{processingStatus}</p>
            {isProcessing && (
              <div className="mt-3 text-center">
                <Loader2 className="inline-block h-6 w-6 animate-spin text-neutral-700" />
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="flex-shrink-0 mt-4 max-w-xl mx-auto">
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        )}

        {/* Download Results */}
        {result && (
          <div className="flex-shrink-0 text-center mt-4 max-w-xl mx-auto">
            <Card className="rounded-md shadow-none">
              <CardContent >
                <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-1">
                  Audio separation complete! Download your tracks:
                </p>
                <p className="text-xs text-neutral-500 mb-4">
                  {formatDuration(result.duration_seconds)} of audio processed in {formatDuration(result.timings.total)}
                </p>
                <div className="flex gap-2 justify-center">
                  <Button
                    onClick={() => handleDownload('language1', 'language1_track')}
                    variant="default"
                    className="cursor-pointer rounded-sm"
                  >
                    Download Track 1
                  </Button>
                  <Button
                    onClick={() => handleDownload('language2', 'language2_track')}
                    variant="default"
                    className="cursor-pointer rounded-sm"
                  >
                    Download Track 2
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
        {/* Tree Component */}
        <div className="flex flex-1 min-h-0 relative">
          <SimpleTree />
        </div>

        {/* Bible Verse */}
        <div className="flex-shrink-0 text-center mb-8 pt-2">
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto leading-relaxed">
            <span className="font-semibold">20</span> But know this first of all, that no prophecy of Scripture becomes a matter of someone&apos;s own interpretation, <span className="font-semibold">21</span> for no prophecy was ever made by an act of human will, but men moved by the Holy Spirit spoke from God.
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-500 mt-1">
            2 Peter 1:20-21
          </p>
        </div>

        
      </main>
    </div>
  );
}

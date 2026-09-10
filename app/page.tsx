"use client";

import Image from "next/image";
import { SimpleTree } from "@/components/ui/simple-growth-tree";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Upload, X } from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import {
  LANGUAGE_OPTIONS,
  type SeparationRequest,
  type SeparationResult,
  type UploadStart,
} from "@/lib/types";
import { cn } from "@/lib/utils";

const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;
const UPLOAD_PARALLELISM = 3;
const UPLOAD_RETRIES = 3;
const AUTO = "auto";

const formatFileSize = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

const apiBase = () => {
  const base = process.env.NEXT_PUBLIC_MODAL_ENDPOINT;
  if (!base) throw new Error('Modal endpoint not configured');
  return base.replace(/\/+$/, '');
};

const apiError = async (response: Response, fallback: string) => {
  try {
    const body = await response.json();
    return new Error(body.detail ?? body.message ?? fallback);
  } catch {
    return new Error(fallback);
  }
};

// Upload the file to the API in fixed-size chunks (a few in flight, each retried) and
// return the job id the server assembled it under.
const uploadFile = async (file: File, onProgress: (sentBytes: number) => void) => {
  const base = apiBase();
  const startRes = await fetch(`${base}/upload`, { method: 'POST' });
  if (!startRes.ok) throw await apiError(startRes, 'Failed to start upload');
  const { job_id, chunk_bytes }: UploadStart = await startRes.json();

  const chunkCount = Math.max(1, Math.ceil(file.size / chunk_bytes));
  let sent = 0;
  let next = 0;

  const putChunk = async (index: number) => {
    const blob = file.slice(index * chunk_bytes, (index + 1) * chunk_bytes);
    for (let attempt = 1; ; attempt++) {
      try {
        const res = await fetch(`${base}/upload/${job_id}/${index}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: blob,
        });
        if (!res.ok) throw await apiError(res, `Upload failed (chunk ${index + 1})`);
        break;
      } catch (err) {
        if (attempt >= UPLOAD_RETRIES) throw err;
      }
    }
    sent += blob.size;
    onProgress(sent);
  };

  const worker = async () => {
    while (next < chunkCount) await putChunk(next++);
  };
  await Promise.all(Array.from({ length: Math.min(UPLOAD_PARALLELISM, chunkCount) }, worker));

  const doneRes = await fetch(`${base}/upload/${job_id}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chunks: chunkCount }),
  });
  if (!doneRes.ok) throw await apiError(doneRes, 'Failed to finish upload');
  return job_id;
};

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
  const [language1, setLanguage1] = useState<string>("en");
  const [language2, setLanguage2] = useState<string>(AUTO);

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
      // Upload occupies the first 20% of the progress bar
      setProcessingStatus(`Uploading ${formatFileSize(audioFile.size)}...`);
      const jobId = await uploadFile(audioFile, (sent) => {
        setProgress(Math.round((sent / audioFile.size) * 20));
        setProcessingStatus(`Uploading ${formatFileSize(sent)} of ${formatFileSize(audioFile.size)}...`);
      });

      const requestBody: SeparationRequest = {
        job_id: jobId,
        languages: [language1, language2].filter((code) => code !== AUTO),
      };
      const response = await fetch(`${apiBase()}/separate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw await apiError(response, 'Failed to start processing');
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let delimiter: number;
        while ((delimiter = buffer.indexOf('\n\n')) !== -1) {
          const message = buffer.slice(0, delimiter);
          buffer = buffer.slice(delimiter + 2);

          const dataStart = message.indexOf('\ndata: ');
          if (!message.startsWith('event: ') || dataStart === -1) continue;

          const eventType = message.slice(7, dataStart);
          const data = JSON.parse(message.slice(dataStart + 7));

          if (eventType === 'progress') {
            setProgress(20 + Math.round(data.progress * 0.8));
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
      }

      if (!finished) {
        throw new Error('Connection closed before processing finished');
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process audio');
      setProcessingStatus("");
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  // The API serves the track as an attachment named after its language, so the
  // browser downloads it directly without the file passing through JS memory.
  const handleDownload = (track: 'lang1' | 'lang2') => {
    if (!result) return;
    const a = document.createElement('a');
    a.href = `${apiBase()}${result.downloads[track]}`;
    a.download = `${result.languages[track].name.toLowerCase()}.mp3`;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
                disabled={isProcessing || !audioFile || (language1 !== AUTO && language1 === language2)}
              >
                {isProcessing ? 'Processing...' : 'Separate'}
              </Button>
            </div>
            <div className="flex items-center gap-2 mt-2 text-xs text-neutral-500">
              <span>Languages:</span>
              {([
                [language1, setLanguage1, 'Track 1 language'],
                [language2, setLanguage2, 'Track 2 language'],
              ] as const).map(([value, setValue, label]) => (
                <select
                  key={label}
                  aria-label={label}
                  value={value}
                  disabled={isProcessing}
                  onChange={(e) => setValue(e.target.value)}
                  className="h-7 rounded-md border border-input bg-transparent px-2 text-xs text-neutral-700 disabled:opacity-50"
                >
                  <option value={AUTO}>Auto-detect</option>
                  {LANGUAGE_OPTIONS.map((lang) => (
                    <option key={lang.code} value={lang.code}>{lang.name}</option>
                  ))}
                </select>
              ))}
              <span className="ml-auto">MP3 files up to {formatFileSize(MAX_UPLOAD_BYTES)}</span>
            </div>
            {language1 !== AUTO && language1 === language2 && (
              <p className="text-xs text-red-600 mt-1">Pick two different languages (or Auto-detect).</p>
            )}
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
                <p className="text-xs text-neutral-500 mb-1">
                  {formatDuration(result.duration_seconds)} of audio processed in {formatDuration(result.timings.total)}
                  {' · '}{result.num_speakers} {result.num_speakers === 1 ? 'voice' : 'voices'}
                </p>
                {result.uncertain_seconds > 0 && (
                  <p className="text-xs text-neutral-500 mb-1">
                    {formatDuration(result.uncertain_seconds)} of short fragments were routed by voice rather than by language
                  </p>
                )}
                <div className="flex gap-2 justify-center mt-4">
                  {(['lang1', 'lang2'] as const).map((track) => (
                    <Button
                      key={track}
                      onClick={() => handleDownload(track)}
                      variant="default"
                      className="cursor-pointer rounded-sm"
                    >
                      Download {result.languages[track].name} ({formatDuration(result.languages[track].seconds)})
                    </Button>
                  ))}
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

"use client";

import Image from "next/image";
import { SimpleTree } from "@/components/ui/simple-growth-tree";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";
import { useState } from "react";

interface ProcessingResult {
  language1: string; // Base64 encoded MP3
  language2: string; // Base64 encoded MP3
}

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const [progress, setProgress] = useState<number>(0);
  const [youtubeUrl, setYoutubeUrl] = useState<string>("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!youtubeUrl.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setProcessingStatus("Initializing...");

    try {
      // Validate YouTube URL format
      const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|live\/)|youtu\.be\/)[\w-]+/;
      if (!youtubeRegex.test(youtubeUrl)) {
        throw new Error('Invalid YouTube URL. Please enter a valid YouTube link.');
      }

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
        body: JSON.stringify({ youtube_url: youtubeUrl }),
      });

      if (!response.ok) {
        throw new Error('Failed to start processing');
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      // Read the SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');

        // Keep incomplete message in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          // Parse SSE format: "event: type\ndata: {...}"
          // Use a character class instead of the dotAll (/s) flag for broader TS target compatibility
          const eventMatch = line.match(/event: (\w+)\ndata: ([\s\S]+)/);
          if (!eventMatch) continue;

          const [, eventType, dataStr] = eventMatch;
          const data = JSON.parse(dataStr);

          if (eventType === 'progress') {
            setProgress(data.progress);
            setProcessingStatus(data.message);
          } else if (eventType === 'complete') {
            setResult({
              language1: data.language1,
              language2: data.language2,
            });
            setProgress(100);
            setProcessingStatus("Processing complete!");
            setIsProcessing(false);
            break;
          } else if (eventType === 'error') {
            throw new Error(data.message);
          }
        }
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process video');
      setProcessingStatus("");
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = (language: 'language1' | 'language2', trackName: string) => {
    if (!result) return;

    try {
      // Convert base64 to blob
      const base64Data = language === 'language1' ? result.language1 : result.language2;
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'audio/mpeg' });

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
      <main className="flex-1 flex flex-col p-6 overflow-hidden">
        {/* YouTube URL Input Form */}
        <div className="flex-shrink-0">
          <form onSubmit={handleSubmit} className="max-w-xl mx-auto">
            <div className="flex gap-2">
              <Input
                type="url"
                placeholder="Paste YouTube link here..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                disabled={isProcessing}
                className="flex-1"
              />
              <Button
                type="submit"
                className="cursor-pointer"
                disabled={isProcessing || !youtubeUrl.trim()}
              >
                {isProcessing ? 'Processing...' : 'Separate'}
              </Button>
            </div>
            <p className="text-xs text-neutral-500 mt-2">
              Supports youtube.com and youtu.be URLs
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
                  <span>{progress < 25 ? 'Downloading' : progress < 45 ? 'Loading' : progress < 80 ? 'Processing' : progress < 95 ? 'Building' : 'Finalizing'}</span>
                </div>
              </div>
            )}
            {/* Status Message */}
            <p className="text-sm text-neutral-600 text-center">{processingStatus}</p>
            {isProcessing && (
              <div className="mt-2 text-center">
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
            <Card>
              <CardContent className="">
                <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-4">
                  Audio separation complete! Download your tracks:
                </p>
                <div className="flex gap-2 justify-center">
                  <Button
                    onClick={() => handleDownload('language1', 'language1_track')}
                    variant="default"
                  >
                    Download Track 1
                  </Button>
                  <Button
                    onClick={() => handleDownload('language2', 'language2_track')}
                    variant="default"
                  >
                    Download Track 2
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
        {/* Tree Component */}
        <div className="hidden md:flex flex-1 min-h-0 relative">
          <SimpleTree />
        </div>

        {/* Bible Verse */}
        <div className="flex-shrink-0 text-center p-15">
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-2xl mx-auto leading-relaxed">
            <span className="font-semibold">20</span> But know this first of all, that no prophecy of Scripture becomes a matter of someone's own interpretation, <span className="font-semibold">21</span> for no prophecy was ever made by an act of human will, but men moved by the Holy Spirit spoke from God.
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-500 mt-1">
            2 Peter 1:20-21
          </p>
        </div>

        
      </main>
    </div>
  );
}

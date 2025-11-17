"use client";

import { SimpleTree } from "@/components/ui/simple-growth-tree";
import { FileUpload } from "@/components/ui/file-upload";
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

  const handleFileUpload = async (files: File[]) => {
    if (files.length === 0) return;

    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProcessingStatus("Uploading and processing audio... This may take several minutes for long files.");

    try {
      const formData = new FormData();
      formData.append('file', files[0]);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Processing failed');
      }

      const data = await response.json();

      if (data.success) {
        setResult({
          language1: data.language1,
          language2: data.language2,
        });
        setProcessingStatus("Processing complete!");
      } else {
        throw new Error(data.error || 'Processing failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process file');
      setProcessingStatus("");
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
        <h1 className="text-2xl font-bold text-neutral-700 dark:text-neutral-300">
          Interpret
        </h1>
        <p className="text-sm text-neutral-500 dark:text-neutral-400 mt-1">
          Split bilingual sermons into two clean one language audio tracks
        </p>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col px-6 overflow-hidden">
        {/* YouTube Link Input */}
        {/* <div className="flex-shrink-0 mb-2">
          <Input
            type="url"
            placeholder="Insert YouTube link"
            value={youtubeLink}
            onChange={(e) => setYoutubeLink(e.target.value)}
            className="max-w-xl mx-auto"
          />
        </div> */}

    
        {/* File Upload */}
        <div className="flex-shrink-0">
          <div className="max-w-xl mx-auto">
            <FileUpload onChange={handleFileUpload} />
          </div>
        </div>

        {/* Processing Status */}
        {(isProcessing || processingStatus) && (
          <div className="flex-shrink-0 text-center mt-4">
            <p className="text-sm text-neutral-600">{processingStatus}</p>
            {isProcessing && (
              <div className="mt-2">
                <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-neutral-700"></div>
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="flex-shrink-0 text-center mt-4">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Download Results */}
        {result && (
          <div className="flex-shrink-0 text-center mt-4 max-w-xl mx-auto">
            <div className="bg-white dark:bg-neutral-900 rounded-lg p-4 shadow-sm">
              <p className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-4">
                Audio separation complete! Download your tracks:
              </p>
              <div className="flex gap-2 justify-center">
                <button
                  onClick={() => handleDownload('language1', 'language1_track')}
                  className="px-4 py-2 bg-neutral-700 text-white rounded-md text-sm hover:bg-neutral-800 transition-colors"
                >
                  Download Track 1
                </button>
                <button
                  onClick={() => handleDownload('language2', 'language2_track')}
                  className="px-4 py-2 bg-neutral-700 text-white rounded-md text-sm hover:bg-neutral-800 transition-colors"
                >
                  Download Track 2
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Bible Verse */}
        <div className="flex-shrink-0 text-center p-15">
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-2xl mx-auto leading-relaxed">
            <span className="font-semibold">20</span> But know this first of all, that no prophecy of Scripture becomes a matter of someone's own interpretation, <span className="font-semibold">21</span> for no prophecy was ever made by an act of human will, but men moved by the Holy Spirit spoke from God.
          </p>
          <p className="text-xs text-neutral-500 dark:text-neutral-500 mt-1">
            2 Peter 1:20-21
          </p>
        </div>

        {/* Tree Component */}
        <div className="flex-1 min-h-0 mb-2 relative">
          <SimpleTree />
        </div>
      </main>
    </div>
  );
}

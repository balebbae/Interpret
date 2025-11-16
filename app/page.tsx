"use client";

import { SimpleTree } from "@/components/ui/simple-growth-tree";
import { FileUpload } from "@/components/ui/file-upload";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import { ProcessingJob } from "@/lib/types";

export default function Home() {
  const [youtubeLink, setYoutubeLink] = useState("");
  const [currentJob, setCurrentJob] = useState<ProcessingJob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll for job status
  useEffect(() => {
    if (!currentJob || currentJob.status === 'completed' || currentJob.status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/status/${currentJob.jobId}`);
        if (response.ok) {
          const data = await response.json();
          setCurrentJob(data.job);
        }
      } catch (err) {
        console.error('Failed to poll status:', err);
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(pollInterval);
  }, [currentJob]);

  const handleFileUpload = async (files: File[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', files[0]);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const data = await response.json();

      // Create initial job object
      setCurrentJob({
        jobId: data.jobId,
        status: data.status,
        inputFile: files[0].name,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDownload = async (language: 'language1' | 'language2') => {
    if (!currentJob) return;

    try {
      const response = await fetch(`/api/download/${currentJob.jobId}`);
      if (response.ok) {
        const data = await response.json();
        const url = data.urls[language];

        // Open download URL in new tab
        window.open(url, '_blank');
      }
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

        {/* Upload Status */}
        {isUploading && (
          <div className="flex-shrink-0 text-center mt-4">
            <p className="text-sm text-neutral-600">Uploading...</p>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="flex-shrink-0 text-center mt-4">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Processing Status */}
        {currentJob && (
          <div className="flex-shrink-0 text-center mt-4 max-w-xl mx-auto">
            <div className="bg-white dark:bg-neutral-900 rounded-lg p-4 shadow-sm">
              <p className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-2">
                {currentJob.status === 'pending' && 'Processing starting...'}
                {currentJob.status === 'processing' && 'Separating audio tracks...'}
                {currentJob.status === 'completed' && 'Processing complete!'}
                {currentJob.status === 'failed' && 'Processing failed'}
              </p>

              {currentJob.progress !== undefined && (
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${currentJob.progress}%` }}
                  />
                </div>
              )}

              {currentJob.status === 'completed' && currentJob.outputFiles && (
                <div className="flex gap-2 justify-center mt-4">
                  <button
                    onClick={() => handleDownload('language1')}
                    className="px-4 py-2 bg-neutral-700 text-white rounded-md text-sm hover:bg-neutral-800 transition-colors"
                  >
                    Download Track 1
                  </button>
                  <button
                    onClick={() => handleDownload('language2')}
                    className="px-4 py-2 bg-neutral-700 text-white rounded-md text-sm hover:bg-neutral-800 transition-colors"
                  >
                    Download Track 2
                  </button>
                </div>
              )}
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

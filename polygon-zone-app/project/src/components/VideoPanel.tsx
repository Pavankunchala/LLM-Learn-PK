// components/VideoPanel.tsx

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Camera, Upload, Eye, EyeOff, Loader2, XCircle, PlayCircle, PauseCircle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Video, Info, AlertTriangle, Download, X } from 'lucide-react';
import PolygonDrawer from './PolygonDrawer';
import { useDetectionStore } from '../store/detectionStore';
import axios from 'axios';
import WebSocketManager from '../utils/websocketManager';

// --- Configuration ---
const HTTP_HOST = import.meta.env.VITE_BACKEND_HTTP_HOST || 'localhost';
const HTTP_PORT = import.meta.env.VITE_BACKEND_HTTP_PORT || 8000;
const HTTP_UPLOAD_URL = `http://${HTTP_HOST}:${HTTP_PORT}/upload`;
const HTTP_OUTPUT_URL_BASE = `http://${HTTP_HOST}:${HTTP_PORT}/outputs`; // Base URL for final videos

const VideoPanel = () => {
  const videoRef = useRef<HTMLVideoElement>(null); // Ref for the <video> element
  const containerRef = useRef<HTMLDivElement>(null); // Ref for drag-and-drop

  // --- Component State ---
  const [showPolygons, setShowPolygons] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [localVideoUrl, setLocalVideoUrl] = useState<string | null>(null);
  const [isPlayingLocal, setIsPlayingLocal] = useState(false);
  const [currentTimeLocal, setCurrentTimeLocal] = useState(0);
  const [durationLocal, setDurationLocal] = useState(0);
  const [isSeekingLocal, setIsSeekingLocal] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [playbackError, setPlaybackError] = useState<MediaError | null>(null);
  // *** RESTORED STATE ***
  const [showFinalVideo, setShowFinalVideo] = useState(false); // Explicit control for final video display

  // --- Zustand Store ---
  const {
    videoSource, setVideoSource,
    uploadedFilePath, setUploadedFilePath,
    uploadedFileName, setUploadedFileName,
    startFrameTime, setStartFrameTime,
    isProcessing,
    processedFrameUrl, // Live feed Base64
    backendStatus, backendStatusMessage,
    // *** RESTORED STATE ***
    finalOutputFilename, // Filename from backend
    clearPolygons,
    polygons // Read polygons state
  } = useDetectionStore();

  // --- Effects ---

  // WebSocket Connection
  useEffect(() => {
    console.log("[VP] Mounted. Connecting WS...");
    WebSocketManager.connect();
  }, []);

  // *** RESTORED EFFECT: Handle Backend Status -> Show Final Video ***
  useEffect(() => {
    console.log(`[VP] Status Check: Proc=${isProcessing}, Status=${backendStatus}, File=${finalOutputFilename}, ShowFinal=${showFinalVideo}`);
    // Condition: Processing just finished successfully AND we got a filename AND we're not already showing it.
    if (!isProcessing && backendStatus === 'stopped' && finalOutputFilename && !showFinalVideo) {
      console.log(`[VP] Triggering final video display for: ${finalOutputFilename}`);
      setShowFinalVideo(true); // Explicitly switch view
      setShowPolygons(false); // Hide polygon layer
      setIsPlayingLocal(false); // Ensure local state reflects not playing preview
      // Clear the video element's current source to prepare for the final URL
      if (videoRef.current) {
        videoRef.current.removeAttribute('src');
        videoRef.current.srcObject = null;
        videoRef.current.load(); // Ask browser to update state
        console.log("[VP] Cleared video element for final video.");
      }
    }
    // Reset final video view if state changes away from the final stopped condition
     if ((isProcessing || isUploading || videoSource === 'webcam' || !finalOutputFilename || backendStatus !== 'stopped') && showFinalVideo) {
        console.log("[VP] Hiding final video due to state change.");
        setShowFinalVideo(false);
     }
  }, [isProcessing, backendStatus, finalOutputFilename, showFinalVideo, isUploading, videoSource]); // Dependencies updated

  // File Upload Logic (Reset showFinalVideo)
  const handleFileUpload = useCallback(async (file: File) => {
    if (!file) return;
    console.log(`[VP] Uploading: ${file.name}`);
    setUploadError(null); setPlaybackError(null); setIsUploading(true);
    setShowFinalVideo(false); // *** Reset showFinalVideo ***
    setVideoSource('file'); setUploadedFilePath(null); setUploadedFileName(null);
    if (localVideoUrl) URL.revokeObjectURL(localVideoUrl);
    setLocalVideoUrl(null); setStartFrameTime(0); setCurrentTimeLocal(0); setDurationLocal(0); setIsPlayingLocal(false);
    if (videoRef.current) { videoRef.current.removeAttribute('src'); videoRef.current.srcObject = null; videoRef.current.load(); }
    clearPolygons();
    const formData = new FormData(); formData.append('videoFile', file);
    try {
      const response = await axios.post(HTTP_UPLOAD_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      if (response.data?.status === 'success' && response.data.filePath) {
        console.log("[VP] Upload successful:", response.data);
        setUploadedFilePath(response.data.filePath); setUploadedFileName(response.data.filename);
        const url = URL.createObjectURL(file); console.log(`[VP] Created local object URL: ${url}`);
        setLocalVideoUrl(url); // Trigger local preview effect
      } else { throw new Error(response.data?.message || 'Upload failed structure'); }
    } catch (error: any) { console.error('[VP] Upload error:', error); setUploadError(`Upload failed: ${error.message}`); setVideoSource(null); }
    finally { setIsUploading(false); }
  }, [localVideoUrl, setVideoSource, setUploadedFilePath, setUploadedFileName, setLocalVideoUrl, setStartFrameTime, clearPolygons]);

  // File Input Change Handler
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) handleFileUpload(file); if (e.target) e.target.value = ''; }, [handleFileUpload]);

  // Webcam Local Preview Setup & Cleanup (Consider showFinalVideo)
  useEffect(() => {
    let stream: MediaStream | null = null; let cancelled = false;
    const setupWebcam = async () => {
      // *** Condition: Only run if webcam selected AND NOT processing AND NOT showing final video ***
      if (videoSource === 'webcam' && !isProcessing && !showFinalVideo) {
         if (!videoRef.current) return;
         console.log("[VP] Setting up webcam preview...");
         setLocalVideoUrl(null); setUploadedFilePath(0); setUploadedFileName("Webcam"); setStartFrameTime(0);
         setCurrentTimeLocal(0); setDurationLocal(0); setIsPlayingLocal(true); setPlaybackError(null); setUploadError(null);
         try {
           stream = await navigator.mediaDevices.getUserMedia({ video: true });
           if (!cancelled && videoRef.current) { videoRef.current.src = ''; videoRef.current.srcObject = stream; console.log("[VP] Webcam stream assigned."); }
           else if (stream) { stream.getTracks().forEach(t => t.stop()); }
         } catch (err) { console.error('[VP] Webcam access error:', err); if (!cancelled) { setUploadError('Webcam access failed.'); setVideoSource(null); } }
      } else if (videoSource === 'webcam' && videoRef.current && (isProcessing || showFinalVideo)) {
          // Stop local stream if processing starts or final video shown
          if (videoRef.current.srcObject) { console.log("[VP] Stopping local webcam stream."); (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop()); videoRef.current.srcObject = null; setIsPlayingLocal(false); }
      }
    };
    setupWebcam();
    return () => { // Cleanup
      cancelled = true; if (stream) { stream.getTracks().forEach(t => t.stop()); }
    };
  }, [videoSource, isProcessing, showFinalVideo, setVideoSource, setUploadedFilePath, setUploadedFileName, setStartFrameTime]);

  // Local File Preview Setup & Cleanup (Consider showFinalVideo)
  useEffect(() => {
    const videoElement = videoRef.current;
    // *** Condition: Only run for file source, URL exists, NOT processing, AND NOT showing final video ***
    if (videoSource === 'file' && localVideoUrl && !isProcessing && !showFinalVideo && videoElement) {
        console.log("[VP] Setting up local file preview for:", localVideoUrl);
        setPlaybackError(null);
        if (videoElement.srcObject) videoElement.srcObject = null; // Ensure no webcam stream
        if (videoElement.src !== localVideoUrl) { videoElement.src = localVideoUrl; videoElement.load(); console.log("[VP] Assigned local URL to src."); }
        // Event Listeners
        const listeners = { timeupdate: () => { if (!isSeekingLocal && videoElement) setCurrentTimeLocal(videoElement.currentTime); }, loadedmetadata: () => { if (videoElement) { console.log(`[VP] Metadata loaded. Dur: ${videoElement.duration}`); setDurationLocal(videoElement.duration); setCurrentTimeLocal(0); setStartFrameTime(0); setPlaybackError(null); }}, play: () => setIsPlayingLocal(true), pause: () => setIsPlayingLocal(false), ended: () => { setIsPlayingLocal(false); if(videoElement) setCurrentTimeLocal(durationLocal); }, error: (e: Event) => { if (videoElement?.error) { console.error("[VP] Playback Error:", videoElement.error); setPlaybackError(videoElement.error); setIsPlayingLocal(false); setDurationLocal(0); setCurrentTimeLocal(0); setStartFrameTime(0); }} };
        Object.entries(listeners).forEach(([evt, handler]) => videoElement.addEventListener(evt, handler as EventListener));
        return () => { if (videoElement) { console.log("[VP] Removing local video listeners."); Object.entries(listeners).forEach(([evt, handler]) => videoElement.removeEventListener(evt, handler as EventListener)); } };
    } else if (videoElement && !isProcessing && !showFinalVideo && videoSource !== 'webcam') {
        // Clear element if not in local file preview mode or webcam preview mode
         if (videoElement.src || videoElement.srcObject) { console.log("[VP] Clearing video element (not preview/webcam)."); videoElement.removeAttribute('src'); videoElement.srcObject = null; videoElement.load(); videoElement.pause(); }
    }
     // Explicitly clear src if processing starts or final video shown, to avoid conflicts
     if(videoElement && (isProcessing || showFinalVideo)) {
          if(videoElement.src === localVideoUrl || videoElement.srcObject){ console.log("[VP] Clearing local src/stream due to processing/final video view."); videoElement.removeAttribute('src'); videoElement.srcObject = null; videoElement.load(); videoElement.pause(); }
     }
  }, [videoSource, localVideoUrl, isProcessing, showFinalVideo, isSeekingLocal, setStartFrameTime, durationLocal]);


  // --- Playback Control Handlers ---
  const togglePlayLocal = useCallback(() => { const v = videoRef.current; if(v?.readyState >= v.HAVE_METADATA){ if(isPlayingLocal) v.pause(); else v.play().catch(e=>console.error("Play failed",e)); }}, [isPlayingLocal]);
  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => { const v = videoRef.current; const t = parseFloat(e.target.value); if(v?.readyState >= v.HAVE_METADATA){ v.currentTime = t; setCurrentTimeLocal(t); }}, []);
  const handleSeekStart = useCallback(() => { setIsSeekingLocal(true); if(videoRef.current && isPlayingLocal) videoRef.current.pause(); }, [isPlayingLocal]);
  const handleSeekEnd = useCallback(() => { setIsSeekingLocal(false); const v = videoRef.current; if(v?.paused && isPlayingLocal && v.readyState >= v.HAVE_METADATA){ v.play().catch(e=>console.error("Resume fail",e)); }}, [isPlayingLocal]);
  const seekBy = useCallback((s: number) => { const v = videoRef.current; if(v && !isSeekingLocal && v.readyState >= v.HAVE_METADATA){ const t = Math.max(0, Math.min(durationLocal, v.currentTime + s)); v.currentTime = t; setCurrentTimeLocal(t); }}, [isSeekingLocal, durationLocal]);
  const formatTime = (t: number): string => { if(isNaN(t)||!isFinite(t)||t<0) return "00:00.00"; const m=Math.floor(t/60); const s=Math.floor(t%60); const h=Math.floor((t-Math.floor(t))*100); return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(h).padStart(2,'0')}`; };

  // --- Drag and Drop Handlers ---
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => { if (isProcessing || isUploading || backendStatus === 'connecting' || videoSource || showFinalVideo) return; e.preventDefault(); e.stopPropagation(); setIsDraggingOver(true); }, [isProcessing, isUploading, backendStatus, videoSource, showFinalVideo]);
  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => { if (isProcessing || isUploading || backendStatus === 'connecting' || videoSource || showFinalVideo) return; e.preventDefault(); e.stopPropagation(); setIsDraggingOver(false); }, [isProcessing, isUploading, backendStatus, videoSource, showFinalVideo]);
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => { if (isProcessing || isUploading || backendStatus === 'connecting' || videoSource || showFinalVideo) return; e.preventDefault(); e.stopPropagation(); setIsDraggingOver(false); const files = e.dataTransfer.files; if (files?.length > 0) { const vf = Array.from(files).find(f => f.type.startsWith('video/')); if (vf) handleFileUpload(vf); else setUploadError("Only video files accepted."); } }, [isProcessing, isUploading, backendStatus, videoSource, showFinalVideo, handleFileUpload]);

  // --- Error Display Logic ---
  const currentErrorMessage = uploadError || backendStatusMessage;
  const shouldShowError = (currentErrorMessage && (uploadError || ['error', 'disconnected', 'warning'].includes(backendStatus))) || (playbackError && !isProcessing && !showFinalVideo); // Only show playback error if relevant

  // --- Final Video URL ---
  const finalVideoUrl = finalOutputFilename ? `${HTTP_OUTPUT_URL_BASE}/${encodeURIComponent(finalOutputFilename)}` : null;

  // --- Button Handlers ---
  // *** RESTORED HANDLER ***
  const handleCloseFinalVideo = useCallback(() => {
      console.log("[VP] Closing final video view.");
      setShowFinalVideo(false);
      setPlaybackError(null); // Clear potential final video load errors
      // Let effects handle reloading local preview if applicable
  }, []);

  // Determine if polygon drawing should be possible/visible
  const canDrawPolygons = !isProcessing && !showFinalVideo && videoSource === 'file' && !playbackError && localVideoUrl;


  // --- JSX ---
  return (
    <div className="bg-gray-800 rounded-xl shadow-lg overflow-hidden flex flex-col h-full text-white">
      {/* Header */}
      <div className="p-3 bg-gray-700 flex justify-between items-center flex-shrink-0">
         {/* Title */}
         <h2 className="text-base font-medium flex items-center mr-2 truncate" title={showFinalVideo ? finalOutputFilename ?? "Result" : uploadedFileName ?? (videoSource === 'webcam' ? "Webcam" : "Feed")}>
            <Video size={18} className="mr-1.5 flex-shrink-0"/>
            <span className="truncate">{showFinalVideo ? "Processed Result" : (isProcessing ? "Processing..." : (uploadedFileName ? uploadedFileName : (videoSource === 'webcam' ? "Webcam Preview" : "Video Feed")))}</span>
        </h2>
         {/* Header Buttons */}
         <div className="flex items-center space-x-2">
           <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className={`px-2 py-1 rounded-md flex items-center text-xs ${ videoSource === 'webcam' && !showFinalVideo ? 'bg-primary-600' : 'bg-gray-600 hover:bg-gray-500' } ${isProcessing || isUploading || backendStatus === 'connecting' || showFinalVideo ? 'opacity-50 cursor-not-allowed' : ''}`} onClick={() => { if (!showFinalVideo) setVideoSource('webcam'); }} disabled={isProcessing || isUploading || backendStatus === 'connecting' || showFinalVideo} title="Use Webcam"> <Camera size={14} className="mr-1" /> Webcam </motion.button>
           <motion.label whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className={`px-2 py-1 rounded-md flex items-center text-xs cursor-pointer ${ videoSource === 'file' && !showFinalVideo ? 'bg-primary-600' : 'bg-gray-600 hover:bg-gray-500' } ${isProcessing || isUploading || backendStatus === 'connecting' || showFinalVideo ? 'opacity-50 cursor-not-allowed' : ''}`} title="Upload Video"> {isUploading ? <Loader2 size={14} className="mr-1 animate-spin" /> : <Upload size={14} className="mr-1" />} <span className='truncate max-w-[80px]'>{isUploading ? 'Uploading' : (uploadedFileName ? uploadedFileName : 'Upload')}</span> <input type="file" accept="video/*" className="hidden" onChange={handleFileInputChange} disabled={isProcessing || isUploading || backendStatus === 'connecting' || showFinalVideo} /> </motion.label>
           {/* Toggle Polygons Button */}
           {canDrawPolygons && ( <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="p-1.5 rounded-md bg-gray-600 hover:bg-gray-500" onClick={() => setShowPolygons(!showPolygons)} title={showPolygons ? "Hide Polygons" : "Show Polygons"}> {showPolygons ? <EyeOff size={14} /> : <Eye size={14} />} </motion.button> )}
           {/* Close Final Video Button */}
           {showFinalVideo && ( <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="p-1.5 rounded-md bg-error-600 hover:bg-error-500 text-white" onClick={handleCloseFinalVideo} title="Close Result View"> <X size={14} /> </motion.button> )}
        </div>
      </div>

      {/* Media Display Area */}
      <div ref={containerRef} className={`relative aspect-video bg-black flex-grow flex items-center justify-center overflow-hidden ${!videoSource && !showFinalVideo ? 'border-2 border-dashed border-gray-600' : ''} ${isDraggingOver ? 'border-primary-500 bg-primary-950 bg-opacity-20' : ''}`} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} >

          {/* --- Media Element (Video or Img) --- */}

          {/* 1. Final Processed Video */}
          {showFinalVideo && finalVideoUrl && (
              <video
                  ref={videoRef} // Attach ref
                  key={finalVideoUrl} // Force re-render
                  className="block w-full h-full object-contain"
                  src={finalVideoUrl}
                  controls autoPlay loop playsInline preload="auto"
                  onError={(e) => { console.error("Error loading final video:", e); setPlaybackError(new MediaError()); setShowFinalVideo(false); setUploadError("Failed to load processed video."); }}
              />
          )}

          {/* 2. Live Processing Feed */}
          {isProcessing && processedFrameUrl && !showFinalVideo && (
              <img
                  key="processing-feed"
                  src={processedFrameUrl}
                  alt="Processing Feed"
                  className="block w-full h-full object-contain"
              />
          )}

          {/* 3. Local Preview (File or Webcam) */}
          {/* Render this video tag only if NOT processing AND NOT showing final video */}
          {!isProcessing && !showFinalVideo && videoSource && !playbackError && (
              <video
                  ref={videoRef} // Attach ref
                  key={localVideoUrl || videoSource}
                  className="block w-full h-full object-contain"
                  // src/srcObject set by useEffects
                  autoPlay={videoSource === 'webcam'}
                  playsInline muted preload="metadata"
              />
          )}

          {/* 4. Playback Error Placeholder (Shown when local preview fails) */}
          {!isProcessing && !showFinalVideo && videoSource && playbackError && (
              <div className="absolute inset-0 flex items-center justify-center text-error-300 p-4 text-center text-sm bg-black bg-opacity-70"> Cannot display local preview.<br />(Unsupported format?) </div>
          )}


          {/* --- Overlays --- */}

          {/* Polygon Drawing Overlay (Show only on local file preview) */}
          {canDrawPolygons && showPolygons && videoRef.current && (
              <div className="absolute inset-0 z-20 pointer-events-auto"> {/* Ensure events pass through */}
                  <PolygonDrawer videoRef={videoRef} />
              </div>
          )}

          {/* No Source Placeholder (Show only if nothing else is displayed) */}
          {!videoSource && !isUploading && !showFinalVideo && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900 bg-opacity-80 z-30 p-4 text-center">
                 <p className="text-gray-400 mb-4">Select source or drag & drop video</p>
                 {backendStatus === 'connecting' ? (<div className="flex items-center text-primary-400"><Loader2 size={20} className="mr-2 animate-spin" />Connecting...</div>)
                  : (<div className="flex flex-col sm:flex-row gap-3 w-full max-w-xs"> <motion.button className="px-4 py-2 bg-primary-600 rounded-lg flex items-center justify-center w-full disabled:opacity-50 text-sm" onClick={() => setVideoSource('webcam')} disabled={backendStatus === 'connecting'} > <Camera size={16} className="mr-2" /> Use Webcam </motion.button> <motion.label className={`px-4 py-2 rounded-lg flex items-center justify-center cursor-pointer w-full text-sm ${ backendStatus === 'connecting' ? 'opacity-50 cursor-not-allowed bg-gray-700' : 'bg-gray-700 hover:bg-gray-600' }`} > <Upload size={16} className="mr-2" /> Upload Video <input type="file" accept="video/*" className="hidden" onChange={handleFileInputChange} disabled={backendStatus === 'connecting'} /> </motion.label> </div>)}
              </div>
          )}
          {/* Uploading Spinner */}
          {isUploading && ( <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-80 z-40"><Loader2 size={24} className="mr-3 animate-spin text-primary-400" /> <span className="text-lg text-primary-400">Uploading...</span></div> )}
          {/* Processing Spinner */}
          {isProcessing && (backendStatus === 'started' || backendStatus === 'stopping') && ( <div className="absolute top-2 left-2 bg-gray-900 bg-opacity-75 text-white text-xs px-2 py-1 rounded-md flex items-center z-30"><Loader2 size={12} className="mr-1.5 animate-spin" />{backendStatus === 'started' ? 'Processing...' : 'Stopping...'}</div> )}
          {/* Error Message Display */}
          {shouldShowError && ( <div className={`absolute inset-x-2 bottom-2 p-2 rounded-lg shadow-lg flex items-start z-40 text-xs ${ playbackError && !isProcessing && !showFinalVideo ? 'bg-error-900 text-error-300' : uploadError || ['error', 'disconnected'].includes(backendStatus) ? 'bg-error-900 text-error-300' : backendStatus === 'warning' ? 'bg-warning-900 text-warning-300' : 'bg-gray-700 text-gray-300' }`}> {(playbackError && !isProcessing && !showFinalVideo) || uploadError || ['error', 'disconnected'].includes(backendStatus) ? <XCircle size={16} className="mr-1.5 flex-shrink-0 mt-0.5" /> : <AlertTriangle size={16} className="mr-1.5 flex-shrink-0 mt-0.5 text-warning-400" />} <div className="flex-1 break-words"> {(playbackError && !isProcessing && !showFinalVideo) ? ( <span>Local playback error.</span> ) : ( <span>{currentErrorMessage || 'Unknown error'}</span> )} </div> </div> )}

      </div> {/* End Media Display Area */}

      {/* Footer Area: Local Controls or Download Button */}
      <div className="flex-shrink-0 bg-gray-700">
          {/* Local Playback Controls */}
          {canDrawPolygons && durationLocal > 0 && !showFinalVideo && ( // Added !showFinalVideo condition
               <div className="p-3 flex flex-col space-y-2">
                   {/* Slider */}
                   <div className="flex items-center space-x-2"> <span className="text-xs font-mono text-gray-300 w-16 text-right">{formatTime(currentTimeLocal)}</span> <input type="range" min="0" max={durationLocal} step="0.01" value={currentTimeLocal} onChange={handleSeek} onMouseDown={handleSeekStart} onMouseUp={handleSeekEnd} className="flex-1 h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-primary-500" disabled={!videoRef.current} /> <span className="text-xs font-mono text-gray-300 w-16">{formatTime(durationLocal)}</span> </div>
                   {/* Buttons */}
                   <div className="flex items-center justify-between"> <div className='flex items-center space-x-2'> <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={() => seekBy(-5)} className="p-1 rounded-full hover:bg-gray-600 disabled:opacity-50" title="-5s" disabled={isSeekingLocal || !videoRef.current}> <ChevronLeft size={18} /> </motion.button> <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={() => seekBy(-0.1)} className="p-1 rounded-full hover:bg-gray-600 disabled:opacity-50" title="-0.1s" disabled={isSeekingLocal || !videoRef.current}> <ChevronsLeft size={18} /> </motion.button> <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={togglePlayLocal} className="p-1 rounded-full bg-primary-600 hover:bg-primary-500 disabled:opacity-50" title={isPlayingLocal ? "Pause" : "Play"} disabled={isSeekingLocal || !videoRef.current}> {isPlayingLocal ? <PauseCircle size={20} /> : <PlayCircle size={20} />} </motion.button> <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={() => seekBy(0.1)} className="p-1 rounded-full hover:bg-gray-600 disabled:opacity-50" title="+0.1s" disabled={isSeekingLocal || !videoRef.current}> <ChevronsRight size={18} /> </motion.button> <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }} onClick={() => seekBy(5)} className="p-1 rounded-full hover:bg-gray-600 disabled:opacity-50" title="+5s" disabled={isSeekingLocal || !videoRef.current}> <ChevronRight size={18} /> </motion.button> </div> <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className={`px-3 py-1.5 rounded-lg flex items-center text-xs transition-colors ${ (Math.abs(currentTimeLocal - startFrameTime) < 0.05 && currentTimeLocal > 0) ? 'bg-gray-600 text-gray-400 cursor-default' : 'bg-success-600 hover:bg-success-500 text-white'} ${isSeekingLocal || durationLocal <= 0 || !videoRef.current ? 'opacity-50 cursor-not-allowed' : ''} `} onClick={() => setStartFrameTime(currentTimeLocal)} disabled={(Math.abs(currentTimeLocal - startFrameTime) < 0.05 && currentTimeLocal > 0) || isSeekingLocal || durationLocal <= 0 || !videoRef.current} title="Set start time"> <Video size={14} className="mr-1.5" /> Set Start ({formatTime(startFrameTime)}) </motion.button> </div>
               </div>
          )}
          {/* Download Button */}
          {showFinalVideo && finalVideoUrl && (
               <div className="p-3 flex items-center justify-center"> <a href={finalVideoUrl} download={finalOutputFilename || 'processed_video.mp4'} className="px-3 py-1.5 bg-secondary-600 hover:bg-secondary-500 text-white rounded-lg text-sm flex items-center" title="Download Processed Video"> <Download size={16} className='mr-2'/> Download Result </a> </div>
          )}
      </div> {/* End Footer Area */}
    </div> // End Main Component Div
  );
};

export default VideoPanel;
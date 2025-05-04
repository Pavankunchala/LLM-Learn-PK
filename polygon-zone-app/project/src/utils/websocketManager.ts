// utils/websocketManager.ts

import { BackendMessage, ConfigurationMessage, ControlMessage } from '../types/detection';
import { useDetectionStore } from '../store/detectionStore';

const WS_URL = `ws://${import.meta.env.VITE_BACKEND_WS_HOST || 'localhost'}:${import.meta.env.VITE_BACKEND_WS_PORT || 8765}`;

class WebSocketManager {
  private static instance: WebSocketManager;
  private socket: WebSocket | null = null;
  private isConnected = false;
  private messageQueue: any[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;

  private constructor() { console.log(`WebSocketManager initialized. WS_URL: ${WS_URL}`); }
  public static getInstance(): WebSocketManager { if (!WebSocketManager.instance) WebSocketManager.instance = new WebSocketManager(); return WebSocketManager.instance; }

  public connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) { console.log("WS already connected/connecting."); return; }
    console.log(`Attempting WS connect: ${WS_URL}`);
    useDetectionStore.getState().setBackendStatus("connecting");
    this.socket = new WebSocket(WS_URL);
    this.socket.onopen = () => {
      this.isConnected = true; this.reconnectAttempts = 0; console.log('WS connection established');
      useDetectionStore.getState().setBackendStatus("connected");
      while (this.messageQueue.length > 0) this.send(this.messageQueue.shift());
    };
    this.socket.onclose = (event) => {
      this.isConnected = false; this.socket = null; console.log(`WS connection closed: ${event.code} ${event.reason || ''}`);
      const currentStatus = useDetectionStore.getState().backendStatus;
      // Avoid setting disconnected if it was already error or stopping
      if (currentStatus !== 'error' && currentStatus !== 'stopping') {
         useDetectionStore.getState().setBackendStatus("disconnected");
      }
      if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
         console.log(`Attempting reconnect (#${this.reconnectAttempts + 1}) in ${this.reconnectDelay}ms`);
         setTimeout(() => this.connect(), this.reconnectDelay); this.reconnectAttempts++; this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
      } else if (event.code !== 1000) { console.error("Max reconnect attempts reached."); useDetectionStore.getState().setBackendStatus("error", "Connection failed after multiple retries."); }
    };
    this.socket.onerror = (error) => { console.error('WS error:', error); useDetectionStore.getState().setBackendStatus("error", "WebSocket connection error."); }; // Set error status on WS error
    this.socket.onmessage = (event) => {
      try {
        const message: BackendMessage = JSON.parse(event.data);
        this.handleMessage(message); // Process the parsed message
      } catch (error) {
        console.error('Error parsing WS message:', event.data, error);
        useDetectionStore.getState().setBackendStatus("error", "Received malformed message from backend.");
      }
    };
  }

  public disconnect(): void {
    if (this.socket) { console.log("Disconnecting WS..."); this.socket.close(1000, "Client disconnecting"); }
    this.isConnected = false; this.socket = null; useDetectionStore.getState().setBackendStatus("disconnected"); this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
  }

  private send(message: any): void {
    if (this.isConnected && this.socket) {
      try { this.socket.send(JSON.stringify(message)); /* console.log("Sent WS:", message); */ }
      catch (error) { console.error("Error sending WS message:", error); }
    } else { console.warn("WS not connected, queuing:", message); this.messageQueue.push(message); }
  }

  public sendConfiguration(config: Omit<ConfigurationMessage, 'type'>): void { this.send({ type: 'configuration', ...config }); }
  public sendControl(action: ControlMessage['action']): void { this.send({ type: 'control', action }); }

  // MODIFIED handleMessage
  private handleMessage(message: BackendMessage): void {
    console.log("DEBUG: Received WS message:", message);
    const state = useDetectionStore.getState();

    switch (message.type) {
      case 'processedFrame':
        // console.log("DEBUG: Handling processedFrame"); // Keep if needed, can be spammy
        // Only update if still processing, prevents late frame overwriting final video display
        if (state.isProcessing) {
            state.setProcessedFrameUrl(message.imageUrl);
        }
        break;
      case 'resultsUpdate':
        // console.log("DEBUG: Handling resultsUpdate"); // Keep if needed
        if (state.isProcessing) {
            state.updateDetectionResults(message.results);
        }
        break;
      case 'status':
        console.log("DEBUG: Handling status:", message.status);
        // *** Pass the whole message object to setBackendStatus ***
        // It now contains status, message, and potentially outputFilename
        state.setBackendStatus(message.status, message.message, message); // Pass entire message as extraData

        // Trigger zustand actions based on status
        if (message.status === 'started') {
             state.startProcessing();
        } else if (message.status === 'stopped' || message.status === 'error') {
             state.stopProcessing(); // stopProcessing now clears finalOutputFilename too
        }
        break;
      default:
        console.warn('Received unknown message type:', (message as any).type, message);
    }
  }
}

export default WebSocketManager.getInstance();
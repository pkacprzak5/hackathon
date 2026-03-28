import type { WSIncomingMessage, WSOutgoingMessage } from "@/lib/types";

export type WSMessageHandler = (message: WSIncomingMessage) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private handler: WSMessageHandler;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(url: string, handler: WSMessageHandler) {
    this.url = url;
    this.handler = handler;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WSIncomingMessage;
          this.handler(message);
        } catch {
          console.warn("Failed to parse WebSocket message:", event.data);
        }
      };

      this.ws.onclose = () => {
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  send(message: WSOutgoingMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectAttempts = this.maxReconnectAttempts;
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }
}

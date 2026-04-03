# AI Interview Monitoring Backend

Production-ready FastAPI backend for real-time interview monitoring with webcam analysis.

## Features

- Real-time webcam frame processing using OpenCV
- Eye direction classification (`center`, `left`, `right`, `down`) using MediaPipe Face Mesh + iris landmarks
- Head pose estimation (`yaw`, `pitch`, `direction`) with `solvePnP`
- Mobile phone detection with YOLOv8 (`yolov8n.pt`, class `cell phone`)
- WebSocket stream for live monitoring updates (`/ws/monitor`)
- REST endpoints for health and current monitoring status
- Structured JSON logging
- Environment-driven configuration via `.env`
- Basic service-level unit tests with mocked camera/model components

## Tech Stack

- Python 3.10+
- FastAPI
- OpenCV
- MediaPipe
- YOLOv8 (Ultralytics)
- WebSockets
- Pydantic
- Uvicorn

## Project Structure

```text
backend/
│
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   └── monitoring.py
│   │   └── deps.py
│   ├── services/
│   │   ├── camera_service.py
│   │   ├── eye_tracking.py
│   │   ├── head_pose.py
│   │   ├── object_detection.py
│   │   └── monitoring_service.py
│   ├── models/
│   │   └── schemas.py
│   └── utils/
│       └── helpers.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create environment file:

```bash
cp .env.example .env
```

4. Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at `http://localhost:8000/docs`.

### OpenAPI Specification

- Canonical API spec file: `openapi.yaml`
- You can import this file directly into Postman, Stoplight, Insomnia, or openapi-generator.

Generate a TypeScript REST client from the spec (example with OpenAPI Generator):

```bash
npx @openapitools/openapi-generator-cli generate \
  -i openapi.yaml \
  -g typescript-fetch \
  -o ./generated/api-client
```

## API Endpoints

### `GET /health`

Returns service health.

Example response:

```json
{
  "status": "ok",
  "monitoring_running": true,
  "last_error": null,
  "timestamp": "2026-04-18T15:40:10.259746+00:00"
}
```

### `GET /status`

Returns the latest monitoring snapshot.

Example response:

```json
{
  "timestamp": "2026-04-18T15:40:11.173120+00:00",
  "eye": "center",
  "head": "left",
  "phone": true,
  "confidence": {
    "eye": 0.87,
    "head": 0.91,
    "phone": 0.95
  },
  "suspicion_score": 0.71,
  "yaw": -19.4,
  "pitch": 4.2,
  "error": null
}
```

### `WS /ws/monitor`

Streams the same JSON payload as `/status` every `WS_INTERVAL_MS` (200-500ms).

## Frontend Integration (Next.js)

### Basic WebSocket client

```tsx
"use client";

import { useEffect, useState } from "react";

type Snapshot = {
  timestamp: string;
  eye: "center" | "left" | "right" | "down" | "unknown";
  head: "center" | "left" | "right" | "up" | "down" | "unknown";
  phone: boolean;
  confidence: { eye: number; head: number; phone: number };
  suspicion_score: number;
  yaw: number | null;
  pitch: number | null;
  error: string | null;
};

export default function MonitorPanel() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/monitor");

    ws.onmessage = (event) => {
      const payload: Snapshot = JSON.parse(event.data);
      setSnapshot(payload);
    };

    ws.onerror = (event) => {
      console.error("WebSocket error", event);
    };

    return () => ws.close();
  }, []);

  return (
    <pre style={{ whiteSpace: "pre-wrap" }}>
      {snapshot ? JSON.stringify(snapshot, null, 2) : "Waiting for data..."}
    </pre>
  );
}
```

### Expected JSON format

```json
{
  "timestamp": "ISO-8601 string",
  "eye": "center|left|right|down|unknown",
  "head": "center|left|right|up|down|unknown",
  "phone": true,
  "confidence": {
    "eye": 0.0,
    "head": 0.0,
    "phone": 0.0
  },
  "suspicion_score": 0.0,
  "yaw": 0.0,
  "pitch": 0.0,
  "error": null
}
```

## Frontend Implementation Guide (Next.js)

This section gives a production-ready integration approach for App Router projects.

### 1) Environment variables

Create `.env.local` in your frontend app:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/monitor
```

### 2) Shared types

Create `src/types/monitoring.ts`:

```ts
export type EyeDirection = "center" | "left" | "right" | "down" | "unknown";
export type HeadDirection =
  | "center"
  | "left"
  | "right"
  | "up"
  | "down"
  | "unknown";

export type MonitoringSnapshot = {
  timestamp: string;
  eye: EyeDirection;
  head: HeadDirection;
  phone: boolean;
  confidence: {
    eye: number;
    head: number;
    phone: number;
  };
  suspicion_score: number;
  yaw: number | null;
  pitch: number | null;
  error: string | null;
};

export type HealthResponse = {
  status: "ok" | "degraded";
  monitoring_running: boolean;
  last_error: string | null;
  timestamp: string;
};
```

### 3) REST helper

Create `src/lib/api.ts`:

```ts
import type { HealthResponse, MonitoringSnapshot } from "@/types/monitoring";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function requestJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export const api = {
  getHealth: () => requestJson<HealthResponse>("/health"),
  getStatus: () => requestJson<MonitoringSnapshot>("/status"),
};
```

### 4) Real-time WebSocket hook with reconnect

Create `src/hooks/useMonitoringSocket.ts`:

```ts
"use client";

import { useEffect, useRef, useState } from "react";
import type { MonitoringSnapshot } from "@/types/monitoring";

type SocketState = "connecting" | "open" | "closed" | "error";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/monitor";

export function useMonitoringSocket() {
  const [snapshot, setSnapshot] = useState<MonitoringSnapshot | null>(null);
  const [socketState, setSocketState] = useState<SocketState>("connecting");
  const reconnectDelayRef = useRef(1000);
  const wsRef = useRef<WebSocket | null>(null);
  const retryTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let disposed = false;

    const connect = () => {
      if (disposed) return;

      setSocketState("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setSocketState("open");
        reconnectDelayRef.current = 1000;
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as MonitoringSnapshot;
          setSnapshot(parsed);
        } catch {
          setSocketState("error");
        }
      };

      ws.onerror = () => {
        setSocketState("error");
      };

      ws.onclose = () => {
        setSocketState("closed");
        if (!disposed) {
          const delay = reconnectDelayRef.current;
          retryTimerRef.current = window.setTimeout(connect, delay);
          reconnectDelayRef.current = Math.min(delay * 2, 10000);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, []);

  return { snapshot, socketState };
}
```

### 5) Dashboard component example

Create `src/components/MonitorCard.tsx`:

```tsx
"use client";

import { useMonitoringSocket } from "@/hooks/useMonitoringSocket";

function riskLabel(score: number): "low" | "medium" | "high" {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

export function MonitorCard() {
  const { snapshot, socketState } = useMonitoringSocket();

  if (!snapshot) {
    return <p>Socket: {socketState} - waiting for monitoring data...</p>;
  }

  return (
    <section>
      <h2>Interview Monitoring</h2>
      <p>Socket: {socketState}</p>
      <p>Eye: {snapshot.eye}</p>
      <p>Head: {snapshot.head}</p>
      <p>Phone detected: {String(snapshot.phone)}</p>
      <p>
        Suspicion: {snapshot.suspicion_score.toFixed(2)} ({riskLabel(snapshot.suspicion_score)})
      </p>
      <p>Yaw/Pitch: {snapshot.yaw ?? 0} / {snapshot.pitch ?? 0}</p>
      {snapshot.error ? <p>Error: {snapshot.error}</p> : null}
    </section>
  );
}
```

### 6) App Router page usage

```tsx
import { MonitorCard } from "@/components/MonitorCard";

export default function Page() {
  return <MonitorCard />;
}
```

### 7) Production frontend checklist

- Use `wss://` in production and route WebSocket through your reverse proxy.
- Restrict backend CORS to the real frontend domain(s).
- Add auth headers/tokens before exposing endpoints outside trusted networks.
- Handle stale connections (reconnect + status badge in UI).
- Buffer/aggregate UI updates if rendering every frame causes jank.

## Performance Notes

- `TARGET_FPS` controls processing load. Start around `5-8` FPS.
- `PHONE_DETECTION_FRAME_SKIP` reduces YOLO inference frequency.
- `WS_INTERVAL_MS` controls update frequency for clients.
- Keep frame resolution moderate (`640x480`) for lower latency.
- Use `YOLO_DEVICE=cuda` when GPU is available.

## Error Handling Behavior

- Camera unavailable: service remains up, `/health` shows `degraded` and latest error.
- YOLO load/inference failure: pipeline continues with phone detection disabled.
- WebSocket disconnects are handled cleanly per client.
- Missing face in frame: eye/head become `unknown` with low confidence.

## Running Tests

```bash
pytest -q
```

Tests include:

- Camera service with mocked `VideoCapture`
- Eye tracking directional classification
- Head pose direction classification + fallback behavior
- Object detection parsing with mocked YOLO outputs
- Monitoring aggregation logic with mocked dependencies

## Docker (Bonus)

Build and run:

```bash
docker build -t interview-monitor-backend .
docker run --rm -p 8000:8000 interview-monitor-backend
```

To enable GPU in compatible environments, run with NVIDIA runtime and set `YOLO_DEVICE=cuda`.

## Production Recommendations

- Run behind a reverse proxy (Nginx/Traefik) with TLS termination.
- Use process supervision (systemd, Docker restart policies, Kubernetes).
- Set explicit CORS origins for production frontend domains.
- Centralize logs into a platform like ELK/OpenSearch/Datadog.
- Add auth for websocket and REST endpoints if used outside trusted networks.

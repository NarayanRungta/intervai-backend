from __future__ import annotations

import asyncio
import base64
import cv2
import numpy as np

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.deps import get_monitoring_service
from app.core.config import Settings
from app.core.logger import get_logger
from app.models.schemas import MonitoringSnapshot
from app.services.monitoring_service import MonitoringService

router = APIRouter()
logger = get_logger(__name__)


class WebSocketConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def connection_count(self) -> int:
        async with self._lock:
            return len(self._connections)


connection_manager = WebSocketConnectionManager()


@router.get("/status", response_model=MonitoringSnapshot)
async def get_latest_status(
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringSnapshot:
    return monitoring_service.get_latest_snapshot()


@router.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket) -> None:
    monitoring_service: MonitoringService = websocket.app.state.monitoring_service
    settings: Settings = websocket.app.state.settings

    await connection_manager.connect(websocket)
    logger.info("websocket_connected", extra={"active_clients": await connection_manager.connection_count()})

    try:
        while True:
            try:
                # Receive message with timeout to allow sending periodic snapshots
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                
                if "frame" in message:
                    # Decode base64 frame
                    frame_data = message["frame"]
                    if "," in frame_data:
                        frame_data = frame_data.split(",")[1]
                    
                    header_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(header_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        snapshot = monitoring_service.process_frame(frame)
                        monitoring_service.update_snapshot(snapshot)
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                logger.warning("frame_processing_failed", extra={"error": str(exc)})

            # Send latest snapshot
            payload = monitoring_service.get_latest_snapshot().model_dump(mode="json")
            await websocket.send_json(payload)
            await asyncio.sleep(settings.ws_interval_seconds)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as exc:
        logger.exception("websocket_stream_failed", extra={"error": str(exc)})
        await websocket.close(code=1011)
    finally:
        await connection_manager.disconnect(websocket)
        logger.info(
            "websocket_cleanup_done",
            extra={"active_clients": await connection_manager.connection_count()},
        )

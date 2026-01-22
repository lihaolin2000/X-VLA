from typing import Any, Dict
import logging
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import json_numpy
import msgpack
import msgpack_numpy as m
from abc import ABC, abstractmethod
m.patch()

class ModelServer(ABC):
    def __init__(self):
        self.app: FastAPI | None = None

    @abstractmethod
    def inference_api(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Abstract method for model inference API.

        Parameters
        ----------
        payload : Dict[str, Any]
            The input payload for inference.

        Returns
        -------
        Dict[str, Any]
            The inference result.
        """
        pass


    def _build_app(self, **infer_kwargs):
        """
        Minimal FastAPI app for XVLA inference.
        kwargs are passed to inference_api.
        """
        if self.app is not None: return
        app = FastAPI()
        
        # ODL VERSION With Json Response
        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                for key, value in payload.items():
                    if isinstance(value, (str, bytes)):
                        try: payload[key] = json_numpy.loads(value)
                        except Exception: pass
                action = self.inference_api(payload, **infer_kwargs)
                return JSONResponse({"action": action.tolist()})
            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        @app.websocket("/act")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_bytes(msgpack.packb({"type": "welcome", "ok": True}, 
                                                     use_bin_type=True))
            try:
                while True:
                    data = await websocket.receive_bytes()
                    payload = msgpack.unpackb(data, raw=False)
                    try: action_pred = self.inference_api(payload, **infer_kwargs)
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        response = {"error": f"Inference failed: {e}"}
                        await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
                        continue
                    # 4. Pack & Send Response
                    response = {"action": action_pred}
                    await websocket.send_bytes(msgpack.packb(response, use_bin_type=True))
            except WebSocketDisconnect:
                logging.info("WS disconnected")
            except Exception:
                logging.error(traceback.format_exc())
        self.app = app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """
        Launch the FastAPI service.
        """
        logging.info(f"🚀 XVLAServer listening on http://{host}:{port}/act")
        logging.info(f"🚀 XVLAServer listening on ws://{host}:{port}/act")
        self._build_app(**kwargs)
        assert self.app is not None
        uvicorn.run(self.app, 
                    host=host, 
                    port=port, 
                    log_level="info",
                    ws_ping_interval=20,
                    ws_ping_timeout=20)
        




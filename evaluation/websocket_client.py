import numpy as np
import threading
import logging
import msgpack
import msgpack_numpy as m
import asyncio
import websockets
from typing import Optional, List, Dict
m.patch()
logger = logging.getLogger(__name__)

# ==============================================================================
# === 1. Action Buffer (Pure Control Logic)                                  ===
# ==============================================================================
class ActionBuffer:
    """
    Thread-safe buffer for Receding Horizon Control (RHC).
    Lightweight: Only stores action arrays and timestamps. No images.
    """
    def __init__(self, 
                 action_dim: int = 20,
                 merge_strategy: str = "average", 
                 merge_weight: float = 0.5):
        assert merge_strategy in ["replace", "average"], "unsupport merge strategy"
        self.merge_strategy = merge_strategy
        self.merge_weight = merge_weight
        self.action_dim = action_dim
        self._lock = threading.Lock()
        
        # Internal State
        self.action_plan: Dict[int, np.ndarray] = {}
        self.current_time = 0

    def reset(self):
        """Resets the buffer state."""
        with self._lock:
            self.action_plan = {}
            self.current_time = 0

    def add_actions(self, actions: List[np.ndarray] | np.ndarray, timestamp: int):
        """
        Integrates new inference results with temporal alignment.
        """
        if isinstance(actions, np.ndarray):
            actions = [actions[i] for i in range(actions.shape[0])]
        try:
            assert all([act.shape[-1] == self.action_dim for act in actions]), "action dimension mismatch"
        except Exception as e:
            logger.error(f"ActionBuffer add_actions error: {e}")
            return
        with self._lock:
            # Temporal Ensembling
            for i, new_act in enumerate(actions):
                idx = timestamp + i
                if idx not in self.action_plan: self.action_plan[idx] = new_act
                else:
                    if self.merge_strategy == "replace":
                        self.action_plan[idx] = new_act
                    elif self.merge_strategy == "average":
                        curr = self.action_plan[idx]
                        self.action_plan[idx] = curr * (1 - self.merge_weight) + new_act * self.merge_weight
                    else:
                        raise RuntimeError("Unsupported merge strategy")

    def step(self) -> Optional[np.ndarray]:
        """Consumes one action step. Called by Robot Control Loop."""
        with self._lock:
            try:
                assert self.current_time in self.action_plan, "No action for current time"
            except Exception as e:
                logger.warning(f"ActionBuffer step warning: {e}")
                return None
            current_action = self.action_plan[self.current_time]
            self.current_time += 1
            return current_action
    
    def snapshot(self) -> np.ndarray:
        """export the current buffer state."""
        with self._lock:
            times = sorted(self.action_plan.keys())
            actions = np.stack([self.action_plan[t] for t in times], axis=0)
        return actions

# ==============================================================================
# === 2. Network Client (Pure I/O)                                           ===
# ==============================================================================
class WebSocketClient:
    def __init__(self, action_buffer: ActionBuffer, host: str, port: int):
        self.url = f"ws://{host}:{port}/act"
        self.buffer = action_buffer
        self.loop = asyncio.new_event_loop()
        self.websocket = None
        self._future = None
        threading.Thread(target=self._run_loop, daemon=True).start()
        try: self._sync(self._connect())
        except: logger.error("Connect failed")

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _sync(self, coro): return asyncio.run_coroutine_threadsafe(coro, self.loop).result()
    def _async(self, coro): return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _connect(self):
        if self.websocket and self.websocket.open: return
        self.websocket = await websockets.connect(self.url, max_size=None)
        await self.websocket.recv() # Handshake

    async def _send(self, payload, timestamp: int):
        try:
            if not self.websocket or self.websocket.closed: await self._connect()
            await self.websocket.send(msgpack.packb(payload, use_bin_type=True))
            resp = msgpack.unpackb(await self.websocket.recv(), raw=False)
            if "action" in resp: self.buffer.add_actions(resp["action"], timestamp)
        except Exception: self.websocket = None

    def get_action(self): return self.buffer.step()
    
    def update(self, payload: Dict, sync=False):
        """
        Sends observation to server.
        """
        if sync: self._sync(self._send(payload, self.buffer.current_time))
        else:
            if self._future and not self._future.done(): return # Drop frame
            self._future = self._async(self._send(payload, self.buffer.current_time))


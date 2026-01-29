import numpy as np
import threading
import logging
import msgpack
import msgpack_numpy as m
import asyncio
import websockets
import json_numpy
from typing import Optional, List, Dict
m.patch()
logger = logging.getLogger(__name__)



# ==============================================================================
# === 1. Action Buffer (Pure Control Logic)                                  ===
# ==============================================================================
class ActionBuffer:
    """
    High-performance Ring Buffer for Receding Horizon Control (RHC).
    """
    def __init__(self, 
                 action_dim: int = 20, 
                 max_steps: int = 256, 
                 merge_weight: float = 1.0):
        
        self.action_dim = action_dim
        self.capacity = max_steps
        self.merge_weight = merge_weight
        self._lock = threading.Lock()
        # Pre-allocated Memory (Zero GC)
        self.buffer = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.valid_mask = np.zeros(self.capacity, dtype=bool)
        
        # Time tracking
        self.current_time = 0
        self.min_valid_time = float('inf')
        self.max_valid_time = float('-inf')

    def reset(self):
        """Resets the buffer state instantly."""
        with self._lock:
            self.buffer.fill(0)
            self.valid_mask.fill(False)
            self.current_time = 0
            self.min_valid_time = float('inf')
            self.max_valid_time = float('-inf')
    def add_actions(self, actions: np.ndarray, timestamp: int):
        """
        Integrates new predictions into the ring buffer with temporal ensembling.
        """
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)
        
        num_steps = actions.shape[0]

        with self._lock:
            # 1. Update Time Boundaries
            self.min_valid_time = min(self.min_valid_time, timestamp)
            self.max_valid_time = max(self.max_valid_time, timestamp + num_steps)
            self.min_valid_time = max(self.min_valid_time, self.max_valid_time - self.capacity)

            # 2. Calculate Ring Indices
            indices = (np.arange(num_steps) + timestamp) % self.capacity

            # 3. Vectorized Merge (Average Strategy)


            existing_mask = self.valid_mask[indices]
            # Update existing slots: old * (1 - w) + new * w
            if np.any(existing_mask):
                idx_exist = indices[existing_mask]
                self.buffer[idx_exist] = (
                    self.buffer[idx_exist] * (1 - self.merge_weight) + 
                    actions[existing_mask] * self.merge_weight
                )
            # Write to new slots
            if np.any(~existing_mask):
                idx_new = indices[~existing_mask]
                self.buffer[idx_new] = actions[~existing_mask]
                self.valid_mask[idx_new] = True

    def step(self):
        """
        Consumes one action for the current time step.
        """
        with self._lock:
            idx = self.current_time % self.capacity
            # Return None if no valid action exists for current time
            if not self.valid_mask[idx]: return None
            action = self.buffer[idx].copy()
            self.current_time += 1
            return action

    def left_valid_time(self):
        '''
        Returns the number of steps left to the end of the buffer.
        '''
        return self.max_valid_time - self.current_time

    def snapshot(self):
        """
        Export all valid actions currently in the buffer.
        Returns: (actions_numpy, start_timestamp)
        """
        with self._lock:
            if self.max_valid_time == float('-inf'): return np.empty((0, self.action_dim)), 0
            start_time = int(self.min_valid_time)
            end_time = int(self.max_valid_time)
            
            if start_time >= end_time: return np.empty((0, self.action_dim)), 0
            start_idx = start_time % self.capacity
            end_idx = end_time % self.capacity
            if start_idx < end_idx:
                data = self.buffer[start_idx:end_idx].copy()
            else:
                part1 = self.buffer[start_idx:]
                part2 = self.buffer[:end_idx]
                data = np.concatenate((part1, part2), axis=0)
            return data, start_time
# ==============================================================================
# === 2. Network Client (Pure I/O)                                           ===
# ==============================================================================



class HttpClient:
    def __init__(self, action_buffer: ActionBuffer, host: str, port: int):
        import requests
        self.url = f"http://{host}:{port}/act"
        self.buffer = action_buffer
        self.session = requests.Session()

    def get_action(self): return self.buffer.step()
    
    def update(self, payload: Dict):
        """
        Sends observation to server.
        """
        try:
            for key in payload.keys():
                if isinstance(payload[key], np.ndarray):
                    payload[key] = json_numpy.dumps(payload[key])
            resp = self.session.post(self.url, json=payload, timeout=5.0)
            resp.raise_for_status()
            resp_json = resp.json()
            if "error" in resp_json:
                print(f"[HTTP] server error: {resp_json['error']}", flush=True)
                return
            if "action" in resp_json:
                self.buffer.add_actions(np.asarray(resp_json["action"]), self.buffer.current_time)
        except Exception as e:
            import traceback
            print("[HTTP] exception:", repr(e), flush=True)
            print(traceback.format_exc(), flush=True)




class WebSocketClient:
    def __init__(self, action_buffer: ActionBuffer, host: str, port: int):
        self.url = f"ws://{host}:{port}/act"
        self.buffer = action_buffer
        self.loop = asyncio.new_event_loop()
        self.websocket = None
        self._future = None
        threading.Thread(target=self._run_loop, daemon=True).start()
        self._sync(self._connect())

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _sync(self, coro): return asyncio.run_coroutine_threadsafe(coro, self.loop).result()
    def _async(self, coro): return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _connect(self):
        if self.websocket: return
        self.websocket = await websockets.connect(self.url, max_size=None)
        await self.websocket.recv() # Handshake

    async def _send(self, payload, timestamp: int):
        try:
            if not self.websocket: await self._connect()
            await self.websocket.send(msgpack.packb(payload, use_bin_type=True))

            resp_raw = await self.websocket.recv()
            resp = msgpack.unpackb(resp_raw, raw=False)
            if resp.get("type") == "welcome":
                resp = msgpack.unpackb(await self.websocket.recv(), raw=False)

            if "error" in resp:
                print(f"[WS] server error: {resp['error']}", flush=True)
                return

            if "action" in resp:
                self.buffer.add_actions(np.asarray(resp["action"]), timestamp)

        except Exception as e:
            import traceback
            print("[WS] exception:", repr(e), flush=True)
            print(traceback.format_exc(), flush=True)
            self.websocket = None

    def get_action(self): return self.buffer.step()
    
    def update(self, payload: Dict, sync=False):
        """
        Sends observation to server.
        """
        if sync: return self._sync(self._send(payload, self.buffer.current_time))
        else:
            if self._future and not self._future.done(): return "skip" # Drop frame
            self._future = self._async(self._send(payload, self.buffer.current_time))

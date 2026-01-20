import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patheffects as pe
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
    def __init__(self, merge_strategy: str = "average", merge_weight: float = 0.5):
        self.merge_strategy = merge_strategy
        self.merge_weight = merge_weight
        self._lock = threading.Lock()
        
        # Internal State
        self.action_plan: List[np.ndarray] = []
        self.current_time = 0

    def reset(self):
        """Resets the buffer state."""
        with self._lock:
            self.action_plan = []
            self.current_time = 0

    def add_actions(self, actions: List[np.ndarray], timestamp: int):
        """
        Integrates new inference results with temporal alignment.
        """
        if not actions: return
        with self._lock:
            # Latency Compensation logic
            elapsed = self.current_time - timestamp
            
            if elapsed > 0:
                # Packet arrived late, discard past actions
                if elapsed >= len(actions): return
                actions = actions[elapsed:]
                offset = 0
            else:
                # Packet is a forecast
                offset = -elapsed

            # Temporal Ensembling
            for i, new_act in enumerate(actions):
                idx = offset + i
                if idx >= len(self.action_plan):
                    self.action_plan.append(new_act)
                else:
                    if self.merge_strategy == "replace":
                        self.action_plan[idx] = new_act
                    else: # average
                        curr = self.action_plan[idx]
                        self.action_plan[idx] = curr * (1 - self.merge_weight) + new_act * self.merge_weight

    def step(self) -> Optional[np.ndarray]:
        """Consumes one action step. Called by Robot Control Loop."""
        with self._lock:
            if not self.action_plan: return None
            self.current_time += 1
            return self.action_plan.pop(0)
    
    def get_snapshot(self) -> np.ndarray:
        """Returns a snapshot for visualization (Thread-safe copy)."""
        with self._lock:
            if not self.action_plan: return np.empty((0, 0))
            return np.array(self.action_plan)

# ==============================================================================
# === 2. Auto Visualizer (Decoupled Rendering)                               ===
# ==============================================================================
class AutoVisualizer:
    """
    High-Performance Dashboard.
    - Images are pushed explicitly via update_image() (Bypassing ActionBuffer).
    - Actions are pulled from ActionBuffers.
    """
    # Fixed Palette
    NEON_PALETTE = ['#00f0ff', '#ff00ff', '#fcee0a', '#00ff41', '#ff4d00']
    
    def __init__(self, buffers: Dict[str, ActionBuffer], horizon=50, interval=20):
        self.buffers = buffers
        self.horizon = horizon
        self.interval = interval
        
        # State
        self.is_initialized = False
        self._dim = 0
        
        # Image State (Independent of Action Buffer)
        self._img_lock = threading.Lock()
        self.latest_image: Optional[np.ndarray] = None
        
        # UI Components
        plt.style.use('dark_background')
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['font.family'] = 'monospace'
        
        self.fig = plt.figure(figsize=(16, 9), facecolor='#080808')
        self.fig.canvas.manager.set_window_title("VLA Telemetry System")
        
        self.lines_map: List[Dict[str, plt.Line2D]] = [] 
        self.axes: List[plt.Axes] = []
        self.im_display = None
        self.txt_status = self.fig.text(0.5, 0.5, "INITIALIZING...", color='#00ff41', fontsize=16)

    def update_image(self, image_np: np.ndarray):
        """
        Explicitly update the camera feed.
        Call this from your loop, e.g., every 5 steps.
        """
        if image_np is None: return
        with self._img_lock:
            # Copy to prevent data race if capture thread overwrites buffer
            self.latest_image = image_np.copy()

    def start(self, robot_loop_callback):
        threading.Thread(target=robot_loop_callback, daemon=True, name="RobotLogic").start()
        animation.FuncAnimation(self.fig, self._update_frame, interval=self.interval, blit=False, cache_frame_data=False)
        plt.show()

    def _init_canvas(self, dim: int):
        self._dim = dim
        self.txt_status.set_visible(False)
        self.fig.clf()

        # Layout: [Cam 40%] | [Charts 60%]
        gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[1, 1.2], wspace=0.15)
        
        # 1. Camera Panel
        ax_img = self.fig.add_subplot(gs[0])
        ax_img.axis('off')
        ax_img.set_title("[ OPTICAL FEED ]", color='#888888', fontsize=10, pad=10)
        self.im_display = ax_img.imshow(np.zeros((224, 224, 3), dtype=np.uint8), aspect='auto')

        # 2. Charts Panel
        cols = int(np.ceil(np.sqrt(dim)))
        rows = int(np.ceil(dim / cols))
        gs_charts = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs[1], wspace=0.25, hspace=0.4)

        self.lines_map = []
        self.axes = []
        buffer_names = list(self.buffers.keys())

        for i in range(dim):
            ax = self.fig.add_subplot(gs_charts[i])
            ax.set_facecolor('#0f0f0f')
            ax.grid(True, color='#333333', linestyle=':', linewidth=0.5)
            ax.axhline(0, color='white', linestyle='--', linewidth=0.5, alpha=0.15)
            ax.text(0.02, 0.9, f"DIM_{i:02d}", transform=ax.transAxes, color='#666666', fontsize=8, weight='bold')
            ax.set_xlim(0, self.horizon)
            ax.tick_params(colors='#666666', labelsize=7)
            if i < dim - cols: ax.set_xticklabels([])

            lines_in_dim = {}
            for idx, name in enumerate(buffer_names):
                color = self.NEON_PALETTE[idx % len(self.NEON_PALETTE)]
                line, = ax.plot([], [], label=name, color=color, lw=1.5)
                line.set_path_effects([pe.SimpleLineShadow(rho=2, alpha=0.3), pe.Normal()])
                lines_in_dim[name] = line
            
            if i == 0: ax.legend(loc='upper right', fontsize=6, facecolor='#080808', edgecolor='none', labelcolor='white')
            self.lines_map.append(lines_in_dim)
            self.axes.append(ax)
        
        self.is_initialized = True

    def _smart_scale(self, ax, data_list):
        valid = [d for d in data_list if d.size > 0]
        if not valid: return
        comb = np.concatenate(valid)
        if len(comb) < 2: return
        
        mn, mx = np.min(comb), np.max(comb)
        span = max(mx - mn, 0.1)
        margin = span * 0.15
        mid = (mx + mn) / 2
        ax.set_ylim(mid - span/2 - margin, mid + span/2 + margin)
        ax.set_xlim(0, self.horizon)

    def _update_frame(self, _):
        """Rendering Loop."""
        # 1. Update Image (From local state, not Buffer)
        with self._img_lock:
            if self.latest_image is not None and self.im_display:
                self.im_display.set_data(self.latest_image)

        # 2. Update Charts (From Buffers)
        snapshots = {k: v.get_snapshot() for k, v in self.buffers.items()}
        primary_key = list(self.buffers.keys())[0]
        if snapshots[primary_key].shape[0] == 0: return []

        _, dim = snapshots[primary_key].shape
        if not self.is_initialized:
            self._init_canvas(dim)
            return []
        if dim != self._dim: return []

        artists = []
        if self.im_display: artists.append(self.im_display)

        for i in range(dim):
            scaling_data = []
            for name, snap in snapshots.items():
                line = self.lines_map[i][name]
                if snap.shape[0] > 0 and i < snap.shape[1]:
                    d = snap[:, i]
                    line.set_data(np.arange(len(d)), d)
                    scaling_data.append(d)
                    artists.append(line)
            self._smart_scale(self.axes[i], scaling_data)
        
        return artists

# ==============================================================================
# === 3. Network Client (Pure I/O)                                           ===
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

    async def _send(self, payload: bytes, ts: int):
        try:
            if not self.websocket or self.websocket.closed: await self._connect()
            await self.websocket.send(payload)
            resp = msgpack.unpackb(await self.websocket.recv(), raw=False)
            if "action" in resp: self.buffer.add_actions(resp["action"], ts)
        except Exception: self.websocket = None

    def get_action(self): return self.buffer.step()
    
    def update(self, obs: Dict, sync=False):
        """
        Sends observation to server.
        NOTE: Does NOT handle visualization. Pass image to visualizer manually.
        """
        ts = self.buffer.current_time
        
        # Optimization: Flatten images for MsgPack
        payload = {k:v for k,v in obs.items() if k != "image_list"}
        for i, img in enumerate(obs.get("image_list", [])): payload[f"image{i}"] = img
        
        packed = msgpack.packb(payload, use_bin_type=True)
        
        if sync:
            self._sync(self._send(packed, ts))
        else:
            if self._future and not self._future.done(): return # Drop frame
            self._future = self._async(self._send(packed, ts))
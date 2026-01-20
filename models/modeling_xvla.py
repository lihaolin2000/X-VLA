# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations



from typing import Any, Dict, List
import torch

import numpy as np
from PIL import Image
from fastapi import FastAPI
import cv2

from transformers import PreTrainedModel
from .server import ModelServer
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig


class XVLA(PreTrainedModel, ModelServer):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Florence2 encoder-only backbone (vision-language)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # Florence2 backbone (encoder only)
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # Temporal/action head
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)         # [B*V]
        flat_images = pixel_values.flatten(0, 1)                # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        valid_feats = self.vlm._encode_image(valid_images)      # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)

    # =============================== FastAPI service =============================


    def inference_api(self, processor, payload: Dict[str, Any]) -> np.ndarray:
        """
        XVLA inference supporting:
        - Single sample: payload is a dict of scalars/arrays.
        - Grouped batch: payload is a dict where some fields are list/tuple of length B.
            In grouped batch mode, ALL list/tuple fields must share the same length B.

        Returns:
        - (T, D) for single sample
        - (B, T, D) for grouped batch
        """

        # -------------------------
        # 1) Normalize payload -> List[Dict[str, Any]]
        # -------------------------
        denoiseing_steps = payload.pop("steps", 10)
        batch_fields = [k for k, v in payload.items() if isinstance(v, (list, tuple))]
        if not batch_fields:
            batch_payloads: List[Dict[str, Any]] = [payload]
            batch_size = 1
        else:
            lengths = {k: len(payload[k]) for k in batch_fields}
            if len(set(lengths.values())) != 1: raise ValueError(f"Grouped batch size mismatch among fields: {lengths}")
            batch_size = next(iter(lengths.values()))
            batch_payloads = [
                {k: (payload[k][i] if k in batch_fields else payload[k]) for k in payload}
                for i in range(batch_size)
            ]
        # -------------------------
        # 2) Utilities
        # -------------------------
        def move_to_device(x: Any) -> torch.Tensor:
            """Convert to tensor and move to model device/dtype."""
            tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            if tensor.is_floating_point():
                return tensor.to(device=self.device, dtype=self.dtype)
            return tensor.to(device=self.device)

        def decode_image_list(sample: Dict[str, Any]) -> List[Image.Image]:
            """Decode image0/image1/... from np.ndarray into PIL Images."""
            images: List[Image.Image] = []
            idx = 0
            while f"image{idx}" in sample:
                arr = sample[f"image{idx}"]
                if not isinstance(arr, np.ndarray): raise ValueError(f"image{idx} must be np.ndarray, got {type(arr)}")
                if arr.ndim == 1:  # encoded buffer
                    arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if arr is None: raise ValueError(f"cv2.imdecode failed for image{idx}")
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(arr))
                idx += 1
            if not images:
                raise ValueError("Missing images: expected keys image0, image1, ...")
            return images

        # -------------------------
        # 3) Per-sample preprocessing + strict collation (no padding)
        # -------------------------
        processor_outputs: List[Dict[str, Any]] = []
        proprio_batch: List[torch.Tensor] = []
        domain_id_list: List[int] = []

        for sample in batch_payloads:
            images = decode_image_list(sample)
            processor_outputs.append(processor(images, sample["language_instruction"]))
            proprio_batch.append(move_to_device(sample["proprio"]))
            domain_id_list.append(int(sample.get("domain_id", 0)))

        model_inputs = {
            k: torch.stack([move_to_device(out[k]) for out in processor_outputs], dim=0)
            for k in processor_outputs[0].keys()
        }
        model_inputs.update(
            proprio=torch.stack(proprio_batch, dim=0),  # (B, state_dim)
            domain_id=torch.tensor(domain_id_list, dtype=torch.long, device=self.device),  # (B,)
            steps=denoiseing_steps,  # one scalar for whole batch
        )
        # -------------------------
        # 4) Inference
        # -------------------------
        self.eval()
        with torch.inference_mode():
            actions = self.generate_actions(**model_inputs)  # expected: (B, T, D)
        actions_np = actions.float().cpu().numpy()
        return actions_np[0] if batch_size == 1 else actions_np
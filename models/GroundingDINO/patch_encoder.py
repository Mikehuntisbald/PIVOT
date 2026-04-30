# ------------------------------------------------------------------------
# Patch Encoder
# ------------------------------------------------------------------------
# A lightweight encoder for exemplar patches that mirrors the GroundingDINO
# visual encoder output format. It produces patch tokens, a global descriptor,
# and an optional text-driven gate (alpha) so the downstream scorer can downweight
# patch evidence for queries like "knee"/"leftmost" where local appearance is
# ambiguous (see CVPR - Main.pdf notes).
# ------------------------------------------------------------------------

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list


class PatchEncoder(nn.Module):
    """
    Encode one or more exemplar patches into the same hidden dimension used by the
    GroundingDINO transformer (d_model). The encoder is intentionally lightweight
    and reuses the provided visual backbone so the patch branch learns "what this
    object looks like" while the text branch handles fine-grained parts/relations.

    The optional text gate follows the PDF suggestion:
        score = alpha(text) * score_patch + (1 - alpha(text)) * score_text
    where alpha is predicted from the text tokens, letting the model lower
    patch influence on part/relational prompts.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 256,
        gate_with_text: bool = True,
        max_text_len: int = 256,
    ) -> None:
        """
        Args:
            backbone: A visual Joiner (backbone + position embedding) that returns
                (features, pos) like the main GroundingDINO backbone.
            hidden_dim: Output embedding dimension (should match transformer.d_model).
            gate_with_text: If True, expose a text-driven gate alpha(text).
            max_text_len: Used only for type hints; gate computation respects the
                provided masks rather than this limit.
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.gate_with_text = gate_with_text
        self.max_text_len = max_text_len

        # Project the last backbone feature map to d_model.
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(hidden_dim)

        if gate_with_text:
            self.text_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.text_gate = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_text(self, text_dict: Dict[str, Tensor]) -> Optional[Tensor]:
        """Return pooled text embedding for gating (masked mean over tokens)."""
        if (not self.gate_with_text) or text_dict is None:
            return None
        encoded = text_dict.get("encoded_text", None)
        mask = text_dict.get("text_token_mask", None)
        if encoded is None or mask is None:
            return None

        valid = mask.unsqueeze(-1)  # bs, L, 1
        pooled = (encoded * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.norm(pooled)

    def forward(
        self,
        patches: Union[NestedTensor, List[Tensor], Tensor],
        text_dict: Optional[Dict[str, Tensor]] = None,
        return_tokens: bool = True,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Args:
            patches: Either a NestedTensor or a list/tensor of RGB patch crops
                shaped [3, H, W] or [B, 3, H, W].
            text_dict: Optional text features/masks used to compute alpha(text).
        Returns:
            {
                "patch_tokens": (bs, L, hidden_dim),
                "patch_pos":    (bs, L, hidden_dim),
                "patch_mask":   (bs, L) bool, True for valid tokens,
                "patch_global": (bs, hidden_dim),
                "patch_gate":   (bs,) in [0,1] or None when disabled.
            }
        """
        if patches is None:
            return {
                "patch_tokens": None,
                "patch_pos": None,
                "patch_mask": None,
                "patch_global": None,
                "patch_gate": None,
            }

        # Multi-patch case: [B,K,3,H,W]. Encode as (B*K) patches, then reshape back.
        is_multi = False
        B = K = None
        if isinstance(patches, torch.Tensor) and patches.dim() == 5:
            is_multi = True
            B, K, C, H, W = patches.shape
            if int(C) != 3:
                raise ValueError(f"Expected RGB patches with C=3, got C={C}")
            patches_flat = patches.view(int(B) * int(K), int(C), int(H), int(W))
            patch_tensor = nested_tensor_from_tensor_list(list(patches_flat))
        else:
            if isinstance(patches, NestedTensor):
                patch_tensor = patches
            elif isinstance(patches, torch.Tensor):
                if patches.dim() == 3:
                    patch_tensor = nested_tensor_from_tensor_list([patches])
                elif patches.dim() == 4:
                    patch_tensor = nested_tensor_from_tensor_list(list(patches))
                else:
                    raise ValueError(f"Unexpected patch tensor dim: {patches.dim()}")
            elif isinstance(patches, (list, tuple)):
                patch_tensor = nested_tensor_from_tensor_list(list(patches))
            else:
                raise TypeError(f"Unsupported patches type: {type(patches)}")

        features, pos = self.backbone(patch_tensor)
        src, mask = features[-1].decompose()  # use highest-level feature

        src = self.input_proj(src)
        patch_tokens = patch_pos = patch_mask = None
        if return_tokens:
            pos_embed = pos[-1]
            # Flatten spatial tokens; mask is True for padding, so invert to mark valid.
            patch_tokens = src.flatten(2).transpose(1, 2)
            patch_pos = pos_embed.flatten(2).transpose(1, 2)
            patch_mask = (~mask).flatten(1)

        patch_global = self.pool(src).flatten(1)
        patch_global = self.norm(patch_global)

        gate = None
        pooled_text = self._encode_text(text_dict)
        if pooled_text is not None and self.text_gate is not None:
            gate = self.text_gate(pooled_text).squeeze(-1)

        # Reshape back for multi-patch.
        if is_multi:
            assert B is not None and K is not None
            patch_global = patch_global.view(int(B), int(K), -1)
            if patch_tokens is not None:
                patch_tokens = patch_tokens.view(int(B), int(K), patch_tokens.shape[1], patch_tokens.shape[2])
            if patch_pos is not None:
                patch_pos = patch_pos.view(int(B), int(K), patch_pos.shape[1], patch_pos.shape[2])
            if patch_mask is not None:
                patch_mask = patch_mask.view(int(B), int(K), patch_mask.shape[1])
            if gate is not None:
                gate = gate.view(int(B), 1).expand(int(B), int(K))

        return {
            "patch_tokens": patch_tokens,
            "patch_pos": patch_pos,
            "patch_mask": patch_mask,
            "patch_global": F.normalize(patch_global, dim=-1),
            "patch_gate": gate,
        }

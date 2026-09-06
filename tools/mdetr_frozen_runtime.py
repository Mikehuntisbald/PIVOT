"""Pinned official MDETR EMA runtime with an explicit Native query identity.

No upstream source files are modified. Construction skips pretrained downloads
because strict EMA loading replaces every parameter. Modern Roberta is forced
to the eager attention implementation; the exact environment is receipted.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

PINNED_COMMIT = "ea09acc44ca067072c4b143b726447ee7ff66f5f"
CHECKPOINT_MD5 = "3219e03af7709cd15ab0d0db521b9070"
LOCALIZER = "mdetr_r101_refcoco_ema"


def file_digest(path, algorithm="sha256"):
    out = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            out.update(block)
    return out.hexdigest()


def cxcywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - .5 * w, y - .5 * h, x + .5 * w, y + .5 * h), -1)


def official_native_index(scores, boxes_xyxy_abs, mask=None):
    """Exactly the official RefExpEvaluator score/box tuple ordering.

    Python stable sorting keeps the earliest query when both keys coincide.
    No quantization, thresholding, clipping, NMS or torch argmax tie shortcut.
    """
    if scores.ndim != 1 or boxes_xyxy_abs.shape != (len(scores), 4):
        raise ValueError("Native selection requires Q scores and Qx4 pixel boxes")
    if not torch.isfinite(scores).all() or not torch.isfinite(boxes_xyxy_abs).all():
        raise ValueError("nonfinite Native output")
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != scores.shape or not mask.any():
        raise ValueError("empty or malformed Native candidate mask")
    values = scores.detach().cpu().tolist()
    boxes = boxes_xyxy_abs.detach().cpu().tolist()
    valid = mask.detach().cpu().tolist()
    return sorted((i for i, ok in enumerate(valid) if ok), key=lambda i: (values[i], boxes[i]), reverse=True)[0]


def official_resize_shape(width, height, short_side=800, max_size=1333):
    if min(width, height) <= 0:
        raise ValueError("invalid image size")
    size = short_side
    if max(width, height) / min(width, height) * size > max_size:
        size = int(round(max_size * min(width, height) / max(width, height)))
    if (width <= height and width == size) or (height <= width and height == size):
        return height, width
    if width < height:
        return int(size * height / width), size
    return size, int(size * width / height)


def preprocess(image):
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode
    image = image.convert("RGB")
    shape = official_resize_shape(*image.size)
    resized = F.resize(image, list(shape), interpolation=InterpolationMode.BILINEAR)
    return F.normalize(F.to_tensor(resized), [0.485, .456, .406], [.229, .224, .225])


def require_ema(checkpoint):
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_ema"), dict):
        raise ValueError("official model_ema is mandatory; raw model fallback forbidden")
    state = checkpoint["model_ema"]
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("malformed EMA state")
    if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError("EMA contains nonfinite tensors")
    return state


@dataclass(frozen=True)
class MDETRHookBatch:
    query_features: torch.Tensor
    native_score: torch.Tensor
    boxes: torch.Tensor
    candidate_mask: torch.Tensor
    native_selected_index: int
    image_size: tuple[int, int]
    native_boxes_xyxy_abs: torch.Tensor


class MDETRFrozenRuntime:
    def __init__(self, *, upstream_root, checkpoint_path, text_assets, device="cuda:0", expected_checkpoint_sha256=None):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torchvision
        import transformers
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
        self.upstream_root = Path(upstream_root).resolve(strict=True)
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        text_assets = Path(text_assets).resolve(strict=True)
        text_receipt = json.loads((text_assets / "receipt.json").read_text())
        if text_receipt.get("revision") != "e2da8e2f811d1448a5b465c236feacd80ffbac7b":
            raise ValueError("RoBERTa revision drift")
        if {v["name"] for v in text_receipt.get("files", [])} != {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}:
            raise ValueError("RoBERTa asset list drift")
        for item in text_receipt["files"]:
            if file_digest(text_assets / item["name"]) != item["sha256"]:
                raise ValueError("RoBERTa tokenizer/config hash drift")
        commit = subprocess.check_output(["git", "-C", str(self.upstream_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(self.upstream_root), "status", "--porcelain", "--untracked-files=no"], text=True)
        if commit != PINNED_COMMIT or dirty:
            raise ValueError("MDETR pinned upstream source drift")
        if file_digest(checkpoint_path, "md5") != CHECKPOINT_MD5:
            raise ValueError("official MDETR MD5 mismatch")
        ckpt_sha = file_digest(checkpoint_path)
        if expected_checkpoint_sha256 is not None and ckpt_sha != expected_checkpoint_sha256:
            raise ValueError("MDETR checkpoint SHA drift")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("formal MDETR runtime requires explicit CUDA device")
        torch.manual_seed(20260911)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        if "util" in sys.modules and not str(getattr(sys.modules["util"], "__file__", "")).startswith(str(self.upstream_root)):
            raise ValueError("another project's util module is loaded; use independent process")
        sys.path.insert(0, str(self.upstream_root))
        config = RobertaConfig.from_pretrained(str(text_assets), local_files_only=True)
        config._attn_implementation = "eager"
        tokenizer = RobertaTokenizerFast.from_pretrained(str(text_assets), local_files_only=True)
        resnet = torchvision.models.resnet101
        def no_download_resnet(*args, **kwargs):
            kwargs.pop("pretrained", None)
            kwargs["weights"] = None
            return resnet(*args, **kwargs)
        spec = importlib.util.spec_from_file_location("arrow_pinned_mdetr_hub", self.upstream_root / "hubconf.py")
        hub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hub)
        with patch.object(torchvision.models, "resnet101", no_download_resnet), patch.object(RobertaTokenizerFast, "from_pretrained", return_value=tokenizer), patch.object(RobertaModel, "from_pretrained", side_effect=lambda *a, **kw: RobertaModel(config)):
            self.model, self.postprocessor = hub.mdetr_resnet101(pretrained=False, return_postprocessor=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = require_ema(checkpoint)
        # HF 4.5 stored position_ids persistently; HF 4.40 registers the same
        # used buffer as nonpersistent. Restore only persistence, not its value
        # or operator, so the original full EMA loads with strict=True.
        compatibility = []
        for name, module in self.model.named_modules():
            for key in tuple(module._non_persistent_buffers_set):
                full_name = f"{name}.{key}" if name else key
                if full_name in state:
                    if key not in {"position_ids", "token_type_ids"}:
                        raise ValueError(f"unapproved legacy persistent buffer: {full_name}")
                    module._non_persistent_buffers_set.remove(key)
                    compatibility.append({"persistent_buffer": full_name})
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        if self.model.num_queries != 100 or self.model.query_embed.weight.shape != (100, 256):
            raise ValueError("official MDETR query geometry mismatch")
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("MDETR must be fully frozen")
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("MDETR model parameters must remain FP32")
        self._last_features = None
        self._last_logits = None
        self._handle = self.model.class_embed.register_forward_hook(self._capture)
        self.receipt = {"schema": "arrow.confidence_readout.mdetr_runtime/v1", "localizer": LOCALIZER,
                        "checkpoint": {"path": str(checkpoint_path), "sha256": ckpt_sha, "md5": CHECKPOINT_MD5},
                        "state_key": "model_ema", "strict_state_load": True, "state_tensors": len(state),
                        "upstream_commit": commit, "upstream_source_changed": False, "compatibility": compatibility,
                        "text_assets": {"path": str(text_assets / "receipt.json"), "sha256": file_digest(text_assets / "receipt.json")},
                        "construction_only_adapters": ["skip_resnet_pretrained_download", "instantiate_roberta_config_then_strict_ema"],
                        "torch": torch.__version__, "torchvision": torchvision.__version__, "transformers": transformers.__version__,
                        "text_attention": "eager", "model_dtype": "float32", "cache_features_dtype": "float16",
                        "query_count": 100, "feature_dim": 256, "all_frozen": True, "native_sort": "score_then_pixel_xyxy_python_stable_descending",
                        "metric_note": "study uses IoU>=0.5; upstream RefExpEvaluator aggregates GIoU"}
        del checkpoint, state

    def _capture(self, module, inputs, output):
        if len(inputs) != 1 or inputs[0].ndim != 4:
            raise ValueError("MDETR class_embed input must be layers,B,Q,256")
        self._last_features = inputs[0][-1].detach().clone()
        self._last_logits = output[-1].detach().clone()

    def _forward(self, tensor, caption):
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            memory = self.model(tensor, [caption], encode_and_save=True)
            outputs = self.model(tensor, [caption], encode_and_save=False, memory_cache=memory)
        return outputs

    def infer(self, image_path, caption):
        from PIL import Image
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("full expression must be nonempty")
        with Image.open(image_path) as image:
            width, height = image.size
            tensor = preprocess(image).unsqueeze(0).to(self.device)
        self._last_features = None
        outputs = self._forward(tensor, caption)
        features = self._last_features
        if features is None or features.shape != (1, 100, 256):
            raise ValueError("last-layer feature hook failed")
        with torch.inference_mode():
            post = self.postprocessor(outputs, torch.tensor([[height, width]], device=self.device))[0]
            scores = 1 - outputs["pred_logits"].softmax(-1)[0, :, -1]
            if not torch.equal(scores, post["scores"]):
                raise ValueError("official PostProcess score parity failed")
            pixel = cxcywh_to_xyxy(outputs["pred_boxes"])[0] * scores.new_tensor([width, height, width, height])
            if not torch.equal(pixel, post["boxes"]):
                raise ValueError("official PostProcess box parity failed")
            if not torch.equal(self._last_logits, outputs["pred_logits"]):
                raise ValueError("class_embed feature hook/logit parity failed")
        selected = official_native_index(scores, pixel)
        batch = MDETRHookBatch(features[0].cpu().half().contiguous(), scores.cpu().float().contiguous(),
                               outputs["pred_boxes"][0].cpu().float().contiguous(), torch.ones(100, dtype=torch.bool),
                               selected, (height, width), pixel.cpu().float().contiguous())
        for value in (batch.query_features, batch.native_score, batch.boxes):
            if value.requires_grad or not torch.isfinite(value).all():
                raise ValueError("nonfinite or connected frozen output")
        return batch

    def close(self):
        self._handle.remove()

from __future__ import annotations

import copy
import math
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .transformer import TransformerDecoder
from .utils import ContrastiveEmbed


ExpressionContextProvider = Callable[[List[str], Tensor], Dict[str, Any]]
CONFIDENCE_OUTPUT_MODES = ("base_plus_gate", "gate_only")
SCORE_OWNERSHIP_MODES = (
    "shared_score",
    "shared_trunk_two_heads",
    "independent_decoders_joint",
    "independent_decoders_two_phase",
    "rank_tower_stopgrad_token_adapter_two_phase",
)


def normalize_stage_b_score_ownership(value: Any) -> str:
    """Normalize the explicit Table-D ownership label, or keep legacy mode."""
    ownership = str(value or "").strip().lower().replace("-", "_")
    if ownership and ownership not in SCORE_OWNERSHIP_MODES:
        raise ValueError(
            "score_ownership must be empty (legacy) or one of "
            f"{SCORE_OWNERSHIP_MODES}, got {ownership!r}"
        )
    return ownership


def select_stage_b_rank_confidence_logits(
    outputs: Mapping[str, Tensor],
    *,
    score_ownership: str = "",
    legacy_decoupled_confidence: bool = False,
    legacy_validity_head: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    """Select training tensors under an explicit or historical score contract."""
    ownership = normalize_stage_b_score_ownership(score_ownership)
    try:
        phrase_logits = outputs["stage_b_v11_final_phrase_logits"]
    except KeyError as error:
        raise KeyError(
            "Stage-B fixed scorer output is missing final phrase logits"
        ) from error
    validity_logits = outputs.get("stage_b_v14_final_validity_logits")
    if ownership == "shared_score":
        return phrase_logits, phrase_logits
    if ownership:
        if validity_logits is None:
            raise KeyError(
                f"Stage-B ownership {ownership!r} requires final validity logits"
            )
        return phrase_logits, validity_logits
    if legacy_decoupled_confidence:
        if validity_logits is None:
            raise KeyError(
                "Legacy decoupled confidence requires final validity logits"
            )
        return phrase_logits, validity_logits
    if legacy_validity_head:
        if validity_logits is None:
            raise KeyError("Legacy validity-head scoring requires validity logits")
        return validity_logits, None
    return phrase_logits, None


def validate_stage_b_fixed_text_scorer_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Require an exact, shape-compatible scorer state for resume/evaluation."""
    root = model.module if hasattr(model, "module") else model
    scorer = getattr(root, "stage_b_fixed_text_scorer", None)
    if scorer is None:
        raise ValueError(
            f"{checkpoint_label}: stage_b_v11_fixed_text is enabled but the model "
            "has no stage_b_fixed_text_scorer"
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{checkpoint_label}: checkpoint model state must be a mapping")

    prefix = "stage_b_fixed_text_scorer."
    expected = {
        prefix + key: value for key, value in scorer.state_dict().items()
    }
    provided = {
        str(key): value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }
    missing = sorted(set(expected).difference(provided))
    unexpected = sorted(set(provided).difference(expected))
    shape_mismatches = []
    contract_mismatches = []
    for key in sorted(set(expected).intersection(provided)):
        expected_value = expected[key]
        provided_value = provided[key]
        if not torch.is_tensor(provided_value):
            shape_mismatches.append(
                (key, tuple(expected_value.shape), type(provided_value).__name__)
            )
            continue
        if tuple(provided_value.shape) != tuple(expected_value.shape):
            shape_mismatches.append(
                (key, tuple(expected_value.shape), tuple(provided_value.shape))
            )
            continue
        if (
            key.startswith(prefix + "_score_contract_")
            and not torch.equal(
                provided_value.detach().to(
                    device="cpu", dtype=expected_value.dtype
                ),
                expected_value.detach().to(device="cpu"),
            )
        ):
            contract_mismatches.append(
                (key, expected_value.detach().cpu().tolist(), provided_value.detach().cpu().tolist())
            )

    if missing or unexpected or shape_mismatches or contract_mismatches:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        if shape_mismatches:
            details.append(f"shape_mismatches={shape_mismatches[:8]}")
        if contract_mismatches:
            details.append(f"contract_mismatches={contract_mismatches[:8]}")
        raise ValueError(
            f"{checkpoint_label}: incompatible stage_b_fixed_text_scorer state "
            f"({'; '.join(details)}). Use --pretrain_model_path, not --resume, "
            "to initialize v11 from a StageA/v10 checkpoint without scorer state."
        )


def build_stage_b_pair_token_diff_masks_from_ids(
    input_ids: Tensor,
    attention_mask: Tensor,
    expression_valid_mask: Tensor,
    eligible_token_mask: Optional[Tensor] = None,
    *,
    max_text_len: int,
) -> tuple[Tensor, Tensor]:
    """Build independent positive/TN token masks from paired token-id diffs.

    Equal subsequences, including shared object words, are deliberately omitted.
    A pair is valid only when both expressions contain at least one eligible
    changed token; insertion/deletion-only pairs keep their existing phrase loss
    but do not receive this symmetric auxiliary loss.
    """
    if input_ids.dim() != 3 or input_ids.shape[1] != 2:
        raise ValueError(
            "input_ids must have shape (B,2,L), "
            f"got {tuple(input_ids.shape)}"
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if tuple(expression_valid_mask.shape) != tuple(input_ids.shape[:2]):
        raise ValueError("expression_valid_mask must have shape (B,2)")
    if eligible_token_mask is None:
        eligible_token_mask = attention_mask
    if eligible_token_mask.shape != input_ids.shape:
        raise ValueError("eligible_token_mask must match input_ids")
    if int(max_text_len) <= 0:
        raise ValueError("max_text_len must be positive")

    batch_size = int(input_ids.shape[0])
    width = min(int(input_ids.shape[2]), int(max_text_len))
    device = input_ids.device
    result = torch.zeros(
        (batch_size, 2, int(max_text_len)), dtype=torch.bool, device=device
    )
    pair_valid = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    ids_cpu = input_ids.detach().to(device="cpu")[:, :, :width]
    attention_cpu = attention_mask.detach().to(device="cpu", dtype=torch.bool)[
        :, :, :width
    ]
    eligible_cpu = eligible_token_mask.detach().to(
        device="cpu", dtype=torch.bool
    )[:, :, :width]
    valid_cpu = expression_valid_mask.detach().to(device="cpu", dtype=torch.bool)

    for batch_idx in range(batch_size):
        if not bool(valid_cpu[batch_idx].all().item()):
            continue
        active_positions = [
            attention_cpu[batch_idx, slot_idx].nonzero(as_tuple=False)
            .flatten()
            .tolist()
            for slot_idx in range(2)
        ]
        sequences = [
            ids_cpu[batch_idx, slot_idx, active_positions[slot_idx]].tolist()
            for slot_idx in range(2)
        ]
        for tag, pos_start, pos_end, tn_start, tn_end in SequenceMatcher(
            None, sequences[0], sequences[1], autojunk=False
        ).get_opcodes():
            if tag in {"replace", "delete"}:
                for sequence_idx in range(pos_start, pos_end):
                    token_idx = int(active_positions[0][sequence_idx])
                    if bool(eligible_cpu[batch_idx, 0, token_idx].item()):
                        result[batch_idx, 0, token_idx] = True
            if tag in {"replace", "insert"}:
                for sequence_idx in range(tn_start, tn_end):
                    token_idx = int(active_positions[1][sequence_idx])
                    if bool(eligible_cpu[batch_idx, 1, token_idx].item()):
                        result[batch_idx, 1, token_idx] = True

        valid_pair = bool(result[batch_idx, 0].any().item()) and bool(
            result[batch_idx, 1].any().item()
        )
        if valid_pair:
            pair_valid[batch_idx] = True
        else:
            result[batch_idx].zero_()

    return result, pair_valid


def build_stage_b_noncanonical_token_masks_from_ids(
    expression_input_ids: Tensor,
    expression_attention_mask: Tensor,
    canonical_input_ids: Tensor,
    canonical_attention_mask: Tensor,
    expression_valid_mask: Tensor,
    eligible_token_mask: Optional[Tensor] = None,
    *,
    max_text_len: int,
    fallback_to_eligible: bool = True,
) -> Tensor:
    """Select expression tokens not shared with the Stage-A canonical phrase.

    Matching is done on token ids rather than decoded strings so the returned
    positions align exactly with the scorer tokenizer. By default,
    canonical-only expressions fall back to their eligible tokens
    so legacy scorers remain defined. Responsibility-separated callers disable
    that fallback: an empty mask is then the explicit signal that patch scores
    own category-only ranking.
    """
    if expression_input_ids.dim() != 3:
        raise ValueError("expression_input_ids must have shape (B,K,L)")
    if expression_attention_mask.shape != expression_input_ids.shape:
        raise ValueError("expression_attention_mask must match expression_input_ids")
    if canonical_input_ids.dim() != 2:
        raise ValueError("canonical_input_ids must have shape (B,Lc)")
    if canonical_attention_mask.shape != canonical_input_ids.shape:
        raise ValueError("canonical_attention_mask must match canonical_input_ids")
    if canonical_input_ids.shape[0] != expression_input_ids.shape[0]:
        raise ValueError("canonical and expression batches must align")
    if tuple(expression_valid_mask.shape) != tuple(expression_input_ids.shape[:2]):
        raise ValueError("expression_valid_mask must have shape (B,K)")
    if eligible_token_mask is None:
        eligible_token_mask = expression_attention_mask
    if eligible_token_mask.shape != expression_input_ids.shape:
        raise ValueError("eligible_token_mask must match expression_input_ids")
    if int(max_text_len) <= 0:
        raise ValueError("max_text_len must be positive")
    if type(fallback_to_eligible) is not bool:
        raise TypeError("fallback_to_eligible must be a boolean")

    batch_size, slot_count, expression_width_raw = expression_input_ids.shape
    expression_width = min(int(expression_width_raw), int(max_text_len))
    output = torch.zeros(
        (batch_size, slot_count, int(max_text_len)),
        dtype=torch.bool,
        device=expression_input_ids.device,
    )

    expression_ids_cpu = expression_input_ids.detach().to(device="cpu")
    expression_attention_cpu = expression_attention_mask.detach().to(
        device="cpu", dtype=torch.bool
    )
    canonical_ids_cpu = canonical_input_ids.detach().to(device="cpu")
    canonical_attention_cpu = canonical_attention_mask.detach().to(
        device="cpu", dtype=torch.bool
    )
    eligible_cpu = eligible_token_mask.detach().to(device="cpu", dtype=torch.bool)
    valid_cpu = expression_valid_mask.detach().to(device="cpu", dtype=torch.bool)

    for batch_idx in range(int(batch_size)):
        canonical_positions = (
            canonical_attention_cpu[batch_idx].nonzero(as_tuple=False).flatten().tolist()
        )
        canonical_sequence = canonical_ids_cpu[
            batch_idx, canonical_positions
        ].tolist()
        for slot_idx in range(int(slot_count)):
            if not bool(valid_cpu[batch_idx, slot_idx].item()):
                continue
            expression_positions = (
                expression_attention_cpu[batch_idx, slot_idx]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            expression_sequence = expression_ids_cpu[
                batch_idx, slot_idx, expression_positions
            ].tolist()
            keep = eligible_cpu[batch_idx, slot_idx].clone()
            for expression_start, _canonical_start, size in SequenceMatcher(
                None, expression_sequence, canonical_sequence, autojunk=False
            ).get_matching_blocks():
                for sequence_idx in range(expression_start, expression_start + size):
                    token_idx = int(expression_positions[sequence_idx])
                    keep[token_idx] = False
            keep = keep[:expression_width]
            if fallback_to_eligible and not bool(keep.any().item()):
                keep = eligible_cpu[batch_idx, slot_idx, :expression_width].clone()
            output[batch_idx, slot_idx, :expression_width] = keep.to(
                device=output.device
            )
    return output


class FixedBoxFullTextScorer(nn.Module):
    """Full-text decoder scorer over immutable external candidate boxes.

    The module owns a scoring decoder, a parameter-free token head, and an
    optional scalar validity residual. It deliberately does not own BERT, a
    visual encoder, patch modules, or a box head. ``context_provider`` supplies
    expression-conditioned text and image memory for each expression microbatch.
    """

    def __init__(
        self,
        source_decoder: TransformerDecoder,
        *,
        num_layers: int = 3,
        max_text_len: int = 256,
        expression_microbatch: int = 0,
        use_validity_head: bool = False,
        decouple_validity_from_ranking: bool = False,
        validity_pool_temperature: float = 0.2,
        patch_rank_fusion: bool = False,
        patch_rank_weight: float = 1.0,
        exclude_canonical_from_score: bool = False,
        candidate_topk: int = 50,
        confidence_output_mode: str = "base_plus_gate",
        explicit_confidence_output_contract: bool = False,
        score_ownership: str = "",
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.max_text_len = int(max_text_len)
        self.expression_microbatch = int(expression_microbatch)
        self.use_validity_head = bool(use_validity_head)
        self.decouple_validity_from_ranking = bool(decouple_validity_from_ranking)
        self.validity_pool_temperature = float(validity_pool_temperature)
        self.patch_rank_fusion = bool(patch_rank_fusion)
        self.patch_rank_weight = float(patch_rank_weight)
        self.exclude_canonical_from_score = bool(exclude_canonical_from_score)
        self.candidate_topk = int(candidate_topk)
        self.confidence_output_mode = str(confidence_output_mode).strip().lower()
        self.explicit_confidence_output_contract = bool(
            explicit_confidence_output_contract
        )
        self.score_ownership = normalize_stage_b_score_ownership(score_ownership)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.max_text_len <= 0:
            raise ValueError("max_text_len must be positive")
        if self.expression_microbatch < 0:
            raise ValueError("expression_microbatch must be non-negative")
        if self.validity_pool_temperature <= 0.0:
            raise ValueError("validity_pool_temperature must be positive")
        if not math.isfinite(self.patch_rank_weight) or self.patch_rank_weight < 0.0:
            raise ValueError("patch_rank_weight must be finite and non-negative")
        if self.candidate_topk <= 0:
            raise ValueError("candidate_topk must be positive")
        if self.confidence_output_mode not in CONFIDENCE_OUTPUT_MODES:
            raise ValueError(
                "confidence_output_mode must be one of "
                f"{CONFIDENCE_OUTPUT_MODES}, got {self.confidence_output_mode!r}"
            )
        if self.decouple_validity_from_ranking and not self.use_validity_head:
            raise ValueError(
                "decouple_validity_from_ranking requires use_validity_head=True"
            )
        ownership_contracts = {
            "shared_score": (False, False),
            "shared_trunk_two_heads": (True, False),
            "independent_decoders_joint": (True, True),
            "independent_decoders_two_phase": (True, True),
        }
        if self.score_ownership:
            expected = ownership_contracts[self.score_ownership]
            actual = (
                self.use_validity_head,
                self.decouple_validity_from_ranking,
            )
            if actual != expected:
                raise ValueError(
                    f"score_ownership={self.score_ownership!r} requires "
                    "(use_validity_head, decouple_validity_from_ranking)="
                    f"{expected}, got {actual}"
                )
        if self.confidence_output_mode == "gate_only" and not (
            self.decouple_validity_from_ranking and self.use_validity_head
        ):
            raise ValueError(
                "confidence_output_mode='gate_only' requires decoupled confidence "
                "and use_validity_head=True"
            )
        if self.explicit_confidence_output_contract and not (
            self.confidence_output_mode == "base_plus_gate"
            and self.decouple_validity_from_ranking
            and self.use_validity_head
        ):
            raise ValueError(
                "explicit_confidence_output_contract requires decoupled "
                "confidence_output_mode='base_plus_gate' and use_validity_head=True"
            )
        self._validate_source_decoder(source_decoder)

        decoder = copy.deepcopy(source_decoder)
        decoder.layers = nn.ModuleList(list(decoder.layers)[-self.num_layers :])
        decoder.num_layers = self.num_layers
        decoder.bbox_embed = None
        decoder.class_embed = None
        self.decoder = decoder
        self.confidence_decoder: Optional[TransformerDecoder]
        if self.decouple_validity_from_ranking:
            self.confidence_decoder = copy.deepcopy(decoder)
            self._freeze_confidence_decoder()
        else:
            self.confidence_decoder = None
        if (
            self.decouple_validity_from_ranking
            or self.patch_rank_fusion
            or self.exclude_canonical_from_score
        ):
            gate_only_output = self.confidence_output_mode == "gate_only"
            explicit_base_plus_gate = self.explicit_confidence_output_contract
            contract_values = {
                "_score_contract_version": torch.tensor(
                    5
                    if explicit_base_plus_gate
                    else (4 if gate_only_output else 3),
                    dtype=torch.int64,
                ),
                "_score_contract_decoupled_confidence": torch.tensor(
                    self.decouple_validity_from_ranking, dtype=torch.bool
                ),
                "_score_contract_validity_pool_temperature": torch.tensor(
                    self.validity_pool_temperature, dtype=torch.float32
                ),
                "_score_contract_patch_rank_fusion": torch.tensor(
                    self.patch_rank_fusion, dtype=torch.bool
                ),
                "_score_contract_patch_rank_weight": torch.tensor(
                    self.patch_rank_weight, dtype=torch.float32
                ),
                "_score_contract_exclude_canonical": torch.tensor(
                    self.exclude_canonical_from_score, dtype=torch.bool
                ),
                "_score_contract_candidate_topk": torch.tensor(
                    self.candidate_topk, dtype=torch.int64
                ),
            }
            # Legacy v15 checkpoints retain their exact v3 key set. New output
            # contracts persist an explicit code: v16 gate-only=1, v19
            # base-plus-gate=0.
            if gate_only_output or explicit_base_plus_gate:
                contract_values["_score_contract_confidence_output_mode"] = (
                    torch.tensor(1 if gate_only_output else 0, dtype=torch.int64)
                )
            for name, value in contract_values.items():
                self.register_buffer(name, value, persistent=True)
        if self.score_ownership:
            self.register_buffer(
                "_score_contract_ownership",
                torch.tensor(
                    SCORE_OWNERSHIP_MODES.index(self.score_ownership),
                    dtype=torch.int64,
                ),
                persistent=True,
            )
        self.token_head = ContrastiveEmbed(max_text_len=self.max_text_len)
        self.validity_head: Optional[nn.Module]
        if self.use_validity_head:
            hidden_dim = int(source_decoder.d_model)
            self.validity_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.validity_head[-1].weight)
            nn.init.zeros_(self.validity_head[-1].bias)
        else:
            # Keep the v11-v13 state dict byte-for-byte compatible when disabled.
            self.validity_head = None

    def _freeze_confidence_decoder(self) -> None:
        if self.confidence_decoder is None:
            return
        self.confidence_decoder.eval()
        for parameter in self.confidence_decoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Confidence is an immutable initialization snapshot, including its
        # dropout state. Only validity_head may learn absolute calibration.
        self._freeze_confidence_decoder()
        return self

    def _validate_source_decoder(self, source_decoder: TransformerDecoder) -> None:
        layers = list(getattr(source_decoder, "layers", []))
        if len(layers) < self.num_layers:
            raise ValueError(
                f"source decoder has {len(layers)} layers, cannot take the last {self.num_layers}"
            )
        if not bool(getattr(source_decoder, "return_intermediate", False)):
            raise ValueError("source decoder must return intermediate layer outputs")
        missing_text_ca = [
            idx
            for idx, layer in enumerate(layers[-self.num_layers :])
            if not bool(getattr(layer, "use_text_cross_attention", False))
        ]
        if missing_text_ca:
            raise ValueError(
                "fixed full-text scorer requires text cross-attention in every selected layer; "
                f"missing at selected layer offsets {missing_text_ca}"
            )

    @torch.no_grad()
    def load_from_decoder(self, source_decoder: TransformerDecoder) -> None:
        """Initialize the existing scorer parameters after a base checkpoint load."""
        self._validate_source_decoder(source_decoder)
        source_layers = list(source_decoder.layers)[-self.num_layers :]
        for target_layer, source_layer in zip(self.decoder.layers, source_layers):
            target_layer.load_state_dict(source_layer.state_dict(), strict=True)
        self.decoder.ref_point_head.load_state_dict(
            source_decoder.ref_point_head.state_dict(), strict=True
        )
        if self.decoder.norm is not None:
            if source_decoder.norm is None:
                raise ValueError("source decoder has no final norm")
            self.decoder.norm.load_state_dict(source_decoder.norm.state_dict(), strict=True)
        self.decoder.bbox_embed = None
        self.decoder.class_embed = None
        if self.confidence_decoder is not None:
            self.confidence_decoder.load_state_dict(
                self.decoder.state_dict(), strict=True
            )
            self._freeze_confidence_decoder()

    @torch.no_grad()
    def load_from_full_text_checkpoint_state(
        self,
        source_state_dict: Mapping[str, Any],
        *,
        checkpoint_label: str,
        source_decoder_prefix: str = "transformer.decoder",
    ) -> Dict[str, Any]:
        """Warm-start only the full-text decoder components from a checkpoint.

        The source checkpoint is deliberately treated as a tensor namespace,
        not as a model state to load wholesale. Only the last ``num_layers``
        decoder layers, ``ref_point_head``, and final ``norm`` are accepted.
        Every selected component must have an exact key set and compatible
        shapes before any scorer tensor is mutated.
        """
        if not isinstance(source_state_dict, Mapping):
            raise TypeError(f"{checkpoint_label}: source model state must be a mapping")
        source_decoder_prefix = str(source_decoder_prefix).rstrip(".")
        if not source_decoder_prefix:
            raise ValueError("source_decoder_prefix must be non-empty")
        if self.decoder.norm is None:
            raise ValueError(
                f"{checkpoint_label}: scorer decoder has no final norm to warm-start"
            )

        layer_root = source_decoder_prefix + ".layers."
        source_layer_indices = set()
        malformed_layer_keys = []
        for raw_key in source_state_dict:
            key = str(raw_key)
            if not key.startswith(layer_root):
                continue
            remainder = key[len(layer_root) :]
            layer_text, separator, suffix = remainder.partition(".")
            if not separator or not suffix or not layer_text.isdigit():
                malformed_layer_keys.append(key)
                continue
            source_layer_indices.add(int(layer_text))
        if malformed_layer_keys:
            raise ValueError(
                f"{checkpoint_label}: malformed decoder layer keys "
                f"{sorted(malformed_layer_keys)[:8]}"
            )
        if not source_layer_indices:
            raise ValueError(
                f"{checkpoint_label}: no keys found below {layer_root!r}"
            )
        ordered_source_layers = sorted(source_layer_indices)
        expected_source_layers = list(range(ordered_source_layers[-1] + 1))
        if ordered_source_layers != expected_source_layers:
            raise ValueError(
                f"{checkpoint_label}: source decoder layer indices must be contiguous "
                f"from zero, got {ordered_source_layers}"
            )
        if len(ordered_source_layers) < self.num_layers:
            raise ValueError(
                f"{checkpoint_label}: source decoder has {len(ordered_source_layers)} "
                f"layers, cannot take the last {self.num_layers}"
            )
        selected_source_layers = ordered_source_layers[-self.num_layers :]

        mapped_state: Dict[str, Tensor] = {}
        validation_errors: List[str] = []

        def map_component(
            *,
            source_prefix: str,
            target_prefix: str,
            target_module: nn.Module,
        ) -> None:
            expected = target_module.state_dict()
            source_root = source_prefix + "."
            provided = {
                str(key)[len(source_root) :]: value
                for key, value in source_state_dict.items()
                if str(key).startswith(source_root)
            }
            missing = sorted(set(expected).difference(provided))
            unexpected = sorted(set(provided).difference(expected))
            shape_mismatches = []
            for suffix in sorted(set(expected).intersection(provided)):
                value = provided[suffix]
                if not torch.is_tensor(value):
                    shape_mismatches.append(
                        (suffix, tuple(expected[suffix].shape), type(value).__name__)
                    )
                elif tuple(value.shape) != tuple(expected[suffix].shape):
                    shape_mismatches.append(
                        (suffix, tuple(expected[suffix].shape), tuple(value.shape))
                    )
                else:
                    mapped_state[target_prefix + "." + suffix] = value
            if missing or unexpected or shape_mismatches:
                details = []
                if missing:
                    details.append(f"missing={missing[:8]}")
                if unexpected:
                    details.append(f"unexpected={unexpected[:8]}")
                if shape_mismatches:
                    details.append(f"shape_mismatches={shape_mismatches[:8]}")
                validation_errors.append(
                    f"{source_prefix} ({'; '.join(details)})"
                )

        for target_idx, (source_idx, target_layer) in enumerate(
            zip(selected_source_layers, self.decoder.layers)
        ):
            map_component(
                source_prefix=f"{source_decoder_prefix}.layers.{source_idx}",
                target_prefix=f"layers.{target_idx}",
                target_module=target_layer,
            )
        map_component(
            source_prefix=f"{source_decoder_prefix}.ref_point_head",
            target_prefix="ref_point_head",
            target_module=self.decoder.ref_point_head,
        )
        map_component(
            source_prefix=f"{source_decoder_prefix}.norm",
            target_prefix="norm",
            target_module=self.decoder.norm,
        )
        if validation_errors:
            raise ValueError(
                f"{checkpoint_label}: incompatible full-text decoder warm-start "
                f"({'; '.join(validation_errors)})"
            )

        expected_target_keys = set(self.decoder.state_dict())
        if set(mapped_state) != expected_target_keys:
            missing = sorted(expected_target_keys.difference(mapped_state))
            unexpected = sorted(set(mapped_state).difference(expected_target_keys))
            raise ValueError(
                f"{checkpoint_label}: internal decoder mapping was not exact "
                f"(missing={missing[:8]}, unexpected={unexpected[:8]})"
            )

        self.decoder.load_state_dict(mapped_state, strict=True)
        self.decoder.bbox_embed = None
        self.decoder.class_embed = None
        if self.confidence_decoder is not None:
            self.confidence_decoder.load_state_dict(
                self.decoder.state_dict(), strict=True
            )
            self._freeze_confidence_decoder()
        return {
            "source_decoder_num_layers": len(ordered_source_layers),
            "selected_source_layer_indices": selected_source_layers,
            "loaded_num_layers": self.num_layers,
            "loaded_tensor_count": len(mapped_state),
            "loaded_components": [
                "decoder.layers[-N:]",
                "decoder.ref_point_head",
                "decoder.norm",
            ],
        }

    @staticmethod
    def _normalize_captions(
        expression_captions: Sequence[Sequence[str]],
        *,
        batch_size: int,
    ) -> tuple[List[str], int]:
        if len(expression_captions) != batch_size:
            raise ValueError(
                f"expression_captions must have B={batch_size} rows, got {len(expression_captions)}"
            )
        if batch_size <= 0:
            raise ValueError("candidate batch must be non-empty")
        slot_count = len(expression_captions[0])
        if slot_count <= 0:
            raise ValueError("each sample must provide at least one expression slot")
        flattened: List[str] = []
        for batch_idx, row in enumerate(expression_captions):
            if len(row) != slot_count:
                raise ValueError(
                    "expression_captions must be rectangular; "
                    f"row 0 has {slot_count} slots while row {batch_idx} has {len(row)}"
                )
            for caption in row:
                if not isinstance(caption, str):
                    raise TypeError(f"expression caption must be str, got {type(caption).__name__}")
                flattened.append(caption if caption.strip() else "object .")
        return flattened, slot_count

    def _validate_context(
        self,
        context: Dict[str, Any],
        *,
        batch_size: int,
        hidden_dim: int,
        device: torch.device,
    ) -> tuple[Dict[str, Tensor], Tensor]:
        required = {
            "memory",
            "memory_key_padding_mask",
            "level_start_index",
            "spatial_shapes",
            "valid_ratios",
            "text_dict",
        }
        missing = sorted(required.difference(context))
        if missing:
            raise KeyError(f"context_provider omitted required keys: {missing}")
        memory = context["memory"]
        if not torch.is_tensor(memory) or memory.dim() != 3:
            raise ValueError("context memory must be a tensor shaped (M,S,D)")
        if memory.shape[0] != batch_size or memory.shape[-1] != hidden_dim:
            raise ValueError(
                f"context memory must be ({batch_size},S,{hidden_dim}), got {tuple(memory.shape)}"
            )
        if memory.device != device:
            raise ValueError(f"context memory is on {memory.device}, expected {device}")

        text_dict = context["text_dict"]
        if not isinstance(text_dict, dict):
            raise TypeError("context text_dict must be a dict")
        encoded_text = text_dict.get("encoded_text")
        text_token_mask = text_dict.get("text_token_mask")
        if not torch.is_tensor(encoded_text) or encoded_text.dim() != 3:
            raise ValueError("text_dict['encoded_text'] must be (M,T,D)")
        if encoded_text.shape[0] != batch_size or encoded_text.shape[-1] != hidden_dim:
            raise ValueError(
                "encoded_text must be "
                f"({batch_size},T,{hidden_dim}), got {tuple(encoded_text.shape)}"
            )
        if encoded_text.shape[1] > self.max_text_len:
            raise ValueError(
                f"encoded_text length {encoded_text.shape[1]} exceeds "
                f"max_text_len={self.max_text_len}"
            )
        if not torch.is_tensor(text_token_mask) or text_token_mask.shape != encoded_text.shape[:2]:
            raise ValueError("text_dict['text_token_mask'] must be (M,T)")
        if encoded_text.device != device or text_token_mask.device != device:
            raise ValueError("text tensors must be on the candidate device")

        phrase_token_mask = context.get("phrase_token_mask", text_dict.get("phrase_token_mask"))
        if phrase_token_mask is None:
            phrase_token_mask = text_token_mask
        if (
            not torch.is_tensor(phrase_token_mask)
            or phrase_token_mask.shape != encoded_text.shape[:2]
        ):
            raise ValueError("phrase_token_mask must be (M,T) and align with encoded_text")

        tensor_context = {
            "memory": memory,
            "memory_key_padding_mask": context["memory_key_padding_mask"],
            "level_start_index": context["level_start_index"],
            "spatial_shapes": context["spatial_shapes"],
            "valid_ratios": context["valid_ratios"],
            "memory_pos": context.get("memory_pos"),
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
        }
        for key, value in tensor_context.items():
            if value is not None and not torch.is_tensor(value):
                raise TypeError(f"context {key!r} must be a tensor or None")
        return tensor_context, phrase_token_mask.to(device=device, dtype=torch.bool)

    def _pad_phrase_mask(self, mask: Tensor) -> Tensor:
        out = torch.zeros(
            (mask.shape[0], self.max_text_len),
            dtype=torch.bool,
            device=mask.device,
        )
        width = min(int(mask.shape[1]), self.max_text_len)
        if width > 0:
            out[:, :width] = mask[:, :width]
        return out

    @staticmethod
    def _aggregate_phrase_logits(token_logits: Tensor, phrase_mask: Tensor) -> Tensor:
        finite_logits = torch.where(
            torch.isfinite(token_logits), token_logits, torch.full_like(token_logits, -20.0)
        )
        weight = phrase_mask[:, None, :].to(dtype=finite_logits.dtype)
        denom = weight.sum(dim=-1).clamp(min=1.0)
        probability = (finite_logits.sigmoid() * weight).sum(dim=-1) / denom
        eps = max(float(torch.finfo(finite_logits.dtype).eps), 1e-6)
        phrase_logits = torch.logit(probability.clamp(min=eps, max=1.0 - eps))
        valid = phrase_mask.any(dim=-1)
        return phrase_logits.masked_fill(~valid[:, None], torch.finfo(phrase_logits.dtype).min)

    def _fuse_patch_prior(self, text_logits: Tensor, patch_logits: Tensor) -> Tensor:
        if not self.patch_rank_fusion:
            return text_logits
        return text_logits + self.patch_rank_weight * patch_logits.unsqueeze(0)

    def forward(
        self,
        *,
        candidate_hs: Tensor,
        candidate_boxes: Tensor,
        expression_captions: Sequence[Sequence[str]],
        expression_valid_mask: Tensor,
        expression_predicate_token_mask: Optional[Tensor] = None,
        expression_score_token_mask: Optional[Tensor] = None,
        candidate_patch_logits: Optional[Tensor] = None,
        context_provider: ExpressionContextProvider,
        expression_microbatch: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if candidate_hs.dim() != 3:
            raise ValueError(f"candidate_hs must be (B,N,D), got {tuple(candidate_hs.shape)}")
        if candidate_boxes.dim() != 3 or candidate_boxes.shape[-1] != 4:
            raise ValueError(f"candidate_boxes must be (B,N,4), got {tuple(candidate_boxes.shape)}")
        if candidate_hs.shape[:2] != candidate_boxes.shape[:2]:
            raise ValueError(
                "candidate_hs/boxes disagree: "
                f"{tuple(candidate_hs.shape)} vs {tuple(candidate_boxes.shape)}"
            )
        if candidate_boxes.device != candidate_hs.device:
            raise ValueError(
                "candidate_hs and candidate_boxes must share a device, got "
                f"{candidate_hs.device} and {candidate_boxes.device}"
            )
        if not candidate_hs.is_floating_point() or not candidate_boxes.is_floating_point():
            raise TypeError("candidate_hs and candidate_boxes must be floating-point tensors")
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")

        batch_size, num_candidates, hidden_dim = candidate_hs.shape
        flat_captions, slot_count = self._normalize_captions(
            expression_captions, batch_size=batch_size
        )
        if tuple(expression_valid_mask.shape) != (batch_size, slot_count):
            raise ValueError(
                "expression_valid_mask must align with captions, expected "
                f"{(batch_size, slot_count)}, got {tuple(expression_valid_mask.shape)}"
            )
        expression_valid_mask = expression_valid_mask.to(
            device=candidate_hs.device, dtype=torch.bool
        )
        if self.patch_rank_fusion:
            if candidate_patch_logits is None:
                raise ValueError(
                    "patch_rank_fusion requires candidate_patch_logits shaped (B,N)"
                )
            if tuple(candidate_patch_logits.shape) != tuple(candidate_hs.shape[:2]):
                raise ValueError("candidate_patch_logits must align with candidate_hs (B,N)")
            if (
                candidate_patch_logits.device != candidate_hs.device
                or not candidate_patch_logits.is_floating_point()
            ):
                raise ValueError(
                    "candidate_patch_logits must be floating point on the candidate device"
                )
            candidate_patch_logits = candidate_patch_logits.detach()
        if expression_predicate_token_mask is None:
            expression_predicate_token_mask = torch.zeros(
                (batch_size, slot_count, self.max_text_len),
                dtype=torch.bool,
                device=candidate_hs.device,
            )
        elif tuple(expression_predicate_token_mask.shape) != (
            batch_size,
            slot_count,
            self.max_text_len,
        ):
            raise ValueError(
                "expression_predicate_token_mask must be "
                f"{(batch_size, slot_count, self.max_text_len)}, got "
                f"{tuple(expression_predicate_token_mask.shape)}"
            )
        else:
            expression_predicate_token_mask = expression_predicate_token_mask.to(
                device=candidate_hs.device, dtype=torch.bool
            )
        if expression_score_token_mask is None:
            flat_score_mask = None
        elif tuple(expression_score_token_mask.shape) != (
            batch_size,
            slot_count,
            self.max_text_len,
        ):
            raise ValueError(
                "expression_score_token_mask must be "
                f"{(batch_size, slot_count, self.max_text_len)}, got "
                f"{tuple(expression_score_token_mask.shape)}"
            )
        else:
            flat_score_mask = expression_score_token_mask.to(
                device=candidate_hs.device, dtype=torch.bool
            ).reshape(batch_size * slot_count, self.max_text_len)

        flat_count = batch_size * slot_count
        flat_predicate_mask = expression_predicate_token_mask.reshape(
            flat_count, self.max_text_len
        )
        owner_indices = torch.arange(
            batch_size, device=candidate_hs.device, dtype=torch.long
        ).repeat_interleave(slot_count)
        microbatch = (
            self.expression_microbatch
            if expression_microbatch is None
            else int(expression_microbatch)
        )
        if microbatch <= 0:
            microbatch = flat_count

        layer_token_chunks: List[Tensor] = []
        layer_phrase_chunks: List[Tensor] = []
        layer_predicate_chunks: List[Tensor] = []
        layer_validity_chunks: List[Tensor] = []
        layer_validity_gate_chunks: List[Tensor] = []
        layer_confidence_base_chunks: List[Tensor] = []
        effective_predicate_mask_chunks: List[Tensor] = []
        effective_score_mask_chunks: List[Tensor] = []
        for start in range(0, flat_count, microbatch):
            end = min(start + microbatch, flat_count)
            owners = owner_indices[start:end]
            captions = flat_captions[start:end]
            context_raw = context_provider(captions, owners)
            if not isinstance(context_raw, dict):
                raise TypeError("context_provider must return a dict")
            context, phrase_mask = self._validate_context(
                context_raw,
                batch_size=end - start,
                hidden_dim=hidden_dim,
                device=candidate_hs.device,
            )
            chunk_hs = candidate_hs.index_select(0, owners)
            chunk_boxes = candidate_boxes.index_select(0, owners)
            decoded_layers, _references = self.decoder.forward_fixed_external(
                tgt=chunk_hs,
                reference_boxes=chunk_boxes,
                memory=context["memory"],
                memory_key_padding_mask=context["memory_key_padding_mask"],
                memory_pos=context["memory_pos"],
                level_start_index=context["level_start_index"],
                spatial_shapes=context["spatial_shapes"],
                valid_ratios=context["valid_ratios"],
                memory_text=context["encoded_text"],
                text_attention_mask=~context["text_token_mask"].to(dtype=torch.bool),
            )
            if len(decoded_layers) != self.num_layers:
                raise RuntimeError(
                    f"scoring decoder returned {len(decoded_layers)} layers, "
                    f"expected {self.num_layers}"
                )
            text_dict = {
                "encoded_text": context["encoded_text"],
                "text_token_mask": context["text_token_mask"].to(dtype=torch.bool),
            }
            token_layers = torch.stack(
                [self.token_head(layer_hs, text_dict) for layer_hs in decoded_layers], dim=0
            )
            padded_phrase_mask = self._pad_phrase_mask(phrase_mask)
            effective_score_mask = padded_phrase_mask
            if flat_score_mask is not None:
                effective_score_mask = (
                    flat_score_mask[start:end] & padded_phrase_mask
                )
            effective_predicate_mask = (
                flat_predicate_mask[start:end] & padded_phrase_mask
            )
            text_phrase_layers = torch.stack(
                [
                    self._aggregate_phrase_logits(
                        token_layers[layer_idx], effective_score_mask
                    )
                    for layer_idx in range(self.num_layers)
                ],
                dim=0,
            )
            chunk_patch_logits = (
                candidate_patch_logits.index_select(0, owners)
                if candidate_patch_logits is not None
                else text_phrase_layers.new_zeros(text_phrase_layers.shape[1:])
            )
            phrase_layers = self._fuse_patch_prior(
                text_phrase_layers, chunk_patch_logits
            )
            predicate_layers = torch.stack(
                [
                    self._aggregate_phrase_logits(
                        token_layers[layer_idx], effective_predicate_mask
                    )
                    for layer_idx in range(self.num_layers)
                ],
                dim=0,
            )
            if self.validity_head is not None:
                if self.decouple_validity_from_ranking:
                    if self.confidence_decoder is None:
                        raise RuntimeError("decoupled confidence decoder is missing")
                    with torch.no_grad():
                        confidence_decoded_layers, _ = (
                            self.confidence_decoder.forward_fixed_external(
                                tgt=chunk_hs,
                                reference_boxes=chunk_boxes,
                                memory=context["memory"],
                                memory_key_padding_mask=context[
                                    "memory_key_padding_mask"
                                ],
                                memory_pos=context["memory_pos"],
                                level_start_index=context["level_start_index"],
                                spatial_shapes=context["spatial_shapes"],
                                valid_ratios=context["valid_ratios"],
                                memory_text=context["encoded_text"],
                                text_attention_mask=~context[
                                    "text_token_mask"
                                ].to(dtype=torch.bool),
                            )
                        )
                        confidence_token_layers = torch.stack(
                            [
                                self.token_head(layer_hs, text_dict)
                                for layer_hs in confidence_decoded_layers
                            ],
                            dim=0,
                        )
                        confidence_text_phrase_layers = torch.stack(
                            [
                                self._aggregate_phrase_logits(
                                    confidence_token_layers[layer_idx],
                                    effective_score_mask,
                                )
                                for layer_idx in range(self.num_layers)
                            ],
                            dim=0,
                        )
                        confidence_phrase_layers = self._fuse_patch_prior(
                            confidence_text_phrase_layers, chunk_patch_logits
                        )
                        validity_input = torch.stack(
                            list(confidence_decoded_layers), dim=0
                        )
                else:
                    confidence_phrase_layers = phrase_layers.detach()
                    validity_input = torch.stack(list(decoded_layers), dim=0)
                validity_residual = self.validity_head(validity_input).squeeze(-1)
                # The legacy score is only an initialization prior. Detaching it
                # prevents the validity objective from falling back to token mean.
                if self.decouple_validity_from_ranking:
                    # Pool an image-expression scalar and broadcast it to every
                    # candidate. Confidence calibration can then change the
                    # global detection score without changing the box order.
                    score_mask_valid = effective_score_mask.any(dim=-1)
                    pool_logits = torch.where(
                        score_mask_valid[None, :, None],
                        confidence_phrase_layers.float(),
                        torch.zeros_like(confidence_phrase_layers.float()),
                    )
                    pool_weight = torch.softmax(
                        pool_logits / self.validity_pool_temperature, dim=-1
                    ).to(dtype=validity_residual.dtype)
                    validity_gate_layers = (
                        pool_weight * validity_residual
                    ).sum(dim=-1)
                    if self.confidence_output_mode == "gate_only":
                        validity_layers = validity_gate_layers.unsqueeze(-1).expand_as(
                            confidence_phrase_layers
                        )
                    else:
                        validity_layers = (
                            confidence_phrase_layers
                            + validity_gate_layers.unsqueeze(-1)
                        )
                    layer_validity_gate_chunks.append(validity_gate_layers)
                else:
                    validity_layers = phrase_layers.detach() + validity_residual
                    layer_validity_gate_chunks.append(
                        torch.zeros_like(phrase_layers[..., 0])
                    )
                layer_validity_chunks.append(validity_layers)
                layer_confidence_base_chunks.append(confidence_phrase_layers)
            layer_token_chunks.append(token_layers)
            layer_phrase_chunks.append(phrase_layers)
            layer_predicate_chunks.append(predicate_layers)
            effective_predicate_mask_chunks.append(effective_predicate_mask)
            effective_score_mask_chunks.append(effective_score_mask)

        flat_token = torch.cat(layer_token_chunks, dim=1)
        flat_phrase = torch.cat(layer_phrase_chunks, dim=1)
        flat_predicate = torch.cat(layer_predicate_chunks, dim=1)
        effective_predicate_token_mask = torch.cat(
            effective_predicate_mask_chunks, dim=0
        ).view(batch_size, slot_count, self.max_text_len)
        effective_score_token_mask = torch.cat(
            effective_score_mask_chunks, dim=0
        ).view(batch_size, slot_count, self.max_text_len)
        layer_token_logits = (
            flat_token.view(
                self.num_layers,
                batch_size,
                slot_count,
                num_candidates,
                self.max_text_len,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        layer_phrase_logits = (
            flat_phrase.view(self.num_layers, batch_size, slot_count, num_candidates)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        layer_predicate_logits = (
            flat_predicate.view(
                self.num_layers, batch_size, slot_count, num_candidates
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        if self.validity_head is None:
            layer_validity_logits = layer_phrase_logits
            layer_validity_gate_logits = layer_phrase_logits.new_zeros(
                (self.num_layers, batch_size, slot_count)
            )
        else:
            flat_validity = torch.cat(layer_validity_chunks, dim=1)
            layer_validity_logits = (
                flat_validity.view(
                    self.num_layers, batch_size, slot_count, num_candidates
                )
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            flat_validity_gate = torch.cat(layer_validity_gate_chunks, dim=1)
            layer_validity_gate_logits = (
                flat_validity_gate.view(
                    self.num_layers, batch_size, slot_count
                )
                .contiguous()
            )
        if layer_confidence_base_chunks:
            flat_confidence_base = torch.cat(layer_confidence_base_chunks, dim=1)
            layer_confidence_base_logits = (
                flat_confidence_base.view(
                    self.num_layers, batch_size, slot_count, num_candidates
                )
                .permute(0, 1, 3, 2)
                .contiguous()
            )
        else:
            layer_confidence_base_logits = layer_phrase_logits

        valid_token = expression_valid_mask[None, :, None, :, None]
        valid_phrase = expression_valid_mask[None, :, None, :]
        predicate_valid_mask = (
            expression_valid_mask & effective_predicate_token_mask.any(dim=-1)
        )
        valid_predicate = predicate_valid_mask[None, :, None, :]
        layer_token_logits = layer_token_logits.masked_fill(
            ~valid_token, torch.finfo(layer_token_logits.dtype).min
        )
        layer_phrase_logits = layer_phrase_logits.masked_fill(
            ~valid_phrase, torch.finfo(layer_phrase_logits.dtype).min
        )
        if self.validity_head is None:
            # Preserve exact tensor values and the legacy graph when v14 is off.
            layer_validity_logits = layer_phrase_logits
        else:
            layer_validity_logits = layer_validity_logits.masked_fill(
                ~valid_phrase, torch.finfo(layer_validity_logits.dtype).min
            )
        layer_predicate_logits = layer_predicate_logits.masked_fill(
            ~valid_predicate, torch.finfo(layer_predicate_logits.dtype).min
        )
        final_phrase_logits = layer_phrase_logits[-1]
        final_validity_logits = layer_validity_logits[-1]
        final_validity_gate_logits = layer_validity_gate_logits[-1].masked_fill(
            ~expression_valid_mask, 0.0
        )
        final_rank_score = final_phrase_logits.sigmoid().masked_fill(
            ~expression_valid_mask[:, None, :], 0.0
        )
        final_score = final_validity_logits.sigmoid().masked_fill(
            ~expression_valid_mask[:, None, :], 0.0
        )
        return {
            "layer_token_logits": layer_token_logits,
            "final_token_logits": layer_token_logits[-1],
            "layer_phrase_logits": layer_phrase_logits,
            "final_phrase_logits": final_phrase_logits,
            "layer_validity_logits": layer_validity_logits,
            "final_validity_logits": final_validity_logits,
            "layer_validity_gate_logits": layer_validity_gate_logits,
            "final_validity_gate_logits": final_validity_gate_logits,
            "layer_confidence_base_logits": layer_confidence_base_logits,
            "final_confidence_base_logits": layer_confidence_base_logits[-1],
            "layer_predicate_logits": layer_predicate_logits,
            "final_predicate_logits": layer_predicate_logits[-1],
            "predicate_token_mask": effective_predicate_token_mask,
            "score_token_mask": effective_score_token_mask,
            "predicate_valid_mask": predicate_valid_mask,
            "final_rank_score": final_rank_score,
            "final_score": final_score,
            "expression_valid_mask": expression_valid_mask,
        }


__all__ = [
    "CONFIDENCE_OUTPUT_MODES",
    "SCORE_OWNERSHIP_MODES",
    "FixedBoxFullTextScorer",
    "build_stage_b_noncanonical_token_masks_from_ids",
    "build_stage_b_pair_token_diff_masks_from_ids",
    "normalize_stage_b_score_ownership",
    "select_stage_b_rank_confidence_logits",
    "validate_stage_b_fixed_text_scorer_checkpoint",
]

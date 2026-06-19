from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
from difflib import SequenceMatcher
import re

import datasets.transforms as T

_WS_RE = re.compile(r"\s+")
_PUNC_RE = re.compile(r"[^a-z0-9 _-]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _clean_phrase(value):
    text = str(value or "").replace("_", " ").strip()
    text = text[:-1].strip() if text.endswith(".") else text
    return _WS_RE.sub(" ", text).strip()


def _norm_text(value):
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = _PUNC_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize_with_offsets(value):
    text = _clean_phrase(value)
    out = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        norm = _norm_text(token)
        if norm:
            out.append({"text": token, "norm": norm, "start": int(match.start()), "end": int(match.end())})
    return out


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _find_word_span(text, candidate, *, ignore_case=False):
    if not text or not candidate:
        return None
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(candidate)}(?![0-9A-Za-z])", flags=flags)
    match = pattern.search(text)
    if match is None:
        return None
    return int(match.start()), int(match.end())


def _char_span_to_token_mask(tokenized, span, max_text_len):
    mask = torch.zeros((int(max_text_len),), dtype=torch.bool)
    start, end = int(span[0]), int(span[1])
    if end <= start:
        return mask
    beg_pos = tokenized.char_to_token(start)
    if beg_pos is None:
        for delta in (1, 2):
            if start + delta < end:
                beg_pos = tokenized.char_to_token(start + delta)
                if beg_pos is not None:
                    break
    end_pos = tokenized.char_to_token(end - 1)
    if end_pos is None:
        for delta in (2, 3):
            if end - delta >= start:
                end_pos = tokenized.char_to_token(end - delta)
                if end_pos is not None:
                    break
    if beg_pos is None or end_pos is None:
        return mask
    beg_pos = max(0, min(int(beg_pos), int(max_text_len) - 1))
    end_pos = max(0, min(int(end_pos), int(max_text_len) - 1))
    if end_pos >= beg_pos:
        mask[beg_pos : end_pos + 1] = True
    return mask


def _find_token_subsequence_start(haystack_tokens, needle_tokens):
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return None
    hay = [t["norm"] for t in haystack_tokens]
    needle = [t["norm"] for t in needle_tokens]
    for idx in range(0, len(hay) - len(needle) + 1):
        if hay[idx : idx + len(needle)] == needle:
            return int(idx)
    return None


def _changed_attribute_token_spans(phrase_text, replace_from, replace_to):
    from_tokens = _tokenize_with_offsets(replace_from)
    to_tokens = _tokenize_with_offsets(replace_to)
    if not to_tokens:
        return []
    changed_indices = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(
        None, [t["norm"] for t in from_tokens], [t["norm"] for t in to_tokens]
    ).get_opcodes():
        if tag in {"replace", "insert"}:
            changed_indices.extend(range(int(j1), int(j2)))
    if not changed_indices:
        return []
    local_span = _find_word_span(_clean_phrase(phrase_text), _clean_phrase(replace_to), ignore_case=False)
    if local_span is None:
        local_span = _find_word_span(_clean_phrase(phrase_text), _clean_phrase(replace_to), ignore_case=True)
    out = []
    if local_span is not None:
        base = int(local_span[0])
        for idx in changed_indices:
            tok = to_tokens[idx]
            out.append({"text": tok["text"], "norm": tok["norm"], "start": base + int(tok["start"]), "end": base + int(tok["end"])})
        return out
    phrase_tokens = _tokenize_with_offsets(phrase_text)
    start_idx = _find_token_subsequence_start(phrase_tokens, to_tokens)
    if start_idx is None:
        return None
    for idx in changed_indices:
        tok = phrase_tokens[start_idx + idx]
        out.append({"text": tok["text"], "norm": tok["norm"], "start": int(tok["start"]), "end": int(tok["end"])})
    return out


def _mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, local_spans, max_text_len):
    out = torch.zeros_like(phrase_mask)
    span_start = int(phrase_span[0])
    for local_start, local_end in local_spans:
        out = out | _char_span_to_token_mask(tokenized, (span_start + int(local_start), span_start + int(local_end)), max_text_len)
    return out & phrase_mask


def _build_caption_from_phrases(phrases):
    parts = []
    spans = []
    cursor = 0
    for idx, phrase in enumerate(phrases):
        if idx > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(phrase)
        cursor += len(phrase)
        spans.append((start, cursor))
        parts.append(" .")
        cursor += 2
    return "".join(parts), spans

class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        args=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        self.max_text_len = int(getattr(args, "max_text_len", 256)) if args is not None else 256
        self._text_tokenizer = None
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def _get_text_tokenizer(self):
        if self._text_tokenizer is None:
            from transformers import AutoTokenizer

            self._text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return self._text_tokenizer

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with  open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            print(f"  == total labels: {len(self.label_map)}")

    def _build_tn_token_masks(self, caption, caption_list, tn_records):
        phrases = [_clean_phrase(x) for x in (caption_list or []) if _clean_phrase(x)]
        if not phrases:
            phrases = [_clean_phrase(caption[:-1] if caption.endswith(".") else caption) or "object"]
        records = list(tn_records or [{} for _ in phrases])
        canonical_texts = []
        for idx, phrase in enumerate(phrases):
            rec = records[idx] if idx < len(records) and isinstance(records[idx], dict) else {}
            canonical = _clean_phrase(
                rec.get("head")
                or rec.get("head_phrase")
                or rec.get("canonical_name")
                or phrase
            )
            canonical_texts.append(canonical)

        canonical_caption, spans = _build_caption_from_phrases(phrases)
        if not str(caption or "").strip():
            caption = canonical_caption
        tokenized = self._get_text_tokenizer()(
            caption,
            truncation=True,
            max_length=int(self.max_text_len),
        )
        k = len(phrases)
        t = int(self.max_text_len)
        phrase_to_token_mask = torch.zeros((k, t), dtype=torch.bool)
        canonical_to_token_mask = torch.zeros((k, t), dtype=torch.bool)
        content_to_token_mask = torch.zeros((k, t), dtype=torch.bool)
        attr_neg_to_token_mask = torch.zeros((k, t), dtype=torch.bool)
        is_tn = torch.ones((k,), dtype=torch.bool)

        for idx, (phrase, span, canonical) in enumerate(zip(phrases, spans, canonical_texts)):
            phrase_mask = _char_span_to_token_mask(tokenized, span, t)
            phrase_to_token_mask[idx] = phrase_mask
            if not phrase_mask.any():
                continue

            candidates = [canonical]
            rec = records[idx] if idx < len(records) and isinstance(records[idx], dict) else {}
            for extra in (rec.get("head_phrase"), rec.get("canonical_name"), rec.get("head")):
                extra = _clean_phrase(extra)
                if extra and extra not in candidates:
                    candidates.append(extra)

            local_span = None
            for cand in candidates:
                local_span = _find_word_span(phrase, cand, ignore_case=False)
                if local_span is not None:
                    break
            if local_span is None:
                for cand in candidates:
                    local_span = _find_word_span(phrase, cand, ignore_case=True)
                    if local_span is not None:
                        break
            if local_span is not None:
                canonical_span = (int(span[0]) + int(local_span[0]), int(span[0]) + int(local_span[1]))
                canonical_to_token_mask[idx] = _char_span_to_token_mask(tokenized, canonical_span, t) & phrase_mask

            neg_mask = torch.zeros((t,), dtype=torch.bool)
            replace_from_values = _as_list(rec.get("replace_from", None))
            replace_to_values = _as_list(rec.get("replace_to", None))
            max_replacements = max(len(replace_from_values), len(replace_to_values))
            for ridx in range(max_replacements):
                replace_from = replace_from_values[ridx] if ridx < len(replace_from_values) else ""
                replace_to = replace_to_values[ridx] if ridx < len(replace_to_values) else ""
                changed_tokens = _changed_attribute_token_spans(phrase, replace_from, replace_to)
                if not changed_tokens:
                    continue
                token_spans = [
                    (int(tok["start"]), int(tok["end"]))
                    for tok in changed_tokens
                    if str(tok.get("norm", "")) not in {"a", "an", "the"}
                ]
                neg_mask = neg_mask | _mask_from_phrase_local_spans(tokenized, span, phrase_mask, token_spans, t)
            neg_mask = neg_mask & (~canonical_to_token_mask[idx])
            attr_neg_to_token_mask[idx] = neg_mask
            content_to_token_mask[idx] = phrase_mask & (~canonical_to_token_mask[idx]) & (~neg_mask)

        return {
            "phrase_to_token_mask": phrase_to_token_mask,
            "canonical_to_token_mask": canonical_to_token_mask,
            "content_to_token_mask": content_to_token_mask,
            "attr_neg_to_token_mask": attr_neg_to_token_mask,
            "negative_to_token_mask": attr_neg_to_token_mask,
            "is_tn": is_tn,
        }

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        is_negative = False
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            not_exhaustive = set(str(x) for x in meta.get("not_exhaustive_labels", []) or [])
            neg_labels = list(self.label_index.difference(pos_labels).difference(not_exhaustive))

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))
            
            # shuffle
            for i in range(len(vg_labels)-1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item:index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            is_negative = bool(anno.get("is_negative", meta.get("is_negative", False)))
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            if boxes:
                c = list(zip(boxes, caption_list))
                random.shuffle(c)
                boxes[:], caption_list[:] = zip(*c)
                uni_caption_list  = list(set(caption_list))
                label_map = {}
                for idx in range(len(uni_caption_list)):
                    label_map[uni_caption_list[idx]] = idx
                classes = [label_map[cap] for cap in caption_list]
                caption = ' . '.join(uni_caption_list) + ' .'
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                classes = torch.tensor(classes, dtype=torch.int64)
                caption_list = uni_caption_list
            else:
                caption_list = anno.get("caption_list", []) or meta.get("cap_list", [])
                if not caption_list:
                    caption = str(anno.get("caption", meta.get("caption", "object ."))).strip()
                    caption_list = [caption[:-1].strip() if caption.endswith(".") else caption]
                caption_list = [str(x).strip() for x in caption_list if str(x).strip()]
                if not caption_list:
                    caption_list = ["object"]
                caption = str(anno.get("caption", meta.get("caption", ""))).strip()
                if not caption:
                    caption = ' . '.join(caption_list) + ' .'
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                classes = torch.zeros((0,), dtype=torch.int64)
        target = {}
        image_id = meta.get("image_id", index)
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["image_id"] = torch.as_tensor([int(image_id)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        target["is_negative"] = torch.as_tensor([1 if self.dataset_mode == "VG" and is_negative else 0], dtype=torch.int64)
        if self.dataset_mode == "VG" and is_negative and boxes.numel() == 0:
            tn_records = anno.get("tn_records", []) if isinstance(anno, dict) else []
            if tn_records:
                target.update(self._build_tn_token_masks(caption, caption_list, tn_records))
        # size, cap_list, caption, bboxes, labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    # datadict_for_print = {
    #     'scales': scales,
    #     'max_size': max_size,
    #     'scales2_resize': scales2_resize,
    #     'scales2_crop': scales2_crop
    # }
    # print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = ODVGDataset(img_folder, ann_file, label_map, max_labels=args.max_labels,
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
            args=args,
    )
    return dataset


if __name__=="__main__":
    dataset_vg = ODVGDataset("path/GRIT-20M/data/","path/GRIT-20M/anno/grit_odvg_10k.jsonl",)
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)] 
    print(data)
    dataset_od = ODVGDataset("pathl/V3Det/",
        "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
        "path/V3Det/annotations/v3det_label_map.json",
    )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)] 
    print(data)

# python test.py \
#   --checkpoint /home/haoyi/Open-GroundingDino/exp_vg_multiclass_clean/multiclass_bertwarper_focal_epoch8.pt \
#   --canonical_json canonical_classes_with_aliases.json \
#   --max_len 24 \
#   --topk 5

import argparse
import json
from pathlib import Path

import torch
from transformers import BertTokenizerFast, BertModel

# 按你自己的工程路径改这行，比如:
# from groundingdino.models.bertwarper import BertModelWarper
from bertwarper import BertModelWarper


# 和训练时保持一致的模型定义
class MultiClassPhraseClassifier(torch.nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=2048):
        super().__init__()
        bert = BertModel.from_pretrained(bert_model_name)
        self.text_encoder = BertModelWarper(bert)
        hidden_size = self.text_encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def load_id2name_from_canonical(canonical_path):
    canonical_path = Path(canonical_path)
    with canonical_path.open("r", encoding="utf-8") as f:
        canonical_list = json.load(f)

    id2name = {}
    for cls in canonical_list:
        cid = cls["id"]
        base_name = cls.get("base_name") or cls.get("norm_name")
        if base_name is None:
            base_name = cls.get("raw_name", f"class_{cid}")
        id2name[int(cid)] = base_name
    return id2name


def build_id2name(ckpt, canonical_path=None):
    # 1) 优先用训练时保存在 ckpt 里的 id2name
    if "id2name" in ckpt:
        raw = ckpt["id2name"]
        id2name = {}
        for k, v in raw.items():
            # 可能是字符串 key，转成 int
            id2name[int(k)] = v
        return id2name

    # 2) 否则从 canonical json 里重建
    if canonical_path is None:
        raise ValueError(
            "Checkpoint 里没有 id2name，请用 --canonical_json 提供 canonical_classes_with_aliases.json"
        )
    return load_id2name_from_canonical(canonical_path)


def prepare_text(text: str) -> str:
    text = text.strip()
    # 训练时我们在 phrase 后都会补一个标点，这里保持一致
    if not text.endswith((".", "?", "!", "。", "？", "！")):
        text = text + "."
    return text


def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    bert_model_name = ckpt.get("bert_model_name", args.bert_model_name)
    num_classes = ckpt.get("num_classes", args.num_classes)

    print(f"[INFO] Using BERT model: {bert_model_name}")
    print(f"[INFO] num_classes from checkpoint: {num_classes}")

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

    model = MultiClassPhraseClassifier(
        bert_model_name=bert_model_name,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    id2name = build_id2name(ckpt, args.canonical_json)
    print(f"[INFO] Loaded {len(id2name)} class names.")

    topk = args.topk

    print("\n===== 测试模式启动 =====")
    print("输入一句短语，我帮你输出 top-k 类别预测。")
    print("输入 空行 或 'exit' / 'quit' 退出。\n")

    while True:
        text = input(">>> phrase: ").strip()
        if text == "" or text.lower() in ["exit", "quit"]:
            print("Bye.")
            break

        text_proc = prepare_text(text)

        enc = tokenizer(
            text_proc,
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.softmax(logits, dim=-1)[0]  # (C,)

        # top-k
        k = min(topk, probs.size(0))
        top_probs, top_indices = torch.topk(probs, k=k, dim=-1)

        print(f"\n[Input] {text_proc}")
        print("[Top-{} prediction]".format(k))
        for rank, (p, idx) in enumerate(zip(top_probs, top_indices), start=1):
            cid = int(idx.item())
            cname = id2name.get(cid, f"class_{cid}")
            print(f"  #{rank}:  id={cid:4d}  prob={p.item():.4f}  name={cname}")
        print("")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的分类器 checkpoint 路径，比如 best.pt",
    )
    parser.add_argument(
        "--canonical_json",
        type=str,
        default=None,
        help="canonical_classes_with_aliases.json，如果 ckpt 里已有 id2name 可以不填",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="备用的 BERT 名字（如果 ckpt 里没有 bert_model_name 就用这个）",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2048,
        help="备用 num_classes（如果 ckpt 里没有 num_classes 就用这个）",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=24,
        help="tokenizer 的 max_length，要和训练时保持一致",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="输出前 top-k 个类别",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制用 CPU 进行推理",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_loop(args)

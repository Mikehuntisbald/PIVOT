# python train_classifier_clean.py \      
#   --train_jsonl /media/haoyi/T9/data/vg_text_pairs_clean.jsonl \
#   --canonical_json /media/haoyi/T9/data/canonical_classes_with_aliases.json \
#   --bert_model_name bert-base-uncased \
#   --batch_size 256 \
#   --epochs 8 \
#   --max_len 24 \
#   --val_ratio 0.05 \
#   --output_dir ../../exp_vg_multiclass_clean \
#   --use_head_phrase \
#   --focal_gamma 2.0 \
#   --lr 5e-5 \
#   --lr_milestones 3 5 \
#   --lr_gamma 0.1 \
#   --early_stop_patience 3

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertModel

# 按你自己的工程路径改这行，比如:
# from groundingdino.models.bertwarper import BertModelWarper
from bertwarper import BertModelWarper


# =========================
# Dataset
# =========================

class VGTextDataset(Dataset):
    """
    从 vg_text_pairs.jsonl 里读：
      - phrase: 默认优先用 head_phrase，退化到 raw_phrase
      - label: class_id (要求在 canonical 里存在)

    这里会在 phrase 最后补一个 '.'，保持和其他 text encoder 输入风格统一。
    """

    def __init__(
        self,
        jsonl_path,
        canonical_json=None,
        tokenizer=None,
        max_length=24,
        use_head_phrase=False,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_head_phrase = use_head_phrase

        # 读 canonical：用来过滤合法 class_id，顺便存个 id->name 做 debug / 使用
        self.valid_class_ids = None
        self.id2name = None
        if canonical_json is not None:
            canonical_json = Path(canonical_json)
            with canonical_json.open("r", encoding="utf-8") as f:
                canonical_list = json.load(f)
            self.valid_class_ids = set()
            self.id2name = {}
            for cls in canonical_list:
                cid = cls["id"]
                base_name = cls.get("base_name") or cls.get("norm_name")
                if base_name is None:
                    base_name = cls.get("raw_name", f"class_{cid}")
                self.valid_class_ids.add(cid)
                self.id2name[cid] = base_name
            print(f"[INFO] Loaded {len(self.valid_class_ids)} canonical classes.")

        # 读 jsonl，构造样本
        self.samples = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                if use_head_phrase:
                    phrase = item.get("head_phrase") or item.get("raw_phrase")
                else:
                    phrase = item.get("raw_phrase") or item.get("head_phrase")

                if phrase is None:
                    continue

                cid = item.get("class_id")
                if cid is None:
                    continue

                if self.valid_class_ids is not None and cid not in self.valid_class_ids:
                    continue

                self.samples.append(
                    {
                        "phrase": phrase,
                        "class_id": int(cid),
                    }
                )

        if len(self.samples) == 0:
            raise ValueError("No valid samples loaded from jsonl.")

        # 推断 num_classes（假设 class_id 连续从 0 开始）
        self.num_classes = max(s["class_id"] for s in self.samples) + 1

        print(
            f"[INFO] Loaded {len(self.samples)} samples "
            f"from {self.jsonl_path.name}, num_classes={self.num_classes}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        phrase = item["phrase"]
        label = item["class_id"]

        # —— 在短语结尾加 '.'，避免重复加多次 —— 
        phrase = phrase.strip()
        if not phrase.endswith((".", "?", "!", "。", "？", "！")):
            phrase = phrase + "."

        enc = self.tokenizer(
            phrase,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": torch.tensor(label, dtype=torch.long),
        }


# =========================
# Focal Loss (multi-class)
# =========================

class FocalLoss(nn.Module):
    """
    多分类 Focal Loss：
      logits: (B, C)
      target: (B,)   int64 class id

    标准公式：
      FL = - alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha:
      - None: 不做 class reweight
      - scalar: 所有类同一个 alpha
      - tensor(C,): 每类一个 alpha
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # log_softmax & softmax 概率
        log_prob = nn.functional.log_softmax(logits, dim=-1)      # (B, C)
        prob = log_prob.exp()                                     # (B, C)

        # 取出每个样本对应的正确类概率 p_t 和 log(p_t)
        target = target.long()
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1)       # (B,)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.full_like(pt, float(self.alpha))
            else:
                # alpha 是一个 (C,) tensor
                alpha_t = self.alpha[target]
            loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        else:
            loss = -(1.0 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =========================
# Model: BertWarper + Linear
# =========================

class MultiClassPhraseClassifier(nn.Module):
    """
    使用 BertModelWarper 做 text encoder：

      - 文本 encoder: BertModelWarper(BertModel)
      - pooled_output 走 Dropout + Linear -> num_classes logits
    """

    def __init__(self, bert_model_name="bert-base-uncased", num_classes=2048):
        super().__init__()
        # 先加载 HuggingFace 的 BertModel
        bert = BertModel.from_pretrained(bert_model_name)
        # 再包一层 BertModelWarper（跟 GroundingDINO 的 text encoder 一致风格）
        self.text_encoder = BertModelWarper(bert)

        hidden_size = self.text_encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output  # (B, hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, C)
        return logits


# =========================
# Eval
# =========================

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = ce(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# =========================
# Train
# =========================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)

    dataset = VGTextDataset(
        jsonl_path=args.train_jsonl,
        canonical_json=args.canonical_json,
        tokenizer=tokenizer,
        max_length=args.max_len,
        use_head_phrase=args.use_head_phrase,
    )

    # train / val 划分
    if args.val_ratio > 0.0:
        val_size = int(len(dataset) * args.val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"[INFO] Train size: {train_size}, Val size: {val_size}")
    else:
        train_dataset = dataset
        val_dataset = None
        print(f"[INFO] Using all {len(train_dataset)} samples for training (no val).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = MultiClassPhraseClassifier(
        bert_model_name=args.bert_model_name,
        num_classes=dataset.num_classes,
    )

    if args.freeze_bert:
        for p in model.text_encoder.parameters():
            p.requires_grad = False
        print("[INFO] Freeze BERT parameters, only train classifier head.")

    model.to(device)

    # 用 FocalLoss
    criterion = FocalLoss(gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler（按 epoch 里程碑衰减）
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )

    best_val_acc = 0.0
    best_epoch = -1
    no_improve = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"]
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            global_step += 1
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                acc = running_correct / running_total if running_total > 0 else 0.0
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] "
                    f"Step {global_step} "
                    f"Train Loss: {avg_loss:.4f} "
                    f"Train Acc: {acc:.4f}"
                )
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        # ===== 每个 epoch 结束后做验证 / 保存 =====
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
            )

            # 保存当前 epoch 的 checkpoint（方便对比）
            ckpt = {
                "model_state_dict": model.state_dict(),
                "bert_model_name": args.bert_model_name,
                "num_classes": dataset.num_classes,
                "config": vars(args),
            }
            if dataset.id2name is not None:
                ckpt["id2name"] = dataset.id2name

            epoch_ckpt_path = out_dir / f"multiclass_bertwarper_focal_epoch{epoch+1}.pt"
            torch.save(ckpt, epoch_ckpt_path)
            print(f"[INFO] Saved epoch checkpoint to {epoch_ckpt_path}")

            # 维护 best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_ckpt_path = out_dir / "best.pt"
                torch.save(ckpt, best_ckpt_path)
                print(
                    f"[INFO] New best model at epoch {epoch+1}, "
                    f"Val Acc = {val_acc:.4f}, saved to {best_ckpt_path}"
                )
                no_improve = 0
            else:
                no_improve += 1
                print(
                    f"[INFO] No val improvement for {no_improve} epoch(s). "
                    f"Best so far: epoch {best_epoch} (Val Acc={best_val_acc:.4f})"
                )

            # Early stopping
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(
                    f"[INFO] Early stopping triggered at epoch {epoch+1}. "
                    f"Best epoch: {best_epoch} with Val Acc={best_val_acc:.4f}"
                )
                scheduler.step()
                break

        else:
            # 没有 val_loader 的情况也可以存一下最后的模型（可选）
            ckpt = {
                "model_state_dict": model.state_dict(),
                "bert_model_name": args.bert_model_name,
                "num_classes": dataset.num_classes,
                "config": vars(args),
            }
            if dataset.id2name is not None:
                ckpt["id2name"] = dataset.id2name
            epoch_ckpt_path = out_dir / f"multiclass_bertwarper_focal_epoch{epoch+1}.pt"
            torch.save(ckpt, epoch_ckpt_path)
            print(f"[INFO] Saved epoch checkpoint to {epoch_ckpt_path}")

        # 每个 epoch 结束后，scheduler 往前走一步
        scheduler.step()


# =========================
# Argparse
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="vg_text_pairs.jsonl 路径",
    )
    parser.add_argument(
        "--canonical_json",
        type=str,
        default="canonical_classes_with_aliases.json",
        help="canonical classes json，用来过滤 id 和保存名字",
    )

    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace 的 BERT 名字或本地路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_multiclass_bertwarper_focal",
        help="checkpoint 输出目录",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="从训练数据里划一部分做 val，0 表示不做验证",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="只训练最后的分类层",
    )
    parser.add_argument(
        "--use_head_phrase",
        action="store_true",
        help="优先用 head_phrase (默认用 raw_phrase)",
    )
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU 训练")

    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss 的 gamma 超参数",
    )

    parser.add_argument(
        "--lr_milestones",
        type=int,
        nargs="*",
        default=[3, 5],
        help="在这些 epoch 结束后做一次 lr 衰减，比如 [3,5] 表示在 3 和 5 之后 * gamma",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="学习率衰减倍率，配合 lr_milestones 使用",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="验证集 acc 连续多少个 epoch 没提升就 early stop；<=0 表示不开",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train(args)

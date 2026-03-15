"""
train.py  —  Full training pipeline with HuggingFace Trainer + custom loop.
Usage:
    python src/train.py --model roberta-base --epochs 4 --batch_size 16
"""

import os
import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from tqdm import tqdm

from dataset import load_isot, load_liar, load_custom_csv, get_demo_dataframe, split_dataset, FakeNewsDataset, dataset_stats
from model import build_model, FakeNewsClassifier
from evaluate import compute_metrics_from_logits


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Training Arguments Parser
# ──────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Train Fake News Detector")
    parser.add_argument("--model", type=str, default="roberta-base",
                        help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="demo",
                        choices=["demo", "isot", "liar", "custom"],
                        help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing dataset files")
    parser.add_argument("--custom_csv", type=str, default=None,
                        help="Path to custom CSV file (if --dataset custom)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--use_custom_head", action="store_true",
                        help="Use FakeNewsClassifier instead of HF head")
    parser.add_argument("--use_hf_trainer", action="store_true",
                        help="Use HuggingFace Trainer API")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# HuggingFace Trainer-based Training
# ──────────────────────────────────────────────────────────────
def train_with_hf_trainer(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")

    # Load data
    df = _load_data(args)
    dataset_stats(df)
    train_df, val_df, test_df = split_dataset(df)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = FakeNewsDataset(train_df, tokenizer, args.max_length)
    val_dataset   = FakeNewsDataset(val_df,   tokenizer, args.max_length)
    test_dataset  = FakeNewsDataset(test_df,  tokenizer, args.max_length)

    # Model
    model = build_model(args.model, num_labels=2, dropout=args.dropout)
    model.to(device)

    output_path = Path(args.output_dir) / args.model.replace("/", "_")

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_dir=str(output_path / "logs"),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_from_logits,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n🚀 Starting training with HuggingFace Trainer...")
    trainer.train()

    # Evaluate on test set
    print("\n📊 Final evaluation on test set:")
    results = trainer.evaluate(test_dataset)
    print(json.dumps(results, indent=2))

    # Save tokenizer alongside model
    tokenizer.save_pretrained(str(output_path))
    print(f"\n✅ Model saved to {output_path}")
    return trainer, results


# ──────────────────────────────────────────────────────────────
# Manual PyTorch Training Loop (more control / educational)
# ──────────────────────────────────────────────────────────────
def train_manual_loop(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    # Load data
    df = _load_data(args)
    dataset_stats(df)
    train_df, val_df, test_df = split_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = FakeNewsDataset(train_df, tokenizer, args.max_length)
    val_dataset   = FakeNewsDataset(val_df,   tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    # Model
    if args.use_custom_head:
        model = FakeNewsClassifier(backbone_name=args.model, dropout=args.dropout)
    else:
        model = build_model(args.model, num_labels=2, dropout=args.dropout)
    model.to(device)

    # Optimizer — separate weight decay for biases/norms
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    output_path = Path(args.output_dir) / args.model.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training for {args.epochs} epochs | {total_steps} total steps\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = total_loss / len(train_loader)

        # ── Validate ──
        val_metrics = _evaluate_loop(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"\nEpoch {epoch} → "
              f"train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f}\n")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_pretrained(str(output_path)) if hasattr(model, "save_pretrained") else torch.save(model.state_dict(), str(output_path / "best_weights.pt"))
            tokenizer.save_pretrained(str(output_path))
            print(f"  💾 New best model saved (F1={best_val_f1:.4f})")

    # Save training history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Best Val F1: {best_val_f1:.4f}")
    return history


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _load_data(args):
    if args.dataset == "demo":
        print("⚠️  Using demo data (8 samples). Use --dataset isot/liar for real training.")
        return get_demo_dataframe()
    elif args.dataset == "isot":
        return load_isot(args.data_dir)
    elif args.dataset == "liar":
        return load_liar(args.data_dir)
    elif args.dataset == "custom":
        return load_custom_csv(args.custom_csv)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def _evaluate_loop(model, loader, device) -> dict:
    """Run validation loop and return metrics."""
    import torch.nn.functional as F
    from evaluate import compute_metrics_from_arrays

    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            logits  = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics_from_arrays(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    if args.use_hf_trainer:
        train_with_hf_trainer(args)
    else:
        train_manual_loop(args)

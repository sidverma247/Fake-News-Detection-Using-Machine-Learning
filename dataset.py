"""
dataset.py  —  Data loading, cleaning & preprocessing
Supports: LIAR, ISOT, FakeNewsNet, and custom CSV datasets.
"""

import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────
# Label Maps
# ──────────────────────────────────────────────────────────────
BINARY_LABEL_MAP = {"real": 1, "true": 1, "1": 1,
                    "fake": 0, "false": 0, "0": 0}

LIAR_6_CLASS_MAP = {
    "true": 5, "mostly-true": 4, "half-true": 3,
    "barely-true": 2, "false": 1, "pants-fire": 0
}


# ──────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove URLs, special chars, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)           # Remove URLs
    text = re.sub(r"<[^>]+>", "", text)                   # Remove HTML tags
    text = re.sub(r"[^\w\s.,!?'\"-]", " ", text)          # Keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()              # Collapse whitespace
    return text


# ──────────────────────────────────────────────────────────────
# Dataset Loaders
# ──────────────────────────────────────────────────────────────
def load_isot(data_dir: str) -> pd.DataFrame:
    """
    Load ISOT Fake News dataset.
    Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    Files: True.csv, Fake.csv
    """
    true_path = Path(data_dir) / "True.csv"
    fake_path = Path(data_dir) / "Fake.csv"

    true_df = pd.read_csv(true_path)
    true_df["label"] = 1

    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
    return df[["text", "label"]].dropna()


def load_liar(data_dir: str, split: str = "train") -> pd.DataFrame:
    """
    Load LIAR dataset (TSV format).
    Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    """
    path = Path(data_dir) / f"{split}.tsv"
    columns = [
        "id", "label", "statement", "subject", "speaker",
        "job", "state", "party", "barely_true_cnt", "false_cnt",
        "half_true_cnt", "mostly_true_cnt", "pants_fire_cnt", "context"
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=columns)
    df["text"] = df["statement"].apply(clean_text)

    # Binary: map to real/fake
    real_labels = {"true", "mostly-true", "half-true"}
    df["label"] = df["label"].apply(lambda x: 1 if x in real_labels else 0)
    return df[["text", "label"]].dropna()


def load_custom_csv(path: str,
                    text_col: str = "text",
                    label_col: str = "label") -> pd.DataFrame:
    """Load any CSV with a text column and a binary label column."""
    df = pd.read_csv(path)
    df["text"] = df[text_col].apply(clean_text)
    df["label"] = df[label_col].map(BINARY_LABEL_MAP).fillna(df[label_col]).astype(int)
    return df[["text", "label"]].dropna()


# ──────────────────────────────────────────────────────────────
# Dataset Statistics
# ──────────────────────────────────────────────────────────────
def dataset_stats(df: pd.DataFrame) -> Dict:
    """Print and return dataset statistics."""
    stats = {
        "total_samples": len(df),
        "real_news": int((df["label"] == 1).sum()),
        "fake_news": int((df["label"] == 0).sum()),
        "avg_text_length": float(df["text"].apply(len).mean()),
        "max_text_length": int(df["text"].apply(len).max()),
        "class_balance": float((df["label"] == 1).sum() / len(df)),
    }
    print("=" * 45)
    print("         DATASET STATISTICS")
    print("=" * 45)
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    print("=" * 45)
    return stats


# ──────────────────────────────────────────────────────────────
# Train / Val / Test Split
# ──────────────────────────────────────────────────────────────
def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train / val / test."""
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df["label"], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size),
        stratify=temp_df["label"], random_state=random_state
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    """PyTorch Dataset for tokenized news articles."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Demo data (for testing without real datasets)
# ──────────────────────────────────────────────────────────────
DEMO_DATA = [
    {"text": "Scientists confirm new renewable energy breakthrough reduces costs by 40%.", "label": 1},
    {"text": "Government releases annual budget report with detailed financial projections.", "label": 1},
    {"text": "New study shows regular exercise reduces risk of heart disease significantly.", "label": 1},
    {"text": "SHOCKING: Doctors don't want you to know this one weird trick to cure cancer!", "label": 0},
    {"text": "EXPOSED: World leaders secretly meeting to control global weather patterns!", "label": 0},
    {"text": "Scientists REVEAL vaccines contain mind-control microchips, government coverup!", "label": 0},
    {"text": "Central bank raises interest rates to combat rising inflation pressures.", "label": 1},
    {"text": "BREAKING: Aliens have landed and the media is covering it up RIGHT NOW!", "label": 0},
]


def get_demo_dataframe() -> pd.DataFrame:
    """Return a small demo dataframe for quick testing."""
    return pd.DataFrame(DEMO_DATA)

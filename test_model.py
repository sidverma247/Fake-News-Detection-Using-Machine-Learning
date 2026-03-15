"""tests/test_model.py — Unit tests for dataset and model modules."""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import clean_text, get_demo_dataframe, split_dataset, FakeNewsDataset
from evaluate import compute_metrics_from_arrays


# ──────────────────────────────────────────────────────────────
# Dataset Tests
# ──────────────────────────────────────────────────────────────
class TestCleanText:
    def test_removes_urls(self):
        assert "http" not in clean_text("Visit http://example.com for more")

    def test_removes_html(self):
        assert "<b>" not in clean_text("<b>Bold</b> text")

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""


class TestDatasetSplit:
    def test_split_sizes(self):
        df = get_demo_dataframe()
        # Duplicate for larger split
        import pandas as pd
        df = pd.concat([df] * 20, ignore_index=True)
        train, val, test = split_dataset(df, val_size=0.1, test_size=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(df)

    def test_no_label_leakage(self):
        df = get_demo_dataframe()
        import pandas as pd
        df = pd.concat([df] * 20, ignore_index=True)
        train, val, test = split_dataset(df)
        # Indices should be disjoint
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0


class TestFakeNewsDataset:
    def test_len(self):
        from transformers import AutoTokenizer
        df = get_demo_dataframe()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds = FakeNewsDataset(df, tokenizer, max_length=64)
        assert len(ds) == len(df)

    def test_item_keys(self):
        from transformers import AutoTokenizer
        df = get_demo_dataframe()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds = FakeNewsDataset(df, tokenizer, max_length=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_shapes(self):
        from transformers import AutoTokenizer
        df = get_demo_dataframe()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds = FakeNewsDataset(df, tokenizer, max_length=64)
        item = ds[0]
        assert item["input_ids"].shape == torch.Size([64])
        assert item["attention_mask"].shape == torch.Size([64])


# ──────────────────────────────────────────────────────────────
# Metrics Tests
# ──────────────────────────────────────────────────────────────
class TestMetrics:
    def test_perfect_predictions(self):
        logits = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        labels = np.array([1, 0, 1])
        metrics = compute_metrics_from_arrays(logits, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_metric_keys(self):
        logits = np.random.randn(20, 2)
        labels = np.random.randint(0, 2, 20)
        metrics = compute_metrics_from_arrays(logits, labels)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_metric_range(self):
        logits = np.random.randn(50, 2)
        labels = np.random.randint(0, 2, 50)
        metrics = compute_metrics_from_arrays(logits, labels)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

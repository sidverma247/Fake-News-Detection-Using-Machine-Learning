"""
model.py  —  Transformer model definition for Fake News Detection.
Supports BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa out-of-the-box.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,
    DebertaV2ForSequenceClassification,
    PreTrainedModel,
)
from typing import Optional, Dict, Any


# ──────────────────────────────────────────────────────────────
# Supported Models
# ──────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "bert-base-uncased":          BertForSequenceClassification,
    "bert-large-uncased":         BertForSequenceClassification,
    "roberta-base":               RobertaForSequenceClassification,
    "roberta-large":              RobertaForSequenceClassification,
    "distilbert-base-uncased":    DistilBertForSequenceClassification,
    "albert-base-v2":             AlbertForSequenceClassification,
    "microsoft/deberta-v3-base":  DebertaV2ForSequenceClassification,
}


# ──────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────
def build_model(model_name: str, num_labels: int = 2, dropout: float = 0.1) -> PreTrainedModel:
    """
    Instantiate a transformer classifier from HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of output classes (2 for binary fake/real).
        dropout: Dropout rate for classifier head.

    Returns:
        A HuggingFace PreTrainedModel ready for fine-tuning.
    """
    if model_name in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY[model_name]
    else:
        # Fallback: use AutoModel — works for any HF checkpoint
        model_class = None

    if model_class:
        model = model_class.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            ignore_mismatched_sizes=True,
        )
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model       : {model_name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable   : {trainable_params:,}")
    return model


# ──────────────────────────────────────────────────────────────
# Custom Model with Extra Attention Head (Optional)
# ──────────────────────────────────────────────────────────────
class FakeNewsClassifier(nn.Module):
    """
    Custom wrapper: transformer backbone + multi-layer classification head.
    Useful when you want more control over the classifier (e.g. extra layers,
    feature concatenation, multi-task outputs).
    """

    def __init__(
        self,
        backbone_name: str = "roberta-base",
        num_labels: int = 2,
        dropout: float = 0.3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_labels),
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        logits = self.classifier(cls_output)              # (batch, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def save(self, path: str):
        """Save model weights and backbone config."""
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/classifier_weights.pt")
        self.backbone.config.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, backbone_name: str, **kwargs) -> "FakeNewsClassifier":
        """Load model from saved weights."""
        model = cls(backbone_name=backbone_name, **kwargs)
        model.load_state_dict(torch.load(f"{path}/classifier_weights.pt", map_location="cpu"))
        model.eval()
        return model


# ──────────────────────────────────────────────────────────────
# Freeze / Unfreeze Utilities
# ──────────────────────────────────────────────────────────────
def freeze_backbone(model: nn.Module, num_layers_to_unfreeze: int = 2):
    """
    Freeze all backbone layers except the last N transformer layers.
    Useful for fine-tuning with limited data.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Unfreeze last N encoder layers (works for BERT-like models)
    encoder = None
    if hasattr(model, "backbone"):
        backbone = model.backbone
    else:
        backbone = model

    for attr in ["encoder", "transformer", "roberta", "bert", "distilbert"]:
        if hasattr(backbone, attr):
            encoder = getattr(backbone, attr)
            break

    if encoder and hasattr(encoder, "layer"):
        layers = encoder.layer
        for layer in layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params after freezing: {trainable:,}")

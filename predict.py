"""
predict.py  —  Inference utilities for single text or batch prediction.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_NAMES = {0: "FAKE", 1: "REAL"}
LABEL_EMOJIS = {0: "🔴", 1: "🟢"}


class FakeNewsPredictor:
    """
    High-level predictor class.
    Loads a fine-tuned model and tokenizer for inference.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")

    def predict(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        return_all_scores: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict whether news is real or fake.

        Args:
            text: Single string or list of strings.
            max_length: Max token length.
            return_all_scores: Include probability for both classes.

        Returns:
            dict (single) or list of dicts with prediction results.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits.cpu().numpy()

        probs = self._softmax(logits)
        results = []

        for i, (prob, logit) in enumerate(zip(probs, logits)):
            pred_label = int(np.argmax(prob))
            result = {
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "prediction": LABEL_NAMES[pred_label],
                "confidence": float(prob[pred_label]),
                "emoji": LABEL_EMOJIS[pred_label],
            }
            if return_all_scores:
                result["scores"] = {
                    "FAKE": float(prob[0]),
                    "REAL": float(prob[1]),
                }
            results.append(result)

        return results[0] if single else results

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def explain(self, text: str, top_k: int = 10) -> Dict:
        """
        Basic attention-based explanation (which tokens mattered most).
        Returns top-k tokens by mean attention weight across heads/layers.
        """
        encoding = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=True)

        # Average attention across all layers and heads
        # attentions: tuple of (1, num_heads, seq_len, seq_len)
        all_attn = torch.stack(outputs.attentions)  # (num_layers, 1, heads, seq, seq)
        avg_attn = all_attn.mean(dim=[0, 1, 2])     # (seq, seq)
        token_importance = avg_attn.mean(dim=0).cpu().numpy()  # (seq,)

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        token_scores = list(zip(tokens, token_importance.tolist()))

        # Filter out special tokens
        token_scores = [(t, s) for t, s in token_scores
                        if t not in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]]
        token_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "top_tokens": token_scores[:top_k],
            "prediction": self.predict(text),
        }


# ──────────────────────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict fake/real news")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    predictor = FakeNewsPredictor(args.model_path)
    result = predictor.predict(args.text)
    print(f"\n{result['emoji']} Prediction : {result['prediction']}")
    print(f"   Confidence : {result['confidence']:.2%}")
    print(f"   FAKE score : {result['scores']['FAKE']:.4f}")
    print(f"   REAL score : {result['scores']['REAL']:.4f}")

"""Model and baseline predictors for fill-mask evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

from src.baseline.frequency import FrequencyBaseline
from src.baseline.ngram import NgramContextBaseline
from src.eval.mask_window import build_masked_window


class Predictor(ABC):
    @abstractmethod
    def predict_topk(self, row: pd.Series, k: int = 10) -> list[str]:
        raise NotImplementedError


class BaselinePredictor(Predictor):
    def __init__(self, baseline: FrequencyBaseline | NgramContextBaseline) -> None:
        self.baseline = baseline

    def predict_topk(self, row: pd.Series, k: int = 10) -> list[str]:
        return self.baseline.predict_topk(row, k=k)


class FillMaskPredictor(Predictor):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        batch_size: int = 8,
        use_pipeline: bool = False,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.mask_token = self.tokenizer.mask_token or "[MASK]"

        if use_pipeline and resolved_device == "cpu":
            self.pipe = pipeline(
                "fill-mask",
                model=model_name_or_path,
                tokenizer=self.tokenizer,
                top_k=10,
                device=-1,
            )
            self.model = None
        else:
            self.pipe = None
            self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
            self.model.to(resolved_device)
            self.model.eval()

    def _window_for_row(self, row: pd.Series) -> str | None:
        built = build_masked_window(
            str(row["original_text"]),
            int(row["mask_position"]),
            str(row["label_char"]),
            self.mask_token,
        )
        if built is None:
            return None
        return built[0]

    def predict_topk(self, row: pd.Series, k: int = 10) -> list[str]:
        text = self._window_for_row(row)
        if text is None:
            return []
        if self.pipe is not None:
            results = self.pipe(text, top_k=k)
            chars: list[str] = []
            for item in results:
                token = str(item["token_str"]).strip()
                if len(token) == 1 and token not in chars:
                    chars.append(token)
            return chars[:k]
        return self._predict_single_model(text, k)

    def predict_batch(self, rows: list[pd.Series], k: int = 10) -> list[list[str]]:
        scored = self.predict_batch_scored(rows, k=k)
        return [chars for chars, _scores in scored]

    def predict_batch_scored(
        self, rows: list[pd.Series], k: int = 10
    ) -> list[tuple[list[str], list[float]]]:
        if self.pipe is not None:
            outputs: list[tuple[list[str], list[float]]] = []
            for row in rows:
                text = self._window_for_row(row)
                if text is None:
                    outputs.append(([], []))
                    continue
                results = self.pipe(text, top_k=k)
                chars: list[str] = []
                scores: list[float] = []
                for item in results:
                    token = str(item["token_str"]).strip()
                    if len(token) == 1 and token not in chars:
                        chars.append(token)
                        scores.append(float(item.get("score", 0.0)))
                    if len(chars) >= k:
                        break
                outputs.append((chars[:k], scores[:k]))
            return outputs

        texts: list[str] = []
        valid_idx: list[int] = []
        outputs = [([], []) for _ in rows]
        for idx, row in enumerate(rows):
            text = self._window_for_row(row)
            if text is None:
                continue
            texts.append(text)
            valid_idx.append(idx)

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batch_idx = valid_idx[start : start + self.batch_size]
            batch_preds = self._predict_batch_model_scored(batch_texts, k)
            for idx, preds in zip(batch_idx, batch_preds):
                outputs[idx] = preds
        return outputs

    def _predict_single_model(self, text: str, k: int) -> list[str]:
        chars, _scores = self._predict_batch_model_scored([text], k)[0]
        return chars

    def _predict_batch_model(self, texts: list[str], k: int) -> list[list[str]]:
        return [chars for chars, _scores in self._predict_batch_model_scored(texts, k)]

    def _predict_batch_model_scored(
        self, texts: list[str], k: int
    ) -> list[tuple[list[str], list[float]]]:
        assert self.model is not None
        mask_id = self.tokenizer.mask_token_id
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits

        results: list[tuple[list[str], list[float]]] = []
        for row_idx in range(logits.shape[0]):
            input_ids = encoded["input_ids"][row_idx]
            mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                results.append(([], []))
                continue
            mask_pos = int(mask_positions[0])
            token_logits = logits[row_idx, mask_pos]
            probs = torch.softmax(token_logits, dim=-1)
            top_ids = torch.topk(probs, k=min(k * 3, probs.shape[-1])).indices.tolist()
            chars: list[str] = []
            scores: list[float] = []
            for token_id in top_ids:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                if token.startswith("##"):
                    continue
                ch = token.strip()
                if len(ch) == 1 and ch not in chars:
                    chars.append(ch)
                    scores.append(float(probs[token_id].item()))
                if len(chars) >= k:
                    break
            results.append((chars[:k], scores[:k]))
        return results

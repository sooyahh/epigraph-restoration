"""
EXP-012 SillokBERT — masked-Hanja prediction inference stub (FastAPI).

Contract (matches handoff/fixtures/predict_samples.json):

  POST /predict
    body : { "text": "...貧道[MASK]居...", "top_k": 10 }
    200  : {
             "input_text": "...",
             "mask_index": <int char-offset of [MASK]>,
             "candidates": [ { "rank": 1, "char": "所", "score": 0.0435 }, ... ],
             "model_version": "exp012_sillokbert_gpu_best_valid_loss",
             "meta": { "max_seq_len": 512, "context_window": 64 }
           }

Run:
  pip install -r requirements.txt
  set MODEL_DIR=..\..\notebooks\best        # Windows (PowerShell: $env:MODEL_DIR="...")
  uvicorn app:app --host 0.0.0.0 --port 8000

Notes:
- Loads a local fine-tuned checkpoint (model.safetensors + tokenizer.*), NOT the HF hub.
- Replicates EXP-012 preprocessing: keep only ±CONTEXT_WINDOW chars around the mask.
- Performance is an early baseline (Exact@1 ~33.75%); this stub exists to unblock the
  pipeline, not to represent final model quality.
"""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "best"))
MODEL_VERSION = "exp012_sillokbert_gpu_best_valid_loss"
MAX_SEQ_LEN = 512
CONTEXT_WINDOW = 64          # chars kept on each side of [MASK]  (EXP-012 setting)
MASK_LITERAL = "[MASK]"

app = FastAPI(title="SillokBERT masked-Hanja prediction", version=MODEL_VERSION)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw text containing exactly one [MASK] token.")
    top_k: int = Field(10, ge=1, le=50)


class Candidate(BaseModel):
    rank: int
    char: str
    score: float


class PredictResponse(BaseModel):
    input_text: str
    mask_index: int
    candidates: list[Candidate]
    model_version: str = MODEL_VERSION
    meta: dict = {"max_seq_len": MAX_SEQ_LEN, "context_window": CONTEXT_WINDOW}


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def _windowed(text: str) -> str:
    """Keep ±CONTEXT_WINDOW chars around [MASK] so the mask survives 512-token truncation."""
    i = text.find(MASK_LITERAL)
    left = text[:i][-CONTEXT_WINDOW:]
    right = text[i + len(MASK_LITERAL):][:CONTEXT_WINDOW]
    return left + MASK_LITERAL + right


@app.get("/health")
def health():
    ok = os.path.isdir(MODEL_DIR)
    return {"status": "ok" if ok else "model_dir_missing", "model_dir": MODEL_DIR, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.text.count(MASK_LITERAL) != 1:
        raise HTTPException(status_code=422, detail=f"text must contain exactly one {MASK_LITERAL} token.")

    tokenizer, model, device = _load()
    mask_char_index = len(req.text[: req.text.find(MASK_LITERAL)])

    enc = tokenizer(_windowed(req.text), return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    enc = {k: v.to(device) for k, v in enc.items()}

    mask_positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_positions.numel() != 1:
        raise HTTPException(status_code=422, detail="[MASK] did not tokenize to exactly one mask position.")
    mask_pos = mask_positions.item()

    with torch.no_grad():
        logits = model(**enc).logits[0, mask_pos]
    probs = F.softmax(logits, dim=-1)
    scores, ids = probs.topk(req.top_k)

    candidates = [
        Candidate(rank=r + 1, char=tokenizer.convert_ids_to_tokens(int(tid)), score=round(float(s), 6))
        for r, (s, tid) in enumerate(zip(scores.tolist(), ids.tolist()))
    ]
    return PredictResponse(input_text=req.text, mask_index=mask_char_index, candidates=candidates)

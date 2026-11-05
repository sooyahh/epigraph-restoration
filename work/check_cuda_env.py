"""Check CUDA / PyTorch / transformers before PC bang session."""

from __future__ import annotations

import sys


def main() -> None:
    print("python:", sys.version.replace("\n", " "))
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda_device:", torch.cuda.get_device_name(0))
            print("cuda_memory_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    except ImportError as exc:
        print("torch: NOT INSTALLED", exc)
        sys.exit(1)

    try:
        import transformers

        print("transformers:", transformers.__version__)
    except ImportError as exc:
        print("transformers: NOT INSTALLED", exc)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("ddokbaro/SillokBert")
        print("SillokBert tokenizer: OK", "vocab", tok.vocab_size)
    except Exception as exc:
        print("SillokBert download/tokenizer: FAIL", exc)
        sys.exit(1)

    print("\nEnvironment OK for GPU fine-tuning." if torch.cuda.is_available() else "\nCUDA not available — use CPU scripts only.")


if __name__ == "__main__":
    main()

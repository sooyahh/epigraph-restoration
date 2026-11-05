"""DEPRECATED: legacy GPU FT entrypoint that used EXP-007 (conflicts with zero-shot).

Use: python work/run_exp012_gpu.py
Official IDs: EXP-007=zero-shot SillokBERT; EXP-011=CPU FT baseline; EXP-012=planned GPU FT.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "work/run_exp007_gpu.py is retired for fine-tuning.\n"
        "EXP-007 remains SillokBERT zero-shot in docs/EXPERIMENT_LOG.md.\n"
        "For planned GPU full-data fine-tuning use:\n"
        "  python work/run_exp012_gpu.py\n"
        "See reports/local_gpu_runbook.md."
    )


if __name__ == "__main__":
    main()
    sys.exit(1)

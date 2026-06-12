from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.lm.char_ngram import build_char_ngram_lm
from harness.src.pipeline.fusion import run_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Harness v2 offline multi-model fusion.")
    parser.add_argument("--set-name", default="debug100")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--anchor-model", default="paraformer-zh")
    parser.add_argument("--candidate-mode", choices=["baseline_top1", "top1", "all"], default="all")
    parser.add_argument(
        "--strategy",
        choices=["priority", "lm", "support", "consensus", "prior_consensus"],
        default="prior_consensus",
    )
    parser.add_argument("--train-manifest", default="data/manifests/full/train.jsonl")
    parser.add_argument("--output-name")
    args = parser.parse_args()

    name = args.output_name or f"{args.split}.{args.candidate_mode}.{args.strategy}"
    output_dir = ROOT / "outputs" / args.set_name / "harness_v2" / name
    prediction_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    lm = build_char_ngram_lm(ROOT / args.train_manifest)
    summary = run_fusion(
        project_root=ROOT,
        set_name=args.set_name,
        split=args.split,
        models=args.models,
        anchor_model=args.anchor_model,
        candidate_mode=args.candidate_mode,
        strategy=args.strategy,
        lm=lm,
        prediction_path=prediction_path,
        summary_path=summary_path,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print(f"predictions: {prediction_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


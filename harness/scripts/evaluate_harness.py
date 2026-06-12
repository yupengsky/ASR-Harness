from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.audio.variants import DEFAULT_VARIANTS
from harness.src.lm.char_ngram import build_char_ngram_lm
from harness.src.models.base import BackendDependencyError
from harness.src.models.factory import create_backend
from harness.src.pipeline.harness import run_harness


def default_output_paths(model: str, manifest: Path, output_root: Path) -> tuple[Path, Path]:
    set_name = manifest.parent.name
    split_name = manifest.stem
    out_dir = output_root / set_name / "harness_v1" / model
    return out_dir / f"{split_name}_predictions.jsonl", out_dir / f"{split_name}_summary.json"


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR Harness v1: audio multi-view + char n-gram reranking.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="data/manifests/debug100/test.jsonl")
    parser.add_argument("--train-manifest", default="data/manifests/full/train.jsonl")
    parser.add_argument("--registry", default="harness/configs/models.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ngram-order", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--score-margin", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    train_manifest_path = ROOT / args.train_manifest
    registry_path = ROOT / args.registry
    prediction_path, summary_path = default_output_paths(args.model, manifest_path, ROOT / args.output_root)

    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "manifest": _relative_or_absolute(manifest_path),
            "train_manifest": _relative_or_absolute(train_manifest_path),
            "predictions": _relative_or_absolute(prediction_path),
            "summary": _relative_or_absolute(summary_path),
            "device": args.device,
            "variants": args.variants,
            "score_margin": args.score_margin,
            "limit": args.limit,
        }, ensure_ascii=False, indent=2))
        return

    try:
        load_started = time.perf_counter()
        backend = create_backend(args.model, ROOT, registry_path, device=args.device)
        model_load_time = time.perf_counter() - load_started
        lm = build_char_ngram_lm(train_manifest_path, order=args.ngram_order, alpha=args.alpha)
        summary = run_harness(
            backend=backend,
            project_root=ROOT,
            manifest_path=manifest_path,
            train_manifest_path=train_manifest_path,
            prediction_path=prediction_path,
            summary_path=summary_path,
            lm=lm,
            device=args.device,
            model_load_time_s=model_load_time,
            limit=args.limit,
            variant_names=tuple(args.variants),
            score_margin=args.score_margin,
        )
    except BackendDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc

    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print(f"predictions: {prediction_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

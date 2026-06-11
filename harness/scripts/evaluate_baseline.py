from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.models.base import BackendDependencyError
from harness.src.models.factory import create_backend
from harness.src.models.registry import load_model_registry
from harness.src.pipeline.baseline import run_baseline


def default_output_paths(model: str, manifest: Path, output_root: Path) -> tuple[Path, Path]:
    set_name = manifest.parent.name
    split_name = manifest.stem
    out_dir = output_root / set_name / "baseline" / model
    return out_dir / f"{split_name}_predictions.jsonl", out_dir / f"{split_name}_summary.json"


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified ASR baseline evaluation.")
    parser.add_argument("--model", help="Model name from harness/configs/models.json")
    parser.add_argument("--manifest", default="data/manifests/smoke/test.jsonl")
    parser.add_argument("--registry", default="harness/configs/models.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without loading model weights.")
    args = parser.parse_args()

    registry_path = ROOT / args.registry
    registry = load_model_registry(registry_path)
    if args.list_models:
        for name, spec in registry.items():
            print(f"{name}\t{spec.get('type')}\t{spec.get('priority')}\t{spec.get('path')}")
        return

    if not args.model:
        raise SystemExit("--model is required unless --list-models is used")

    manifest_path = ROOT / args.manifest
    prediction_path, summary_path = default_output_paths(
        args.model,
        manifest_path,
        ROOT / args.output_root,
    )
    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "manifest": _relative_or_absolute(manifest_path),
            "predictions": _relative_or_absolute(prediction_path),
            "summary": _relative_or_absolute(summary_path),
            "device": args.device,
            "limit": args.limit,
        }, ensure_ascii=False, indent=2))
        return

    try:
        load_started = time.perf_counter()
        backend = create_backend(args.model, ROOT, registry_path, device=args.device)
        model_load_time = time.perf_counter() - load_started
        summary = run_baseline(
            backend=backend,
            project_root=ROOT,
            manifest_path=manifest_path,
            prediction_path=prediction_path,
            summary_path=summary_path,
            device=args.device,
            model_load_time_s=model_load_time,
            limit=args.limit,
        )
    except BackendDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    print(f"predictions: {prediction_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

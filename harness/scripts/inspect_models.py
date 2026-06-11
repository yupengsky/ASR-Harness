from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BY_TYPE = {
    "funasr": ["config.yaml", "model.pt"],
    "transformers-whisper": ["config.json", "preprocessor_config.json", "tokenizer.json"],
    "transformers": ["config.json"],
    "vosk": ["am", "conf", "graph"],
    "sherpa-onnx": ["config.yaml", "tokens.txt"],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, spec: dict[str, Any]) -> Path:
    path = root / spec["path"]
    if path.exists():
        return path
    for legacy in spec.get("legacy_paths", []):
        legacy_path = root / legacy
        if legacy_path.exists():
            return legacy_path
    return path


def inspect_model(root: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    model_path = resolve_path(root, spec)
    required = REQUIRED_BY_TYPE.get(spec.get("type", ""), [])
    missing = [entry for entry in required if not (model_path / entry).exists()]
    status = "ok" if model_path.exists() and not missing else "needs_attention"
    return {
        "name": name,
        "type": spec.get("type", ""),
        "priority": spec.get("priority", ""),
        "path": model_path.relative_to(root).as_posix() if model_path.exists() else spec["path"],
        "exists": model_path.exists(),
        "missing_required": missing,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local model directories without loading model weights.")
    parser.add_argument("--registry", default="harness/configs/models.json")
    parser.add_argument("--output", default="outputs/logs/model_inventory.json")
    args = parser.parse_args()

    root = project_root()
    registry = load_registry(root / args.registry)
    results = [inspect_model(root, name, spec) for name, spec in registry.items()]

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for item in results:
        missing = ",".join(item["missing_required"]) if item["missing_required"] else "-"
        print(f"{item['name']}: {item['status']} exists={item['exists']} missing={missing}")
    print(f"wrote {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()


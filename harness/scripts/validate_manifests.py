from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_jsonl(root: Path, path: Path) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            utt_id = obj.get("utt_id", "")
            wav_path = obj.get("wav_path", "")
            text = obj.get("text", "")
            if not utt_id:
                errors.append(f"{path}:{line_no}: empty utt_id")
            if utt_id in seen:
                errors.append(f"{path}:{line_no}: duplicate utt_id {utt_id}")
            seen.add(utt_id)
            if not wav_path:
                errors.append(f"{path}:{line_no}: empty wav_path")
            elif Path(wav_path).is_absolute():
                errors.append(f"{path}:{line_no}: wav_path must be relative: {wav_path}")
            elif not (root / wav_path).is_file():
                errors.append(f"{path}:{line_no}: missing wav {wav_path}")
            if not text:
                errors.append(f"{path}:{line_no}: empty text")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated ASR manifests.")
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--sets", nargs="+", default=["smoke", "debug100", "full"])
    args = parser.parse_args()

    root = project_root()
    manifest_root = root / args.manifest_root
    all_errors: list[str] = []
    for set_name in args.sets:
        for split in SPLITS:
            path = manifest_root / set_name / f"{split}.jsonl"
            if not path.is_file():
                all_errors.append(f"missing manifest {path}")
                continue
            all_errors.extend(validate_jsonl(root, path))

    if all_errors:
        for error in all_errors[:100]:
            print(error)
        if len(all_errors) > 100:
            print(f"... {len(all_errors) - 100} more errors")
        raise SystemExit(1)

    print("manifest validation passed")


if __name__ == "__main__":
    main()


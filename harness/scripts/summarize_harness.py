from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Harness v1 summary files into one JSON table.")
    parser.add_argument("--summary-root", default="outputs")
    parser.add_argument("--output", default="outputs/logs/harness_v1_summary.json")
    args = parser.parse_args()

    root = project_root()
    rows = []
    for path in sorted((root / args.summary_root).glob("*/harness_v1/*/*_summary*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = path.relative_to(root).parts
        if not data.get("set"):
            data["set"] = parts[1] if len(parts) > 1 else ""
        if not data.get("model"):
            data["model"] = parts[3] if len(parts) > 3 else ""
        data["summary_path"] = path.relative_to(root).as_posix()
        rows.append(data)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        margin = row.get("score_margin", "-")
        print(
            f"{row.get('model')}\t{row.get('set')}\tcount={row.get('count')}\t"
            f"cer={row.get('cer'):.4f}\tmargin={margin}\t{row['summary_path']}"
        )
    print(f"wrote {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()

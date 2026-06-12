from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved Harness candidate predictions with a score margin.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--score-margin", type=float, required=True)
    args = parser.parse_args()

    root = project_root()
    input_path = root / args.input
    output_path = root / args.output
    summary_path = root / args.summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_distance = 0
    total_ref_chars = 0
    changed = 0
    improved = 0
    worsened = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            row = json.loads(line)
            orig = row["candidates"][0]
            best = max(row["candidates"], key=lambda c: (float(c["score"]), -int(c["variant_order"])))
            if best["norm"] != orig["norm"] and float(best["score"]) - float(orig["score"]) < args.score_margin:
                best = orig

            from harness.src.metrics.cer import cer_stats, normalize_chinese_text

            stats = cer_stats(row["ref"], best["text"])
            orig_stats = cer_stats(row["ref"], orig["text"])
            row["hyp"] = best["text"]
            row["hyp_norm"] = normalize_chinese_text(best["text"])
            row["edit_distance"] = stats.distance
            row["ref_chars"] = stats.ref_chars
            row["cer"] = stats.cer
            row["orig_hyp"] = orig["text"]
            row["orig_cer"] = orig_stats.cer
            row["orig_edit_distance"] = orig_stats.distance
            row["selected_variant"] = best["variant"]
            row["selected_score"] = best["score"]
            row["changed_from_orig"] = best["norm"] != orig["norm"]
            row["improved_vs_orig"] = stats.distance < orig_stats.distance
            row["worsened_vs_orig"] = stats.distance > orig_stats.distance

            total_distance += stats.distance
            total_ref_chars += stats.ref_chars
            changed += int(row["changed_from_orig"])
            improved += int(row["improved_vs_orig"])
            worsened += int(row["worsened_vs_orig"])
            rows.append(row)
            dst.write(json.dumps(row, ensure_ascii=False))
            dst.write("\n")

    rel_parts = input_path.relative_to(root).parts
    model_name = rel_parts[3] if len(rel_parts) > 3 else None
    summary = {
        "source_predictions": input_path.relative_to(root).as_posix(),
        "rescored_predictions": output_path.relative_to(root).as_posix(),
        "score_margin": args.score_margin,
        "count": len(rows),
        "total_ref_chars": total_ref_chars,
        "total_edit_distance": total_distance,
        "cer": (total_distance / total_ref_chars) if total_ref_chars else 0.0,
        "changed_from_orig": changed,
        "improved_vs_orig": improved,
        "worsened_vs_orig": worsened,
        "model": model_name,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

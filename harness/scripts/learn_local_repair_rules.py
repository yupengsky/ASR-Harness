from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.metrics.cer import cer_stats


def extract_contextual_edits(anchor: str, target: str, context: int) -> list[dict[str, str]]:
    matcher = difflib.SequenceMatcher(a=anchor, b=target, autojunk=False)
    edits: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = anchor[max(0, i1 - context) : i1]
        right = anchor[i2 : min(len(anchor), i2 + context)]
        edits.append(
            {
                "left": left,
                "from": anchor[i1:i2],
                "to": target[j1:j2],
                "right": right,
                "tag": tag,
            }
        )
    return edits


def candidate_supports_edit(candidate_norms: list[str], edit: dict[str, str]) -> bool:
    target_pattern = edit["left"] + edit["to"] + edit["right"]
    if not target_pattern:
        return False
    return any(target_pattern in norm for norm in candidate_norms)


def learn_rules(
    calibration_predictions: Path,
    anchor_name: str,
    context: int,
    max_candidate_distance: int,
    min_gain: int,
) -> dict[str, object]:
    rules: dict[tuple[str, str, str, str], dict[str, object]] = {}
    with calibration_predictions.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = str(row["ref"])
            candidates = row["candidates"]
            anchor = next(candidate for candidate in candidates if candidate["name"] == anchor_name)
            anchor_norm = str(anchor["norm"])
            anchor_distance = cer_stats(ref, anchor_norm).distance
            candidate_norms = [str(candidate["norm"]) for candidate in candidates if candidate["name"] != anchor_name]

            for candidate in candidates:
                if candidate["name"] == anchor_name:
                    continue
                candidate_norm = str(candidate["norm"])
                candidate_distance = cer_stats(ref, candidate_norm).distance
                if anchor_distance - candidate_distance < min_gain:
                    continue
                if abs(len(anchor_norm) - len(candidate_norm)) > max_candidate_distance:
                    continue
                if sum(1 for a, b in zip(anchor_norm, candidate_norm) if a != b) > max_candidate_distance:
                    if len(anchor_norm) == len(candidate_norm):
                        continue
                edits = extract_contextual_edits(anchor_norm, candidate_norm, context)
                if not edits or len(edits) > max_candidate_distance:
                    continue
                for edit in edits:
                    if not candidate_supports_edit(candidate_norms, edit):
                        continue
                    key = (edit["left"], edit["from"], edit["to"], edit["right"])
                    entry = rules.setdefault(
                        key,
                        {
                            "left": edit["left"],
                            "from": edit["from"],
                            "to": edit["to"],
                            "right": edit["right"],
                            "support": 0,
                            "examples": [],
                        },
                    )
                    entry["support"] = int(entry["support"]) + 1
                    entry["examples"].append(
                        {
                            "utt_id": row["utt_id"],
                            "candidate": candidate["name"],
                            "anchor_distance": anchor_distance,
                            "candidate_distance": candidate_distance,
                        }
                    )

    contextual_replacements = sorted(
        rules.values(),
        key=lambda rule: (-int(rule["support"]), str(rule["left"]), str(rule["from"]), str(rule["to"]), str(rule["right"])),
    )
    return {
        "source": str(calibration_predictions.relative_to(ROOT)),
        "anchor": anchor_name,
        "context": context,
        "max_candidate_distance": max_candidate_distance,
        "min_gain": min_gain,
        "contextual_replacements": contextual_replacements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn contextual local repair rules from a calibration set.")
    parser.add_argument("--calibration-predictions", default="outputs/representative100/harness_v3/oracle/predictions.jsonl")
    parser.add_argument("--anchor", default="qwen3-asr-1.7b")
    parser.add_argument("--output", default="harness/configs/local_repair_rules.learned.json")
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--max-candidate-distance", type=int, default=4)
    parser.add_argument("--min-gain", type=int, default=1)
    args = parser.parse_args()

    output_path = ROOT / args.output
    rules = learn_rules(
        calibration_predictions=ROOT / args.calibration_predictions,
        anchor_name=args.anchor,
        context=args.context,
        max_candidate_distance=args.max_candidate_distance,
        min_gain=args.min_gain,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.relative_to(ROOT)), "rules": len(rules["contextual_replacements"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

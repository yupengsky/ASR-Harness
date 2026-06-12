from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.src.data.manifest import read_jsonl
from harness.src.metrics.cer import cer_stats, edit_distance, normalize_chinese_text


@dataclass(frozen=True)
class PredictionInput:
    name: str
    weight: float
    path: Path


def parse_prediction(value: str) -> PredictionInput:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use NAME=WEIGHT=PATH.")
    name, weight, raw_path = parts
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("NAME cannot be empty.")
    return PredictionInput(name=name, weight=float(weight), path=ROOT / raw_path)


def read_predictions(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["utt_id"])] = row
    return rows


def align_sequences(left: list[str], right: list[str]) -> list[tuple[str | None, str | None]]:
    n = len(left)
    m = len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if left[i - 1] == right[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + sub_cost, "diag"),
                (dp[i - 1][j] + 1, "up"),
                (dp[i][j - 1] + 1, "left"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    aligned: list[tuple[str | None, str | None]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            aligned.append((left[i - 1], right[j - 1]))
            i -= 1
            j -= 1
        elif move == "up":
            aligned.append((left[i - 1], None))
            i -= 1
        else:
            aligned.append((None, right[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def bin_consensus_char(vote_bin: dict[str, float], anchor_char: str | None) -> str:
    best_weight = max(vote_bin.values())
    winners = [char for char, weight in vote_bin.items() if weight == best_weight]
    if anchor_char in winners:
        return anchor_char or ""
    non_empty = sorted(char for char in winners if char)
    return non_empty[0] if non_empty else ""


def bins_to_chars(bins: list[dict[str, object]]) -> list[str]:
    chars: list[str] = []
    for vote_bin in bins:
        char = bin_consensus_char(vote_bin["votes"], vote_bin.get("anchor_char"))  # type: ignore[arg-type]
        chars.append(char)
    return chars


def weighted_char_vote(candidates: list[dict[str, object]], anchor_name: str) -> str:
    ordered = sorted(candidates, key=lambda item: 0 if item["name"] == anchor_name else 1)
    anchor = ordered[0]
    anchor_chars = list(str(anchor["norm"]))
    bins: list[dict[str, object]] = [
        {"anchor_char": char, "votes": {char: float(anchor["weight"])}}
        for char in anchor_chars
    ]
    processed_weight = float(anchor["weight"])

    for candidate in ordered[1:]:
        current_chars = bins_to_chars(bins)
        candidate_chars = list(str(candidate["norm"]))
        alignment = align_sequences(current_chars, candidate_chars)
        old_bins = bins
        new_bins: list[dict[str, object]] = []
        old_index = 0
        for current_char, candidate_char in alignment:
            if current_char is None and candidate_char is not None:
                new_bins.append(
                    {
                        "anchor_char": None,
                        "votes": {"": processed_weight, candidate_char: float(candidate["weight"])},
                    }
                )
                continue

            vote_bin = old_bins[old_index]
            old_index += 1
            votes = vote_bin["votes"]  # type: ignore[assignment]
            if candidate_char is None:
                votes[""] = votes.get("", 0.0) + float(candidate["weight"])
            else:
                votes[candidate_char] = votes.get(candidate_char, 0.0) + float(candidate["weight"])
            new_bins.append(vote_bin)
        bins = new_bins
        processed_weight += float(candidate["weight"])

    return "".join(char for char in bins_to_chars(bins) if char)


def candidate_consensus(candidates: list[dict[str, object]], anchor_name: str) -> dict[str, object]:
    scored: list[dict[str, object]] = []
    for candidate in candidates:
        score = float(candidate["weight"])
        text = str(candidate["norm"])
        for other in candidates:
            distance = edit_distance(text, str(other["norm"])) / max(len(text), len(str(other["norm"])), 1)
            score -= 0.75 * float(other["weight"]) * distance
        if candidate["name"] == anchor_name:
            score += 0.05
        scored.append({**candidate, "score": score})
    return max(scored, key=lambda item: float(item["score"]))


def load_repair_rules(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"phrase_replacements": [], "enable_supported_suffix_completion": False}
    return json.loads(path.read_text(encoding="utf-8"))


def local_repair(candidates: list[dict[str, object]], anchor_name: str, rules: dict[str, object]) -> tuple[str, list[str]]:
    anchor = next(candidate for candidate in candidates if candidate["name"] == anchor_name)
    current = str(anchor["norm"])
    applied: list[str] = []
    non_anchor_norms = [str(candidate["norm"]) for candidate in candidates if candidate["name"] != anchor_name]

    for rule in rules.get("contextual_replacements", []):
        left = str(rule["left"])
        source = str(rule["from"])
        target = str(rule["to"])
        right = str(rule["right"])
        source_pattern = left + source + right
        target_pattern = left + target + right
        support_norms = [norm for norm in non_anchor_norms if target_pattern in norm]
        if source_pattern in current and support_norms:
            replacement_count = 0
            while source_pattern in current:
                repaired = _apply_best_supported_replacement(current, source_pattern, target_pattern, support_norms)
                current_distance = min(edit_distance(current, norm) for norm in support_norms)
                repaired_distance = min(edit_distance(repaired, norm) for norm in support_norms)
                if repaired == current or repaired_distance >= current_distance:
                    break
                current = repaired
                replacement_count += 1
            if replacement_count:
                applied.append(f"{source_pattern}->{target_pattern}x{replacement_count}")

    for rule in rules.get("phrase_replacements", []):
        source = str(rule["from"])
        target = str(rule["to"])
        if source in current and any(target in norm for norm in non_anchor_norms):
            current = current.replace(source, target)
            applied.append(f"{source}->{target}")

    if bool(rules.get("enable_supported_suffix_completion", False)):
        support_counts: dict[str, int] = {}
        for norm in non_anchor_norms:
            support_counts[norm] = support_counts.get(norm, 0) + 1
        for norm, support in support_counts.items():
            if support < 2:
                continue
            if len(norm) == len(current) + 1 and norm.startswith(current) and norm[-1] in norm[max(0, len(norm) - 10) : -1]:
                current = norm
                applied.append("supported_suffix_completion")
                break

    return current, applied


def _apply_best_supported_replacement(current: str, source_pattern: str, target_pattern: str, support_norms: list[str]) -> str:
    best_text = current
    best_score: tuple[int, int] | None = None
    start = 0
    while True:
        position = current.find(source_pattern, start)
        if position < 0:
            break
        candidate = current[:position] + target_pattern + current[position + len(source_pattern) :]
        distance_to_support = min(edit_distance(candidate, norm) for norm in support_norms)
        score = (distance_to_support, position)
        if best_score is None or score < best_score:
            best_score = score
            best_text = candidate
        start = position + 1
    return best_text


def select_hypothesis(
    strategy: str,
    candidates: list[dict[str, object]],
    anchor_name: str,
    ref: str,
    repair_rules: dict[str, object],
) -> tuple[str, dict[str, object]]:
    anchor = next(candidate for candidate in candidates if candidate["name"] == anchor_name)
    if strategy == "anchor":
        return str(anchor["hyp"]), {"selected_source": anchor_name, "score": None}
    if strategy == "candidate_consensus":
        selected = candidate_consensus(candidates, anchor_name)
        return str(selected["hyp"]), {"selected_source": selected["name"], "score": selected["score"]}
    if strategy == "weighted_vote":
        voted_norm = weighted_char_vote(candidates, anchor_name)
        return voted_norm, {"selected_source": "weighted_vote", "score": None}
    if strategy == "cascade_fast_exact":
        fast_candidates = [candidate for candidate in candidates if candidate["name"] != anchor_name]
        if fast_candidates and len({candidate["norm"] for candidate in fast_candidates}) == 1:
            selected = max(fast_candidates, key=lambda candidate: float(candidate["weight"]))
            return str(selected["hyp"]), {"selected_source": f"fast_exact:{selected['name']}", "score": None}
        return str(anchor["hyp"]), {"selected_source": anchor_name, "score": None}
    if strategy == "local_repair":
        repaired, applied = local_repair(candidates, anchor_name, repair_rules)
        source = "local_repair" if applied else anchor_name
        return repaired, {"selected_source": source, "score": None, "applied_repairs": applied}
    if strategy == "oracle":
        selected = min(candidates, key=lambda candidate: cer_stats(ref, str(candidate["hyp"])).distance)
        return str(selected["hyp"]), {"selected_source": selected["name"], "score": None}
    raise ValueError(f"Unknown strategy: {strategy}")


def default_output_paths(manifest_path: Path, strategy: str, output_root: Path, harness_name: str) -> tuple[Path, Path]:
    set_name = manifest_path.parent.name
    out_dir = output_root / set_name / harness_name / strategy
    return out_dir / "predictions.jsonl", out_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline multi-model ensemble harness.")
    parser.add_argument("--manifest", default="data/manifests/representative100/test.jsonl")
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument(
        "--strategy",
        choices=["anchor", "candidate_consensus", "weighted_vote", "cascade_fast_exact", "local_repair", "oracle"],
        required=True,
    )
    parser.add_argument("--repair-rules", default="harness/configs/local_repair_rules.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output-name")
    parser.add_argument("--harness-name", default="harness_v3")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_name = args.output_name or args.strategy
    prediction_path, summary_path = default_output_paths(
        manifest_path,
        output_name,
        ROOT / args.output_root,
        args.harness_name,
    )

    items = read_jsonl(manifest_path)
    inputs: list[PredictionInput] = args.prediction
    if args.anchor not in {item.name for item in inputs}:
        raise SystemExit(f"Anchor is not in predictions: {args.anchor}")
    rows_by_model = {item.name: read_predictions(item.path) for item in inputs}
    weights = {item.name: item.weight for item in inputs}
    repair_rules_path = ROOT / args.repair_rules if args.repair_rules else None
    repair_rules = load_repair_rules(repair_rules_path)

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    total_distance = 0
    total_ref_chars = 0
    anchor_distance = 0
    oracle_distance = 0
    changed = 0
    improved = 0
    worsened = 0
    selected_source_counts: dict[str, int] = {}

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, item in enumerate(items, start=1):
            candidates: list[dict[str, object]] = []
            for model_input in inputs:
                row = rows_by_model[model_input.name][item.utt_id]
                hyp = str(row["hyp"])
                candidates.append(
                    {
                        "name": model_input.name,
                        "weight": weights[model_input.name],
                        "hyp": hyp,
                        "norm": normalize_chinese_text(hyp),
                        "edit_distance": int(row.get("edit_distance", cer_stats(item.text, hyp).distance)),
                    }
                )

            selected_hyp, selection_meta = select_hypothesis(
                args.strategy,
                candidates,
                args.anchor,
                item.text,
                repair_rules,
            )
            anchor_hyp = str(next(candidate for candidate in candidates if candidate["name"] == args.anchor)["hyp"])
            selected_stats = cer_stats(item.text, selected_hyp)
            anchor_stats = cer_stats(item.text, anchor_hyp)
            oracle_ed = min(cer_stats(item.text, str(candidate["hyp"])).distance for candidate in candidates)
            source = str(selection_meta["selected_source"])
            selected_source_counts[source] = selected_source_counts.get(source, 0) + 1

            total_distance += selected_stats.distance
            total_ref_chars += selected_stats.ref_chars
            anchor_distance += anchor_stats.distance
            oracle_distance += oracle_ed
            changed += int(normalize_chinese_text(selected_hyp) != normalize_chinese_text(anchor_hyp))
            improved += int(selected_stats.distance < anchor_stats.distance)
            worsened += int(selected_stats.distance > anchor_stats.distance)

            f.write(
                json.dumps(
                    {
                        "index": index,
                        "utt_id": item.utt_id,
                        "split": item.split,
                        "wav_path": item.wav_path,
                        "ref": item.text,
                        "anchor": args.anchor,
                        "anchor_hyp": anchor_hyp,
                        "hyp": selected_hyp,
                        "hyp_norm": normalize_chinese_text(selected_hyp),
                        "edit_distance": selected_stats.distance,
                        "ref_chars": selected_stats.ref_chars,
                        "cer": selected_stats.cer,
                        "selected_source": source,
                        "selected_score": selection_meta["score"],
                        "applied_repairs": selection_meta.get("applied_repairs", []),
                        "anchor_edit_distance": anchor_stats.distance,
                        "oracle_edit_distance": oracle_ed,
                        "changed_from_anchor": normalize_chinese_text(selected_hyp) != normalize_chinese_text(anchor_hyp),
                        "improved_vs_anchor": selected_stats.distance < anchor_stats.distance,
                        "worsened_vs_anchor": selected_stats.distance > anchor_stats.distance,
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

    summary = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "strategy": args.strategy,
        "anchor": args.anchor,
        "models": [{"name": item.name, "weight": item.weight, "path": str(item.path.relative_to(ROOT))} for item in inputs],
        "repair_rules": str(repair_rules_path.relative_to(ROOT)) if repair_rules_path else None,
        "count": len(items),
        "total_ref_chars": total_ref_chars,
        "anchor_edit_distance": anchor_distance,
        "anchor_cer": anchor_distance / total_ref_chars if total_ref_chars else 0.0,
        "oracle_edit_distance": oracle_distance,
        "oracle_cer": oracle_distance / total_ref_chars if total_ref_chars else 0.0,
        "total_edit_distance": total_distance,
        "cer": total_distance / total_ref_chars if total_ref_chars else 0.0,
        "changed_from_anchor": changed,
        "improved_vs_anchor": improved,
        "worsened_vs_anchor": worsened,
        "selected_source_counts": selected_source_counts,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"predictions: {prediction_path.relative_to(ROOT)}")
    print(f"summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

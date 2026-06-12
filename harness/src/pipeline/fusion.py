from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.src.lm.char_ngram import CharNGramLM
from harness.src.metrics.cer import cer_stats, edit_distance, normalize_chinese_text


DEFAULT_MODEL_PRIORS = {
    "paraformer-zh": 3.0,
    "sensevoice-small": 2.2,
    "sherpa-onnx-paraformer-zh-small": 2.0,
    "vosk-full": 1.4,
    "moonshine-tiny-zh": 1.1,
    "whisper-small": 0.8,
    "vosk-small": 0.7,
    "whisper-tiny": 0.4,
}


@dataclass(frozen=True)
class FusionSummary:
    set_name: str
    split: str
    models: list[str]
    anchor_model: str
    candidate_mode: str
    strategy: str
    count: int
    total_ref_chars: int
    anchor_edit_distance: int
    anchor_cer: float
    oracle_edit_distance: int
    oracle_cer: float
    total_edit_distance: int
    cer: float
    changed_from_anchor: int
    improved_vs_anchor: int
    worsened_vs_anchor: int


def run_fusion(
    project_root: Path,
    set_name: str,
    split: str,
    models: list[str],
    anchor_model: str,
    candidate_mode: str,
    strategy: str,
    lm: CharNGramLM,
    prediction_path: Path,
    summary_path: Path,
    priors: dict[str, float] | None = None,
) -> FusionSummary:
    priors = priors or DEFAULT_MODEL_PRIORS
    rows_by_model = {
        model: _load_model_rows(project_root, set_name, split, model, candidate_mode)
        for model in models
    }
    common_utts = sorted(set.intersection(*(set(rows) for rows in rows_by_model.values())))
    if not common_utts:
        raise ValueError("No common utterances across selected models")

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    total_distance = 0
    total_ref_chars = 0
    anchor_distance = 0
    oracle_distance = 0
    changed = 0
    improved = 0
    worsened = 0

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for index, utt_id in enumerate(common_utts, start=1):
            anchor_row = rows_by_model[anchor_model][utt_id]
            ref = anchor_row["ref"]
            ref_chars = int(anchor_row["ref_chars"])
            candidates = _collect_candidates(rows_by_model, models, utt_id, candidate_mode)
            anchor_candidate = _anchor_candidate(candidates, anchor_model)
            selected = _select_candidate(candidates, strategy, lm, priors)

            selected_stats = cer_stats(ref, selected["text"])
            anchor_stats = cer_stats(ref, anchor_candidate["text"])
            oracle_ed = min(cer_stats(ref, candidate["text"]).distance for candidate in candidates)

            total_distance += selected_stats.distance
            total_ref_chars += ref_chars
            anchor_distance += anchor_stats.distance
            oracle_distance += oracle_ed
            changed += int(selected["norm"] != anchor_candidate["norm"])
            improved += int(selected_stats.distance < anchor_stats.distance)
            worsened += int(selected_stats.distance > anchor_stats.distance)

            f.write(json.dumps({
                "index": index,
                "utt_id": utt_id,
                "split": split,
                "wav_path": anchor_row["wav_path"],
                "ref": ref,
                "anchor_model": anchor_model,
                "anchor_hyp": anchor_candidate["text"],
                "anchor_norm": anchor_candidate["norm"],
                "anchor_edit_distance": anchor_stats.distance,
                "hyp": selected["text"],
                "hyp_norm": normalize_chinese_text(selected["text"]),
                "selected_models": sorted(selected["models"]),
                "selected_sources": selected["sources"],
                "selected_score": selected["score"],
                "edit_distance": selected_stats.distance,
                "ref_chars": selected_stats.ref_chars,
                "cer": selected_stats.cer,
                "oracle_edit_distance": oracle_ed,
                "changed_from_anchor": selected["norm"] != anchor_candidate["norm"],
                "improved_vs_anchor": selected_stats.distance < anchor_stats.distance,
                "worsened_vs_anchor": selected_stats.distance > anchor_stats.distance,
                "candidates": candidates,
            }, ensure_ascii=False))
            f.write("\n")

    summary = FusionSummary(
        set_name=set_name,
        split=split,
        models=models,
        anchor_model=anchor_model,
        candidate_mode=candidate_mode,
        strategy=strategy,
        count=len(common_utts),
        total_ref_chars=total_ref_chars,
        anchor_edit_distance=anchor_distance,
        anchor_cer=anchor_distance / total_ref_chars if total_ref_chars else 0.0,
        oracle_edit_distance=oracle_distance,
        oracle_cer=oracle_distance / total_ref_chars if total_ref_chars else 0.0,
        total_edit_distance=total_distance,
        cer=total_distance / total_ref_chars if total_ref_chars else 0.0,
        changed_from_anchor=changed,
        improved_vs_anchor=improved,
        worsened_vs_anchor=worsened,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_model_rows(
    project_root: Path,
    set_name: str,
    split: str,
    model: str,
    candidate_mode: str,
) -> dict[str, dict[str, object]]:
    if candidate_mode == "baseline_top1":
        path = project_root / "outputs" / set_name / "baseline" / model / f"{split}_predictions.jsonl"
    else:
        path = project_root / "outputs" / set_name / "harness_v1" / model / f"{split}_predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["utt_id"]] = row
    return rows


def _collect_candidates(
    rows_by_model: dict[str, dict[str, dict[str, object]]],
    models: list[str],
    utt_id: str,
    candidate_mode: str,
) -> list[dict[str, object]]:
    dedup: dict[str, dict[str, object]] = {}
    for model in models:
        row = rows_by_model[model][utt_id]
        if candidate_mode in ("baseline_top1", "top1"):
            raw_candidates = [{"variant": "orig", "text": row.get("orig_hyp", row["hyp"])}]
        elif candidate_mode == "all":
            raw_candidates = row["candidates"]
        else:
            raise ValueError(f"Unknown candidate_mode: {candidate_mode}")

        for candidate in raw_candidates:
            text = str(candidate["text"])
            norm = normalize_chinese_text(text)
            if not norm:
                continue
            entry = dedup.setdefault(norm, {
                "text": text,
                "norm": norm,
                "models": set(),
                "sources": [],
            })
            entry["models"].add(model)
            entry["sources"].append({
                "model": model,
                "variant": candidate.get("variant", "orig"),
            })

    candidates = list(dedup.values())
    for candidate in candidates:
        candidate["models"] = sorted(candidate["models"])
    return candidates


def _anchor_candidate(candidates: list[dict[str, object]], anchor_model: str) -> dict[str, object]:
    for candidate in candidates:
        if anchor_model in candidate["models"]:
            for source in candidate["sources"]:
                if source["model"] == anchor_model and source["variant"] == "orig":
                    return candidate
    raise ValueError(f"Anchor model candidate not found: {anchor_model}")


def _select_candidate(
    candidates: list[dict[str, object]],
    strategy: str,
    lm: CharNGramLM,
    priors: dict[str, float],
) -> dict[str, object]:
    scored = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["score"] = _score_candidate(candidate, candidates, strategy, lm, priors)
        scored.append(candidate)
    return max(scored, key=lambda c: c["score"])


def _score_candidate(
    candidate: dict[str, object],
    candidates: list[dict[str, object]],
    strategy: str,
    lm: CharNGramLM,
    priors: dict[str, float],
) -> float:
    source_prior = max(priors.get(model, 0.0) for model in candidate["models"])
    lm_score = lm.score_avg_logprob(str(candidate["norm"]))
    model_support = len(candidate["models"])
    source_support = len(candidate["sources"])

    if strategy == "priority":
        return source_prior
    if strategy == "lm":
        return lm_score
    if strategy == "support":
        return 10.0 * model_support + source_support + 0.01 * lm_score
    if strategy == "consensus":
        return _consensus_score(candidate, candidates, priors) + 0.05 * lm_score + 0.3 * model_support
    if strategy == "prior_consensus":
        return (
            source_prior
            + 0.7 * model_support
            + 0.25 * _consensus_score(candidate, candidates, priors)
            + 0.03 * lm_score
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _consensus_score(
    candidate: dict[str, object],
    candidates: list[dict[str, object]],
    priors: dict[str, float],
) -> float:
    score = 0.0
    text = str(candidate["norm"])
    for other in candidates:
        other_text = str(other["norm"])
        weight = sum(priors.get(source["model"], 0.0) for source in other["sources"])
        distance = edit_distance(text, other_text) / max(len(text), len(other_text), 1)
        score -= weight * distance
    return score


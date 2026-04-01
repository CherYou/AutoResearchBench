import argparse
import json
import logging
import os
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import configure_logging, ensure_parent_dir, load_env_file


LOGGER = logging.getLogger(__name__)
ENV_PATH = ROOT / ".env"
load_env_file(ENV_PATH)

JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
MAX_IOU_K_VALUES = (1, 2, 4, 8, 16)


def search_arxiv_id_by_title(title: str, max_retries: int = 3) -> Optional[str]:
    query = f"arxiv {title}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://s.jina.ai/?q={encoded_query}"
    headers = {
        "X-Respond-With": "no-content",
        "Accept": "application/json",
    }
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            arxiv_ids = re.findall(r"\b(\d{4}\.\d{4,5})\b", response.text)
            return arxiv_ids[0] if arxiv_ids else None
        except Exception as exc:
            LOGGER.warning("Jina title search failed for %r (attempt %s/%s): %s", title[:80], attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2)

    return None


def load_gt_file(gt_file: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with Path(gt_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid GT JSONL line %s: %s", line_number, exc)
                continue
            question = data.get("question", "").strip()
            arxiv_ids = [normalize_arxiv_id(str(arxiv_id)) for arxiv_id in data.get("arxiv_id", []) if arxiv_id]
            arxiv_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id]
            if question:
                mapping[question] = arxiv_ids
    return mapping


def max_iou_at_k_sampling(iou_list: Sequence[float], k: int, sample_times: int = 1000) -> float:
    if len(iou_list) < k or k <= 0:
        return 0.0

    total_max = 0.0
    for _ in range(sample_times):
        total_max += max(random.sample(list(iou_list), k))
    return round(total_max / sample_times, 4)


def normalize_arxiv_id(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    cleaned = re.sub(r"(?i)arxiv:", "", arxiv_id).strip()
    match = re.search(r"(\d{4}\.\d{4,5})", cleaned)
    return match.group(1) if match else cleaned.strip()


def get_gt_arxiv_ids(input_data: Dict[str, Any], use_jina: bool = True) -> Set[str]:
    raw_ids = input_data.get("arxiv_id", [])
    if raw_ids:
        normalized = {normalize_arxiv_id(str(raw_id)) for raw_id in raw_ids if raw_id}
        normalized.discard("")
        if normalized:
            return normalized

    if use_jina and JINA_API_KEY:
        answers = input_data.get("answer", [])
        found_ids: Set[str] = set()
        LOGGER.info("GT arxiv_id missing. Trying Jina lookup for %s answer titles.", len(answers))
        for title in answers:
            arxiv_id = search_arxiv_id_by_title(title)
            if arxiv_id:
                found_ids.add(arxiv_id)
        return found_ids

    return set()


def get_predicted_arxiv_ids(final_candidates: Sequence[Dict[str, Any]]) -> Set[str]:
    predicted: Set[str] = set()
    for candidate in final_candidates or []:
        normalized_id = normalize_arxiv_id(str(candidate.get("arxiv_id", "")))
        if normalized_id:
            predicted.add(normalized_id)
    return predicted


def compute_iou_recall(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0

    intersection = len(gt_ids & pred_ids)
    union = len(gt_ids | pred_ids)
    iou = intersection / union if union else 0.0
    recall = intersection / len(gt_ids) if gt_ids else 0.0
    precision = intersection / len(pred_ids) if pred_ids else 0.0
    return iou, recall, precision


def extract_turn_stats(turn_details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not turn_details:
        return {
            "last_tokens": 0,
            "total_duration": 0.0,
            "tool_call_count": 0,
            "turn_count": 0,
            "per_turn_duration": [],
            "last_input_tokens": 0,
            "last_output_tokens": 0,
        }

    last_turn = turn_details[-1]
    last_input = last_turn.get("input_tokens", 0)
    last_output = last_turn.get("output_tokens", 0)
    return {
        "last_tokens": last_input + last_output,
        "last_input_tokens": last_input,
        "last_output_tokens": last_output,
        "total_duration": sum(turn.get("duration", 0.0) for turn in turn_details),
        "tool_call_count": sum(1 for turn in turn_details if turn.get("action") == "tool"),
        "turn_count": len(turn_details),
        "per_turn_duration": [turn.get("duration", 0.0) for turn in turn_details],
    }


def load_records(input_file: str) -> List[Tuple[int, Dict[str, Any]]]:
    records: List[Tuple[int, Dict[str, Any]]] = []
    with Path(input_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append((line_number, json.loads(stripped)))
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid inference JSONL line %s: %s", line_number, exc)
    return records


def evaluate_file(
    input_file: str,
    use_jina: bool = True,
    gt_mapping: Optional[Dict[str, List[str]]] = None,
    max_records: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    records = load_records(input_file)
    if gt_mapping is not None:
        filtered_records = [
            (line_number, record)
            for line_number, record in records
            if record.get("input_data", {}).get("question", "").strip() in gt_mapping
        ]
        skipped = len(records) - len(filtered_records)
        LOGGER.info("Loaded %s records, matched %s GT entries, skipped %s unmatched records.", len(records), len(filtered_records), skipped)
        records = filtered_records
    else:
        LOGGER.info("Loaded %s records from %s", len(records), input_file)

    if max_records is not None:
        records = records[:max_records]
        LOGGER.info("Limiting evaluation to the first %s records.", len(records))

    per_record_results: List[Dict[str, Any]] = []
    for line_number, data in records:
        input_data = data.get("input_data", {})
        inference_results = data.get("inference_results", [])
        question = input_data.get("question", "")

        if gt_mapping is not None:
            gt_ids = set(gt_mapping.get(question.strip(), []))
            gt_source = "gt_file"
        else:
            gt_ids = get_gt_arxiv_ids(input_data, use_jina=use_jina)
            gt_source = "input_data"

        pass_results: List[Dict[str, Any]] = []
        for pass_data in inference_results:
            pass_id = pass_data.get("pass_id", 0)
            final_candidates = pass_data.get("final_candidates", []) or []
            turn_details = pass_data.get("turn_details", []) or []
            predicted_ids = get_predicted_arxiv_ids(final_candidates)
            iou, recall, precision = compute_iou_recall(gt_ids, predicted_ids)
            turn_stats = extract_turn_stats(turn_details)

            pass_results.append(
                {
                    "pass_id": pass_id,
                    "status": pass_data.get("status", "unknown"),
                    "gt_source": gt_source,
                    "gt_arxiv_ids": sorted(gt_ids),
                    "predicted_arxiv_ids": sorted(predicted_ids),
                    "hit_ids": sorted(gt_ids & predicted_ids),
                    "missed_ids": sorted(gt_ids - predicted_ids),
                    "extra_ids": sorted(predicted_ids - gt_ids),
                    "gt_count": len(gt_ids),
                    "predicted_count": len(predicted_ids),
                    "hit_count": len(gt_ids & predicted_ids),
                    "iou": round(iou, 6),
                    "recall": round(recall, 6),
                    "precision": round(precision, 6),
                    "last_tokens": turn_stats["last_tokens"],
                    "last_input_tokens": turn_stats["last_input_tokens"],
                    "last_output_tokens": turn_stats["last_output_tokens"],
                    "total_duration": round(turn_stats["total_duration"], 4),
                    "per_turn_duration": [round(duration, 4) for duration in turn_stats["per_turn_duration"]],
                    "tool_call_count": turn_stats["tool_call_count"],
                    "turn_count": turn_stats["turn_count"],
                }
            )

        record_result = {
            "line_num": line_number,
            "question": question,
            "pass_results": pass_results,
        }
        per_record_results.append(record_result)

        if verbose:
            LOGGER.info("Record %s | question=%r | gt=%s | passes=%s", line_number, question[:120], sorted(gt_ids), len(pass_results))

    return per_record_results


def compute_aggregate_stats(per_record_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_record_results:
        return {
            "total_records": 0,
            "total_passes": 0,
            "total_pass_results": 0,
            "avg_max_iou_at_1": 0.0,
            "avg_max_iou_at_2": 0.0,
            "avg_max_iou_at_4": 0.0,
            "avg_max_iou_at_8": 0.0,
            "avg_max_iou_at_16": 0.0,
            "avg_iou": 0.0,
            "avg_recall": 0.0,
            "avg_precision": 0.0,
            "avg_last_tokens": 0.0,
            "avg_total_duration_sec": 0.0,
            "avg_turn_count": 0.0,
            "avg_tool_call_count": 0.0,
            "total_tokens": 0,
            "total_duration_sec": 0.0,
            "iou_distribution": {"max": 0.0, "min": 0.0},
            "recall_distribution": {"max": 0.0, "min": 0.0},
        }

    iou_list: List[float] = []
    recall_list: List[float] = []
    precision_list: List[float] = []
    token_list: List[int] = []
    duration_list: List[float] = []
    turn_list: List[int] = []
    tool_call_list: List[int] = []
    max_iou_lists: Dict[int, List[float]] = {k: [] for k in MAX_IOU_K_VALUES}

    for record in per_record_results:
        record_ious = [pass_result["iou"] for pass_result in record["pass_results"]]
        for pass_result in record["pass_results"]:
            iou_list.append(pass_result["iou"])
            recall_list.append(pass_result["recall"])
            precision_list.append(pass_result["precision"])
            token_list.append(pass_result["last_tokens"])
            duration_list.append(pass_result["total_duration"])
            turn_list.append(pass_result["turn_count"])
            tool_call_list.append(pass_result["tool_call_count"])
        for k in MAX_IOU_K_VALUES:
            if len(record_ious) >= k:
                max_iou_lists[k].append(max_iou_at_k_sampling(record_ious, k))

    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    pass_count = len(per_record_results[0]["pass_results"]) if per_record_results else 0
    total_pass_results = sum(len(record["pass_results"]) for record in per_record_results)

    return {
        "total_records": len(per_record_results),
        "total_passes": pass_count,
        "total_pass_results": total_pass_results,
        "avg_max_iou_at_1": round(average(max_iou_lists[1]), 6),
        "avg_max_iou_at_2": round(average(max_iou_lists[2]), 6),
        "avg_max_iou_at_4": round(average(max_iou_lists[4]), 6),
        "avg_max_iou_at_8": round(average(max_iou_lists[8]), 6),
        "avg_max_iou_at_16": round(average(max_iou_lists[16]), 6),
        "avg_iou": round(average(iou_list), 6),
        "avg_recall": round(average(recall_list), 6),
        "avg_precision": round(average(precision_list), 6),
        "avg_last_tokens": round(average(token_list), 2),
        "avg_total_duration_sec": round(average(duration_list), 4),
        "avg_turn_count": round(average(turn_list), 4),
        "avg_tool_call_count": round(average(tool_call_list), 4),
        "total_tokens": sum(token_list),
        "total_duration_sec": round(sum(duration_list), 4),
        "iou_distribution": {
            "max": round(max(iou_list), 6) if iou_list else 0.0,
            "min": round(min(iou_list), 6) if iou_list else 0.0,
        },
        "recall_distribution": {
            "max": round(max(recall_list), 6) if recall_list else 0.0,
            "min": round(min(recall_list), 6) if recall_list else 0.0,
        },
    }


def print_summary(aggregate_stats: Dict[str, Any]) -> None:
    print("\n=== Wide Search Evaluation Summary ===")
    print(json.dumps(aggregate_stats, indent=2, ensure_ascii=False))


def save_results(per_record_results: Sequence[Dict[str, Any]], aggregate_stats: Dict[str, Any], output_file: str) -> None:
    output = {
        "aggregate_stats": aggregate_stats,
        "per_record_results": list(per_record_results),
    }
    with Path(output_file).open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Evaluation results saved to %s", output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate wide-search inference outputs.")
    parser.add_argument("--input", "--input-file", dest="input_file", required=True, help="Path to the inference JSONL file.")
    parser.add_argument(
        "--output",
        "--output-file",
        dest="output_file",
        default=None,
        help="Path to save evaluation JSON. Defaults to <input>_eval_results.json.",
    )
    parser.add_argument("--no-jina", action="store_true", help="Disable Jina lookup when GT arxiv_id is missing.")
    parser.add_argument(
        "--gt-file",
        type=str,
        default=None,
        help="Optional GT JSONL file. If provided, only records whose question appears in the GT file will be evaluated.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit on the number of records to evaluate.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def build_default_output_path(input_file: str) -> Path:
    input_path = Path(input_file)
    return input_path.parent / f"{input_path.stem}_eval_results.json"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be greater than 0 when provided.")

    if not Path(args.input_file).exists():
        LOGGER.error("Input file does not exist: %s", args.input_file)
        raise SystemExit(1)

    use_jina = not args.no_jina
    LOGGER.info("Input file: %s", args.input_file)
    LOGGER.info("Jina lookup: %s", "enabled" if use_jina else "disabled")

    gt_mapping: Optional[Dict[str, List[str]]] = None
    if args.gt_file:
        if not Path(args.gt_file).exists():
            LOGGER.error("GT file does not exist: %s", args.gt_file)
            raise SystemExit(1)
        gt_mapping = load_gt_file(args.gt_file)
        non_empty_count = sum(1 for arxiv_ids in gt_mapping.values() if arxiv_ids)
        LOGGER.info("Loaded %s GT questions from %s (%s with arxiv_id, %s empty).", len(gt_mapping), args.gt_file, non_empty_count, len(gt_mapping) - non_empty_count)
    else:
        LOGGER.info("No GT file provided. Falling back to input_data ground truth.")

    per_record_results = evaluate_file(
        input_file=args.input_file,
        use_jina=use_jina,
        gt_mapping=gt_mapping,
        max_records=args.max_records,
        verbose=args.verbose,
    )
    aggregate_stats = compute_aggregate_stats(per_record_results)
    print_summary(aggregate_stats)

    output_path = Path(args.output_file) if args.output_file else build_default_output_path(args.input_file)
    ensure_parent_dir(output_path)
    save_results(per_record_results, aggregate_stats, str(output_path))


if __name__ == "__main__":
    main()

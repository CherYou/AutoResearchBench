import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import LLMService, configure_logging, ensure_parent_dir, load_env_file, load_jsonl


LOGGER = logging.getLogger(__name__)
ENV_PATH = ROOT / ".env"
load_env_file(ENV_PATH)

JUDGE_PROMPT_SYSTEM = """
You are an expert verifier for academic paper titles. Your task is to determine if a candidate paper title refers to the exact same paper as a ground truth title.
Note that the returned candidate title may be truncated, such as "The effects of external planets on inner systems: multiplicities, inclinations, and pathways to eccentric warm Jupiters" becoming "The effects of external planets on inner systems: multiplicities ...". Such cases should still be judged as correct.
Ignore minor differences in punctuation, capitalization, or the presence or absence of subtitles unless they fundamentally change the paper's identity.
Respond ONLY with a JSON object in the following format:
{"is_match": boolean, "reason": "A brief explanation for your decision."}
"""

JUDGE_PROMPT_USER_TEMPLATE = """
Please verify if the following two titles refer to the same paper.

Ground Truth Title:
`{ground_truth_title}`

Candidate Title:
`{candidate_title}`
"""

DEFAULT_API_KEY = os.environ.get("EVAL_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
DEFAULT_API_BASE = os.environ.get("EVAL_OPENAI_API_BASE", os.environ.get("OPENAI_API_BASE", ""))
DEFAULT_MODEL_NAME = os.environ.get("EVAL_MODEL", os.environ.get("MODEL", ""))


class LLMJudge:
    def __init__(self, model_name: str, api_key: str, api_base: str):
        self.model_name = model_name
        self.client = LLMService(
            api_key=api_key,
            api_base=api_base,
        )
        self.cache: Dict[tuple[str, str], bool] = {}

    async def close(self) -> None:
        await self.client.close()

    @staticmethod
    def _parse_judge_output(output: str) -> bool:
        try:
            data = json.loads(output)
            if isinstance(data, dict) and "is_match" in data:
                return bool(data["is_match"])
        except (json.JSONDecodeError, TypeError):
            return "true" in output.lower()
        return False

    async def are_titles_matching(
        self,
        gt_title: str,
        candidate_title: str,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        cache_key = (gt_title, candidate_title)
        if cache_key in self.cache:
            return self.cache[cache_key]

        async with semaphore:
            messages = [
                {"role": "system", "content": JUDGE_PROMPT_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_PROMPT_USER_TEMPLATE.format(
                        ground_truth_title=gt_title,
                        candidate_title=candidate_title,
                    ),
                },
            ]
            response = await self.client.send_message(
                messages=messages,
                model=self.model_name,
                temperature=0.5,
                max_completion_tokens=512,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            is_match = self._parse_judge_output(response)
            self.cache[cache_key] = is_match
            return is_match


class Evaluator:
    def __init__(self, judge: LLMJudge, semaphore: asyncio.Semaphore, max_candidates_per_pass: int = 1):
        self.judge = judge
        self.semaphore = semaphore
        self.max_candidates_per_pass = max_candidates_per_pass

    @staticmethod
    def _normalize_titles(titles: Sequence[str]) -> List[str]:
        return [title.strip() for title in titles if isinstance(title, str) and title.strip()]

    @staticmethod
    def _is_none_answer(titles: Sequence[str]) -> bool:
        normalized_titles = Evaluator._normalize_titles(titles)
        return bool(normalized_titles) and all(title.lower() == "none" for title in normalized_titles)

    async def _calculate_score(
        self,
        candidate_titles: Sequence[str],
        ground_truth_titles: Sequence[str],
        candidate_state: str = "empty",
    ) -> float:
        valid_ground_truths = self._normalize_titles(ground_truth_titles)
        valid_candidates = self._normalize_titles(candidate_titles)
        gt_is_none = self._is_none_answer(valid_ground_truths)
        pred_is_none = candidate_state == "none"

        if gt_is_none:
            return 1.0 if pred_is_none else 0.0
        if not valid_ground_truths:
            return 1.0 if (pred_is_none or not valid_candidates) else 0.0
        if pred_is_none or not valid_candidates:
            return 0.0

        hits = 0
        for gt_title in valid_ground_truths:
            match_tasks = [
                self.judge.are_titles_matching(gt_title, candidate_title, self.semaphore)
                for candidate_title in valid_candidates
            ]
            if any(await asyncio.gather(*match_tasks)):
                hits += 1
        return hits / len(valid_ground_truths)

    async def evaluate_item(self, inference_item: Dict[str, Any]) -> Dict[str, Any]:
        input_data = inference_item["input_data"]
        inference_results = inference_item["inference_results"]
        ground_truth_titles = input_data.get("answer", [])

        pass_scores: List[float] = []
        for result_pass in inference_results:
            candidate_state = result_pass.get("final_candidate_state", "empty")
            candidate_papers = result_pass.get("final_candidates", [])[: self.max_candidates_per_pass]
            candidate_titles = [paper.get("metadata", {}).get("title", "") for paper in candidate_papers]
            score = await self._calculate_score(
                candidate_titles=candidate_titles,
                ground_truth_titles=ground_truth_titles,
                candidate_state=candidate_state,
            )
            pass_scores.append(score)

        k = len(pass_scores)
        accuracy_at_1 = 1.0 if pass_scores and pass_scores[0] == 1.0 else 0.0
        pass_at_k = 1.0 if any(score == 1.0 for score in pass_scores) else 0.0
        mean_at_k = sum(pass_scores) / k if k else 0.0

        for result in inference_results:
            if result.get("final_candidate_state") == "none":
                result["final_candidates"] = []
            else:
                result["final_candidates"] = [
                    {"title": item.get("metadata", {}).get("title", "")}
                    for item in result.get("final_candidates", [])
                ]

        inference_item["evaluation"] = {
            "pass_scores": pass_scores,
            "metrics": {
                "accuracy_at_1": accuracy_at_1,
                f"pass_at_{k}": pass_at_k,
                f"mean_at_{k}": round(mean_at_k, 4),
            },
        }
        return inference_item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deep-search inference results with an LLM judge.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the inference output JSONL file.")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results. Defaults to <input>_evaluation.json.",
    )
    parser.add_argument("--max-workers", type=int, default=20, help="Maximum number of concurrent judge requests.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Judge model name.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="Judge API key.")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="Judge API base URL.")
    parser.add_argument("--max-candidates-per-pass", type=int, default=1, help="How many final candidates to score per pass.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def build_default_output_path(input_file: str) -> Path:
    input_path = Path(input_file)
    return input_path.parent / f"{input_path.stem}_evaluation.json"


def build_summary(detailed_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_items = len(detailed_results)
    k = len(detailed_results[0]["inference_results"]) if detailed_results else 0

    overall_accuracy_at_1 = 0.0
    overall_pass_at_k = 0.0
    overall_mean_at_k = 0.0

    if total_items:
        overall_accuracy_at_1 = sum(
            item["evaluation"]["metrics"]["accuracy_at_1"] for item in detailed_results
        ) / total_items
        overall_pass_at_k = sum(
            item["evaluation"]["metrics"][f"pass_at_{k}"] for item in detailed_results
        ) / total_items
        overall_mean_at_k = sum(
            item["evaluation"]["metrics"][f"mean_at_{k}"] for item in detailed_results
        ) / total_items

    return {
        "total_items": total_items,
        "k": k,
        "overall_metrics": {
            "Accuracy@1": f"{overall_accuracy_at_1:.2%}",
            f"pass@{k}": f"{overall_pass_at_k:.2%}",
            f"mean@{k}": f"{overall_mean_at_k:.4f}",
        },
    }


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    if args.max_workers <= 0:
        parser.error("--max-workers must be greater than 0.")
    if args.max_candidates_per_pass <= 0:
        parser.error("--max-candidates-per-pass must be greater than 0.")
    if not args.model:
        parser.error("--model is required. Set it explicitly or provide EVAL_MODEL / MODEL in .env.")
    if not args.api_base:
        parser.error("--api-base is required. Set it explicitly or provide EVAL_OPENAI_API_BASE / OPENAI_API_BASE in .env.")

    try:
        inference_data = load_jsonl(args.input_file)
    except FileNotFoundError:
        LOGGER.error("Input file not found: %s", args.input_file)
        raise SystemExit(1)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1)

    if not inference_data:
        LOGGER.info("No data to evaluate.")
        return

    output_path = Path(args.output_file) if args.output_file else build_default_output_path(args.input_file)
    ensure_parent_dir(output_path)

    LOGGER.info("Loaded %s items from %s", len(inference_data), args.input_file)

    semaphore = asyncio.Semaphore(args.max_workers)
    judge = LLMJudge(
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    evaluator = Evaluator(
        judge=judge,
        semaphore=semaphore,
        max_candidates_per_pass=args.max_candidates_per_pass,
    )

    try:
        tasks = [evaluator.evaluate_item(item) for item in inference_data]
        detailed_results = await tqdm_asyncio.gather(*tasks, desc="Running Evaluation")
    finally:
        await judge.close()

    summary = build_summary(detailed_results)
    LOGGER.info("Evaluation summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    final_output = {
        "summary": summary,
        "detailed_results": detailed_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(final_output, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Detailed evaluation results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())

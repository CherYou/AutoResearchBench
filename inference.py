import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prompts import AGENT_SYSTEM_PROMPT, WARNING_PROMPT
from tool_deepxivsearch import DeepXivSearchEngine
from tool_websearch import WebSearchEngine
from utils import (
    LLMService,
    build_agent_messages,
    build_paper_records,
    configure_logging,
    count_tokens,
    ensure_parent_dir,
    filter_pending_items,
    load_completed_questions,
    load_jsonl,
    make_inference_error_result,
    parse_agent_output,
    validate_slice_bounds,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_TOOL_NAMES = {"search", "PaperSearchTool", "WebSearchTool"}


class AgentService:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        paper_search_tool: str = "deepxiv",
        max_turns: int = 10,
        verbose: bool = False,
    ):
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.visible_paper_number = 10
        self.max_context_tokens = 110000
        self.planner_agent = LLMService(
            api_key=api_key,
            api_base=api_base,
        )
        self.search_engine = self._create_search_engine(paper_search_tool)

    @staticmethod
    def _create_search_engine(tool_name: str):
        if tool_name == "deepxiv":
            LOGGER.info("Using DeepXiv search backend.")
            return DeepXivSearchEngine()
        if tool_name == "websearch":
            LOGGER.info("Using web-search backend.")
            return WebSearchEngine()
        raise ValueError(f"Invalid paper search tool: {tool_name}")

    def _build_messages(self, user_query: str, turn_data: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        return build_agent_messages(
            system_prompt=AGENT_SYSTEM_PROMPT,
            user_query=user_query,
            turn_data=turn_data,
            visible_paper_number=self.visible_paper_number,
        )

    def _check_limits(self, messages: List[Dict[str, str]], turn_count: int, input_tokens: int) -> Tuple[bool, str]:
        if input_tokens > self.max_context_tokens:
            LOGGER.warning("Token quantity exceeds the limit: %s > %s", input_tokens, self.max_context_tokens)
            messages[-1]["content"] += WARNING_PROMPT.format(limit_type="context length you can handle")
            return True, "max_tokens_reached"
        if turn_count == self.max_turns:
            LOGGER.warning("Reached the maximum number of turns: %s", turn_count)
            messages[-1]["content"] += WARNING_PROMPT.format(limit_type="number of turns you can handle")
            return True, "max_turns_reached"
        return False, "unknown"

    @staticmethod
    def _update_candidate_state(
        current_paper_list: Sequence[Dict[str, Any]],
        new_candidate_ids: Sequence[int],
        new_candidate_state: str,
        candidate_paper_list: List[Dict[str, Any]],
        candidate_paper_titles: set,
        current_state: str,
    ) -> str:
        if new_candidate_state == "none":
            candidate_paper_list.clear()
            candidate_paper_titles.clear()
            return "none"

        if not new_candidate_ids or not current_paper_list:
            return current_state

        current_papers_map = {paper["id"]: paper for paper in current_paper_list}
        for paper_id in new_candidate_ids:
            paper_to_add = current_papers_map.get(paper_id)
            if not paper_to_add:
                continue
            title = paper_to_add.get("metadata", {}).get("title")
            if title and title not in candidate_paper_titles:
                candidate_paper_list.append(paper_to_add)
                candidate_paper_titles.add(title)
        return "ids"

    async def _execute_search_tool(self, action_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_name = action_content.get("name")
        if tool_name not in SUPPORTED_TOOL_NAMES:
            raise ValueError(f"Unsupported tool requested: {tool_name}")

        tool_args = action_content.get("arguments", {})
        query = tool_args.get("query", tool_args.get("query_str", ""))
        top_k = tool_args.get("top_k", 10)

        raw_results = await self.search_engine.search(
            query=query,
            top_k=top_k,
        )
        if not raw_results:
            return []

        formatted_results = await asyncio.gather(
            *[self.search_engine.format_paper(result, query) for result in raw_results]
        )
        return build_paper_records(raw_results, formatted_results)

    async def close(self) -> None:
        await self.planner_agent.close()

    async def _run_single_pass(self, user_query: str) -> Dict[str, Any]:
        start_total_time = time.time()

        current_paper_list: List[Dict[str, Any]] = []
        turn_data: List[Dict[str, Any]] = []
        turn_details: List[Dict[str, Any]] = []
        last_messages: List[Dict[str, str]] = []

        candidate_paper_list: List[Dict[str, Any]] = []
        candidate_paper_titles: set = set()
        final_candidate_state = "empty"
        status = "unknown"

        for turn_count in range(1, self.max_turns + 1):
            turn_start_time = time.time()
            messages = self._build_messages(user_query, turn_data)
            input_tokens = count_tokens(messages, self.model)
            finished, status = self._check_limits(messages, turn_count, input_tokens)

            try:
                output = await self.planner_agent.send_message(
                    messages=messages,
                    model=self.model,
                    temperature=0.6,
                )
                if self.verbose:
                    LOGGER.info("Turn %s last user message:\n%s", turn_count, messages[-1]["content"])
                    LOGGER.info("Turn %s model output:\n%s", turn_count, output)
            except Exception as exc:
                error_message = str(exc)
                LOGGER.warning("Turn %s LLM API error: %s", turn_count, error_message)
                if "context length" in error_message.lower() or "400" in error_message:
                    status = "context_length_exceeded"
                else:
                    status = "api_error"
                turn_details.append(
                    {
                        "turn": turn_count,
                        "duration": time.time() - turn_start_time,
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "action": "error",
                        "papers_retrieved_this_turn": 0,
                        "error_message": error_message,
                    }
                )
                break

            output_tokens = count_tokens(output, self.model)
            messages.append({"role": "assistant", "content": output})
            last_messages = copy.deepcopy(messages)

            parsed_output = parse_agent_output(output)
            action_result = parsed_output["action"]
            action_type = action_result["type"]
            action_content = action_result["content"]

            final_candidate_state = self._update_candidate_state(
                current_paper_list=current_paper_list,
                new_candidate_ids=parsed_output["candidate_ids"],
                new_candidate_state=parsed_output["candidate_state"],
                candidate_paper_list=candidate_paper_list,
                candidate_paper_titles=candidate_paper_titles,
                current_state=final_candidate_state,
            )

            if action_type == "finish":
                status = "finished"
                finished = True
            elif action_type == "tool":
                try:
                    current_paper_list = await self._execute_search_tool(action_content)
                    turn_data.append(
                        {
                            "assistant_output": output,
                            "paper_list": current_paper_list,
                        }
                    )
                except Exception as exc:
                    action_type = "error"
                    action_content = f"Tool execution error: {exc}"
                    turn_data.append(
                        {
                            "assistant_output": f"{output}\n<error>{action_content}</error>",
                            "paper_list": current_paper_list,
                        }
                    )
            elif action_type == "error":
                turn_data.append(
                    {
                        "assistant_output": f"{output}\n<error>Parsing Error: {action_content}</error>",
                        "paper_list": current_paper_list,
                    }
                )

            turn_details.append(
                {
                    "turn": turn_count,
                    "duration": time.time() - turn_start_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "action": action_type,
                    "action_content": action_content,
                    "papers_retrieved_this_turn": len(current_paper_list) if action_type == "tool" else 0,
                }
            )

            if finished:
                break

        return {
            "status": status,
            "total_time": time.time() - start_total_time,
            "messages": last_messages,
            "final_candidates": candidate_paper_list,
            "final_candidate_state": final_candidate_state,
            "turn_details": turn_details,
        }

    async def _run_with_retry(self, user_query: str, max_attempts: int = 3) -> Dict[str, Any]:
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._run_single_pass(user_query)
            except Exception as exc:
                LOGGER.warning("Inference attempt %s/%s failed for query %r: %s", attempt, max_attempts, user_query, exc)
        return make_inference_error_result(f"Failed after {max_attempts} attempts for query: {user_query}")

    async def run_inference_for_file(
        self,
        data_item: Dict[str, Any],
        k: int,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        async with semaphore:
            user_query = data_item.get("question", "")
            inference_results = await asyncio.gather(*[self._run_with_retry(user_query) for _ in range(k)])
            for pass_id, result in enumerate(inference_results):
                result["pass_id"] = pass_id
            return {
                "input_data": data_item,
                "inference_results": inference_results,
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch inference for SciBench.")
    parser.add_argument("--input-file", type=str, default="data.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output-file", type=str, default="output.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("-k", type=int, default=1, help="Number of inference passes per input item.")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of concurrent inference tasks.")
    parser.add_argument("--model", type=str, required=True, help="Model name used for inference.")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the LLM service.")
    parser.add_argument("--api-base", type=str, required=True, help="Base URL for the LLM API.")
    parser.add_argument(
        "--search-tool",
        type=str,
        choices=["deepxiv", "websearch"],
        default="deepxiv",
        help="Paper-search backend used during inference.",
    )
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns for agent reasoning.")
    parser.add_argument("--eval-start", type=int, default=0, help="Start index (inclusive) of the evaluation slice.")
    parser.add_argument("--eval-end", type=int, default=None, help="End index (exclusive) of the evaluation slice.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose per-turn model I/O for debugging.")
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    try:
        eval_start, eval_end = validate_slice_bounds(args.eval_start, args.eval_end)
    except ValueError as exc:
        parser.error(str(exc))

    if args.k <= 0:
        parser.error("-k must be greater than 0.")
    if args.max_workers <= 0:
        parser.error("--max-workers must be greater than 0.")

    try:
        input_data = load_jsonl(args.input_file)
    except FileNotFoundError:
        LOGGER.error("Input file not found: %s", args.input_file)
        raise SystemExit(1)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1)

    input_data = input_data[eval_start:eval_end]

    completed_questions, existing_results_count = load_completed_questions(args.output_file)
    if existing_results_count:
        LOGGER.info("Found %s existing results in %s", existing_results_count, args.output_file)
        input_data = filter_pending_items(input_data, completed_questions)

    if not input_data:
        LOGGER.info("No pending items to run. Output file is already up to date.")
        return

    LOGGER.info("Loaded %s items from %s", len(input_data), args.input_file)
    LOGGER.info("Evaluation range: start=%s, end=%s", eval_start, "EOF" if eval_end is None else eval_end)
    LOGGER.info("Running inference with k=%s, max_workers=%s, max_turns=%s", args.k, args.max_workers, args.max_turns)
    LOGGER.info("Model: %s | Search tool: %s", args.model, args.search_tool)

    ensure_parent_dir(args.output_file)
    agent_service = AgentService(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        paper_search_tool=args.search_tool,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )
    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = [
        agent_service.run_inference_for_file(data_item, args.k, semaphore)
        for data_item in input_data
    ]

    try:
        with Path(args.output_file).open("a", encoding="utf-8") as handle:
            progress_bar = tqdm(asyncio.as_completed(tasks), total=len(input_data), desc="Running Inference")
            for future in progress_bar:
                result = await future
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    except Exception as exc:
        LOGGER.error("Inference failed: %s", exc)
        LOGGER.error("Partial results may have been saved to %s", args.output_file)
        raise
    finally:
        await agent_service.close()

    LOGGER.info("Inference complete. Results saved to %s", args.output_file)


if __name__ == "__main__":
    asyncio.run(main())

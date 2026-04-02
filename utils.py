import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import tiktoken
from openai import AsyncOpenAI

try:
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - optional dependency for streaming mode only.
    StreamingResponse = None


LOGGER = logging.getLogger(__name__)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
CANDIDATES_PATTERN = re.compile(r"<candidates>(.*?)</candidates>", re.DOTALL)
PathLike = Union[str, Path]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("openai", "openai._base_client", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def ensure_parent_dir(path: PathLike) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {path}: {exc}") from exc
    return records


def load_completed_questions(output_path: PathLike) -> Tuple[Set[str], int]:
    path = Path(output_path)
    if not path.exists():
        return set(), 0

    completed_questions: Set[str] = set()
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping invalid JSONL line %s in %s", line_number, path)
                continue
            question = payload.get("input_data", {}).get("question", "")
            if question:
                completed_questions.add(question)
            record_count += 1
    return completed_questions, record_count


def filter_pending_items(items: Sequence[Dict[str, Any]], completed_questions: Set[str]) -> List[Dict[str, Any]]:
    if not completed_questions:
        return list(items)
    return [item for item in items if item.get("question") not in completed_questions]


def validate_slice_bounds(start: int, end: Optional[int]) -> Tuple[int, Optional[int]]:
    normalized_start = max(start, 0)
    if end is not None and end < normalized_start:
        raise ValueError(f"--eval-end ({end}) must be greater than or equal to --eval-start ({normalized_start})")
    return normalized_start, end


def count_tokens(content: Any, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0

    try:
        if isinstance(content, str):
            return len(encoding.encode(content))
        if isinstance(content, list):
            return len(encoding.encode(json.dumps(content, ensure_ascii=False)))
    except Exception:
        return 0
    return 0


def format_paper_list_for_prompt(
    paper_list: Sequence[Dict[str, Any]],
    simplify: bool = False,
    search_type: str = "deep",
) -> str:
    formatted_papers: List[Dict[str, Any]] = []
    should_simplify = str(search_type).lower() != "wide" and simplify
    for paper in paper_list:
        if should_simplify:
            formatted_papers.append(
                {
                    "paper_title": paper.get("paper_title", ""),
                    "pub_date": paper.get("pub_date", []),
                }
            )
        else:
            formatted_papers.append(
                {
                    "paper_id": paper.get("paper_id", -1),
                    "paper_title": paper.get("paper_title", ""),
                    "authors": paper.get("authors", []),
                    "tldr": paper.get("tldr", ""),
                    "pub_date": paper.get("pub_date", []),
                    "retrieval_evidence": paper.get("retrieval_evidence", ""),
                }
            )
    return json.dumps(formatted_papers, ensure_ascii=False)


def format_paper_list_for_logging(
    paper_list: Sequence[Dict[str, Any]],
    visible_paper_number: Optional[int] = None,
) -> str:
    papers_to_log = paper_list[:visible_paper_number] if visible_paper_number is not None else paper_list
    return json.dumps(
        [
            {
                "paper_title": paper.get("paper_title", ""),
                "pub_date": paper.get("pub_date", []),
            }
            for paper in papers_to_log
        ],
        ensure_ascii=False,
    )


def build_agent_messages(
    system_prompt: str,
    user_query: str,
    turn_data: Sequence[Dict[str, Any]],
    visible_paper_number: int,
    search_type: str = "deep",
) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {user_query}"},
    ]

    for index, turn in enumerate(turn_data):
        assistant_output = turn["assistant_output"]
        if "<error>" in assistant_output and "</error>" in assistant_output:
            assistant_content, error_content = assistant_output.split("<error>", 1)
            error_message = error_content.split("</error>", 1)[0]
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": f"<error>{error_message}</error>"})
            continue

        messages.append({"role": "assistant", "content": assistant_output})
        is_simplified = index < len(turn_data) - 1
        list_str = format_paper_list_for_prompt(
            turn["paper_list"][:visible_paper_number],
            simplify=is_simplified,
            search_type=search_type,
        )
        messages.append({"role": "user", "content": f"<tool_response>{list_str}</tool_response>"})

    return messages


def parse_agent_output(output: str) -> Dict[str, Any]:
    action: Dict[str, Any] = {"type": "unknown", "content": output}
    if "<answer>" in output and "Done" in output:
        action = {"type": "finish", "content": "Done"}
    else:
        tool_match = TOOL_CALL_PATTERN.search(output)
        if tool_match:
            json_str = tool_match.group(1).strip()
            json_str = re.sub(r"^```json\s*", "", json_str)
            json_str = re.sub(r"\s*```$", "", json_str)
            try:
                action = {"type": "tool", "content": json.loads(json_str)}
            except json.JSONDecodeError as exc:
                LOGGER.warning("Failed to parse tool-call JSON: %s", exc)
                action = {
                    "type": "error",
                    "content": (
                        "Failed to parse tool arguments JSON: "
                        f"{json_str}. You need to think again and provide the correct JSON format."
                    ),
                }
        else:
            LOGGER.warning("Missing <tool_call> block in model output.")
            action = {
                "type": "error",
                "content": (
                    "Failed to parse tool format error: "
                    f"<tool_call>{output[-200:]}</tool_call>. "
                    "You need to think again and provide the correct <tool_call></tool_call> format."
                ),
            }

    candidate_ids: List[int] = []
    candidate_state = "empty"
    candidate_match = CANDIDATES_PATTERN.search(output)
    if candidate_match:
        candidate_text = candidate_match.group(1).strip()
        if not candidate_text:
            candidate_state = "empty"
        else:
            normalized_text = re.sub(r'[`"\']', "", candidate_text).strip()
            if normalized_text.lower() == "none":
                candidate_state = "none"
            elif normalized_text == "[]":
                candidate_state = "empty"
            else:
                candidate_state = "ids"
                try:
                    cleaned_text = re.sub(r"\[|\]", "", candidate_text)
                    candidate_ids = [int(value.strip()) for value in cleaned_text.split(",") if value.strip().isdigit()]
                except Exception as exc:
                    LOGGER.warning("Failed to parse candidate IDs: %s", exc)

    return {
        "action": action,
        "candidate_ids": candidate_ids,
        "candidate_state": candidate_state,
    }


def build_paper_records(
    raw_results: Sequence[Dict[str, Any]],
    formatted_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    papers: List[Dict[str, Any]] = []
    for index, (raw_paper, formatted_paper) in enumerate(zip(raw_results, formatted_results)):
        authors = raw_paper.get("authors") or formatted_paper.get("authors") or []
        if not isinstance(authors, list):
            authors = [str(authors)] if authors else []
        if len(authors) > 10:
            authors = authors[:5] + authors[-5:]

        papers.append(
            {
                "id": index,
                "paper_id": index,
                "paper_title": formatted_paper.get("title", ""),
                "arxiv_id": formatted_paper.get("arxiv_id", ""),
                "authors": authors,
                "tldr": formatted_paper.get("tldr", ""),
                "pub_date": formatted_paper.get("dates", []),
                "retrieval_evidence": formatted_paper.get("content", ""),
                "metadata": {
                    "title": formatted_paper.get("title", ""),
                },
            }
        )
    return papers


def make_inference_error_result(error_message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "total_time": 0.0,
        "messages": [],
        "final_candidates": [],
        "final_candidate_state": "empty",
        "turn_details": [{"action": "error", "error_message": error_message}],
    }


class LLMService:
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.default_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    async def close(self) -> None:
        await self.default_client.close()

    async def send_message(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_completion_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if stream:
            return await self._chat_stream(
                messages,
                model,
                temperature,
                max_completion_tokens,
                response_format,
                extra_body,
            )
        return await self._chat_completion(
            messages,
            model,
            temperature,
            max_completion_tokens,
            response_format,
            extra_body,
        )

    def _build_chat_kwargs(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str],
        temperature: float,
        max_completion_tokens: int,
        response_format: Optional[Dict[str, Any]],
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chat_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": 1800,
        }
        if response_format is not None:
            chat_kwargs["response_format"] = response_format
        if extra_body is not None:
            chat_kwargs["extra_body"] = extra_body
        return chat_kwargs

    async def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str],
        temperature: float = 1.0,
        max_completion_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        max_retries = 5
        chat_extra_body = dict(extra_body) if extra_body else None
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = await self.default_client.chat.completions.create(
                    **self._build_chat_kwargs(
                        messages,
                        model,
                        temperature,
                        max_completion_tokens,
                        response_format,
                        chat_extra_body,
                    )
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_exception = exc
                error_msg = str(exc).lower()
                if "context length" in error_msg or ("400" in error_msg and "longer than" in error_msg) or "badrequest" in error_msg:
                    LOGGER.warning("Aborting retries due to request validation error: %s", exc)
                    raise

                if attempt == 0 and chat_extra_body and "chat_template_kwargs" in chat_extra_body:
                    LOGGER.warning("Retrying LLM request without chat_template_kwargs after failure: %s", exc)
                    chat_extra_body.pop("chat_template_kwargs", None)
                else:
                    LOGGER.warning("LLM request attempt %s/%s failed: %s", attempt + 1, max_retries, exc)

                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {exc}") from exc
                await asyncio.sleep(0.5)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM request failed without an explicit exception.")

    async def _chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str],
        temperature: float = 1.0,
        max_completion_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if StreamingResponse is None:
            raise RuntimeError("fastapi is required for stream=True but is not installed.")

        async def event_generator():
            try:
                stream = await self.default_client.chat.completions.create(
                    **self._build_chat_kwargs(
                        messages,
                        model,
                        temperature,
                        max_completion_tokens,
                        response_format,
                        extra_body,
                    ),
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'token': token})}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

"""Academic paper search utilities backed by the DeepXiv retrieval API."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
from dotenv import load_dotenv
from qwen_agent.tools.base import BaseTool, register_tool

from prompts import SUMMARY_MODEL_PROMPT
from utils import LLMService


load_dotenv(Path(__file__).resolve().with_name(".env"))

SEARCH_API_URL = os.getenv("PAPER_SEARCH_API_URL", "")
OPENAI_API_KEY = os.getenv("PAPER_SEARCH_OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("PAPER_SEARCH_OPENAI_API_BASE", "")
SUMMARY_MODEL_NAME = os.getenv("PAPER_SEARCH_SUMMARY_MODEL_NAME", "gpt-5-mini")

LOGGER = logging.getLogger(__name__)


class DeepXivSearchEngine:
    """Search academic papers via an HTTP retrieval API."""

    def __init__(self, api_url: str = SEARCH_API_URL):
        self.api_url = api_url
        self.timeout = 120
        self.max_retries = 2
        self.retry_delay = 5
        self.llm = LLMService(
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_API_BASE,
        )

    async def search(
        self,
        query: Union[str, List[str]],
        top_k: int = 10,
        id_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for papers relevant to the input query."""
        del id_only  # Retained for backward-compatible signature.

        if not query:
            return []

        payload = {
            "query": query,
            "top_k": top_k,
            "return_contents": True,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                return result.get("result", [])
            except asyncio.TimeoutError:
                LOGGER.error("DeepXiv request timed out (attempt %s/%s): %s", attempt, total_attempts, self.api_url)
            except aiohttp.ClientError as exc:
                LOGGER.error("DeepXiv request failed (attempt %s/%s): %s", attempt, total_attempts, exc)
            except json.JSONDecodeError as exc:
                LOGGER.error(
                    "DeepXiv response JSON parsing failed (attempt %s/%s): %s",
                    attempt,
                    total_attempts,
                    exc,
                )
            except Exception as exc:
                LOGGER.error("DeepXiv search failed (attempt %s/%s): %s", attempt, total_attempts, exc)

            if attempt < total_attempts:
                LOGGER.warning("Retrying DeepXiv search, next attempt: %s/%s", attempt + 1, total_attempts)
                await asyncio.sleep(self.retry_delay)

        return []

    async def format_paper(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Normalize a retrieved paper into the public result schema."""
        authors = paper.get("authors", [])
        if len(authors) > 10:
            authors = authors[:5] + authors[-5:]

        content_text = "No content available."
        contents = paper.get("contents", [])
        if contents and isinstance(contents, list):
            section_contents = contents[0].get("section_contents", [])
            if section_contents:
                content_text = str(section_contents[0])

        messages = [
            {"role": "system", "content": "You are an expert in the field of AI academia."},
            {"role": "user", "content": SUMMARY_MODEL_PROMPT.format(paper_content=content_text, query=query)},
        ]

        content_summary = await self.llm.send_message(
            messages=messages,
            model=SUMMARY_MODEL_NAME,
            temperature=0.6,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        return {
            "arxiv_id": paper.get("arxiv_id", ""),
            "title": paper.get("title", ""),
            "authors": authors,
            "tldr": paper.get("tldr", ""),
            "dates": paper.get("date", ""),
            "content": content_summary,
        }


@register_tool("PaperSearchTool")
class PaperSearchTool(BaseTool):
    """Qwen-agent compatible academic paper search tool."""

    name = "PaperSearchTool"
    description = (
        "Search for relevant academic papers using a retrieval API. "
        "Returns paper metadata including arxiv_id, title, authors, TLDR, and summarized content."
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query used to find relevant papers.",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "integer",
            "description": "Number of top results to return (default: 10).",
            "required": False,
        },
    ]

    def __init__(self, api_url: Optional[str] = None):
        super().__init__()
        self.search_engine = DeepXivSearchEngine(api_url=api_url) if api_url else DeepXivSearchEngine()

    async def call(self, params: str, **kwargs) -> str:
        del kwargs

        try:
            params_dict = json.loads(params) if isinstance(params, str) else params
            query = params_dict.get("query", "")
            top_k = params_dict.get("top_k", 10)

            if not query:
                return json.dumps({"error": "Query parameter is required"}, ensure_ascii=False)

            results = await self.search_engine.search(query, top_k)
            if not results:
                return json.dumps(
                    {
                        "message": f"No papers found related to query: '{query}'",
                        "results": [],
                    },
                    ensure_ascii=False,
                )

            formatted_results = await asyncio.gather(
                *[self.search_engine.format_paper(paper, query) for paper in results]
            )

            output_parts = []
            for index, paper in enumerate(formatted_results, 1):
                output_parts.append(
                    f"""
                    === Paper {index} ===
                    ArXiv ID: {paper['arxiv_id']}
                    Title: {paper['title']}
                    Authors: {', '.join(paper['authors'])}
                    TLDR: {paper['tldr']}
                    Dates: {', '.join(paper['dates']) if paper['dates'] else 'N/A'}
                    Content:
                    {paper['content']}
                    """.strip()
                )

            return "\n".join(output_parts)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON parameters: {exc}"}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"Search failed: {exc}"}, ensure_ascii=False)


async def paper_search(query: Union[str, List[str]], top_k: int = 3) -> str:
    """Compatibility wrapper for direct async use."""
    tool = PaperSearchTool()
    return await tool.call({"query": query, "top_k": top_k})

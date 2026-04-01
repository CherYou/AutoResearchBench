"""Web search utilities backed by Serper and Jina."""

import asyncio
import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv
from qwen_agent.tools.base import BaseTool, register_tool

from prompts import SUMMARY_MODEL_PROMPT
from utils import LLMService


load_dotenv(Path(__file__).resolve().with_name(".env"))

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
OPENAI_API_KEY = os.getenv("WEB_SEARCH_OPENAI_API_KEY", os.getenv("PAPER_SEARCH_OPENAI_API_KEY", ""))
OPENAI_API_BASE = os.getenv("WEB_SEARCH_OPENAI_API_BASE", os.getenv("PAPER_SEARCH_OPENAI_API_BASE", ""))
SUMMARY_MODEL_NAME = os.getenv("WEB_SEARCH_SUMMARY_MODEL_NAME", os.getenv("PAPER_SEARCH_SUMMARY_MODEL_NAME", ""))

SERPER_SEARCH_URL = "https://google.serper.dev/search"
JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"

LOGGER = logging.getLogger(__name__)


class WebSearchEngine:
    """Search the web with Serper or Jina and summarize the results."""

    def __init__(self):
        self.timeout = 30
        self.llm = LLMService(
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_API_BASE,
        )

    async def _serper_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search the web through the Serper API."""
        if not SERPER_API_KEY:
            LOGGER.error("SERPER_API_KEY is not configured.")
            return []

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": f"arxiv:{query}", "num": top_k}

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(SERPER_SEARCH_URL, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

            results: List[Dict[str, Any]] = []
            for item in data.get("organic", [])[:top_k]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0),
                    }
                )
            return results
        except asyncio.TimeoutError:
            LOGGER.error("Serper request timed out.")
            return []
        except aiohttp.ClientError as exc:
            LOGGER.error("Serper request failed: %s", exc)
            return []
        except Exception as exc:
            LOGGER.error("Serper search failed: %s", exc)
            return []

    async def _jina_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search the web through the Jina search API."""
        encoded_query = urllib.parse.quote(f"arxiv:{query}")
        url = f"{JINA_SEARCH_URL}{encoded_query}"

        headers = {"Accept": "application/json"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

            results: List[Dict[str, Any]] = []
            for item in data.get("data", [])[:top_k]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "date": item.get("date", ""),
                        "snippet": item.get("description", ""),
                        "content": item.get("content", ""),
                    }
                )
            return results
        except asyncio.TimeoutError:
            LOGGER.error("Jina search request timed out.")
            return []
        except aiohttp.ClientError as exc:
            LOGGER.error("Jina search request failed: %s", exc)
            return []
        except Exception as exc:
            LOGGER.error("Jina search failed: %s", exc)
            return []

    async def _jina_read_url(self, url: str) -> str:
        """Fetch page content for a single URL via Jina Reader."""
        reader_url = f"{JINA_READER_URL}{url}"
        headers = {"Accept": "application/json"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(reader_url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
            return data.get("data", {}).get("content", "")
        except Exception as exc:
            LOGGER.warning("Jina Reader failed for %s: %s", url, exc)
            return ""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_engine: str = "jina",
        fetch_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a web search and optionally fetch full page content."""
        if not query:
            return []

        if search_engine == "jina":
            results = await self._jina_search(query, top_k)
            fetch_content = False
        else:
            results = await self._serper_search(query, top_k)

        if fetch_content and results:
            contents = await asyncio.gather(*[self._jina_read_url(result["url"]) for result in results])
            for result, content in zip(results, contents):
                result["content"] = content
        return results

    async def format_paper(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Normalize a search result into the public paper-like schema."""
        raw_content = (result.get("content", "") or result.get("snippet", ""))[:50000]

        if raw_content:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert research assistant specialized in "
                        "summarizing web content according to the user's query."
                    ),
                },
                {
                    "role": "user",
                    "content": SUMMARY_MODEL_PROMPT.format(
                        paper_content=raw_content,
                        query=query,
                    ),
                },
            ]
            summary = await self.llm.send_message(
                messages=messages,
                model=SUMMARY_MODEL_NAME,
                temperature=0.6,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        else:
            LOGGER.info("No content available for web result summary.")
            summary = json.dumps({"summary": "No content available."}, ensure_ascii=False)

        arxiv_id = result.get("url", "").rstrip("/").split("/")[-1]

        return {
            "title": result.get("title", ""),
            "arxiv_id": arxiv_id,
            "authors": [],
            "dates": result.get("date", ""),
            "tldr": raw_content[:1000] if raw_content else "",
            "content": summary,
        }


@register_tool("WebSearchTool")
class WebSearchTool(BaseTool):
    """Qwen-agent compatible web search tool."""

    name = "WebSearchTool"
    description = (
        "Search the web using Serper or Jina. "
        "Returns top results with titles, arXiv-style identifiers, and summarized content."
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string.",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "integer",
            "description": "Number of top search results to return (default: 10).",
            "required": False,
        },
        {
            "name": "search_engine",
            "type": "string",
            "description": "Search backend to use: 'serper' or 'jina' (default).",
            "required": False,
        },
    ]

    def __init__(self):
        super().__init__()
        self.search_engine = WebSearchEngine()

    async def call(self, params: str, **kwargs) -> str:
        del kwargs

        try:
            params_dict = json.loads(params) if isinstance(params, str) else params
            query = params_dict.get("query", "")
            top_k = params_dict.get("top_k", 10)
            engine = params_dict.get("search_engine", "jina")

            if not query:
                return json.dumps({"error": "Query parameter is required"}, ensure_ascii=False)

            results = await self.search_engine.search(query, top_k=top_k, search_engine=engine)
            if not results:
                return json.dumps(
                    {
                        "message": f"No results found for query: '{query}'",
                        "results": [],
                    },
                    ensure_ascii=False,
                )

            formatted_results = await asyncio.gather(
                *[self.search_engine.format_paper(result, query) for result in results]
            )

            output_parts = []
            for index, item in enumerate(formatted_results, 1):
                output_parts.append(
                    f"""
                    === Result {index} ===
                    ArXiv ID: {item['arxiv_id']}
                    Title: {item['title']}
                    Authors: {item['authors']}
                    Dates: {item['dates']}
                    TLDR: {item['tldr']}
                    Content:
                    {item['content']}
                    """.strip()
                )

            return "\n".join(output_parts)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON parameters: {exc}"}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"Web search failed: {exc}"}, ensure_ascii=False)


async def web_search(query: str, top_k: int = 10, search_engine: str = "jina") -> str:
    """Compatibility wrapper for direct async use."""
    tool = WebSearchTool()
    return await tool.call({"query": query, "top_k": top_k, "search_engine": search_engine})

"""
Web Search — DuckDuckGo search + optional page fetch and summarise.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


class WebSearch:
    def __init__(self, ollama_client=None):
        self._llm = ollama_client  # Optional — used for page summarisation

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Return a list of search results:
          [ { title, url, snippet } ]
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in results
            ]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def search_text(self, query: str, max_results: int = 5) -> str:
        """Formatted string version for injecting into prompts."""
        results = self.search(query, max_results)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Fetch and summarise a page
    # ------------------------------------------------------------------

    def fetch_page(self, url: str, max_chars: int = 8000) -> str:
        """Fetch a URL and return clean plain text (up to max_chars)."""
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:max_chars]
        except Exception as e:
            logger.error(f"Page fetch failed for {url}: {e}")
            return ""

    def summarise_page(self, url: str, question: Optional[str] = None) -> str:
        """Fetch a page and use the LLM to summarise it."""
        if not self._llm:
            return self.fetch_page(url)
        content = self.fetch_page(url)
        if not content:
            return "Could not fetch page."
        focus = f" Focus on answering: {question}" if question else ""
        prompt = f"Summarise the following web page content concisely.{focus}\n\n{content[:6000]}"
        return self._llm.generate(prompt, temperature=0.3)

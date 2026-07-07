from __future__ import annotations

import logging
from typing import Any, Dict, List
import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class JinaWebSearchProvider(WebSearchProvider):
    """Jina Reader web extract provider.

    Uses r.jina.ai to fetch clean markdown content without local docker dependencies or API keys.
    """

    @property
    def name(self) -> str:
        return "jina"

    @property
    def display_name(self) -> str:
        return "Jina Reader"

    def is_available(self) -> bool:
        # Jina is free and needs no API keys. Always available.
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Jina Reader API."""
        results: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in urls:
                logger.info("Jina extract URL: %s", url)
                try:
                    response = await client.get(
                        f"https://r.jina.ai/{url}",
                        headers={"Accept": "application/json"}
                    )

                    if response.status_code == 200:
                        raw_data = response.json()
                        data = raw_data.get("data", {})
                        title = data.get("title", "")
                        content = data.get("content", "")
                        results.append({
                            "url": url,
                            "title": title,
                            "content": content,
                            "raw_content": content,
                            "metadata": data.get("metadata", {}),
                        })
                    else:
                        error_msg = f"Jina Reader returned status code {response.status_code}"
                        logger.warning("Jina extract error for %s: %s", url, error_msg)
                        results.append({
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": error_msg,
                        })
                except Exception as exc:
                    error_msg = f"Jina Reader request failed: {exc}"
                    logger.warning("Jina extract exception for %s: %s", url, error_msg)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": error_msg,
                    })

        return results

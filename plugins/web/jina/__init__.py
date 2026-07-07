"""Jina Reader web extract plugin — bundled, auto-loaded.

Uses Jina Reader API (https://r.jina.ai/) for high-quality, high-speed, 
and zero-dependency markdown web content extraction.
"""

from __future__ import annotations

from plugins.web.jina.provider import JinaWebSearchProvider


def register(ctx) -> None:
    """Register the Jina provider with the plugin context."""
    ctx.register_web_search_provider(JinaWebSearchProvider())

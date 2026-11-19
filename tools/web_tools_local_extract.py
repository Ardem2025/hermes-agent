"""Local-First Tier 1 (SSRF-safe httpx + bs4/lxml + pymupdf/pypdf) and Tier 2 (direct crw-server) extractors."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from plugins.web._common import document, is_origin_http_error, origin_http_error, page_error
from tools.url_safety import (
    SSRFConnectionBlocked,
    async_is_safe_url,
    create_ssrf_safe_async_client,
    redirect_target_from_response,
)
from tools.website_policy import check_website_access

logger = logging.getLogger("tools.web_tools")

TIER1_CONCURRENCY = 5
TIER2_CONCURRENCY = 2
MAX_REDIRECTS = 5
MAX_HTML_BYTES = 10 * 1024 * 1024
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
MAX_PDF_PAGES = 100
DEFAULT_CRW_LOCAL_URL = "http://127.0.0.1:3002/v1/scrape"
DEFAULT_TIER2_BUDGET_SECONDS = 90.0
TIER2_PER_REQUEST_CAP_SECONDS = 45.0

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_BINARY_CONTENT_PREFIXES = ("image/", "video/", "audio/", "font/")
_BINARY_CONTENT_TYPES = {
    "application/octet-stream",
    "application/zip",
    "application/x-gzip",
    "application/gzip",
    "application/x-tar",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/vnd.ms-excel",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_TEXT_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
    "text/xml",
    "application/rss+xml",
    "application/atom+xml",
}
_WAF_MARKERS = (
    "cf-browser-verification",
    "just a moment...",
    "checking your browser before accessing",
    "attention required! | cloudflare",
    "smartcaptcha",
    "ddos-guard",
)
_EMPTY_ROOT_RE = re.compile(r"""<div\s+[^>]*id=["'](?:root|app)["'][^>]*>\s*</div>""", re.IGNORECASE)


@dataclass
class TierOutcome:
    status: str  # "ok" | "terminal_error" | "escalate"
    result: Dict[str, Any]
    reason: str = ""


def _is_pdf_response(content_type: str, url: str) -> bool:
    if content_type in ("application/pdf", "application/x-pdf"):
        return True
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return not content_type and path.endswith(".pdf")


def _is_binary_non_pdf(content_type: str) -> bool:
    if not content_type:
        return False
    if content_type in ("application/pdf", "application/x-pdf"):
        return False
    if content_type.startswith(_BINARY_CONTENT_PREFIXES):
        return True
    return content_type in _BINARY_CONTENT_TYPES


def _extract_pdf_text(raw_bytes: bytes) -> Tuple[str, str]:
    try:
        import pymupdf

        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        try:
            title = str((doc.metadata or {}).get("title") or "").strip()
            parts: List[str] = []
            total = 0
            for idx, page in enumerate(doc):
                if idx >= MAX_PDF_PAGES or total >= MAX_TEXT_CHARS:
                    break
                txt = page.get_text("text") or ""
                if txt:
                    parts.append(txt)
                    total += len(txt)
            text = "\n\n".join(parts).strip()[:MAX_TEXT_CHARS]
            if len("".join(text.split())) >= 100:
                return title, text
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("pymupdf extraction failed, trying pypdf: %s", exc)

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        meta = getattr(reader, "metadata", None) or {}
        title = str(meta.get("/Title") or "").strip() if hasattr(meta, "get") else ""
        parts = []
        total = 0
        for idx, page in enumerate(reader.pages):
            if idx >= MAX_PDF_PAGES or total >= MAX_TEXT_CHARS:
                break
            txt = page.extract_text() or ""
            if txt:
                parts.append(txt)
                total += len(txt)
        text = "\n\n".join(parts).strip()[:MAX_TEXT_CHARS]
        return title, text
    except Exception as exc:
        logger.debug("pypdf extraction failed: %s", exc)
        return "", ""


def _stdlib_html_to_text(html: str) -> Tuple[str, str]:
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._skip_depth = 0
            self._in_title = False
            self.title_parts: List[str] = []
            self.text_parts: List[str] = []

        def handle_starttag(self, tag: str, attrs) -> None:
            t = tag.lower()
            if t in ("script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside"):
                self._skip_depth += 1
            elif t == "title":
                self._in_title = True
            elif t in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
                self.text_parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            t = tag.lower()
            if t in ("script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside") and self._skip_depth > 0:
                self._skip_depth -= 1
            elif t == "title":
                self._in_title = False

        def handle_data(self, data: str) -> None:
            if self._in_title:
                self.title_parts.append(data)
            elif self._skip_depth == 0:
                self.text_parts.append(data)

    parser = _Extractor()
    parser.feed(html)
    title = " ".join("".join(parser.title_parts).split()).strip()
    raw = "".join(parser.text_parts)
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    clean = "\n\n".join(line for line in lines if line).strip()
    return title, clean[:MAX_TEXT_CHARS]


def _bs4_html_to_markdown(html: str) -> Tuple[str, str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _stdlib_html_to_text(html)

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split()).strip()

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside"]):
        tag.decompose()

    root = soup.find("main") or soup.find("article")
    if root is None or len(root.get_text(" ", strip=True)) < 200:
        root = soup.body or soup

    lines: List[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "table"], recursive=True):
        # Skip nested elements already covered by a parent block (e.g. p inside li or table)
        if el.name != "table" and el.find_parent(["table", "pre"]):
            continue
        if el.name == "p" and el.find_parent(["li", "blockquote"]):
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            txt = " ".join(el.get_text(" ", strip=True).split())
            if txt:
                hashes = "#" * int(el.name[1])
                lines.append(f"{hashes} {txt}")
        elif el.name == "li":
            txt = " ".join(el.get_text(" ", strip=True).split())
            if txt:
                lines.append(f"- {txt}")
        elif el.name == "blockquote":
            txt = " ".join(el.get_text(" ", strip=True).split())
            if txt:
                lines.append(f"> {txt}")
        elif el.name == "pre":
            txt = el.get_text("\n", strip=True)
            if txt:
                lines.append(f"```\n{txt}\n```")
        elif el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [" ".join(c.get_text(" ", strip=True).split()) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                lines.append("\n".join(rows))
        else:
            txt = " ".join(el.get_text(" ", strip=True).split())
            if txt:
                lines.append(txt)

    text = "\n\n".join(lines).strip()
    if len(text) < 80:
        fallback = " ".join(root.get_text(" ", strip=True).split())
        if len(fallback) > len(text):
            text = fallback
    return title, text[:MAX_TEXT_CHARS]


def _extract_html_content(html: str) -> Tuple[str, str, bool]:
    """Return ``(title, clean_text, passed_quality_gate)``."""
    lower_html = html.lower()
    has_waf = any(marker in lower_html for marker in _WAF_MARKERS)
    has_empty_root = bool(_EMPTY_ROOT_RE.search(html))
    has_next_data = "__next_data__" in lower_html
    has_enable_js = "you need to enable javascript" in lower_html

    title, bs4_text = _bs4_html_to_markdown(html)
    clean_text = bs4_text

    try:
        import trafilatura  # type: ignore[import-not-found]

        traf_text = trafilatura.extract(
            html,
            include_links=True,
            include_tables=True,
            output_format="markdown",
        ) or ""
        if len(traf_text.strip()) >= 200:
            clean_text = traf_text.strip()[:MAX_TEXT_CHARS]
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("trafilatura extract failed, using bs4 fallback: %s", exc)

    text_len = len(clean_text.strip())
    if has_waf and text_len < 1000:
        return title, clean_text, False
    if has_empty_root and text_len < 300:
        return title, clean_text, False
    if has_next_data and text_len < 300:
        return title, clean_text, False
    if has_enable_js and text_len < 500:
        return title, clean_text, False

    if text_len >= 200:
        return title, clean_text, True
    if len(html) < 4096 and text_len >= 80 and not (has_empty_root or has_next_data or has_enable_js or has_waf):
        return title, clean_text, True
    return title, clean_text, False


async def local_extract_url(url: str, *, client: Optional[httpx.AsyncClient] = None) -> TierOutcome:
    """Tier 1 SSRF-safe local HTTP fetch + HTML/PDF/text extraction."""
    try:
        policy_block = check_website_access(url)
    except Exception:
        policy_block = None
    if policy_block is not None:
        res = page_error(url, policy_block["message"])
        res["blocked_by_policy"] = True
        return TierOutcome("terminal_error", res, "website_policy")

    if not await async_is_safe_url(url):
        return TierOutcome(
            "terminal_error",
            page_error(url, "Blocked: URL targets a private or internal network address"),
            "ssrf",
        )

    owns_client = client is None
    if owns_client:
        client = create_ssrf_safe_async_client(
            follow_redirects=False,
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    current_url = url
    try:
        for hop in range(MAX_REDIRECTS + 1):
            try:
                async with client.stream("GET", current_url, headers=headers) as response:
                    if response.is_redirect:
                        if hop >= MAX_REDIRECTS:
                            return TierOutcome(
                                "terminal_error",
                                page_error(url, f"Too many redirects (>{MAX_REDIRECTS})"),
                                "too_many_redirects",
                            )
                        target = redirect_target_from_response(response)
                        if not target:
                            return TierOutcome(
                                "escalate",
                                page_error(url, "Redirect response missing Location header"),
                                "bad_redirect",
                            )
                        try:
                            redir_policy = check_website_access(target)
                        except Exception:
                            redir_policy = None
                        if redir_policy is not None:
                            res = page_error(url, redir_policy["message"])
                            res["blocked_by_policy"] = True
                            return TierOutcome("terminal_error", res, "redirect_policy")
                        if not await async_is_safe_url(target):
                            return TierOutcome(
                                "terminal_error",
                                page_error(url, "Blocked: redirect target violates URL safety policy"),
                                "redirect_ssrf",
                            )
                        current_url = target
                        continue

                    status = response.status_code
                    if status in (401, 404, 410):
                        return TierOutcome(
                            "terminal_error",
                            page_error(url, origin_http_error(status)),
                            f"origin_http_{status}",
                        )
                    if status in (403, 429, 503) or status >= 400:
                        return TierOutcome(
                            "escalate",
                            page_error(url, f"Tier 1 HTTP {status}"),
                            f"http_{status}",
                        )

                    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                    if _is_binary_non_pdf(content_type):
                        return TierOutcome(
                            "terminal_error",
                            page_error(url, f"Unsupported binary content-type: {content_type}"),
                            "binary_content_type",
                        )

                    is_pdf = _is_pdf_response(content_type, current_url)
                    max_bytes = MAX_PDF_BYTES if is_pdf else MAX_HTML_BYTES
                    cl_header = response.headers.get("content-length")
                    if cl_header:
                        try:
                            if int(cl_header) > max_bytes:
                                limit_mb = max_bytes // (1024 * 1024)
                                return TierOutcome(
                                    "terminal_error",
                                    page_error(url, f"Payload exceeds {limit_mb} MB limit"),
                                    "oversized_content_length",
                                )
                        except ValueError:
                            pass

                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        if not is_pdf and len(buf) < 5 and (bytes(buf) + chunk).startswith(b"%PDF-"):
                            is_pdf = True
                            max_bytes = MAX_PDF_BYTES
                        if len(buf) + len(chunk) > max_bytes:
                            limit_mb = max_bytes // (1024 * 1024)
                            return TierOutcome(
                                "terminal_error",
                                page_error(url, f"Payload exceeds {limit_mb} MB limit"),
                                "oversized_stream",
                            )
                        buf.extend(chunk)

                    raw_bytes = bytes(buf)
                    encoding = response.encoding or "utf-8"
                    break
            except SSRFConnectionBlocked as exc:
                return TierOutcome("terminal_error", page_error(url, f"Blocked: {exc}"), "ssrf_connect")
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return TierOutcome("escalate", page_error(url, f"Tier 1 request error: {exc}"), "request_error")
        else:
            return TierOutcome("terminal_error", page_error(url, f"Too many redirects (>{MAX_REDIRECTS})"), "too_many_redirects")
    finally:
        if owns_client:
            await client.aclose()

    if is_pdf or raw_bytes.startswith(b"%PDF-"):
        title, pdf_text = _extract_pdf_text(raw_bytes)
        if len("".join(pdf_text.split())) < 100:
            return TierOutcome("escalate", page_error(url, "PDF yielded insufficient text (<100 chars)"), "pdf_low_text")
        return TierOutcome("ok", document(url, title, pdf_text, source_url=current_url), "pdf")

    if content_type in _TEXT_CONTENT_TYPES or (content_type.startswith("text/") and "html" not in content_type):
        text = raw_bytes.decode(encoding, errors="replace").strip()[:MAX_TEXT_CHARS]
        if not text:
            return TierOutcome("escalate", page_error(url, "Empty text response"), "empty_text")
        return TierOutcome("ok", document(url, "", text, source_url=current_url), "text")

    html = raw_bytes.decode(encoding, errors="replace")
    title, clean_text, passed = _extract_html_content(html)
    if not passed:
        return TierOutcome("escalate", page_error(url, "Tier 1 HTML quality gate failed (SPA/WAF/short)"), "quality_gate")
    return TierOutcome("ok", document(url, title, clean_text, source_url=current_url), "html")


async def local_extract_urls(urls: List[str]) -> List[TierOutcome]:
    """Run Tier 1 extraction across *urls* with bounded concurrency (Semaphore(5))."""
    if not urls:
        return []
    sem = asyncio.Semaphore(TIER1_CONCURRENCY)
    async with create_ssrf_safe_async_client(
        follow_redirects=False,
        timeout=httpx.Timeout(15.0, connect=5.0),
    ) as client:
        async def _one(u: str) -> TierOutcome:
            async with sem:
                return await local_extract_url(u, client=client)

        return list(await asyncio.gather(*(_one(u) for u in urls)))


def resolve_crw_local_url() -> str:
    """Normalize CRW_LOCAL_URL (or FIRECRAWL_API_URL) without duplicating /v1/scrape."""
    raw = (os.environ.get("CRW_LOCAL_URL") or "").strip()
    if not raw:
        try:
            from hermes_cli.config import get_env_value
            raw = (get_env_value("CRW_LOCAL_URL") or "").strip()
        except Exception:
            raw = ""
    if not raw:
        raw = (os.environ.get("FIRECRAWL_API_URL") or "").strip()
    if not raw:
        return DEFAULT_CRW_LOCAL_URL
    base = raw.rstrip("/")
    if base.endswith("/v1/scrape"):
        return base
    if base.endswith("/v1"):
        return f"{base}/scrape"
    return f"{base}/v1/scrape"


async def crw_local_extract_urls(
    urls: List[str],
    *,
    budget_seconds: float = DEFAULT_TIER2_BUDGET_SECONDS,
) -> List[TierOutcome]:
    """Tier 2 direct POST to local crw-server (/v1/scrape), bypassing FirecrawlProvider._use_keyless_ring()."""
    if not urls:
        return []
    endpoint = resolve_crw_local_url()
    sem = asyncio.Semaphore(TIER2_CONCURRENCY)
    deadline = time.monotonic() + max(1.0, float(budget_seconds))

    async def _one(u: str) -> TierOutcome:
        async with sem:
            remaining = deadline - time.monotonic()
            if remaining <= 0.5:
                return TierOutcome("escalate", page_error(u, "Tier 2 crw-server budget exhausted"), "tier2_budget")
            req_timeout = min(remaining, TIER2_PER_REQUEST_CAP_SECONDS)
            try:
                async with httpx.AsyncClient(timeout=req_timeout) as client:
                    resp = await client.post(
                        endpoint,
                        json={"url": u, "formats": ["markdown"]},
                        headers={"Content-Type": "application/json"},
                    )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
                return TierOutcome("escalate", page_error(u, f"Tier 2 crw-server error: {exc}"), "tier2_request_error")

            if resp.status_code in (401, 404, 410):
                return TierOutcome("terminal_error", page_error(u, origin_http_error(resp.status_code)), f"origin_http_{resp.status_code}")
            if resp.status_code >= 400:
                return TierOutcome("escalate", page_error(u, f"Tier 2 crw-server HTTP {resp.status_code}"), f"tier2_http_{resp.status_code}")

            try:
                body = resp.json()
            except ValueError as exc:
                return TierOutcome("escalate", page_error(u, f"Tier 2 invalid JSON: {exc}"), "tier2_bad_json")

            if not isinstance(body, dict):
                return TierOutcome("escalate", page_error(u, "Tier 2 unexpected response shape"), "tier2_bad_shape")

            data = body.get("data") if isinstance(body.get("data"), dict) else body
            if body.get("success") is False:
                err = str(body.get("error") or data.get("error") or "Tier 2 crw-server returned success=false")
                if is_origin_http_error(err):
                    return TierOutcome("terminal_error", page_error(u, err), "tier2_origin_http")
                return TierOutcome("escalate", page_error(u, err), "tier2_failed")

            content = str(data.get("markdown") or data.get("content") or data.get("html") or "").strip()
            if not content:
                return TierOutcome("escalate", page_error(u, "Tier 2 crw-server returned empty content"), "tier2_empty")

            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            title = str(meta.get("title") or data.get("title") or "")
            final_url = str(meta.get("sourceURL") or meta.get("url") or data.get("url") or u)
            return TierOutcome("ok", document(u, title, content[:MAX_TEXT_CHARS], source_url=final_url), "tier2_ok")

    return list(await asyncio.gather(*(_one(u) for u in urls)))

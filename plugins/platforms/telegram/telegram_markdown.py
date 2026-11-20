"""Markdown -> Telegram MarkdownV2 conversion (the legacy, non-rich send path)."""

import bisect
from dataclasses import dataclass, field
import re
from typing import Callable, Optional

from gateway.platforms.helpers import (
    convert_table_to_bullets as _wrap_markdown_tables,
    normalize_latex_math_symbols,
    normalize_markdown_bullets,
)

# Stands in for the ** of a line-start **> expandable-quote opener while emphasis is resolved.
_EXPANDABLE_QUOTE = "\x00EQ\x00"

# Every char MarkdownV2 requires backslash-escaped outside code spans/fences.
_MDV2_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#\+\-=|{}.!\\])')
_EMPHASIS_MARK = {"bold": "*", "italic": "_"}


def _resolve_emphasis(text: str, stash, escape) -> str:
    """Resolve **bold** and *italic* delimiter runs together so nested/combined emphasis keeps its
    inner markers (***x*** → _*x*_, **a *i* b** → *a _i_ b*) instead of the bold pass swallowing
    them. Bold may span a single newline within a paragraph; italic stays single-line; unmatched
    markers survive as literal text."""

    tokens = []  # ("t", s) literal | ["d", start, end, can_open, can_close, remaining, opens, closes]
    pos = 0
    for m in re.finditer(r"\*+", text):
        if m.start() > pos:
            tokens.append(("t", text[pos : m.start()]))
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        can_open = bool(after) and not after.isspace()
        can_close = bool(before) and not before.isspace()
        tokens.append([
            "d",
            m.start(),
            m.end(),
            can_open,
            can_close,
            m.end() - m.start(),
            [],
            [],
        ])
        pos = m.end()
    if pos < len(text):
        tokens.append(("t", text[pos:]))

    stack = []  # indexes of delimiter tokens that may still close
    for idx, tok in enumerate(tokens):
        if tok[0] != "d":
            continue
        _, start, _end, can_open, can_close, remaining, _opens, closes = tok
        if can_close:
            while remaining > 0 and stack:
                opener = tokens[stack[-1]]
                o_end, o_remaining = opener[2], opener[5]
                use = 2 if o_remaining >= 2 and remaining >= 2 else 1
                span = text[o_end:start]
                if use == 1 and "\n" in span:
                    break  # italic must not cross a line
                if use == 2 and ("\n\n" in span or _crosses_quote_edge(text, o_end, start)):
                    break  # bold stays within one paragraph and on one side of a quote's edge
                kind = "bold" if use == 2 else "italic"
                opener[6].append((kind, idx))
                closes.append((kind, stack[-1]))
                opener[5] = o_remaining - use
                remaining -= use
                if opener[5] == 0:
                    stack.pop()
        tok[5] = remaining
        if remaining > 0 and can_open:
            stack.append(idx)

    return _render_emphasis(tokens, stash, escape)


def _crosses_quote_edge(text: str, start: int, end: int) -> bool:
    """True when a newline in text[start:end] borders a quote line: Telegram ends a blockquote at a
    newline into non-quote text, an entity still open there is rejected, and an entity spanning
    quote lines would swallow the next line's > marker."""
    quote = re.compile(rf'(?:{_EXPANDABLE_QUOTE})?>')
    is_quote = lambda pos: quote.match(text, text.rfind('\n', 0, pos) + 1) is not None
    return any(is_quote(start + m.start()) or is_quote(start + m.end()) for m in re.finditer('\n', text[start:end]))


def _emphasis_marks(tok, depth: dict) -> tuple:
    """The style kinds a delimiter run closes (innermost first) and opens (outermost first).

    A style that is the last to close and the first to reopen at one run continues unbroken;
    emitting both marks would read as "__" (underline) or an empty bold to Telegram."""
    closes = [kind for kind, _ in tok[7]]
    opens = [kind for kind, _ in reversed(tok[6])]
    for kind in ("bold", "italic"):
        if closes[-1:] == opens[:1] == [kind] and closes.count(kind) == depth[kind]:
            closes.remove(kind)
            opens.remove(kind)
    return closes, opens


def _render_emphasis(tokens: list, stash, escape) -> str:
    """Emit each delimiter's own boundary marks so combined runs that end at different places
    (***a* b**) keep every closing mark. MarkdownV2 cannot nest an entity inside itself, so a kind
    already open adds no further marks (****x**** -> *x*). Each outermost emphasized span is escaped
    and stashed whole, so later passes cannot cross it."""
    depth = {"bold": 0, "italic": 0}
    parts, group = [], None

    def emit_literal(value: str) -> None:
        if group is None:
            parts.append(value)
        else:
            group.append(escape(value))

    for tok in tokens:
        if tok[0] == "t":
            emit_literal(tok[1])
            continue
        closes, opens = _emphasis_marks(tok, depth)
        for kind in closes:
            depth[kind] -= 1
            if depth[kind] == 0:
                group.append(_EMPHASIS_MARK[kind])
        if group is not None and not any(depth.values()):
            parts.append(stash("".join(group)))
            group = None
        emit_literal("*" * tok[5])
        for kind in opens:
            if group is None:
                group = []
            if depth[kind] == 0:
                group.append(_EMPHASIS_MARK[kind])
            depth[kind] += 1
    return "".join(parts)


# A link target Telegram can open: a scheme URL (https:, tg:, mailto:) or a bare domain. A schemeless
# title ("[Title](Title)") or Hermes' desktop-only "@session:" reference is shown as raw syntax.
_RENDERABLE_LINK_TARGET_RE = re.compile(
    r'(?i)^(?:[a-z][a-z0-9+.-]*:[^\s@]\S*|[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:[/?#:]\S*)?)$')
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
# Fences, then inline spans closed by a backtick run of the SAME width (CommonMark, as
# ``_RICH_DOLLAR_RE``'s code branch): ``a [k](v w)`` is one span, not two empty ones.
_MD_CODE_RE = re.compile(r'```[\s\S]*?(?:```|$)|(?P<ticks>`+)[^\n]*?(?<!`)(?P=ticks)(?!`)')


def link_target_renders(target: str) -> bool:
    return bool(_RENDERABLE_LINK_TARGET_RE.match(target.strip()))


def degrade_unrenderable_links(markdown: str) -> str:
    """``[label](target)`` → ``label`` when Telegram cannot open ``target`` (rich-message path; code
    spans and fences stay verbatim)."""
    if '](' not in markdown:
        return markdown
    out, pos = [], 0
    for code in _MD_CODE_RE.finditer(markdown):
        out.append(_MD_LINK_RE.sub(_link_or_label, markdown[pos:code.start()]))
        out.append(code.group(0))
        pos = code.end()
    out.append(_MD_LINK_RE.sub(_link_or_label, markdown[pos:]))
    return ''.join(out)


def _link_or_label(m) -> str:
    return m.group(0) if link_target_renders(m.group(2)) else m.group(1)


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r'\\\1', text)


def to_markdown_v2(content: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2: code is stashed behind placeholders first (never
    modified), markdown constructs become MarkdownV2 syntax, everything else is escaped."""
    if not content:
        return content
    content = normalize_latex_math_symbols(content)
    content = normalize_markdown_bullets(content)
    placeholders: dict = {}
    counter = [0]

    def _ph(value: str) -> str:
        """Stash *value* behind a placeholder token that survives escaping."""
        key = f"\x00PH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = value
        return key

    def _ph_wrap(open_: str, close: str):
        return lambda m: _ph(f"{open_}{_escape_mdv2(m.group(1))}{close}")

    # 0) Rewrite GFM-style pipe tables into Telegram-friendly row groups
    #    before the normal MarkdownV2 conversions run.
    text = _wrap_markdown_tables(content)

    # 1) Protect fenced code blocks (``` ... ```)
    #    Per MarkdownV2 spec, \ and ` inside pre/code must be escaped.
    #    A fence still opens on its own line — the opening ``` must be the
    #    first triple-backtick run on that line and must end it — but the
    #    line may carry arbitrary leading whitespace (list/blockquote-
    #    nested code indents fences by 4+ spaces) or lead-in prose
    #    ("Here is the code: ```"), both of which the line-start-only
    #    anchor silently downgraded from <pre> to escaped literal text.
    #    Requiring the rest of the opening line to be backtick-free is
    #    what keeps *inline* triple backticks (e.g. "the syntax is
    #    ```x``` inline") out of the match: a closing run can never sit
    #    on the same line as the opener, and the tempered prefix cannot
    #    skip past an earlier run to a later one.  The closing fence must
    #    sit on its own line (any indent), with optional trailing
    #    whitespace and an optional ``\r`` so CRLF-terminated fences
    #    (Windows-authored content) match too.
    def _protect_fenced(m):
        prefix = m.group(1)   # lead-in text / indent before the opening fence
        opening = m.group(2)  # opening ``` (with optional language) and newline
        body = m.group(3)     # code body (may be empty)
        closing = m.group(4)   # closing fence (with its indent)
        body = body.replace('\\', '\\\\').replace('`', '\\`')
        return prefix + _ph(opening + body + closing)

    text = re.sub(
        r'(?m)^((?:(?!```)[^\n])*)(```[^`\n]*\n)([\s\S]*?)(^[ \t]*```)[ \t]*\r?$',
        _protect_fenced,
        text,
    )

    # 2) Protect inline code (`...` and GFM's backtick-free ``...``) in one pass.
    text = _stash_inline_code(text, _ph)

    # 3) Convert markdown links – escape the display text; inside the URL
    #    only ')' and '\' need escaping per the MarkdownV2 spec.
    def _convert_link(m):
        if _crosses_quote_edge(text, m.start(), m.end()):
            return m.group(0)  # Telegram ends the quote at the newline, cutting the link: keep it literal
        if not link_target_renders(m.group(2)):
            return _ph(_escape_mdv2(m.group(1)))  # Telegram would show the raw [label](target) syntax
        url = m.group(2).replace('\\', '\\\\').replace(')', '\\)')
        return _ph(f'[{_escape_mdv2(m.group(1))}]({url})')

    text = _MD_LINK_RE.sub(_convert_link, text)
    # 4) Headers (## Title) → bold *Title*, stripping redundant ** inside the header
    def _convert_header(m):
        inner = re.sub(r'\*\*(.+?)\*\*', r'\1', m.group(1).strip())
        return _ph(f'*{_escape_mdv2(inner)}*')

    text = re.sub(r'^#{1,6}\s+(.+)$', _convert_header, text, flags=re.MULTILINE)
    # 4b) An expandable quote opener (**> at line start) is quote syntax, not a bold delimiter: hide its
    #     ** behind a sentinel so step 5 cannot pair it with a later ** in the quoted text.
    text = re.sub(r'(?m)^\*\*(?=>{1,3} )', _EXPANDABLE_QUOTE, text)
    # 5) Bold **text** and *italic* resolve together so nested emphasis survives; unmatched
    # markers stay literal (bullet lists rely on that); 6) Strikethrough; 7) Spoiler kept as-is.
    text = _resolve_emphasis(text, _ph, _escape_mdv2)
    text = re.sub(r'~~(.+?)~~', _ph_wrap('~', '~'), text)
    text = re.sub(r'\|\|(.+?)\|\|', _ph_wrap('||', '||'), text)
    # 9) Blockquotes: protect leading > from escaping; expandable quotes (**> starts, || ends the block's LAST line).
    def _convert_blockquote(m):  # one quote block; a raw || anywhere but its last line is an unclosed spoiler
        lines = [line.split(' ', 1) for line in m.group(0).split('\n')]  # [prefix, content]; prefix: >, >>, **> …
        expandable = lines[0][0].startswith(_EXPANDABLE_QUOTE)
        lines[0][0] = lines[0][0].replace(_EXPANDABLE_QUOTE, '**')
        last = len(lines) - 1 if expandable and lines[-1][1].endswith('||') else -1
        return '\n'.join(_ph(f'{p} {_escape_mdv2(c[:-2])}||' if i == last else f'{p} {_escape_mdv2(c)}')
                         for i, (p, c) in enumerate(lines))

    text = re.sub(rf'^(?:{_EXPANDABLE_QUOTE})?>{{1,3}} .+$(?:\n>{{1,3}} .+$)*', _convert_blockquote, text,
                  flags=re.MULTILINE)
    # A sentinel left over means the opener had no quote text after all: restore its literal **.
    text = text.replace(_EXPANDABLE_QUOTE, '**')
    # 10) Escape remaining special characters in plain text
    text = _escape_mdv2(text)
    # 11) Restore placeholders in reverse insertion order so nested placeholders resolve.
    for key in reversed(list(placeholders.keys())):
        text = text.replace(key, placeholders[key])
    # 12) Safety net: escape bare ( ) { } that slipped through, but never inside ``` or ` spans.
    _safe_parts = []
    for _idx, _seg in enumerate(re.split(r'(```[\s\S]*?```|`[^`\r\n]+`)', text)):
        if _idx % 2 == 1:
            _safe_parts.append(_seg)  # inside code — untouched
        else:
            _safe_parts.append(re.sub(r'[(){}]', lambda m, _seg=_seg: _escape_bare_bracket(m, _seg), _seg))
    return ''.join(_safe_parts)


def _stash_inline_code(text: str, stash) -> str:
    """Stash inline code spans: a backtick run opens a span that the next run of the same width on
    the same line closes (CommonMark), so runs inside an open span are content, not new spans.

    Telegram code spans are single-backtick, so a span keeps its content verbatim with ` and \\
    escaped. A ``...`` span whose content holds backticks, or a ```...``` run, keeps its outer
    ticks as literal text and its content is scanned for spans of its own (`` `code` ``)."""
    runs = list(re.finditer(r'`+', text))
    breaks = [m.start() for m in re.finditer(r'[\r\n]', text)] + [len(text)]
    next_same, last = [None] * len(runs), {}
    for k in range(len(runs) - 1, -1, -1):  # index of the next run of the same width
        width = len(runs[k].group())
        next_same[k], last[width] = last.get(width), k
    literal, out, pos, i = set(), [], 0, 0
    while i < len(runs):
        opener, width, close = runs[i], len(runs[i].group()), next_same[i]
        if close is not None and runs[close].start() > breaks[bisect.bisect_left(breaks, opener.end())]:
            close = None  # a code span never crosses a line
        if i in literal or close is None:
            i += 1
            continue
        content = text[opener.end():runs[close].start()]
        if width > 2 or (width == 2 and '`' in content):
            literal.add(close)
            i += 1
            continue
        out.append(text[pos:opener.start()])
        out.append(stash('`' + content.replace('\\', '\\\\').replace('`', '\\`') + '`'))
        pos, i = runs[close].end(), close + 1
    out.append(text[pos:])
    return ''.join(out)


def _escape_bare_bracket(m, seg: str) -> str:
    """Escape a bare ( ) { } unless it is already escaped or delimits a ``[text](url)`` link."""
    s = m.start()
    ch = m.group(0)
    if s > 0 and seg[s - 1] == '\\':  # already escaped
        return ch
    if ch == '(' and s > 0 and seg[s - 1] == ']':  # opens a link [text](url)
        return ch
    if ch == ')':  # closes a link URL? walk back matching depth
        before = seg[:s]
        if '](http' in before or '](' in before:
            depth = 0
            for j in range(s - 1, max(s - 2000, -1), -1):
                if seg[j] == '(':
                    depth -= 1
                    if depth < 0:
                        if j > 0 and seg[j - 1] == ']':
                            return ch
                        break
                elif seg[j] == ')':
                    depth += 1
    return '\\' + ch


# The formula bodies are possessive: none can consume $, so a match ends a body at the first
# character it can't take, and a failed span must not retry every split of a TeX command (the
# escape runs on the gateway event loop for every streamed draft frame).
# Regions the rich-message dollar escape leaves verbatim (fenced code, inline code delimited by
# matching backtick-run widths, $$ display math, $x$ inline math, digit-led formulas such as $2+2$
# that carry an operator or close tight like $4$ (an amount is never closed by a $ that no digit or
# word follows), spaced formulas such as $2x + 1$, $2\,x$ or $5! = 120$ (any symbol but . , and `,
# or a TeX command; letters only as one-letter variables, so "$5 or 10$" and "$5 - $3" stay prose:
# the closing $ is never followed by a digit, so an amount's $ cannot close it), already-escaped \$),
# and the currency $ it escapes: a lone $ before an
# amount, or after a currency prefix (R$ 1.000, US$ 1,200). Telegram's rich Markdown pairs any two
# bare $ as inline LaTeX, so "$395k vs $483k" otherwise renders as one garbled math span.
_RICH_DOLLAR_RE = re.compile(
    r'(?P<keep>```[\s\S]*?```|(?P<ticks>`+)[^\n]*?(?<!`)(?P=ticks)(?!`)|\\\$|\$\$[\s\S]*?\$\$'
    r'|(?<![\w$])\$(?![\d\s$])(?:\\.|[^\\\n$])+?(?<!\s)\$(?![\d$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?:[-+*/^=<>]|\\[a-zA-Z]++)(?:[\w.,(){}^_+*/=<>-]|\\[a-zA-Z]+)*+(?<![-+*/^=<>])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d[\w.,]*(?<![.,])\$(?![\w$])'
    r'|(?<![\w$\\])\$\d(?=[^$\n]*?(?:[^\w\s$.,`]|\\))'
    r'(?:\\[a-zA-Z]+|\\[^a-zA-Z\s$]|[^\W\d_](?![^\W\d_])|[^$\n\\\w`]|[\d_])*+(?<![-+*/^=<>\s])\$(?![\w$]|\s?\d))'
    r'|(?P<currency>(?<![\\$])\$(?!\$)(?=\d)|(?<=[A-Z])\$(?=\s?\d))'
)


def escape_rich_currency_dollars(text: str) -> str:
    """Escape currency ``$`` for ``sendRichMessage`` markdown; real math, code and ``\\$`` are kept."""
    if '$' not in text:
        return text
    return _RICH_DOLLAR_RE.sub(lambda m: m.group(0) if m.group('keep') else '\\$', text)


# ---------------------------------------------------------------------------
# MarkdownV2-aware message chunking (post-format_message)
# ---------------------------------------------------------------------------

_MDV2_INLINE_MARKERS = ("||", "__", "*", "_", "~")
_MDV2_QUOTE_MARKER_RE = re.compile(r"(?:\*\*)?>+")
_MDV2_LINK_RE = re.compile(r"\[((?:\\.|`(?:\\.|[^`\\])*`|[^\]\\`])*)\]\(((?:\\.|[^)\\])*)\)")
_INDICATOR_RESERVE = 14          # " \(XXX/XXX\)"
_INDICATOR_NEWLINE_RESERVE = 15  # "\n \(XXX/XXX\)"


@dataclass
class _Mdv2State:
    in_fence: bool = False
    fence_lang: str = ""
    in_inline_code: bool = False
    in_quote: bool = False
    quote_prefix: str = ""
    expandable_quote: bool = False
    open_marks: list[str] = field(default_factory=list)

    def copy(self) -> "_Mdv2State":
        return _Mdv2State(
            in_fence=self.in_fence,
            fence_lang=self.fence_lang,
            in_inline_code=self.in_inline_code,
            in_quote=self.in_quote,
            quote_prefix=self.quote_prefix,
            expandable_quote=self.expandable_quote,
            open_marks=list(self.open_marks),
        )


def _unit_to_cp(s: str, budget: int, len_fn: Callable[[str], int]) -> int:
    if budget <= 0:
        return 0
    if len_fn is len:
        return min(len(s), budget)
    if len_fn(s) <= budget:
        return len(s)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _ends_with_unescaped_char(s: str, ch: str) -> bool:
    if not s.endswith(ch):
        return False
    stripped = s.rstrip(ch)
    bs = len(stripped) - len(stripped.rstrip("\\"))
    trailing = len(s) - len(stripped)
    return (trailing - (bs % 2)) >= 1


def _closing_suffix(state: _Mdv2State, chunk_body: str) -> tuple[str, bool]:
    """Return (suffix, ends_with_newline_indicator) for closing open entities at split."""
    parts: list[str] = []
    ends_nl = False
    if state.in_inline_code:
        parts.append("`")
    if state.in_fence:
        parts.append("```" if chunk_body.endswith("\n") else "\n```")
        ends_nl = True
    if state.open_marks:
        rev = list(reversed(state.open_marks))
        closed_parts: list[str] = []
        for mk in rev:
            if closed_parts and {closed_parts[-1], mk} == {"_", "__"}:
                closed_parts.append(" ")
            closed_parts.append(mk)
        if (
            not state.in_inline_code
            and not state.in_fence
            and closed_parts[0] in ("_", "__")
            and _ends_with_unescaped_char(chunk_body, "_")
        ):
            closed_parts.insert(0, " ")
        parts.append("".join(closed_parts))
        ends_nl = False
    if state.expandable_quote:
        if not parts and _ends_with_unescaped_char(chunk_body, "|"):
            parts.append(" ")
        elif parts and parts[-1].endswith("||"):
            parts.append(" ")
        parts.append("||")
        ends_nl = True
    return "".join(parts), ends_nl


def _scan_mdv2(
    s: str,
    init_state: _Mdv2State,
    atomic_budget: int,
) -> tuple[list[tuple[int, _Mdv2State, str]], _Mdv2State, str]:
    """Scan *s* from *init_state* and return:
    - boundaries: list of (pos, state_at_pos, last_token_kind) for safe split positions 1..len(s)
    - end_state: state at len(s)
    - end_tok_kind: kind of the last token in s
    """
    st = init_state.copy()
    n = len(s)
    i = 0
    boundaries: list[tuple[int, _Mdv2State, str]] = []
    # Track trailing opener positions so we never leave a dangling opener at chunk end.
    # Each entry is (start_pos, kind) for consecutive openers with no content after them.
    trailing_openers: list[int] = []
    last_tok = "text"

    def _record(pos: int, tok_kind: str, is_opener: bool = False, opener_start: int = -1) -> None:
        nonlocal last_tok
        last_tok = tok_kind
        if is_opener:
            trailing_openers.append(opener_start)
        else:
            trailing_openers.clear()
        boundaries.append((pos, st.copy(), tok_kind))

    while i < n:
        ch = s[i]

        if st.in_fence:
            if ch == "\\":
                step = 2 if i + 1 < n else 1
                i += step
                _record(i, "escape")
                continue
            if s.startswith("```", i):
                st.in_fence = False
                st.fence_lang = ""
                i += 3
                _record(i, "close_fence")
                continue
            i += 1
            _record(i, "text")
            continue

        if st.in_inline_code:
            if ch == "\\":
                step = 2 if i + 1 < n else 1
                i += step
                _record(i, "escape")
                continue
            if ch == "`":
                st.in_inline_code = False
                i += 1
                _record(i, "close_inline_code")
                continue
            i += 1
            _record(i, "text")
            continue

        # Outside fence and inline code
        if ch == "\n" and st.in_quote and s[i + 1 : i + 2] != ">":
            st.in_quote = False
            st.expandable_quote = False
            st.quote_prefix = ""
            i += 1
            _record(i, "newline")
            continue

        if ch == "\\":
            step = 2 if i + 1 < n else 1
            i += step
            _record(i, "escape")
            continue

        if s.startswith("```", i):
            nl = s.find("\n", i + 3)
            close_same = s.find("```", i + 3)
            if nl != -1 and (close_same == -1 or nl < close_same) and (nl - i) <= max(32, atomic_budget // 2):
                st.in_fence = True
                st.fence_lang = s[i + 3 : nl]
                opener_start = i
                i = nl + 1
                _record(i, "open_fence", is_opener=True, opener_start=opener_start)
            else:
                st.in_fence = True
                st.fence_lang = ""
                opener_start = i
                i += 3
                _record(i, "open_fence", is_opener=True, opener_start=opener_start)
            continue

        if ch == "`":
            # Check if short inline code span fits atomically
            end = i + 1
            while end < n and s[end] != "`":
                end += 2 if s[end] == "\\" else 1
            if end < n and (end + 1 - i) <= atomic_budget:
                i = end + 1
                _record(i, "inline_code_span")
            else:
                st.in_inline_code = True
                opener_start = i
                i += 1
                _record(i, "open_inline_code", is_opener=True, opener_start=opener_start)
            continue

        if ch == "[":
            m = _MDV2_LINK_RE.match(s, i)
            if m and not (st.in_quote and re.search(r"\n(?!>)", m.group(0))):
                if (m.end() - i) <= atomic_budget:
                    i = m.end()
                    _record(i, "link")
                    continue
            i += 1
            _record(i, "text")
            continue

        if i == 0 or s[i - 1] == "\n":
            qm = _MDV2_QUOTE_MARKER_RE.match(s, i)
            if qm:
                st.in_quote = True
                st.expandable_quote = st.expandable_quote or (s[i] == "*")
                st.quote_prefix = qm.group(0)
                opener_start = i
                i = qm.end()
                if i < n and s[i] == " ":
                    i += 1
                _record(i, "quote_marker", is_opener=True, opener_start=opener_start)
                continue

        if (
            st.expandable_quote
            and s.startswith("||", i)
            and s[i + 2 : i + 3] in ("", "\n")
            and st.open_marks[-1:] != ["||"]
            and s[i + 3 : i + 4] != ">"
        ):
            st.expandable_quote = False
            i += 2
            _record(i, "close_expandable_quote")
            continue

        mark = next((mk for mk in _MDV2_INLINE_MARKERS if s.startswith(mk, i)), None)
        if mark is not None:
            opener_start = i
            if st.open_marks and st.open_marks[-1] == mark:
                st.open_marks.pop()
                i += len(mark)
                _record(i, "close_mark")
            else:
                st.open_marks.append(mark)
                i += len(mark)
                _record(i, "open_mark", is_opener=True, opener_start=opener_start)
            continue

        i += 1
        _record(i, "newline" if ch == "\n" else ("space" if ch == " " else "text"))

    return boundaries, st, last_tok


def _is_safe_boundary(
    s: str,
    pos: int,
    st: _Mdv2State,
    tok_kind: str,
) -> bool:
    """Check collision and dangling-opener invariants at boundary *pos*."""
    if tok_kind in ("open_mark", "open_inline_code", "open_fence", "quote_marker"):
        return False
    if st.open_marks[-1:] == ["_"]:
        tail = s[pos:] if st.in_fence else s[pos:].lstrip()
        if tail.startswith("__"):
            return False
    if st.open_marks[-1:] == ["__"]:
        tail = s[pos:] if st.in_fence else s[pos:].lstrip()
        if tail.startswith("_") and not tail.startswith("__"):
            return False
    return True


def _collapse_leading_closers(remaining: str, carry: _Mdv2State) -> str:
    """Strip boundary whitespace and collapse immediate closing markers in *remaining*."""
    if carry.in_fence:
        if remaining.startswith("\n"):
            remaining = remaining[1:]
        if remaining.startswith("```") and (len(remaining) == 3 or remaining[3:4] == "\n"):
            carry.in_fence = False
            carry.fence_lang = ""
            remaining = remaining[3:]
            if remaining.startswith("\n"):
                remaining = remaining[1:]
        return remaining

    remaining = remaining.lstrip()
    while carry.open_marks:
        next_mk = next((mk for mk in _MDV2_INLINE_MARKERS if remaining.startswith(mk)), None)
        if next_mk is not None and next_mk == carry.open_marks[-1]:
            carry.open_marks.pop()
            remaining = remaining[len(next_mk):].lstrip()
        else:
            break

    if (
        carry.expandable_quote
        and remaining.startswith("||")
        and remaining[2:3] in ("", "\n")
        and carry.open_marks[-1:] != ["||"]
        and remaining[3:4] != ">"
    ):
        carry.expandable_quote = False
        remaining = remaining[2:]
        if not remaining or (remaining.startswith("\n") and remaining[1:2] != ">"):
            carry.in_quote = False
            carry.quote_prefix = ""
        remaining = remaining.lstrip()

    return remaining


def _build_prefix(carry: _Mdv2State, remaining: str) -> str:
    parts: list[str] = []
    if carry.in_quote:
        if carry.expandable_quote:
            if remaining.startswith(">"):
                parts.append("**")
            else:
                q = carry.quote_prefix.lstrip("*") or ">"
                parts.append(f"**{q} ")
        else:
            if not remaining.startswith(">"):
                q = carry.quote_prefix or ">"
                parts.append(f"{q} ")
    if carry.open_marks:
        open_parts: list[str] = []
        for mk in carry.open_marks:
            if open_parts and {open_parts[-1], mk} == {"_", "__"}:
                open_parts.append(" ")
            open_parts.append(mk)
        parts.append("".join(open_parts))
        if not carry.in_fence and not carry.in_inline_code:
            last_mk = carry.open_marks[-1]
            if (last_mk == "_" and remaining.startswith("__")) or (
                last_mk == "__" and remaining.startswith("_")
            ):
                parts.append(" ")
    if carry.in_fence:
        parts.append(f"```{carry.fence_lang}\n")
    if carry.in_inline_code:
        parts.append("`")
    return "".join(parts)


def truncate_markdown_v2(
    content: str,
    max_length: int = 4096,
    len_fn: Optional[Callable[[str], int]] = None,
) -> list[str]:
    """Split already-formatted Telegram MarkdownV2 *content* into valid chunks <= *max_length*.

    Preserves paired inline entities (``||``, ``__``, ``*``, ``_``, ``~``), escape pairs,
    inline code, fenced code blocks, links, and blockquotes (``>`` / ``**> ...||``),
    appending escaped `` \\(i/N\\)`` (or ``\\n \\(i/N\\)`` after closing fence / expandable quote)
    when split into multiple chunks.
    """
    _len = len_fn or len
    if _len(content) <= max_length:
        return [content]

    chunks: list[tuple[str, bool]] = []
    remaining = content
    carry = _Mdv2State()

    while remaining or carry.open_marks or carry.in_fence or carry.in_inline_code or carry.expandable_quote:
        if chunks:
            remaining = _collapse_leading_closers(remaining, carry)
            if (
                not remaining
                and not carry.open_marks
                and not carry.in_fence
                and not carry.in_inline_code
                and not carry.expandable_quote
            ):
                break

        prefix = _build_prefix(carry, remaining)
        prefix_len = _len(prefix)
        atomic_budget = max(16, max_length - prefix_len - _INDICATOR_NEWLINE_RESERVE - 16)

        boundaries, end_state, end_tok = _scan_mdv2(remaining, carry, atomic_budget)
        end_suffix, end_suffix_nl = _closing_suffix(end_state, prefix + remaining)
        final_candidate = prefix + remaining + end_suffix
        final_ends_nl = end_suffix_nl or (
            not end_suffix and end_tok in ("close_fence", "close_expandable_quote")
        )
        final_ind_reserve = (
            (_INDICATOR_NEWLINE_RESERVE if final_ends_nl else _INDICATOR_RESERVE)
            if chunks
            else 0
        )
        if _len(final_candidate) + final_ind_reserve <= max_length:
            chunks.append((final_candidate, final_ends_nl))
            break

        # Upper bound on codepoints for remaining[:cp_limit]
        body_budget = max(1, max_length - prefix_len - _INDICATOR_RESERVE)
        cp_limit = _unit_to_cp(remaining, body_budget, _len)
        if cp_limit < 1:
            cp_limit = 1

        # Filter boundaries <= cp_limit that fit exact length constraint
        def _fits(pos: int, st: _Mdv2State, tok_kind: str) -> tuple[bool, str, bool]:
            body = remaining[:pos]
            suf, suf_nl = _closing_suffix(st, prefix + body)
            ends_nl = suf_nl or (not suf and tok_kind in ("close_fence", "close_expandable_quote"))
            ind_res = _INDICATOR_NEWLINE_RESERVE if ends_nl else _INDICATOR_RESERVE
            return (_len(prefix + body + suf) + ind_res <= max_length), suf, ends_nl

        # Collect candidate indices in boundaries with pos <= cp_limit
        cand_indices = [idx for idx, (pos, _, _) in enumerate(boundaries) if pos <= cp_limit]
        if not cand_indices and boundaries:
            cand_indices = [0]

        chosen_idx: Optional[int] = None
        half_cp = cp_limit // 2

        # Priority 1: last newline >= cp_limit // 2 (safe & fits)
        for idx in reversed(cand_indices):
            pos, st, tok = boundaries[idx]
            if pos < half_cp:
                break
            if pos > 0 and remaining[pos - 1] == "\n" and _is_safe_boundary(remaining, pos, st, tok):
                # Prefer splitting BEFORE the newline if pos - 1 is also a safe boundary
                if pos >= 2 and idx > 0 and boundaries[idx - 1][0] == pos - 1:
                    p_pos, p_st, p_tok = boundaries[idx - 1]
                    if p_pos >= half_cp and _is_safe_boundary(remaining, p_pos, p_st, p_tok):
                        ok, _, _ = _fits(p_pos, p_st, p_tok)
                        if ok:
                            chosen_idx = idx - 1
                            break
                ok, _, _ = _fits(pos, st, tok)
                if ok:
                    chosen_idx = idx
                    break

        # Priority 2: last space >= cp_limit // 2 (safe & fits)
        if chosen_idx is None:
            for idx in reversed(cand_indices):
                pos, st, tok = boundaries[idx]
                if pos < half_cp:
                    break
                if pos > 0 and remaining[pos - 1] == " " and _is_safe_boundary(remaining, pos, st, tok):
                    ok, _, _ = _fits(pos, st, tok)
                    if ok:
                        chosen_idx = idx
                        break

        # Priority 3: latest safe boundary <= cp_limit that fits
        if chosen_idx is None:
            for idx in reversed(cand_indices):
                pos, st, tok = boundaries[idx]
                if _is_safe_boundary(remaining, pos, st, tok):
                    ok, _, _ = _fits(pos, st, tok)
                    if ok:
                        chosen_idx = idx
                        break

        # Fallback: latest boundary <= cp_limit that fits (even if unsafe, to guarantee progress)
        if chosen_idx is None:
            for idx in reversed(cand_indices):
                pos, st, tok = boundaries[idx]
                ok, _, _ = _fits(pos, st, tok)
                if ok:
                    chosen_idx = idx
                    break

        if chosen_idx is None:
            chosen_idx = cand_indices[0] if cand_indices else 0

        split_at, split_state, split_tok = boundaries[chosen_idx]
        chunk_body = remaining[:split_at]
        suffix, suf_nl = _closing_suffix(split_state, prefix + chunk_body)
        ends_nl = suf_nl or (not suffix and split_tok in ("close_fence", "close_expandable_quote"))
        chunks.append((prefix + chunk_body + suffix, ends_nl))

        next_remaining = remaining[split_at:]
        next_carry = split_state.copy()
        # If split landed right before a newline that exits a quote, close quote state for next chunk
        if next_carry.in_quote and next_remaining.startswith("\n") and next_remaining[1:2] != ">":
            next_carry.in_quote = False
            next_carry.expandable_quote = False
            next_carry.quote_prefix = ""

        remaining = next_remaining
        carry = next_carry

    total = len(chunks)
    if total > 1:
        return [
            f"{text}\n \\({i + 1}/{total}\\)" if ends_nl else f"{text} \\({i + 1}/{total}\\)"
            for i, (text, ends_nl) in enumerate(chunks)
        ]
    return [text for text, _ in chunks]


"""Show a Markdown guide from docs/ as a page inside the app. Small on purpose: headings, paragraphs, lists,
tables, code, bold, italic, inline code and links. Standard library only; everything is HTML-escaped first.
"""
from __future__ import annotations

import html
import re

CSS = """
:root{color-scheme:dark light;--bg:#0f1614;--ink:#e4ebe8;--dim:#9aa9a3;--line:#2e3a36;--acc:#4fb8dc;--panel:#17201d}
@media (prefers-color-scheme: light){:root{--bg:#f8faf9;--ink:#15211e;--dim:#56645f;--line:#c9d2ce;--acc:#0a6b8f;--panel:#e9edeb}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:46rem;margin:0 auto;padding:20px 16px 60px}
h1,h2,h3{line-height:1.2;margin:1.6em 0 .5em} h1{margin-top:.6em}
a{color:var(--acc)} code{background:var(--panel);padding:1px 5px;border-radius:4px;font-size:.92em}
pre{background:var(--panel);padding:10px 12px;border-radius:6px;overflow-x:auto} pre code{padding:0;background:none}
table{border-collapse:collapse;width:100%;font-size:.93em;margin:.8em 0} th,td{border-bottom:1px solid var(--line);
padding:6px 8px;text-align:left;vertical-align:top} th{color:var(--dim);font-weight:600}
.back{font-size:.9em;color:var(--dim)} .scroll{overflow-x:auto}
"""


def _inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?!\w)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', s)
    return s


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(md: str) -> str:
    """Markdown text -> the HTML of the body (no <html> wrapper)."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            flush()
            head = _cells(line)
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            out.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</tbody></table></div>")
            i = j
            continue
        elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
            flush()
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def page(md: str, title: str, back: str = "/") -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
            f'<a class="back" href="{html.escape(back, quote=True)}">&larr; Back</a>'
            f"{render(md)}</main></body></html>")

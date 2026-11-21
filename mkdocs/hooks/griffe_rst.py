"""Griffe extension: render pandera's reST docstrings as Markdown.

Pandera's docstrings were written for Sphinx (``:param:`` fields, ``:class:``
roles, ``>>>`` doctest blocks, ``.. note::`` directives, ``::`` literal
blocks). mkdocstrings renders docstring text as Markdown, so without
translation those constructs come out as run-on paragraphs and nested
block quotes. Before griffe parses each docstring, this extension rewrites
the reST constructs into their Markdown equivalents. Field lists
(``:param:``, ``:returns:``, ``:raises:``) are left for griffe's ``sphinx``
parser; ``docstring_style: auto`` in ``mkdocs.yml`` picks the ``numpy``
parser for the few docstrings that use numpy-style sections.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

from griffe import Extension, Object

_ADMONITIONS = {
    "note": ("note", None),
    "tip": ("tip", None),
    "hint": ("tip", None),
    "important": ("warning", "Important"),
    "attention": ("warning", "Attention"),
    "warning": ("warning", None),
    "caution": ("warning", "Caution"),
    "danger": ("danger", None),
    "seealso": ("info", "See also"),
    "deprecated": ("warning", "Deprecated"),
    "versionadded": ("info", "Added"),
    "versionchanged": ("info", "Changed"),
}
_CODE_DIRECTIVES = {
    "code-block": None,
    "code": None,
    "sourcecode": None,
    "doctest": "pycon",
    "testcode": "python",
    "testoutput": "text",
}

_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w-]+)::\s*(.*)$")
_OPTION = re.compile(r"^\s*:[\w-]+:.*$")
_ROLE = re.compile(
    r":(?:py:)?(class|func|meth|mod|attr|data|obj|exc|const|ref|doc|term)"
    r":`([^`]+)`"
)
_LINK = re.compile(r"`([^`<]+?)\s*<(https?://[^>]+)>`__?")
_FIELD_EXAMPLE = re.compile(r"^(\s*):(examples?):\s*$", re.I)
# ``:raises: `Error`: text`` / ``:raises: Error if ...`` -> ``:raises Error:``
_BARE_RAISES = re.compile(r"^(\s*):raises:\s*`{0,2}([A-Z]\w*)`{0,2}:?\s+(.*)$")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _collect_block(lines: list[str], start: int, base: int) -> int:
    """Return the end index of the block indented deeper than ``base``."""
    j = start
    while j < len(lines) and (
        not lines[j].strip() or _indent_of(lines[j]) > base
    ):
        j += 1
    while j > start and not lines[j - 1].strip():
        j -= 1
    return j


def _fence(pad: str, lang: str | None, body: list[str]) -> list[str]:
    body = textwrap.dedent("\n".join(body)).split("\n")
    return [
        f"{pad}```{lang or ''}",
        *(pad + b if b else b for b in body),
        f"{pad}```",
    ]


def _role(match: re.Match) -> str:
    kind, text = match.groups()
    labelled = re.match(r"^(.*?)\s*<([^>]+)>$", text)
    if labelled:
        label = labelled.group(1)
        return label if kind in {"ref", "doc", "term"} else f"`{label}`"
    if kind in {"ref", "doc", "term"}:
        return text
    if text.startswith("~"):
        text = text[1:].rsplit(".", 1)[-1]
    return f"`{text.lstrip('.!')}`"


def _inline(line: str) -> str:
    line = _ROLE.sub(_role, line)
    line = _LINK.sub(r"[\1](\2)", line)
    return line.replace("``", "`")


def rst_to_markdown(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        pad = line[: _indent_of(line)]
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        directive = _DIRECTIVE.match(line)
        if directive:
            name, arg = directive.group(2), directive.group(3).strip()
            base = len(directive.group(1))
            j = i + 1
            while j < len(lines) and _OPTION.match(lines[j]):
                j += 1  # directive options, e.g. ``:skipif:``
            end = _collect_block(lines, j, base)
            body = lines[j:end]
            if name in _CODE_DIRECTIVES:
                lang = _CODE_DIRECTIVES[name] or arg or None
                out += _fence(pad, lang, body)
            elif name in _ADMONITIONS:
                kind, title = _ADMONITIONS[name]
                content = textwrap.dedent("\n".join(body)).split("\n")
                if (
                    title
                    and arg
                    and name.startswith(("deprecated", "version"))
                ):
                    title = f"{title} in version {arg}"
                elif arg:
                    content = [arg, *content]
                header = f"{pad}!!! {kind}" + (f' "{title}"' if title else "")
                out.append(header)
                out += [
                    f"{pad}    {_inline(c)}" if c.strip() else ""
                    for c in rst_to_markdown("\n".join(content)).split("\n")
                ]
            else:
                out += lines[i:end]
            i = end
            continue

        if stripped.startswith(">>>"):
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            out += _fence(pad, "pycon", lines[i:j])
            i = j
            continue

        line = _BARE_RAISES.sub(r"\1:raises \2: \3", line)
        example = _FIELD_EXAMPLE.match(line)
        if example:
            out.append(f"{example.group(1)}**{example.group(2).title()}:**")
            i += 1
            continue

        if stripped.endswith("::") and not stripped.startswith(".."):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _indent_of(lines[j]) > _indent_of(line):
                end = _collect_block(lines, j, _indent_of(line))
                head = line.rstrip()[:-2].rstrip()
                if head.strip():
                    out.append(_inline(head + ":"))
                out.append("")
                out += _fence(pad, "python", lines[j:end])
                i = end
                continue

        out.append(_inline(line))
        i += 1
    return "\n".join(out)


class RstDocstrings(Extension):
    """Translate reST docstrings to Markdown before griffe parses them."""

    def on_instance(self, *, obj: Object, **kwargs: Any) -> None:
        docstring = obj.docstring
        if docstring is None or not docstring.value:
            return
        docstring.value = rst_to_markdown(docstring.value)

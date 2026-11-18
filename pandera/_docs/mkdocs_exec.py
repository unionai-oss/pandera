"""MkDocs plugin that executes inline Python code cells at build time.

The pandera documentation was migrated from Sphinx + ``myst-nb`` to Material
for MkDocs. On the old site ``myst-nb`` executed every ``{code-cell}`` block
during the build and rendered its output. This plugin restores that behaviour:
for each page it matches the converted python fences (in order) against the
``{code-cell}`` bodies in the original ``docs/source/`` file, executes the
matched cells in a fresh page-scoped namespace, and injects each cell's stdout
plus the rendered representation of its trailing expression back into the page
after the code block. Cells whose source carries the ``raises-exception`` tag
render their traceback instead of failing the build.

The plugin is deliberately dependency-light: cells run with plain ``exec`` in a
page-scoped module (registered in ``sys.modules`` so ``typing.get_type_hints``
resolves, just as it does in an IPython kernel) and a few standard-library
helpers, so no Jupyter kernel is required at build time. This mirrors the common
pandera doc pattern (``import`` + print/pandas frames) while keeping the docs
build fast and reliable.

Output is styled with the ``.doc-exec-*`` classes (see ``mkdocs/css/pandera.css``).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import html
import io
import re
import sys
import textwrap
import traceback
import types
from pathlib import Path
from typing import Any

from mkdocs.config import config_options
from mkdocs.plugins import BasePlugin

_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)\n"
    r"(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _is_python_fence(info: str) -> bool:
    i = info.strip().lower()
    return bool(i) and i.split()[0] in {"python", "python3", "py", "py3"}


_DIRECTIVE_LINE_RE = re.compile(r"^\s*:[A-Za-z_][A-Za-z0-9_]*:\s")
_CELL_BLOCK_RE = re.compile(
    r"```\{code-cell\}[^\n]*\n(?P<body>.*?)\n```", re.DOTALL
)


def _source_cells(repo_root: Path, src_path: str) -> list[tuple[str, bool]]:
    """Return [(dedented body, raises_exception?), ...] for the Sphinx source.

    The executable cells are matched against the converted MkDocs page by body
    (see :meth:`ExecCodePlugin.on_page_markdown`), so no marker has to be
    embedded in the fence info string (which Material's highlighter cannot
    parse)."""
    src_md = repo_root / "docs" / "source" / src_path
    if not src_md.exists() or src_md.is_dir():
        return []
    text = src_md.read_text()
    out: list[tuple[str, bool]] = []
    for m in _CELL_BLOCK_RE.finditer(text):
        body_lines: list[str] = []
        raises = False
        for ln in m.group("body").splitlines():
            stripped = ln.strip()
            if stripped.startswith(":tags:"):
                if "raises-exception" in stripped:
                    raises = True
                continue
            if _DIRECTIVE_LINE_RE.match(ln):
                continue
            body_lines.append(ln)
        out.append((textwrap.dedent("\n".join(body_lines)).strip(), raises))
    return out


class ExecCodePlugin(BasePlugin):
    """Execute {code-cell}-derived python fences and embed their output.

    Cells are identified without touching the fence's language info: each page's
    python fences are matched, in order, against the ``{code-cell}`` bodies in
    the original ``docs/source/`` file (single source of truth). A fence whose
    dedented body equals the next unmatched source cell is executed in a fresh
    per-page namespace; its stdout plus the trailing expression are injected as
    plain-text <pre> blocks after the fence."""

    config_scheme = (
        # When True, a cell that raises makes the build fail (like myst-nb).
        # When False, the traceback is printed in place of the output instead.
        ("strict", config_options.Type(bool, default=True)),
        ("timeout", config_options.Type(int, default=120)),
    )

    def __init__(self) -> None:
        self._namespaces: dict[str, dict[str, Any]] = {}

    def on_page_markdown(self, markdown: str, page, config, files) -> str:
        src_path = page.file.src_path
        # Only MkDocs content pages have a corresponding Sphinx source.
        cells = _source_cells(
            Path(config["config_file_path"]).parent, src_path
        )
        if not cells:
            return markdown

        namespace = self._namespaces.get(src_path)
        if namespace is None:
            namespace = self._namespaces[src_path] = _fresh_module_namespace(
                src_path
            )

        idx = 0
        result = []
        last = 0
        for m in _FENCE_RE.finditer(markdown):
            result.append(markdown[last : m.start()])
            if idx >= len(cells):
                result.append(m.group(0))
                last = m.end()
                continue
            cell_body, raises = cells[idx]
            if (
                _is_python_fence(m.group("info"))
                and textwrap.dedent(m.group("body")).strip() == cell_body
            ):
                code, output_html = self._run_cell(
                    cell_body, namespace, src_path, expects_exception=raises
                )
                idx += 1
                # Keep the original fence byte-for-byte; append any output after it.
                if output_html:
                    indent = m.group("indent")
                    result.append(m.group(0))
                    result.append("\n" + textwrap.indent(output_html, indent))
                    last = m.end()
                else:
                    result.append(m.group(0))
                    last = m.end()
                continue
            result.append(m.group(0))
            last = m.end()
        result.append(markdown[last:])
        return "".join(result)

    def _run_cell(
        self, code, namespace, src_path, expects_exception: bool = False
    ):
        stream = io.StringIO()
        last_value: Any = None
        error: str | None = None
        try:
            with contextlib.redirect_stdout(stream):
                last_value = _execute(code, namespace)
        except BaseException:  # noqa: BLE001 - capture for rendering/failure
            error = traceback.format_exc()

        if error:
            if self.config["strict"] and not expects_exception:
                # Raising any exception aborts the build with this message.
                raise RuntimeError(
                    f"exec-codes: cell in `{src_path}` raised:\n{error}\n"
                    f"Source:\n{code}"
                )
            rendered = html.escape(error)
            output = (
                '<pre class="doc-exec-error">'
                f"<b>Exception</b>\n{rendered}</pre>"
            )
            return code, output

        parts: list[str] = []
        stdout = stream.getvalue().rstrip("\n")
        if stdout:
            parts.append(
                f'<pre class="doc-exec-stdout">{html.escape(stdout)}</pre>'
            )
        if last_value is not None:
            parts.append(_render_text(last_value))

        if not parts:
            return code, ""
        return code, "\n".join(parts)


def _fresh_module_namespace(src_path: str) -> dict[str, Any]:
    """Build an isolated, per-page cell namespace backed by a real module.

    Cells must run inside a module registered in ``sys.modules`` so that
    ``typing.get_type_hints(...)`` (used heavily by pandera's class-based
    DataFrameModel API) can resolve annotations against the module globals —
    the same reason IPython's user namespace is the ``__main__`` module. Each
    page gets its own uniquely-named module so state never leaks across pages.
    """
    digest = hashlib.sha1(src_path.encode()).hexdigest()[:10]
    name = f"_pandera_docs_cells_{digest}"
    module = types.ModuleType(name)
    ns: dict[str, Any] = module.__dict__
    sys.modules[name] = module
    ns.update(
        {
            "__name__": name,
            "__package__": None,
            "__doc__": None,
            "__builtins__": __import__("builtins"),
            "display": display,
        }
    )
    return ns


def display(obj: Any) -> Any:
    """Tiny stand-in for IPython's ``display``: mark last-value rendering later."""
    return obj


def _execute(code: str, namespace: dict[str, Any]) -> Any:
    """Execute *code* in *namespace*; return the value of the last expression."""
    tree = ast.parse(code, mode="exec")
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        # Run everything except the final standalone expression, then evaluate
        # that expression and return its value (like an IPython cell's implicit
        # trailing output).
        if len(tree.body) > 1:
            head = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(head, "<cell>", "exec"), namespace)  # noqa: S102
        expr = ast.Expression(body=last.value)
        return eval(compile(expr, "<cell>", "eval"), namespace)  # noqa: S307
    exec(compile(tree, "<cell>", "exec"), namespace)  # noqa: S102
    return None


def _render_text(value: Any) -> str:
    """Render a cell's trailing value as plain (escaped) text in a <pre>.

    Rendering is deliberately plain-text rather than raw HTML so the injected
    output can never change the surrounding Markdown structure (a DataFrame's
    ``_repr_html_`` goes into a block-level <table>, which broke adjacent
    headings)."""
    if hasattr(value, "to_string"):  # pandas DataFrame / Series
        try:
            text = value.to_string()
            return f'<pre class="doc-exec-result">{html.escape(text)}</pre>'
        except Exception:
            pass
    return f'<pre class="doc-exec-result">{html.escape(repr(value))}</pre>'

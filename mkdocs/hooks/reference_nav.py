"""MkDocs hook: put the API reference pages into the navigation.

The per-object pages in ``reference/generated/`` are linked from the section
pages (``reference/core.md`` etc.), but pages that are not in ``nav`` get no
left sidebar in Material for MkDocs. Listing ~300 objects by hand in
``mkdocs.yml`` would be unmaintainable, so this hook builds the ``Reference``
nav at config time from the section pages themselves:

* section order comes from the tables in ``reference/index.md``;
* within a section, the ``##`` headings become nav groups and each
  ``[`Name`](./generated/<fqname>.html)`` link becomes an entry, using the
  link text as the label.

Editing the section pages is therefore enough to update the sidebar.

The same applies to "hub" pages in :data:`HUB_PAGES` (the supported
libraries list, the integrations list, and the xarray / pytorch guide
indexes): each becomes a nav section whose children are the local pages it
links to, in link order, titled by their ``# heading``. Pages already
listed in ``nav`` are skipped, so each page appears once.
"""

from __future__ import annotations

import re
from pathlib import Path

REF = "reference"
_SECTION_LINK = re.compile(r"\[([^\]]+)\]\(\./([a-z_]+)\.html\)")
_OBJ_LINK = re.compile(r"\[(.+?)\]\(\./generated/([^)#]+)\.html\)")
_TITLE = re.compile(r"^# (.+)$", re.M)


def _label(text: str) -> str:
    return text.replace("`", "").strip()


def _section_nav(docs_dir: Path, name: str, seen: set[str]) -> dict:
    text = (docs_dir / REF / f"{name}.md").read_text(encoding="utf-8")
    title_match = _TITLE.search(text)
    title = title_match.group(1).strip() if title_match else name
    groups: list[tuple[str | None, list]] = [(None, [])]
    for line in text.splitlines():
        if line.startswith("## "):
            groups.append((_label(line[3:]), []))
            continue
        for label, fqname in _OBJ_LINK.findall(line):
            path = f"{REF}/generated/{fqname}.md"
            if fqname in seen or not (docs_dir / path).exists():
                continue
            seen.add(fqname)
            groups[-1][1].append({_label(label): path})
    groups = [(h, items) for h, items in groups if items]
    children: list = [{"Overview": f"{REF}/{name}.md"}]
    if len(groups) <= 1:
        children += [i for _, items in groups for i in items]
    else:
        for heading, items in groups:
            if heading is None:
                children += items
            else:
                children.append({heading: items})
    return {title: children}


def _build(docs_dir: Path) -> list:
    index = (docs_dir / REF / "index.md").read_text(encoding="utf-8")
    seen: set[str] = set()
    sections: list = [{"Overview": f"{REF}/index.md"}]
    names: list[str] = []
    for _, name in _SECTION_LINK.findall(index):
        if name not in names and (docs_dir / REF / f"{name}.md").exists():
            names.append(name)
    for name in names:
        sections.append(_section_nav(docs_dir, name, seen))
    return sections


HUB_PAGES = (
    "supported_libraries.md",
    "integrations.md",
    "xarray_guide/index.md",
    "pytorch_guide/index.md",
)
# "Data Validation with Polars" -> "Polars" in the sidebar (page H1 unchanged)
_NAV_PREFIX = re.compile(r"^Data Validation with\s+", re.I)
_LOCAL_LINK = re.compile(r"\]\((?:\./)?([\w/.-]+?)\.(?:html|md)(?:#[^)]*)?\)")


def _title(docs_dir: Path, page: str) -> str:
    text = (docs_dir / page).read_text(encoding="utf-8")
    match = _TITLE.search(text)
    return match.group(1).strip() if match else page


def _nav_pages(nav: list) -> set[str]:
    pages: set[str] = set()
    for item in nav:
        values = item.values() if isinstance(item, dict) else [item]
        for value in values:
            if isinstance(value, list):
                pages |= _nav_pages(value)
            elif isinstance(value, str):
                pages.add(value)
    return pages


def _hub_nav(docs_dir: Path, hub: str, claimed: set[str]) -> dict:
    base = Path(hub).parent
    children: list = [{"Overview": hub}]
    text = (docs_dir / hub).read_text(encoding="utf-8")
    for target in _LOCAL_LINK.findall(text):
        page = (base / f"{target}.md").as_posix()
        if ".." in page or page in claimed or not (docs_dir / page).exists():
            continue
        claimed.add(page)
        label = _NAV_PREFIX.sub("", _title(docs_dir, page))
        children.append({label: page})
    return {_title(docs_dir, hub): children}


def _expand_hubs(nav: list, docs_dir: Path, claimed: set[str]) -> None:
    for i, item in enumerate(nav):
        if isinstance(item, str) and item in HUB_PAGES:
            nav[i] = _hub_nav(docs_dir, item, claimed)
        elif isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str) and value in HUB_PAGES:
                    hub = _hub_nav(docs_dir, value, claimed)
                    item[key] = next(iter(hub.values()))
                elif isinstance(value, list):
                    _expand_hubs(value, docs_dir, claimed)


def _replace(nav: list, docs_dir: Path) -> bool:
    for i, item in enumerate(nav):
        if item == f"{REF}/index.md":
            nav[i] = {"API reference": _build(docs_dir)}
            return True
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, list) and _replace(value, docs_dir):
                    return True
    return False


def on_config(config, **kwargs):
    if config.nav:
        docs_dir = Path(config.docs_dir)
        _replace(config.nav, docs_dir)
        _expand_hubs(config.nav, docs_dir, _nav_pages(config.nav))
    return config

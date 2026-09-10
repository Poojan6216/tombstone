"""9.6: the README diagram — the lineage graph and the four verification levels, as one SVG.

Generated, not drawn: the artifact kinds come from ``ArtifactKind`` and the levels from
``VerifyLevel``, and the build fails if either enum gains a member the diagram does not show, so
the picture cannot quietly disagree with the code it explains. The edge tags are the ``via``
values from docs/lineage-model.md.

    uv run python bench/plots/make_diagram.py      # writes docs/lineage.svg
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from tombstone.model.artifacts import ArtifactKind
from tombstone.model.status import VerifyLevel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "lineage.svg"

W, H = 1040, 644
BG = "#fbfbfa"
INK = "#1b1f24"
MUTED = "#6b737c"
LINE = "#b9c0c6"
ACCENT = "#2f6f4f"
WARN = "#8a5a2b"
NW, NH = 140, 48

NODES: dict[str, tuple[int, int, str, str]] = {
    "SOURCE": (40, 225, "SOURCE", "the document"),
    "CHUNK": (250, 225, "CHUNK", "unit of retrieval"),
    "EMBED": (470, 135, "EMBED", "one per index"),
    "CACHE": (470, 225, "CACHE", "exact + semantic"),
    "TRAIN": (470, 315, "TRAIN", "example in a shard"),
    "ADAPTER": (690, 315, "ADAPTER", "shard weights"),
    "MEMORY": (250, 390, "MEMORY", "agent memory"),
}
EDGES: list[tuple[str, str, str]] = [
    ("SOURCE", "CHUNK", "chunk"),
    ("CHUNK", "EMBED", "embed:<model>"),
    ("CHUNK", "CACHE", "cache:*"),
    ("CHUNK", "TRAIN", "train:shard-N"),
    ("TRAIN", "ADAPTER", "adapter:<key>"),
    ("SOURCE", "MEMORY", "memory"),
]
LEVELS: dict[str, tuple[str, str]] = {
    "LOGICAL": ("no query path returns it", "the bytes may still be on disk"),
    "PHYSICAL": (
        "its bytes are not in the store's files",
        "only where the store lets us read them",
    ),
    "SEMANTIC": ("the shape of the hole it left", "reported, never a pass/fail gate"),
    "MODEL": (
        "it is not extractable from the weights",
        "an attacker with more budget may differ",
    ),
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _lines(text: str, width: int, x: int, y: int, dy: int, size: float, fill: str) -> str:
    """``text`` wrapped on word boundaries, one <text> per line."""
    return "".join(
        f'<text x="{x}" y="{y + i * dy}" font-size="{size}" fill="{fill}">{_esc(line)}</text>'
        for i, line in enumerate(textwrap.wrap(text, width) or [""])
    )


def _node(nid: str) -> str:
    x, y, label, sub = NODES[nid]
    return (
        f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="7" fill="#ffffff" '
        f'stroke="{INK}" stroke-width="1.4"/>'
        f'<text x="{x + NW // 2}" y="{y + 21}" text-anchor="middle" font-size="13.5" '
        f'font-weight="700" fill="{INK}">{_esc(label)}</text>'
        f'<text x="{x + NW // 2}" y="{y + 36}" text-anchor="middle" font-size="10" '
        f'fill="{MUTED}">{_esc(sub)}</text>'
    )


def _edge(a: str, b: str, via: str) -> str:
    ax, ay, _, _ = NODES[a]
    bx, by, _, _ = NODES[b]
    x1, y1 = ax + NW, ay + NH // 2
    x2, y2 = bx, by + NH // 2
    mx = (x1 + x2) / 2
    path = f"M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2 - 8} {y2}"
    # a white halo under the label keeps it readable where it crosses a curve
    return (
        f'<path d="{path}" fill="none" stroke="{LINE}" stroke-width="1.5" '
        f'marker-end="url(#arrow)"/>'
        f'<text x="{mx}" y="{(y1 + y2) / 2 - 6}" text-anchor="middle" font-size="9.5" '
        f'fill="{MUTED}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'stroke="{BG}" stroke-width="3.5" paint-order="stroke">{_esc(via)}</text>'
    )


def build() -> str:
    for enum, shown, what in (
        (ArtifactKind, set(NODES), "ArtifactKind"),
        (VerifyLevel, set(LEVELS), "VerifyLevel"),
    ):
        missing = {m.name for m in enum} - shown
        if missing:
            raise SystemExit(f"{what} gained {sorted(missing)}; add it to the diagram")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>',
        f'<text x="40" y="46" font-size="19" font-weight="700" fill="{INK}">'
        "Erasure is reachability, not search</text>",
        f'<text x="40" y="68" font-size="12" fill="{MUTED}">'
        "One subject&#8217;s artifacts, and the strongest layer each store lets us check.</text>",
        f'<text x="40" y="112" font-size="10.5" font-weight="700" fill="{MUTED}" '
        'letter-spacing="0.09em">THE GRAPH</text>',
    ]
    parts += [_edge(a, b, via) for a, b, via in EDGES]
    parts += [_node(n) for n in NODES]

    y0, cw, gap = 500, 228, 16
    parts.append(
        f'<text x="40" y="{y0 - 18}" font-size="10.5" font-weight="700" fill="{MUTED}" '
        'letter-spacing="0.09em">WHAT A RECEIPT CAN CLAIM</text>'
    )
    for i, (name, (proves, limit)) in enumerate(LEVELS.items()):
        x = 40 + i * (cw + gap)
        parts.append(
            f'<rect x="{x}" y="{y0}" width="{cw}" height="86" rx="7" fill="#ffffff" '
            f'stroke="{LINE}" stroke-width="1.2"/>'
            f'<text x="{x + 14}" y="{y0 + 22}" font-size="12" font-weight="700" '
            f'fill="{ACCENT}">{_esc(name)}</text>'
            + _lines(f"proves: {proves}", 34, x + 14, y0 + 40, 12, 9.8, INK)
            + _lines(f"but: {limit}", 36, x + 14, y0 + 70, 11, 9.2, WARN)
        )
    parts.append(
        f'<text x="40" y="{H - 22}" font-size="10.5" fill="{MUTED}">'
        "A store that cannot be checked at a level reports UNVERIFIED with the reason "
        "&#8212; never a pass.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

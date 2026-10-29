"""Move link annotations onto the translated text that replaced their source.

A link is an annotation: a rectangle stored apart from the text stream. The
converter replaces glyphs and reflows them, but never touches ``/Annots``, so
every link kept the rectangle of the source words and pointed beside the
translation. This module maps each link from its source text block to the
translated block that replaced it, keeping the anchor's relative position.

The two sources disagree about coordinates on a rotated page: text extraction
returns page coordinates and ``get_links`` returns viewer coordinates, so a
link rectangle is derotated before mapping and rotated back before it is
installed again. A page with no text on either side is left alone.
"""

from __future__ import annotations

import logging

import pymupdf

logger = logging.getLogger(__name__)

Rect = pymupdf.Rect


def text_blocks(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    """The page's text block rectangles, in reading order."""
    blocks: list[tuple[float, float, float, float]] = []
    for block in page.get_text("blocks", sort=True):
        if len(block) < 7 or block[6] != 0:
            continue
        if not str(block[4]).strip():
            continue
        blocks.append(tuple(float(value) for value in block[:4]))
    return blocks


def _intersection(left, right) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    return width * height if width > 0 and height > 0 else 0.0


def _area(rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _union(left, right) -> float:
    return _area(left) + _area(right) - _intersection(left, right)


def _distance(left, right) -> float:
    return (
        ((left[0] + left[2]) - (right[0] + right[2])) ** 2
        + ((left[1] + left[3]) - (right[1] + right[3])) ** 2
    ) ** 0.5


def _source_block(anchor, blocks):
    """The source block the link sits on, by greatest overlap."""
    best, best_overlap = None, 0.0
    for block in blocks:
        overlap = _intersection(anchor, block)
        if overlap > best_overlap:
            best, best_overlap = block, overlap
    return best


def _output_block(source, blocks):
    """The translated block that replaced ``source``, by overlap or proximity."""
    best, best_score = None, 0.0
    for block in blocks:
        union = _union(source, block)
        if union <= 0:
            continue
        score = _intersection(source, block) / union
        if score > best_score:
            best, best_score = block, score
    if best is not None:
        return best
    # A paragraph that moved too far to overlap still came from the nearest one.
    return min(blocks, key=lambda block: _distance(source, block), default=None)


def _remap(anchor, source, output):
    """Place ``anchor`` at the same relative spot inside the translated block."""
    sx0, sy0, sx1, sy1 = source
    ox0, oy0, ox1, oy1 = output
    width, height = sx1 - sx0, sy1 - sy0
    if width <= 0.01 or height <= 0.01:
        return None

    def x(value: float) -> float:
        return ox0 + (value - sx0) * (ox1 - ox0) / width

    def y(value: float) -> float:
        return oy0 + (value - sy0) * (oy1 - oy0) / height

    moved = Rect(x(anchor.x0), y(anchor.y0), x(anchor.x1), y(anchor.y1))
    if moved.width < 0.5 or moved.height < 0.5:
        return None
    return moved


def _definition(link: dict, rect: Rect) -> dict | None:
    """The link as MuPDF can rebuild it, with ``rect`` in viewer coordinates."""
    kind = link.get("kind")
    if kind == pymupdf.LINK_URI and link.get("uri"):
        return {"kind": kind, "from": rect, "uri": link["uri"]}
    if kind in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED) and link.get("page", -1) >= 0:
        return {
            "kind": pymupdf.LINK_GOTO,
            "from": rect,
            "page": link["page"],
            "to": pymupdf.Point(link.get("to") or (0, 0)),
            "zoom": link.get("zoom") or 0,
        }
    if kind == pymupdf.LINK_LAUNCH and link.get("file"):
        return {"kind": kind, "from": rect, "file": link["file"]}
    return None


def relocate_links(
    page: pymupdf.Page,
    source_blocks: list[tuple[float, float, float, float]],
    output_blocks: list[tuple[float, float, float, float]],
) -> int:
    """Move this page's links from their source text to the translated text."""
    links = page.get_links()
    if not links or not source_blocks or not output_blocks:
        return 0
    moved = 0
    for link in links:
        # get_links reports viewer coordinates; insert_link takes page ones.
        anchor = Rect(link["from"]) * page.derotation_matrix
        original = _definition(link, anchor)
        if original is None:
            continue
        source = _source_block(anchor, source_blocks)
        if source is None:
            continue
        output = _output_block(source, output_blocks)
        if output is None:
            continue
        mapped = _remap(anchor, source, output)
        if mapped is None:
            continue
        if mapped == anchor:
            continue
        updated = _definition(link, mapped)
        if updated is None:
            continue
        page.delete_link(link)
        try:
            page.insert_link(updated)
        except Exception as error:  # noqa: BLE001 - a link is never worth a page
            logger.debug("Could not move a link on page %s: %s", page.number + 1, error)
            try:
                page.insert_link(original)
            except Exception:  # noqa: BLE001
                pass
            continue
        moved += 1
    return moved

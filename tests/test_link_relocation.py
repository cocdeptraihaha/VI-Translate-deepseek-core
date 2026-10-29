from __future__ import annotations

import unittest

import pymupdf

from pdf2zh.links import relocate_links, text_blocks


def reopened(document: pymupdf.Document) -> pymupdf.Document:
    """Annotations added to a fresh page only appear after a round trip."""
    data = document.tobytes()
    document.close()
    return pymupdf.open("pdf", data)


def source_page(text: str, position: tuple[float, float]) -> pymupdf.Document:
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.insert_text(position, text, fontsize=12)
    return document


def output_page_with_link(
    text: str,
    position: tuple[float, float],
    anchor: pymupdf.Rect,
    rotation: int = 0,
    link: dict | None = None,
) -> pymupdf.Document:
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.insert_text(position, text, fontsize=12)
    page.insert_link(
        link
        or {"kind": pymupdf.LINK_URI, "from": anchor, "uri": "https://example.com"}
    )
    if rotation:
        page.set_rotation(rotation)
    return reopened(document)


class LinkRelocationTests(unittest.TestCase):
    def test_a_link_follows_the_translated_block(self):
        source = source_page("Click this link to continue", (72, 100))
        source_blocks = text_blocks(source[0])
        source.close()

        anchor = pymupdf.Rect(72, 88, 170, 104)
        output = output_page_with_link(
            "Nhan vao day de tiep tuc", (72, 160), anchor
        )
        page = output[0]
        output_blocks = text_blocks(page)
        self.assertEqual(relocate_links(page, source_blocks, output_blocks), 1)

        # MuPDF only re-reads the annotation list after the file is saved.
        output = reopened(output)
        page = output[0]
        moved = pymupdf.Rect(page.get_links()[0]["from"])
        self.assertEqual(page.get_links()[0]["uri"], "https://example.com")
        self.assertNotEqual(moved, anchor)
        block = pymupdf.Rect(*output_blocks[0])
        self.assertGreater(moved.x0, block.x0 - 1)
        self.assertLess(moved.x1, block.x1 + 1)
        self.assertGreater(moved.y0, block.y0 - 1)
        self.assertLess(moved.y1, block.y1 + 1)
        output.close()

    def test_the_rotation_frame_is_respected(self):
        source = source_page("Rotated source words", (50, 100))
        source[0].set_rotation(90)
        source = reopened(source)
        source_blocks = text_blocks(source[0])
        source.close()

        anchor = pymupdf.Rect(50, 88, 160, 104)
        output = output_page_with_link(
            "Nhan vao day", (50, 160), anchor, rotation=90
        )
        page = output[0]
        output_blocks = text_blocks(page)
        self.assertEqual(relocate_links(page, source_blocks, output_blocks), 1)

        output = reopened(output)
        page = output[0]
        shown = pymupdf.Rect(page.get_links()[0]["from"]) * page.derotation_matrix
        block = pymupdf.Rect(*output_blocks[0])
        self.assertGreater(shown.y0, block.y0 - 1)
        self.assertLess(shown.y1, block.y1 + 1)
        output.close()

    def test_an_internal_link_keeps_its_target(self):
        source = source_page("Contents", (72, 100))
        source_blocks = text_blocks(source[0])
        source.close()

        document = pymupdf.open()
        document.new_page(width=400, height=300)
        page = document.new_page(width=400, height=300)
        page.insert_text((72, 160), "Mucluc", fontsize=12)
        page.insert_link({
            "kind": pymupdf.LINK_GOTO,
            "from": pymupdf.Rect(72, 88, 140, 104),
            "page": 0,
            "to": pymupdf.Point(72, 100),
        })
        output = reopened(document)
        self.assertEqual(relocate_links(output[1], source_blocks, text_blocks(output[1])), 1)

        output = reopened(output)
        link = output[1].get_links()[0]
        self.assertEqual(link["kind"], pymupdf.LINK_GOTO)
        self.assertEqual(link["page"], 0)
        output.close()

    def test_the_probe_copy_supplies_the_translated_blocks(self):
        """The core reads source blocks before patching, then a throwaway copy
        after it; this is that order end to end."""
        document = pymupdf.open()
        page = document.new_page(width=400, height=300)
        page.insert_text((72, 100), "Click this link to continue", fontsize=12)
        page.insert_link({
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(72, 88, 170, 104),
            "uri": "https://example.com",
        })
        work = reopened(document)
        source_blocks = text_blocks(work[0])

        xref = work[0].get_contents()[0]
        work.update_stream(
            xref, b"BT /helv 12 Tf 72 140 Td (Nhan vao day de tiep tuc) Tj ET"
        )
        probe = pymupdf.open("pdf", work.tobytes())
        output_blocks = text_blocks(probe[0])
        probe.close()

        self.assertEqual(relocate_links(work[0], source_blocks, output_blocks), 1)
        work = reopened(work)
        moved = pymupdf.Rect(work[0].get_links()[0]["from"])
        block = pymupdf.Rect(*output_blocks[0])
        self.assertGreater(moved.y0, block.y0 - 1)
        self.assertLess(moved.y1, block.y1 + 1)
        work.close()

    def test_a_page_without_text_keeps_its_links(self):
        document = pymupdf.open()
        page = document.new_page(width=400, height=300)
        anchor = pymupdf.Rect(72, 88, 170, 104)
        page.insert_link({"kind": pymupdf.LINK_URI, "from": anchor, "uri": "https://example.com"})
        output = reopened(document)
        page = output[0]
        self.assertEqual(relocate_links(page, [(10, 10, 200, 40)], []), 0)
        self.assertEqual(pymupdf.Rect(page.get_links()[0]["from"]), anchor)
        output.close()


if __name__ == "__main__":
    unittest.main()

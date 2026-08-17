"""A tiny, dependency-free, byte-deterministic PNG canvas.

Nova's voice environment carries no plotting library, and adding one to draw a
handful of bars would pull a large dependency tree onto the host for the sake of
one evidence artifact. This draws what the benchmark graph needs — filled
rectangles, lines, and text — straight into an RGB buffer and encodes it by
hand.

Deterministic by construction, which is the point rather than a nicety: the same
measurements must produce a byte-identical file, so a graph that changes in
review means the numbers changed and not the renderer. Nothing here consults the
clock, a hash seed, or a font on disk. zlib is used at a fixed compression level
and PNG carries no timestamp chunk unless one is written, and none is.
"""

from __future__ import annotations

import struct
import zlib

Colour = tuple[int, int, int]

# A 5x7 bitmap font, one string of five bits per row. Deliberately hand-written
# and embedded: any system font would make output depend on the host.
_GLYPHS: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10011", "01111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "01100", "01100", "00100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "@": ("01110", "10001", "10111", "10101", "10111", "10000", "01110"),
    "#": ("01010", "11111", "01010", "01010", "01010", "11111", "01010"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "'": ("00100", "00100", "00000", "00000", "00000", "00000", "00000"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
}
GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7


class Canvas:
    """An RGB pixel buffer that can encode itself as a PNG."""

    def __init__(self, width: int, height: int, background: Colour = (255, 255, 255)):
        if width <= 0 or height <= 0:
            raise ValueError("canvas dimensions must be positive")
        self.width = width
        self.height = height
        self._pixels = bytearray(bytes(background) * (width * height))

    def _set(self, x: int, y: int, colour: Colour) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self._pixels[offset : offset + 3] = bytes(colour)

    def rect(self, x: int, y: int, width: int, height: int, colour: Colour) -> None:
        """Filled rectangle. Negative extents are clamped rather than wrapped."""

        for row in range(max(0, y), min(self.height, y + max(0, height))):
            start = (row * self.width + max(0, x)) * 3
            span = min(self.width, x + max(0, width)) - max(0, x)
            if span > 0:
                self._pixels[start : start + span * 3] = bytes(colour) * span

    def hline(self, x: int, y: int, length: int, colour: Colour) -> None:
        self.rect(x, y, length, 1, colour)

    def vline(self, x: int, y: int, length: int, colour: Colour) -> None:
        self.rect(x, y, 1, length, colour)

    def text(
        self,
        x: int,
        y: int,
        message: str,
        colour: Colour = (0, 0, 0),
        scale: int = 1,
    ) -> int:
        """Draw text and return the x position just past it.

        Unknown characters are drawn as a space rather than raising: a label is
        not worth failing a benchmark report over.
        """

        cursor = x
        for character in message:
            glyph = _GLYPHS.get(character) or _GLYPHS.get(character.upper())
            if glyph is not None:
                for row, bits in enumerate(glyph):
                    for column, bit in enumerate(bits):
                        if bit == "1":
                            self.rect(
                                cursor + column * scale,
                                y + row * scale,
                                scale,
                                scale,
                                colour,
                            )
            cursor += (GLYPH_WIDTH + 1) * scale
        return cursor

    @staticmethod
    def text_width(message: str, scale: int = 1) -> int:
        return len(message) * (GLYPH_WIDTH + 1) * scale

    def to_png(self) -> bytes:
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            # Filter type 0 (None). Deterministic and cheap; the images are
            # small enough that a smarter filter buys nothing.
            raw.append(0)
            raw += self._pixels[row * stride : (row + 1) * stride]

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        header = struct.pack(">2I5B", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            # Fixed level, so the same pixels always give the same bytes.
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b"")
        )

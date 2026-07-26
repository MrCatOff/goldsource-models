"""
``--max-texture-size`` shrinks oversized textures to make the merged .mdl
smaller.  GoldSource stores textures 8-bit palettized, so pixel count *is* the
file; UVs are normalised, so lowering resolution does not move the mapping.
"""

from io import BytesIO

import pytest
from PIL import Image

from goldsource.merger import MergeResult, MergeReport
from goldsource.pipeline import _downscale_textures
from goldsource.qc import QC, TextureRenderMode


def _bmp(w: int, h: int) -> bytes:
    """An 8-bit palettized BMP with a real colour palette and varied pixels."""
    img = Image.new("P", (w, h))
    # A 256-entry colour ramp (not greyscale, so it round-trips as mode 'P').
    img.putpalette([c for i in range(256) for c in (i, (i * 5) % 256, 255 - i)])
    img.putdata(bytes((x + y) % 256 for y in range(h) for x in range(w)))
    out = BytesIO()
    img.save(out, format="BMP")
    return out.getvalue()


def _result(textures, texturemodes=None) -> MergeResult:
    qc = QC(modelname="m.mdl")
    qc.texturemodes = texturemodes or []
    empty_report = MergeReport(
        bone_stats=[], total_unique_bones=0, bone_limit=127,
        conflicts=[], exceeds_limit=False, removal_suggestions=[], warnings=[],
    )
    return MergeResult(
        qc=qc, smds={}, textures=dict(textures),
        renamed_bones={}, renamed_textures={}, report=empty_report,
    )


def _dims(data: bytes) -> tuple[int, int]:
    return Image.open(BytesIO(data)).size


def test_oversized_texture_is_shrunk_to_cap():
    result = _result({"skin.bmp": _bmp(512, 512)})
    resized, before, after = _downscale_textures(result, 256)

    assert resized == 1
    w, h = _dims(result.textures["skin.bmp"])
    assert max(w, h) <= 256
    assert after < before


def test_small_texture_is_left_untouched():
    original = _bmp(128, 128)
    result = _result({"small.bmp": original})
    resized, _, _ = _downscale_textures(result, 256)

    assert resized == 0
    assert result.textures["small.bmp"] == original


def test_masked_textures_are_never_requantised():
    """A masked skin's transparency rides on an exact palette index."""
    brace = _bmp(512, 512)            # '{' name => masked by convention
    flagged = _bmp(512, 512)          # masked via $texrendermode
    result = _result(
        {"{glass.bmp": brace, "grate.bmp": flagged},
        texturemodes=[TextureRenderMode(texture="grate.bmp", mode="masked")],
    )
    resized, _, _ = _downscale_textures(result, 256)

    assert resized == 0
    assert result.textures["{glass.bmp"] == brace
    assert result.textures["grate.bmp"] == flagged


def test_output_stays_8bpp_bmp():
    result = _result({"skin.bmp": _bmp(512, 512)})
    _downscale_textures(result, 256)
    img = Image.open(BytesIO(result.textures["skin.bmp"]))
    assert img.mode in ("P", "L")             # still 8-bit palettized/greyscale
    assert result.textures["skin.bmp"][:2] == b"BM"


def test_dimensions_snap_to_multiple_of_16():
    result = _result({"skin.bmp": _bmp(512, 512)})
    _downscale_textures(result, 200)
    w, h = _dims(result.textures["skin.bmp"])
    assert w % 16 == 0 and h % 16 == 0
    assert max(w, h) <= 200

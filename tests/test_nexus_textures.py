"""
Regression: some CSO/nexus decompiles (e.g. v_infinityex2 in the nexus_pistols
set) write texture files with **no extension** and reference them that way in
the SMD material field.  The Sven Co-op studiomdl selects the image format by
extension and rejects them with "unknown graphics type", so ``from_directory``
must load such a file, give it a ``.bmp`` name, and rewrite the mesh materials
that referenced it.
"""

from goldsource.merger import ModelInput
from goldsource.smd import SMD, Node, SkeletonFrame, BoneTransform, Triangle, Vertex


def _minimal_bmp() -> bytes:
    """A tiny but header-valid 1x1 8bpp BMP (only the ``BM`` magic is checked)."""
    # 14-byte file header + 40-byte info header is plenty; content is irrelevant.
    return b"BM" + b"\x00" * 52


def _write_model(directory, material: str):
    node = Node(id=0, name="root", parent_id=-1)
    frame = SkeletonFrame(time=0, bones=[BoneTransform(0, 0, 0, 0, 0, 0, 0)])
    v = Vertex(0, 0, 0, 0, 0, 0, 1, 0, 0)
    smd = SMD(version=1, nodes=[node], skeleton=[frame],
              triangles=[Triangle(material=material, v0=v, v1=v, v2=v)])
    (directory / "weapon.smd").write_text(smd.to_string(), encoding="utf-8")
    (directory / "v_test.qc").write_text(
        '$modelname "v_test.mdl"\n$cd "."\n$cdtexture "."\n'
        '$bodygroup "weapon"\n{\n\tstudio "weapon"\n}\n',
        encoding="utf-8",
    )


def test_extensionless_texture_is_renamed_and_material_rewritten(tmp_path):
    _write_model(tmp_path, material="dualinfinity_2_02")
    # The texture on disk has NO extension, exactly as nexus decompiles it.
    (tmp_path / "dualinfinity_2_02").write_bytes(_minimal_bmp())

    model = ModelInput.from_directory("v_test", tmp_path)

    # Loaded under a .bmp name so studiomdl accepts it...
    assert "dualinfinity_2_02.bmp" in model.textures
    assert "dualinfinity_2_02" not in model.textures
    # ...and every material that named it was rewritten to match.
    materials = {t.material for smd in model.smds.values() for t in smd.triangles}
    assert materials == {"dualinfinity_2_02.bmp"}


def test_non_bmp_extensionless_file_is_ignored(tmp_path):
    """A referenced extensionless file that is not a BMP must be left alone."""
    _write_model(tmp_path, material="notabmp")
    (tmp_path / "notabmp").write_bytes(b"not an image")

    model = ModelInput.from_directory("v_test", tmp_path)

    assert "notabmp.bmp" not in model.textures
    materials = {t.material for smd in model.smds.values() for t in smd.triangles}
    assert materials == {"notabmp"}  # unchanged


def test_unreferenced_extensionless_bmp_is_not_slurped(tmp_path):
    """Only files a mesh actually names are loaded — not stray BMP-ish files."""
    _write_model(tmp_path, material="used.bmp")
    (tmp_path / "used.bmp").write_bytes(_minimal_bmp())
    (tmp_path / "stray").write_bytes(_minimal_bmp())  # valid BMP, never referenced

    model = ModelInput.from_directory("v_test", tmp_path)

    assert "stray.bmp" not in model.textures
    assert "stray" not in model.textures

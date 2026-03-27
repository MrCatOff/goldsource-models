# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt6.QtOpenGL',
        'PyQt6.QtOpenGLWidgets',
        'OpenGL.GL',
        'OpenGL.GLU',
        'OpenGL.arrays.vbo',
        'OpenGL.platform.win32',
        'OpenGL.platform.darwin',
        'OpenGL.platform.glx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── macOS: one-dir mode → proper .app bundle ────────────────────────────────
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='GoldSourceMerger',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon='assets/512x512.icns',
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='GoldSourceMerger',
    )

    app = BUNDLE(
        coll,
        name='GoldSourceMerger.app',
        icon='assets/512x512.icns',
        bundle_identifier='com.goldsource.merger',
        info_plist={
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
            'LSUIElement': False,
        },
    )

# ── Windows / Linux: single-file executable ──────────────────────────────────
else:
    icon = 'assets/512x512.ico' if sys.platform == 'win32' else 'assets/512x512.png'

    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='GoldSourceMerger',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        icon=icon,
    )

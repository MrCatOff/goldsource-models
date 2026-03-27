"""
Application configuration: save/load the full merger session as JSON.

Stores model paths, per-model skin variants, global skin slots, sequence
rename rules, and output settings.  Use ``AppConfig.build_merge_config()``
to convert into a :class:`~goldsource.merger.MergeConfig` ready for the
merger (replacement texture bytes are read from disk at that point).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from goldsource.merger import MergeConfig, SkinSlot, SkinVariant, TextureReplacement


# ---------------------------------------------------------------------------
# Spec dataclasses (JSON-serialisable, path-based)
# ---------------------------------------------------------------------------

@dataclass
class TextureReplacementSpec:
    """One texture swap entry within a skin variant."""
    original: str       # texture filename as used in the SMD (e.g. "hand.bmp")
    replacement: str    # output filename for the replacement (e.g. "hand_blood.bmp")
    source_path: str    # absolute path to the BMP/TGA file on disk


@dataclass
class SkinVariantSpec:
    """A named variant for one model (e.g. "blood")."""
    name: str
    replacements: list[TextureReplacementSpec] = field(default_factory=list)


@dataclass
class SkinSlotSpec:
    """One global skin slot: maps model_name → variant_name."""
    name: str
    assignments: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelEntry:
    """One model in the session."""
    name: str
    directory: str
    skin_variants: list[SkinVariantSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Complete application session state."""
    models: list[ModelEntry] = field(default_factory=list)
    skin_slots: list[SkinSlotSpec] = field(default_factory=list)
    # Each element is [find, replace_with]
    sequence_renames: list[list[str]] = field(default_factory=list)
    output_model_name: str = ""
    output_directory: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "models": [
                {
                    "name": m.name,
                    "directory": m.directory,
                    "skin_variants": [
                        {
                            "name": v.name,
                            "replacements": [
                                {
                                    "original": r.original,
                                    "replacement": r.replacement,
                                    "source_path": r.source_path,
                                }
                                for r in v.replacements
                            ],
                        }
                        for v in m.skin_variants
                    ],
                }
                for m in self.models
            ],
            "skin_slots": [
                {
                    "name": s.name,
                    "assignments": s.assignments,
                }
                for s in self.skin_slots
            ],
            "sequence_renames": self.sequence_renames,
            "output_model_name": self.output_model_name,
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AppConfig:
        models = []
        for m in d.get("models", []):
            variants = []
            for v in m.get("skin_variants", []):
                reps = [
                    TextureReplacementSpec(
                        original=r["original"],
                        replacement=r["replacement"],
                        source_path=r.get("source_path", ""),
                    )
                    for r in v.get("replacements", [])
                ]
                variants.append(SkinVariantSpec(name=v["name"], replacements=reps))
            models.append(
                ModelEntry(
                    name=m["name"],
                    directory=m["directory"],
                    skin_variants=variants,
                )
            )

        slots = []
        for s in d.get("skin_slots", []):
            slots.append(
                SkinSlotSpec(
                    name=s["name"],
                    assignments=dict(s.get("assignments", {})),
                )
            )

        return cls(
            models=models,
            skin_slots=slots,
            sequence_renames=d.get("sequence_renames", []),
            output_model_name=d.get("output_model_name", ""),
            output_directory=d.get("output_directory", ""),
        )

    def save(self, path: str | Path) -> None:
        """Write the config to a JSON file."""
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        """Load a config from a JSON file."""
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Build merge config (reads replacement texture bytes from disk)
    # ------------------------------------------------------------------

    def build_merge_config(self) -> MergeConfig:
        """
        Convert this config to a :class:`MergeConfig` suitable for
        ``ModelMerger.merge(config=...)``.

        Replacement texture files are read from disk here.  Missing files
        are silently skipped (the replacement will be omitted from the
        texturegroup row).
        """
        skin_variants: dict[str, list[SkinVariant]] = {}

        for model_entry in self.models:
            variants: list[SkinVariant] = []
            for spec in model_entry.skin_variants:
                replacements: list[TextureReplacement] = []
                replacement_data: dict[str, bytes] = {}
                for rep in spec.replacements:
                    src = Path(rep.source_path)
                    if src.exists():
                        replacement_data[rep.replacement] = src.read_bytes()
                        replacements.append(
                            TextureReplacement(
                                original=rep.original,
                                replacement=rep.replacement,
                            )
                        )
                variants.append(
                    SkinVariant(
                        name=spec.name,
                        model_name=model_entry.name,
                        replacements=replacements,
                        replacement_data=replacement_data,
                    )
                )
            skin_variants[model_entry.name] = variants

        skin_slots = [
            SkinSlot(name=s.name, assignments=dict(s.assignments))
            for s in self.skin_slots
        ]

        seq_renames = [
            (pair[0], pair[1]) for pair in self.sequence_renames if len(pair) == 2
        ]

        return MergeConfig(
            sequence_renames=seq_renames,
            skin_variants=skin_variants,
            skin_slots=skin_slots,
        )

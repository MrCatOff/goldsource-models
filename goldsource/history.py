"""
Change history for the GoldSource Model Merger.

Stores per-operation SMD snapshots (gzip + base64) so the user can revert
in-memory model state to any recorded step.

History is persisted alongside the config file as:
  {config_stem}_history.json
"""

from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from goldsource.smd import SMD


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    step_id:     int
    timestamp:   str        # ISO-8601
    description: str
    op_type:     str        # "rename" | "delete" | "apply_all" | "apply_hands"
                            # | "save_all" | "qc_edit"
    model_name:  str
    # smd_key → gzip(smd.to_string()), base64-encoded
    snapshots:   dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class HistoryManager:
    """Append-only history with snapshot-based revert."""

    MAX_ENTRIES = 200

    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []
        self._next_id: int = 0

    # ── Public API ──────────────────────────────────────────────────────

    def record(
        self,
        description: str,
        op_type: str,
        model_name: str,
        smds: dict[str, SMD],
    ) -> HistoryEntry:
        """
        Record an operation and snapshot all SMDs in *smds*.
        Returns the new entry.
        """
        snapshots: dict[str, str] = {}
        for key, smd in smds.items():
            text       = smd.to_string()
            compressed = gzip.compress(text.encode("utf-8"), compresslevel=6)
            snapshots[key] = base64.b64encode(compressed).decode("ascii")

        entry = HistoryEntry(
            step_id=self._next_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            description=description,
            op_type=op_type,
            model_name=model_name,
            snapshots=snapshots,
        )
        self._entries.append(entry)
        self._next_id += 1

        # Trim oldest entries when over the limit
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES :]

        return entry

    def get_entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def restore(self, step_id: int) -> dict[str, SMD]:
        """
        Return the SMD state that was captured at *step_id*.
        Raises KeyError when *step_id* is not found.
        """
        entry = next((e for e in self._entries if e.step_id == step_id), None)
        if entry is None:
            raise KeyError(f"History step {step_id} not found.")
        result: dict[str, SMD] = {}
        for key, b64 in entry.snapshots.items():
            compressed = base64.b64decode(b64.encode("ascii"))
            text       = gzip.decompress(compressed).decode("utf-8")
            result[key] = SMD.from_string(text)
        return result

    def clear(self) -> None:
        self._entries.clear()
        self._next_id = 0

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "version":  1,
            "next_id":  self._next_id,
            "entries": [
                {
                    "step_id":     e.step_id,
                    "timestamp":   e.timestamp,
                    "description": e.description,
                    "op_type":     e.op_type,
                    "model_name":  e.model_name,
                    "snapshots":   e.snapshots,
                }
                for e in self._entries
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        raw  = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        self._next_id = data.get("next_id", 0)
        self._entries = [
            HistoryEntry(
                step_id=e["step_id"],
                timestamp=e["timestamp"],
                description=e["description"],
                op_type=e["op_type"],
                model_name=e["model_name"],
                snapshots=e.get("snapshots", {}),
            )
            for e in data.get("entries", [])
        ]

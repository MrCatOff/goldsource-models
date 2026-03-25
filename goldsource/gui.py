"""
PyQt6 GUI for the GoldSource Model Merger.

Entry point: goldsource.gui.run()
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QObject, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenuBar, QMessageBox, QProgressBar, QProgressDialog,
    QPushButton, QSplitter, QStackedWidget, QStyle,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from goldsource.merger import ModelInput, MergeConfig, MergeReport, MergeResult, ModelMerger
from goldsource.config import (
    AppConfig, ModelEntry, SkinVariantSpec, TextureReplacementSpec, SkinSlotSpec,
)


# ---------------------------------------------------------------------------
# Workers (run off the main thread)
# ---------------------------------------------------------------------------

class _LoadWorker(QObject):
    # Emits (ModelInput, directory_str)
    finished = pyqtSignal(object, str)
    failed   = pyqtSignal(str)

    def __init__(self, name: str, directory: str) -> None:
        super().__init__()
        self._name = name
        self._directory = directory

    @pyqtSlot()
    def run(self) -> None:
        try:
            model = ModelInput.from_directory(self._name, self._directory)
            self.finished.emit(model, self._directory)
        except Exception as exc:
            self.failed.emit(str(exc))


class _AnalysisWorker(QObject):
    finished = pyqtSignal(object)   # MergeReport
    failed   = pyqtSignal(str)

    def __init__(self, models: list[ModelInput]) -> None:
        super().__init__()
        self._models = models

    @pyqtSlot()
    def run(self) -> None:
        try:
            merger = ModelMerger()
            for m in self._models:
                merger.add_model(m)
            self.finished.emit(merger.analyze())
        except Exception as exc:
            self.failed.emit(str(exc))


class _MergeWorker(QObject):
    finished = pyqtSignal(object)   # MergeResult
    failed   = pyqtSignal(str)

    def __init__(
        self,
        models: list[ModelInput],
        modelname: str,
        output_dir: str,
        config: MergeConfig | None = None,
    ) -> None:
        super().__init__()
        self._models    = models
        self._modelname = modelname
        self._output    = output_dir
        self._config    = config

    @pyqtSlot()
    def run(self) -> None:
        try:
            merger = ModelMerger()
            for m in self._models:
                merger.add_model(m)
            result = merger.merge(self._modelname, config=self._config)
            result.save(self._output)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _launch_thread(worker: QObject, on_finish, on_fail, keeper: list) -> None:
    """Wire up a worker to a fresh QThread and start it."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(lambda: keeper.remove(thread) if thread in keeper else None)
    thread.finished.connect(lambda: keeper.remove(worker) if worker in keeper else None)

    worker.finished.connect(on_finish, Qt.ConnectionType.QueuedConnection)
    worker.failed.connect(on_fail,     Qt.ConnectionType.QueuedConnection)

    keeper.append(thread)
    keeper.append(worker)
    thread.start()


# ---------------------------------------------------------------------------
# Model list panel
# ---------------------------------------------------------------------------

class _ModelListPanel(QWidget):
    modelsChanged = pyqtSignal()

    _COL_NAME  = 0
    _COL_DIR   = 1
    _COL_BONES = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._threads: list = []
        self._loading_dlg: QProgressDialog | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Name", "Directory", "Bones"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_NAME,  QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_DIR,   QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_BONES, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Model…")
        self._btn_add.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        self._btn_remove.setEnabled(False)
        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def models(self) -> list[ModelInput]:
        result = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._COL_NAME)
            if item:
                m = item.data(Qt.ItemDataRole.UserRole)
                if m:
                    result.append(m)
        return result

    def add_model_entry(self, name: str, directory: str) -> None:
        """Load a model from directory and add to the list (async)."""
        if any(m.name == name for m in self.models()):
            QMessageBox.warning(self, "Duplicate name",
                                f"A model named '{name}' is already in the list.")
            return

        self._loading_dlg = QProgressDialog(f"Loading {name}…", None, 0, 0, self)
        self._loading_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._loading_dlg.show()

        worker = _LoadWorker(name, directory)
        _launch_thread(
            worker,
            on_finish=self._on_model_loaded,
            on_fail=self._on_load_failed,
            keeper=self._threads,
        )

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        self._btn_remove.setEnabled(bool(self._table.selectedItems()))

    def _on_add(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select decompiled model directory"
        )
        if not directory:
            return

        default_name = Path(directory).name
        name, ok = QInputDialog.getText(
            self, "Model identifier",
            "Short name (used as prefix in merged output):",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        self.add_model_entry(name.strip(), directory)

    def _on_model_loaded(self, model: ModelInput, directory: str) -> None:
        if self._loading_dlg:
            self._loading_dlg.close()
            self._loading_dlg = None

        row = self._table.rowCount()
        self._table.insertRow(row)

        from goldsource.merger import _pick_ref_smd
        ref = _pick_ref_smd(model)
        bones = len(ref.nodes) if ref else 0

        name_item = QTableWidgetItem(model.name)
        name_item.setData(Qt.ItemDataRole.UserRole, model)
        # Store source directory in UserRole+1 for config serialisation
        name_item.setData(Qt.ItemDataRole.UserRole + 1, directory)
        dir_item  = QTableWidgetItem(directory)
        bone_item = QTableWidgetItem(str(bones))
        bone_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self._table.setItem(row, self._COL_NAME,  name_item)
        self._table.setItem(row, self._COL_DIR,   dir_item)
        self._table.setItem(row, self._COL_BONES, bone_item)

        self.modelsChanged.emit()

    def _on_load_failed(self, error: str) -> None:
        if self._loading_dlg:
            self._loading_dlg.close()
            self._loading_dlg = None
        QMessageBox.critical(self, "Failed to load model", error)

    def _on_remove(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._table.removeRow(row)
        self.modelsChanged.emit()

    def model_directories(self) -> dict[str, str]:
        """Return {model_name: source_directory} for all loaded models."""
        result: dict[str, str] = {}
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._COL_NAME)
            if item:
                m = item.data(Qt.ItemDataRole.UserRole)
                d = item.data(Qt.ItemDataRole.UserRole + 1)
                if m and d:
                    result[m.name] = d
        return result


# ---------------------------------------------------------------------------
# Skins panel
# ---------------------------------------------------------------------------

class _SkinsPanel(QWidget):
    """Configure per-model skin variants and global skin slots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # model_name -> list[SkinVariantSpec]
        self._variants: dict[str, list[SkinVariantSpec]] = {}
        # list[SkinSlotSpec]
        self._slots: list[SkinSlotSpec] = []
        self._model_names: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: model/variant tree ─────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Per-Model Variants:"))

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        left_layout.addWidget(self._tree)

        tree_btns = QHBoxLayout()
        self._btn_add_var = QPushButton("+ Variant")
        self._btn_add_var.setEnabled(False)
        self._btn_add_var.clicked.connect(self._on_add_variant)
        self._btn_del_var = QPushButton("- Remove")
        self._btn_del_var.setEnabled(False)
        self._btn_del_var.clicked.connect(self._on_remove_variant)
        tree_btns.addWidget(self._btn_add_var)
        tree_btns.addWidget(self._btn_del_var)
        tree_btns.addStretch()
        left_layout.addLayout(tree_btns)
        top_splitter.addWidget(left)

        # ── Right: texture replacements for selected variant ──────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Texture Replacements (select a variant):"))

        self._rep_table = QTableWidget(0, 2)
        self._rep_table.setHorizontalHeaderLabels(["Original Texture", "Replacement File"])
        self._rep_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._rep_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._rep_table.setAlternatingRowColors(True)
        self._rep_table.verticalHeader().setVisible(False)
        self._rep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self._rep_table)

        rep_btns = QHBoxLayout()
        self._btn_add_rep = QPushButton("+ Add Replacement")
        self._btn_add_rep.setEnabled(False)
        self._btn_add_rep.clicked.connect(self._on_add_replacement)
        self._btn_del_rep = QPushButton("- Remove")
        self._btn_del_rep.setEnabled(False)
        self._btn_del_rep.clicked.connect(self._on_remove_replacement)
        rep_btns.addWidget(self._btn_add_rep)
        rep_btns.addWidget(self._btn_del_rep)
        rep_btns.addStretch()
        right_layout.addLayout(rep_btns)
        top_splitter.addWidget(right)
        top_splitter.setSizes([200, 400])

        layout.addWidget(top_splitter, 3)

        # ── Bottom: global skin slots ─────────────────────────────────
        slot_group = QGroupBox("Global Skin Slots")
        slot_layout = QVBoxLayout(slot_group)

        self._slot_table = QTableWidget(0, 1)
        self._slot_table.setHorizontalHeaderLabels(["Slot Name"])
        self._slot_table.verticalHeader().setVisible(False)
        self._slot_table.setAlternatingRowColors(True)
        slot_layout.addWidget(self._slot_table)

        slot_btns = QHBoxLayout()
        btn_add_slot = QPushButton("+ Add Slot")
        btn_add_slot.clicked.connect(self._on_add_slot)
        btn_del_slot = QPushButton("- Remove Slot")
        btn_del_slot.clicked.connect(self._on_remove_slot)
        slot_btns.addWidget(btn_add_slot)
        slot_btns.addWidget(btn_del_slot)
        slot_btns.addStretch()
        slot_layout.addLayout(slot_btns)
        layout.addWidget(slot_group, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_models(self, models: list[ModelInput]) -> None:
        """Called when the model list changes. Sync internal state."""
        new_names = [m.name for m in models]

        # Remove variants for models that disappeared
        for name in list(self._variants.keys()):
            if name not in new_names:
                del self._variants[name]

        # Add empty variant lists for new models
        for name in new_names:
            if name not in self._variants:
                self._variants[name] = []

        self._model_names = new_names
        self._rebuild_tree()
        self._rebuild_slot_columns()

    def get_skin_variants(self) -> dict[str, list[SkinVariantSpec]]:
        return {k: list(v) for k, v in self._variants.items()}

    def get_skin_slots(self) -> list[SkinSlotSpec]:
        return self._read_slots_from_table()

    def load_from_config(
        self,
        variants: dict[str, list[SkinVariantSpec]],
        slots: list[SkinSlotSpec],
    ) -> None:
        self._variants = {k: list(v) for k, v in variants.items()}
        self._slots = list(slots)
        self._rebuild_tree()
        self._rebuild_slot_table_full()

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        for model_name in self._model_names:
            model_item = QTreeWidgetItem([model_name])
            model_item.setData(0, Qt.ItemDataRole.UserRole, ("model", model_name))
            font = model_item.font(0)
            font.setBold(True)
            model_item.setFont(0, font)
            for var in self._variants.get(model_name, []):
                var_item = QTreeWidgetItem([var.name])
                var_item.setData(0, Qt.ItemDataRole.UserRole, ("variant", model_name, var.name))
                model_item.addChild(var_item)
            self._tree.addTopLevelItem(model_item)
            model_item.setExpanded(True)

    def _selected_model_and_variant(self) -> tuple[str | None, str | None]:
        items = self._tree.selectedItems()
        if not items:
            return None, None
        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return None, None
        if data[0] == "model":
            return data[1], None
        if data[0] == "variant":
            return data[1], data[2]
        return None, None

    def _on_tree_selection(self) -> None:
        model_name, variant_name = self._selected_model_and_variant()
        self._btn_add_var.setEnabled(model_name is not None)
        can_remove = variant_name is not None
        self._btn_del_var.setEnabled(can_remove)
        self._btn_add_rep.setEnabled(can_remove)
        self._btn_del_rep.setEnabled(False)

        if variant_name:
            self._load_replacements(model_name, variant_name)
        else:
            self._rep_table.setRowCount(0)

        self._rep_table.itemSelectionChanged.connect(self._on_rep_selection)

    def _on_rep_selection(self) -> None:
        self._btn_del_rep.setEnabled(bool(self._rep_table.selectedItems()))

    def _load_replacements(self, model_name: str, variant_name: str) -> None:
        self._rep_table.setRowCount(0)
        variants = self._variants.get(model_name, [])
        variant = next((v for v in variants if v.name == variant_name), None)
        if not variant:
            return
        for rep in variant.replacements:
            row = self._rep_table.rowCount()
            self._rep_table.insertRow(row)
            self._rep_table.setItem(row, 0, QTableWidgetItem(rep.original))
            self._rep_table.setItem(row, 1, QTableWidgetItem(rep.replacement))
            # Store source_path in UserRole
            self._rep_table.item(row, 1).setData(
                Qt.ItemDataRole.UserRole, rep.source_path
            )

    def _find_variant(self, model_name: str, variant_name: str) -> SkinVariantSpec | None:
        return next(
            (v for v in self._variants.get(model_name, []) if v.name == variant_name),
            None,
        )

    # ------------------------------------------------------------------
    # Variant CRUD
    # ------------------------------------------------------------------

    def _on_add_variant(self) -> None:
        model_name, _ = self._selected_model_and_variant()
        if not model_name:
            return
        existing_names = [v.name for v in self._variants.get(model_name, [])]
        name, ok = QInputDialog.getText(
            self, "New Skin Variant",
            f"Variant name for '{model_name}':",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in existing_names:
            QMessageBox.warning(self, "Duplicate", f"Variant '{name}' already exists.")
            return
        self._variants.setdefault(model_name, []).append(SkinVariantSpec(name=name))
        self._rebuild_tree()
        self._rebuild_slot_columns()

    def _on_remove_variant(self) -> None:
        model_name, variant_name = self._selected_model_and_variant()
        if not model_name or not variant_name:
            return
        variants = self._variants.get(model_name, [])
        self._variants[model_name] = [v for v in variants if v.name != variant_name]
        self._rep_table.setRowCount(0)
        self._rebuild_tree()
        self._rebuild_slot_columns()

    # ------------------------------------------------------------------
    # Replacement CRUD
    # ------------------------------------------------------------------

    def _on_add_replacement(self) -> None:
        model_name, variant_name = self._selected_model_and_variant()
        if not model_name or not variant_name:
            return

        # Ask for original texture name
        orig, ok = QInputDialog.getText(
            self, "Original Texture",
            "Original texture filename (as in SMD, e.g. hand.bmp):",
        )
        if not ok or not orig.strip():
            return
        orig = orig.strip()

        # Browse for replacement file
        src_path, _ = QFileDialog.getOpenFileName(
            self, "Select replacement texture",
            "",
            "BMP Images (*.bmp *.BMP);;All files (*)",
        )
        if not src_path:
            return

        replacement = Path(src_path).name
        variant = self._find_variant(model_name, variant_name)
        if variant is None:
            return

        rep = TextureReplacementSpec(
            original=orig,
            replacement=replacement,
            source_path=src_path,
        )
        variant.replacements.append(rep)
        self._load_replacements(model_name, variant_name)

    def _on_remove_replacement(self) -> None:
        model_name, variant_name = self._selected_model_and_variant()
        if not model_name or not variant_name:
            return
        rows = sorted(
            {idx.row() for idx in self._rep_table.selectedIndexes()}, reverse=True
        )
        variant = self._find_variant(model_name, variant_name)
        if variant is None:
            return
        for row in rows:
            if 0 <= row < len(variant.replacements):
                variant.replacements.pop(row)
        self._load_replacements(model_name, variant_name)

    # ------------------------------------------------------------------
    # Skin slots
    # ------------------------------------------------------------------

    def _rebuild_slot_columns(self) -> None:
        """Add/remove model columns while preserving existing slot data."""
        existing_slots = self._read_slots_from_table()
        self._rebuild_slot_table_full(existing_slots)

    def _rebuild_slot_table_full(
        self, slots: list[SkinSlotSpec] | None = None
    ) -> None:
        if slots is None:
            slots = self._slots

        # Columns: "Slot Name" + one per model
        cols = ["Slot Name"] + self._model_names
        self._slot_table.setColumnCount(len(cols))
        self._slot_table.setHorizontalHeaderLabels(cols)
        hdr = self._slot_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(1, len(cols)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self._slot_table.setRowCount(0)
        for slot in slots:
            self._add_slot_row(slot.name, slot.assignments)

    def _add_slot_row(
        self, name: str = "", assignments: dict[str, str] | None = None
    ) -> None:
        row = self._slot_table.rowCount()
        self._slot_table.insertRow(row)

        name_item = QTableWidgetItem(name)
        self._slot_table.setItem(row, 0, name_item)

        for col, model_name in enumerate(self._model_names, start=1):
            combo = QComboBox()
            combo.addItem("(default)")
            for v in self._variants.get(model_name, []):
                combo.addItem(v.name)
            if assignments and model_name in assignments:
                idx = combo.findText(assignments[model_name])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._slot_table.setCellWidget(row, col, combo)

    def _read_slots_from_table(self) -> list[SkinSlotSpec]:
        slots: list[SkinSlotSpec] = []
        for row in range(self._slot_table.rowCount()):
            name_item = self._slot_table.item(row, 0)
            name = name_item.text() if name_item else f"Slot {row + 1}"
            assignments: dict[str, str] = {}
            for col, model_name in enumerate(self._model_names, start=1):
                combo = self._slot_table.cellWidget(row, col)
                if combo and combo.currentIndex() > 0:
                    assignments[model_name] = combo.currentText()
            slots.append(SkinSlotSpec(name=name, assignments=assignments))
        return slots

    def _on_add_slot(self) -> None:
        self._add_slot_row(f"Slot {self._slot_table.rowCount() + 1}")

    def _on_remove_slot(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._slot_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._slot_table.removeRow(row)


# ---------------------------------------------------------------------------
# Sequence renames panel
# ---------------------------------------------------------------------------

class _SeqRenamesPanel(QWidget):
    """Configure string replacement rules applied to sequence names."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        info = QLabel(
            "Rules are applied in order. Each rule replaces all occurrences "
            "of 'Find' with 'Replace With' in every sequence name."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(info)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Find", "Replace With"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Add Rule")
        btn_add.clicked.connect(self._on_add)
        btn_rem = QPushButton("- Remove")
        btn_rem.clicked.connect(self._on_remove)
        btn_up = QPushButton("↑ Up")
        btn_up.clicked.connect(self._on_move_up)
        btn_dn = QPushButton("↓ Down")
        btn_dn.clicked.connect(self._on_move_down)
        for b in (btn_add, btn_rem, btn_up, btn_dn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def get_renames(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for row in range(self._table.rowCount()):
            find_item    = self._table.item(row, 0)
            replace_item = self._table.item(row, 1)
            find    = find_item.text()    if find_item    else ""
            replace = replace_item.text() if replace_item else ""
            if find:
                result.append((find, replace))
        return result

    def set_renames(self, renames: list[tuple[str, str]] | list[list[str]]) -> None:
        self._table.setRowCount(0)
        for pair in renames:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(pair[0])))
            self._table.setItem(row, 1, QTableWidgetItem(str(pair[1])))

    # ------------------------------------------------------------------
    def _on_add(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(""))
        self._table.setItem(row, 1, QTableWidgetItem(""))
        self._table.editItem(self._table.item(row, 0))

    def _on_remove(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self._table.removeRow(row)

    def _on_move_up(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        for row in rows:
            if row == 0:
                continue
            self._swap_rows(row - 1, row)

    def _on_move_down(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            if row >= self._table.rowCount() - 1:
                continue
            self._swap_rows(row, row + 1)

    def _swap_rows(self, a: int, b: int) -> None:
        for col in range(self._table.columnCount()):
            ia = self._table.item(a, col)
            ib = self._table.item(b, col)
            ta = ia.text() if ia else ""
            tb = ib.text() if ib else ""
            self._table.setItem(a, col, QTableWidgetItem(tb))
            self._table.setItem(b, col, QTableWidgetItem(ta))


# ---------------------------------------------------------------------------
# Analysis panel (right)
# ---------------------------------------------------------------------------

class _AnalysisPanel(QWidget):
    _COL_MODEL  = 0
    _COL_TOTAL  = 1
    _COL_UNIQUE = 2
    _COL_SHARED = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("Analysis")
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        title.setFont(font)
        layout.addWidget(title)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Page 0 — placeholder
        placeholder = QLabel("Add at least two models to begin analysis.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: grey; font-style: italic;")
        self._stack.addWidget(placeholder)

        # Page 1 — analysis content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        content_layout.addWidget(splitter)

        # ── Bone stats ──────────────────────────────────────────────
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 4)

        stats_layout.addWidget(QLabel("Bone Statistics"))
        self._stats_table = QTableWidget(0, 4)
        self._stats_table.setHorizontalHeaderLabels(
            ["Model", "Total", "Unique", "Shared"]
        )
        self._stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._stats_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._stats_table.setAlternatingRowColors(True)
        self._stats_table.verticalHeader().setVisible(False)
        hdr = self._stats_table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_MODEL, QHeaderView.ResizeMode.Stretch)
        for col in (self._COL_TOTAL, self._COL_UNIQUE, self._COL_SHARED):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        stats_layout.addWidget(self._stats_table)

        bone_row = QHBoxLayout()
        bone_row.addWidget(QLabel("Merged:"))
        self._bone_bar = QProgressBar()
        self._bone_bar.setRange(0, ModelMerger.BONE_LIMIT)
        self._bone_bar.setFormat("%v / 128 bones")
        self._bone_bar.setValue(0)
        self._bone_bar.setMinimumWidth(200)
        bone_row.addWidget(self._bone_bar, 1)
        self._bone_label = QLabel("")
        bone_row.addWidget(self._bone_label)
        stats_layout.addLayout(bone_row)
        splitter.addWidget(stats_widget)

        # ── Conflicts & warnings ─────────────────────────────────────
        bottom_tabs = QTabWidget()

        self._conflict_list = QListWidget()
        self._conflict_list.setAlternatingRowColors(True)
        bottom_tabs.addTab(self._conflict_list, "Conflicts (0)")

        self._warning_list = QListWidget()
        self._warning_list.setAlternatingRowColors(True)
        bottom_tabs.addTab(self._warning_list, "Warnings (0)")

        self._suggestion_box = QGroupBox("Removal Suggestions")
        sugg_layout = QVBoxLayout(self._suggestion_box)
        self._suggestion_list = QListWidget()
        self._suggestion_list.setAlternatingRowColors(True)
        self._suggestion_list.setMaximumHeight(90)
        sugg_layout.addWidget(self._suggestion_list)
        bottom_tabs.addTab(self._suggestion_box, "Suggestions")

        splitter.addWidget(bottom_tabs)
        splitter.setSizes([200, 150])

        self._stack.addWidget(content)
        self._bottom_tabs = bottom_tabs

    # ------------------------------------------------------------------
    def show_placeholder(self) -> None:
        self._stack.setCurrentIndex(0)

    def update_report(self, report: MergeReport) -> None:
        self._stack.setCurrentIndex(1)

        self._stats_table.setRowCount(0)
        for stat in report.bone_stats:
            row = self._stats_table.rowCount()
            self._stats_table.insertRow(row)
            self._stats_table.setItem(row, self._COL_MODEL,
                                      QTableWidgetItem(stat.model_name))
            for col, val in (
                (self._COL_TOTAL,  stat.total_bones),
                (self._COL_UNIQUE, stat.unique_count),
                (self._COL_SHARED, stat.shared_count),
            ):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._stats_table.setItem(row, col, item)

        count = report.total_unique_bones
        self._bone_bar.setValue(min(count, ModelMerger.BONE_LIMIT))
        if report.exceeds_limit:
            self._bone_bar.setFormat(f"{count} / 128 bones  ⚠ EXCEEDS LIMIT")
            color = "#e74c3c"
        elif count >= 116:
            self._bone_bar.setFormat("%v / 128 bones")
            color = "#e74c3c"
        elif count >= 90:
            color = "#e67e22"
            self._bone_bar.setFormat("%v / 128 bones")
        else:
            color = "#27ae60"
            self._bone_bar.setFormat("%v / 128 bones")
        self._bone_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )
        self._bone_label.setText(f"({count})")

        self._conflict_list.clear()
        for conflict in report.conflicts:
            usages = ",  ".join(
                f"{m} → parent={p!r}" for m, p in conflict.usages
            )
            item = QListWidgetItem(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning),
                f"{conflict.bone_name}:  {usages}",
            )
            self._conflict_list.addItem(item)
        self._bottom_tabs.setTabText(0, f"Conflicts ({len(report.conflicts)})")

        self._warning_list.clear()
        for w in report.warnings:
            item = QListWidgetItem(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
                w,
            )
            self._warning_list.addItem(item)
        self._bottom_tabs.setTabText(1, f"Warnings ({len(report.warnings)})")

        self._suggestion_list.clear()
        for name in report.removal_suggestions:
            stat = next((s for s in report.bone_stats if s.model_name == name), None)
            freed = stat.unique_count if stat else "?"
            self._suggestion_list.addItem(f"{name}  ({freed} unique bones freed)")
        tab_label = (
            f"Suggestions ({len(report.removal_suggestions)})"
            if report.removal_suggestions else "Suggestions"
        )
        self._bottom_tabs.setTabText(2, tab_label)
        if report.exceeds_limit:
            self._bottom_tabs.setCurrentIndex(2)


# ---------------------------------------------------------------------------
# Output panel (bottom)
# ---------------------------------------------------------------------------

class _OutputPanel(QGroupBox):
    analyzeRequested = pyqtSignal()
    mergeRequested   = pyqtSignal(str, str)   # modelname, output_dir

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Output", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. v_merged")
        form.addRow("Model name (.mdl):", self._name_edit)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("Select output directory…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_output)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(btn_browse)
        form.addRow("Output directory:", dir_row)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._btn_analyze = QPushButton("Analyze")
        self._btn_analyze.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self._btn_analyze.clicked.connect(self.analyzeRequested)

        self._btn_merge = QPushButton("Merge")
        self._btn_merge.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self._btn_merge.setEnabled(False)
        self._btn_merge.setStyleSheet(
            "QPushButton:enabled { background-color: #2980b9; color: white; "
            "font-weight: bold; padding: 4px 16px; border-radius: 4px; }"
            "QPushButton:enabled:hover { background-color: #3498db; }"
        )
        self._btn_merge.clicked.connect(self._on_merge_clicked)

        btn_row.addWidget(self._btn_analyze)
        btn_row.addWidget(self._btn_merge)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def model_name(self) -> str:
        return self._name_edit.text().strip()

    def output_dir(self) -> str:
        return self._dir_edit.text().strip()

    def set_model_name(self, name: str) -> None:
        self._name_edit.setText(name)

    def set_output_dir(self, path: str) -> None:
        self._dir_edit.setText(path)

    def set_merge_enabled(self, enabled: bool) -> None:
        self._btn_merge.setEnabled(
            enabled and bool(self._name_edit.text().strip()) and bool(self._dir_edit.text())
        )

    def set_analyzing(self, active: bool) -> None:
        self._btn_analyze.setEnabled(not active)
        self._btn_analyze.setText("Analyzing…" if active else "Analyze")

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._dir_edit.setText(path)

    def _on_merge_clicked(self) -> None:
        name = self._name_edit.text().strip()
        directory = self._dir_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a model name.")
            return
        if not directory:
            QMessageBox.warning(self, "Missing directory",
                                "Please select an output directory.")
            return
        mdl_name = name if name.endswith(".mdl") else f"{name}.mdl"
        self.mergeRequested.emit(mdl_name, directory)


# ---------------------------------------------------------------------------
# Post-merge result dialog
# ---------------------------------------------------------------------------

class _MergeResultDialog(QDialog):
    def __init__(self, result: MergeResult, output_dir: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Complete")
        self.setMinimumSize(600, 420)
        self._output_dir = output_dir
        self._setup_ui(result)

    def _setup_ui(self, result: MergeResult) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("✔  Merge complete")
        font = header.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        header.setFont(font)
        header.setStyleSheet("color: #27ae60;")
        layout.addWidget(header)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        # ── Summary tab ──────────────────────────────────────────────
        summary_browser = QTextBrowser()
        smd_count = len(result.smds)
        tex_count  = len(result.textures)
        html = (
            f"<pre>{result.report.summary()}</pre>"
            f"<hr>"
            f"<b>Files written:</b><br>"
            f"&nbsp;&nbsp;SMDs: {smd_count}<br>"
            f"&nbsp;&nbsp;Textures: {tex_count}<br>"
            f"&nbsp;&nbsp;Output: <code>{self._output_dir}</code>"
        )
        summary_browser.setHtml(html)
        tabs.addTab(summary_browser, "Summary")

        # ── Renames tab ──────────────────────────────────────────────
        renames_tree = QTreeWidget()
        renames_tree.setHeaderLabels(["Original", "Renamed to"])
        renames_tree.setAlternatingRowColors(True)

        any_renames = False
        for section_label, remap in (
            ("Bone renames", result.renamed_bones),
            ("Texture renames", result.renamed_textures),
        ):
            if not remap:
                continue
            any_renames = True
            section_item = QTreeWidgetItem([section_label])
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)
            for model_name, pairs in remap.items():
                if not pairs:
                    continue
                model_item = QTreeWidgetItem([model_name])
                for old, new in pairs.items():
                    QTreeWidgetItem(model_item, [old, new])
                section_item.addChild(model_item)
            renames_tree.addTopLevelItem(section_item)
            section_item.setExpanded(True)

        if not any_renames:
            QTreeWidgetItem(renames_tree, ["No renames applied", ""])
        renames_tree.resizeColumnToContents(0)
        tabs.addTab(renames_tree, "Renames")

        # ── Buttons ──────────────────────────────────────────────────
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        open_btn = QPushButton("Open Output Folder")
        open_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        open_btn.clicked.connect(self._open_folder)
        button_box.addButton(open_btn, QDialogButtonBox.ButtonRole.ActionRole)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GoldSource Model Merger")
        self.setMinimumSize(960, 620)
        self.resize(1200, 720)

        self._threads:      list = []
        self._last_report:  MergeReport | None = None
        self._merge_progress: QProgressDialog | None = None
        self._config_path:  str | None = None
        self._analysis_debounce = QTimer(self)
        self._analysis_debounce.setSingleShot(True)
        self._analysis_debounce.timeout.connect(self._run_analysis)

        self._setup_ui()
        self._setup_menu()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        outer_splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(outer_splitter)

        # Top: left tabs + analysis
        inner_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left tab widget
        self._left_tabs = QTabWidget()

        self._model_panel = _ModelListPanel()
        self._skins_panel = _SkinsPanel()
        self._seq_panel   = _SeqRenamesPanel()

        self._left_tabs.addTab(self._model_panel, "Models")
        self._left_tabs.addTab(self._skins_panel, "Skins")
        self._left_tabs.addTab(self._seq_panel,   "Sequences")

        self._analysis_panel = _AnalysisPanel()
        inner_splitter.addWidget(self._left_tabs)
        inner_splitter.addWidget(self._analysis_panel)
        inner_splitter.setSizes([340, 700])
        outer_splitter.addWidget(inner_splitter)

        # Bottom: output controls
        self._output_panel = _OutputPanel()
        outer_splitter.addWidget(self._output_panel)
        outer_splitter.setSizes([500, 130])
        outer_splitter.setCollapsible(1, False)

        # Signals
        self._model_panel.modelsChanged.connect(self._on_models_changed)
        self._output_panel.analyzeRequested.connect(self._run_analysis)
        self._output_panel.mergeRequested.connect(self._on_merge_requested)

        self.statusBar().showMessage("Ready. Add decompiled model directories to begin.")

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_new = QAction("&New", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._on_new)
        file_menu.addAction(act_new)

        file_menu.addSeparator()

        act_open = QAction("&Open Config…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_config)
        file_menu.addAction(act_open)

        act_save = QAction("&Save Config", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._on_save_config)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Config &As…", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._on_save_config_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    # ------------------------------------------------------------------
    # Model change propagation
    # ------------------------------------------------------------------

    def _on_models_changed(self) -> None:
        models = self._model_panel.models()
        self._skins_panel.update_models(models)

        if len(models) < 2:
            self._analysis_panel.show_placeholder()
            self._last_report = None
            self._output_panel.set_merge_enabled(False)
            return
        self._analysis_debounce.start(300)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self) -> None:
        models = self._model_panel.models()
        if len(models) < 2:
            return
        self._output_panel.set_analyzing(True)
        self.statusBar().showMessage("Analyzing…")

        worker = _AnalysisWorker(models)
        _launch_thread(
            worker,
            on_finish=self._on_analysis_done,
            on_fail=self._on_analysis_failed,
            keeper=self._threads,
        )

    def _on_analysis_done(self, report: MergeReport) -> None:
        self._last_report = report
        self._analysis_panel.update_report(report)
        self._output_panel.set_analyzing(False)
        self._output_panel.set_merge_enabled(not report.exceeds_limit)

        msg = (
            f"Analysis complete — {report.total_unique_bones}/128 bones"
            + (f", {len(report.conflicts)} conflict(s)" if report.conflicts else "")
            + (f", {len(report.warnings)} warning(s)" if report.warnings else "")
        )
        if report.exceeds_limit:
            msg += "  ⚠ EXCEEDS BONE LIMIT — see Suggestions tab"
        self.statusBar().showMessage(msg)

    def _on_analysis_failed(self, error: str) -> None:
        self._output_panel.set_analyzing(False)
        self.statusBar().showMessage("Analysis failed.")
        QMessageBox.critical(self, "Analysis error", error)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _build_merge_config(self) -> MergeConfig:
        """Assemble MergeConfig from the current panel state."""
        from goldsource.merger import SkinVariant, SkinSlot, TextureReplacement

        skin_variant_specs = self._skins_panel.get_skin_variants()
        skin_slot_specs    = self._skins_panel.get_skin_slots()
        seq_renames        = self._seq_panel.get_renames()

        # Convert specs → merger objects (reading replacement bytes from disk)
        skin_variants: dict[str, list[SkinVariant]] = {}
        for model_name, specs in skin_variant_specs.items():
            variants: list[SkinVariant] = []
            for spec in specs:
                reps: list[TextureReplacement] = []
                rep_data: dict[str, bytes] = {}
                for r in spec.replacements:
                    src = Path(r.source_path)
                    if src.exists():
                        rep_data[r.replacement] = src.read_bytes()
                        reps.append(TextureReplacement(
                            original=r.original,
                            replacement=r.replacement,
                        ))
                variants.append(SkinVariant(
                    name=spec.name,
                    model_name=model_name,
                    replacements=reps,
                    replacement_data=rep_data,
                ))
            skin_variants[model_name] = variants

        skin_slots = [
            SkinSlot(name=s.name, assignments=dict(s.assignments))
            for s in skin_slot_specs
        ]

        return MergeConfig(
            sequence_renames=seq_renames,
            skin_variants=skin_variants,
            skin_slots=skin_slots,
        )

    def _on_merge_requested(self, modelname: str, output_dir: str) -> None:
        models = self._model_panel.models()
        if not models:
            return

        config = self._build_merge_config()

        self._merge_progress = QProgressDialog("Merging models…", None, 0, 0, self)
        self._merge_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._merge_progress.setMinimumDuration(0)
        self._merge_progress.show()
        self.statusBar().showMessage("Merging…")

        worker = _MergeWorker(models, modelname, output_dir, config)
        _launch_thread(
            worker,
            on_finish=lambda r: self._on_merge_done(r, output_dir),
            on_fail=self._on_merge_failed,
            keeper=self._threads,
        )

    def _on_merge_done(self, result: MergeResult, output_dir: str) -> None:
        if self._merge_progress:
            self._merge_progress.close()
            self._merge_progress = None

        self.statusBar().showMessage(
            f"Merge complete — {len(result.smds)} SMDs, "
            f"{len(result.textures)} textures → {output_dir}"
        )
        dlg = _MergeResultDialog(result, output_dir, self)
        dlg.exec()

    def _on_merge_failed(self, error: str) -> None:
        if self._merge_progress:
            self._merge_progress.close()
            self._merge_progress = None
        self.statusBar().showMessage("Merge failed.")
        QMessageBox.critical(self, "Merge error", error)

    # ------------------------------------------------------------------
    # Config serialisation
    # ------------------------------------------------------------------

    def _build_app_config(self) -> AppConfig:
        """Snapshot current session into an AppConfig."""
        variant_specs = self._skins_panel.get_skin_variants()
        dirs = self._model_panel.model_directories()

        models_out: list[ModelEntry] = [
            ModelEntry(
                name=m.name,
                directory=dirs.get(m.name, ""),
                skin_variants=variant_specs.get(m.name, []),
            )
            for m in self._model_panel.models()
        ]

        slots = self._skins_panel.get_skin_slots()
        seq_renames = [[f, r] for f, r in self._seq_panel.get_renames()]

        return AppConfig(
            models=models_out,
            skin_slots=[SkinSlotSpec(name=s.name, assignments=s.assignments) for s in slots],
            sequence_renames=seq_renames,
            output_model_name=self._output_panel.model_name(),
            output_directory=self._output_panel.output_dir(),
        )

    def _apply_app_config(self, cfg: AppConfig) -> None:
        """Restore UI from an AppConfig (models are reloaded from disk)."""
        # Clear model table
        self._model_panel._table.setRowCount(0)
        self._model_panel.modelsChanged.emit()

        # Reload models (each triggers an async load)
        for entry in cfg.models:
            self._model_panel.add_model_entry(entry.name, entry.directory)

        # Apply seq renames immediately (models load async but these are independent)
        self._seq_panel.set_renames(cfg.sequence_renames)
        self._output_panel.set_model_name(cfg.output_model_name)
        self._output_panel.set_output_dir(cfg.output_directory)

        # Skin variants and slots are applied after models load via a deferred helper
        # stored on self to avoid being GC'd
        self._pending_skin_cfg = cfg
        # We need to wait for all models to load before applying skins.
        # Use a timer that polls until model count matches.
        self._skin_apply_timer = QTimer(self)
        self._skin_apply_timer.setInterval(200)
        expected = len(cfg.models)
        self._skin_apply_timer.timeout.connect(
            lambda: self._try_apply_skins(expected)
        )
        self._skin_apply_timer.start()

    def _try_apply_skins(self, expected: int) -> None:
        if len(self._model_panel.models()) < expected:
            return
        self._skin_apply_timer.stop()
        cfg = self._pending_skin_cfg
        # Build variants dict from config
        variants: dict[str, list[SkinVariantSpec]] = {
            m.name: m.skin_variants for m in cfg.models
        }
        slots = [SkinSlotSpec(name=s.name, assignments=s.assignments)
                 for s in cfg.skin_slots]
        self._skins_panel.load_from_config(variants, slots)

    # ------------------------------------------------------------------
    # File menu actions
    # ------------------------------------------------------------------

    def _on_new(self) -> None:
        reply = QMessageBox.question(
            self, "New Session",
            "Clear all models and settings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._model_panel._table.setRowCount(0)
        self._model_panel.modelsChanged.emit()
        self._seq_panel.set_renames([])
        self._output_panel.set_model_name("")
        self._output_panel.set_output_dir("")
        self._config_path = None
        self.setWindowTitle("GoldSource Model Merger")

    def _on_open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config",
            "",
            "JSON Config (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            cfg = AppConfig.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._config_path = path
        self.setWindowTitle(f"GoldSource Model Merger — {Path(path).name}")
        self._apply_app_config(cfg)
        self.statusBar().showMessage(f"Loaded config: {path}")

    def _on_save_config(self) -> None:
        if not self._config_path:
            self._on_save_config_as()
            return
        try:
            self._build_app_config().save(self._config_path)
            self.statusBar().showMessage(f"Saved: {self._config_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_save_config_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config As",
            self._config_path or "merger_config.json",
            "JSON Config (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        try:
            self._build_app_config().save(path)
            self._config_path = path
            self.setWindowTitle(f"GoldSource Model Merger — {Path(path).name}")
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Launch the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("GoldSource Model Merger")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

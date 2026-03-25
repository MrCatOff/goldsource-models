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
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QProgressDialog,
    QPushButton, QSplitter, QStackedWidget, QStyle,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from goldsource.merger import ModelInput, MergeReport, MergeResult, ModelMerger


# ---------------------------------------------------------------------------
# Workers (run off the main thread)
# ---------------------------------------------------------------------------

class _LoadWorker(QObject):
    finished = pyqtSignal(object)   # ModelInput
    failed   = pyqtSignal(str)

    def __init__(self, name: str, directory: str) -> None:
        super().__init__()
        self._name = name
        self._directory = directory

    @pyqtSlot()
    def run(self) -> None:
        try:
            model = ModelInput.from_directory(self._name, self._directory)
            self.finished.emit(model)
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

    def __init__(self, models: list[ModelInput], modelname: str, output_dir: str) -> None:
        super().__init__()
        self._models    = models
        self._modelname = modelname
        self._output    = output_dir

    @pyqtSlot()
    def run(self) -> None:
        try:
            merger = ModelMerger()
            for m in self._models:
                merger.add_model(m)
            result = merger.merge(self._modelname)
            result.save(self._output)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _launch_thread(worker: QObject, on_finish, on_fail, keeper: list) -> None:
    """Wire up a worker to a fresh QThread and start it.

    Both *thread* and *worker* are stored in *keeper* so Python's GC cannot
    collect them before the worker emits its completion signal.  They remove
    themselves once the thread finishes.
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    # Stop the thread and schedule cleanup when the worker is done.
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(lambda: keeper.remove(thread) if thread in keeper else None)
    thread.finished.connect(lambda: keeper.remove(worker) if worker in keeper else None)

    # Deliver results to the main thread via queued connections.
    worker.finished.connect(on_finish, Qt.ConnectionType.QueuedConnection)
    worker.failed.connect(on_fail,     Qt.ConnectionType.QueuedConnection)

    # Keep hard Python references so the GC cannot collect either object
    # before the worker has a chance to emit finished/failed.
    keeper.append(thread)
    keeper.append(worker)
    thread.start()


# ---------------------------------------------------------------------------
# Model list panel (left)
# ---------------------------------------------------------------------------

class _ModelListPanel(QWidget):
    modelsChanged = pyqtSignal()

    _COL_NAME = 0
    _COL_DIR  = 1
    _COL_BONES = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._threads: list[QThread] = []
        self._loading_dlg: QProgressDialog | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("Models")
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        title.setFont(font)
        layout.addWidget(title)

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
        self._table.setAcceptDrops(True)
        self._table.setDropIndicatorShown(True)
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
        name = name.strip()

        if any(m.name == name for m in self.models()):
            QMessageBox.warning(self, "Duplicate name",
                                f"A model named '{name}' is already in the list.")
            return

        self._loading_dlg = QProgressDialog(
            f"Loading {name}…", None, 0, 0, self
        )
        self._loading_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._loading_dlg.show()

        worker = _LoadWorker(name, directory)
        _launch_thread(
            worker,
            on_finish=self._on_model_loaded,
            on_fail=self._on_load_failed,
            keeper=self._threads,
        )

    def _on_model_loaded(self, model: ModelInput) -> None:
        if self._loading_dlg:
            self._loading_dlg.close()
            self._loading_dlg = None

        row = self._table.rowCount()
        self._table.insertRow(row)

        bones = len(model.smds[next(iter(model.smds))].nodes) if model.smds else 0
        # Try to get bones from the first ref smd
        from goldsource.merger import _pick_ref_smd
        ref = _pick_ref_smd(model)
        if ref:
            bones = len(ref.nodes)

        name_item = QTableWidgetItem(model.name)
        name_item.setData(Qt.ItemDataRole.UserRole, model)
        dir_item  = QTableWidgetItem(str(Path(next(iter(model.smds))).parent) if model.smds else "")
        # Nicer: show the directory we loaded from — reconstruct from smds keys
        dir_item  = QTableWidgetItem(self._infer_dir(model))
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

    @staticmethod
    def _infer_dir(model: ModelInput) -> str:
        """Best-effort: show the common directory prefix of the SMD keys."""
        if not model.smds:
            return ""
        parts = [k.split("/")[0] for k in model.smds]
        return parts[0] if parts else ""


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
        hdr.setSectionResizeMode(self._COL_MODEL,  QHeaderView.ResizeMode.Stretch)
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

        # Bone stats table
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

        # Bone progress bar
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

        # Conflicts
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

        # Warnings
        self._warning_list.clear()
        for w in report.warnings:
            item = QListWidgetItem(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
                w,
            )
            self._warning_list.addItem(item)
        self._bottom_tabs.setTabText(1, f"Warnings ({len(report.warnings)})")

        # Suggestions
        self._suggestion_list.clear()
        for name in report.removal_suggestions:
            stat = next((s for s in report.bone_stats if s.model_name == name), None)
            freed = stat.unique_count if stat else "?"
            self._suggestion_list.addItem(
                f"{name}  ({freed} unique bones freed)"
            )
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
        self._name_edit.textChanged.connect(self._update_merge_state)
        form.addRow("Model name (.mdl):", self._name_edit)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("Select output directory…")
        self._dir_edit.textChanged.connect(self._update_merge_state)
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

    def _update_merge_state(self) -> None:
        # Re-evaluate only the text fields; the caller controls the enabled flag.
        # Trigger a re-check on the main window side via merge button state.
        pass

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
        self.resize(1100, 700)

        self._threads:      list[QThread] = []
        self._last_report:  MergeReport | None = None
        self._merge_progress: QProgressDialog | None = None
        self._analysis_debounce = QTimer(self)
        self._analysis_debounce.setSingleShot(True)
        self._analysis_debounce.timeout.connect(self._run_analysis)

        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        outer_splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(outer_splitter)

        # Top: model list + analysis
        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._model_panel   = _ModelListPanel()
        self._analysis_panel = _AnalysisPanel()
        inner_splitter.addWidget(self._model_panel)
        inner_splitter.addWidget(self._analysis_panel)
        inner_splitter.setSizes([300, 700])
        outer_splitter.addWidget(inner_splitter)

        # Bottom: output controls
        self._output_panel = _OutputPanel()
        outer_splitter.addWidget(self._output_panel)
        outer_splitter.setSizes([480, 130])
        outer_splitter.setCollapsible(1, False)

        # Signals
        self._model_panel.modelsChanged.connect(self._on_models_changed)
        self._output_panel.analyzeRequested.connect(self._run_analysis)
        self._output_panel.mergeRequested.connect(self._on_merge_requested)

        self.statusBar().showMessage("Ready. Add decompiled model directories to begin.")

    # ------------------------------------------------------------------
    def _on_models_changed(self) -> None:
        models = self._model_panel.models()
        if len(models) < 2:
            self._analysis_panel.show_placeholder()
            self._last_report = None
            self._output_panel.set_merge_enabled(False)
            return
        # Debounce: wait 300 ms before analyzing in case multiple models are added quickly
        self._analysis_debounce.start(300)

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
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Analysis error", error)

    # ------------------------------------------------------------------
    def _on_merge_requested(self, modelname: str, output_dir: str) -> None:
        models = self._model_panel.models()
        if not models:
            return

        self._merge_progress = QProgressDialog("Merging models…", None, 0, 0, self)
        self._merge_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._merge_progress.setMinimumDuration(0)
        self._merge_progress.show()
        self.statusBar().showMessage("Merging…")

        worker = _MergeWorker(models, modelname, output_dir)
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
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Merge error", error)


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

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from src.core.anti_cheat_pipeline import CheatEvent
from src.ui.theme import ACCENT, ALERT, TEXT_MUTED

_COLUMNS = ("TIME", "CLASS", "CONF", "TRACK")

# Hard cap on retained incidents so an unattended LIVE session (hours, possibly
# a noisy detector) can't grow this list -- and the CheatEvent.telemetry_data
# path/residuals arrays it carries -- without bound. Oldest rows are evicted
# first; the review queue only ever needs to show what's recent.
_MAX_INCIDENTS = 500


class IncidentQueueModel(QAbstractTableModel):
    """Backs the live incident review queue: one row per detected CheatEvent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events: list[CheatEvent] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._events)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return _COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None

        event = self._events[index.row()]
        column = index.column()

        if role == Qt.DisplayRole:
            if column == 0:
                return datetime.fromisoformat(event.timestamp).strftime("%H:%M:%S")
            if column == 1:
                return event.cheat_category.upper().replace("_", " ")
            if column == 2:
                return f"{event.confidence_score * 100:.1f}%"
            if column == 3:
                return f"#{event.telemetry_data.get('associated_track_id', 'N/A')}"
            return None

        if role == Qt.ForegroundRole:
            if column == 1:
                return QColor(ALERT)
            if column == 2:
                return QColor(ACCENT)
            return QColor(TEXT_MUTED)

        return None

    def event_at(self, row: int) -> CheatEvent | None:
        if 0 <= row < len(self._events):
            return self._events[row]
        return None

    def add_event(self, event: CheatEvent) -> None:
        if len(self._events) >= _MAX_INCIDENTS:
            self.beginRemoveRows(QModelIndex(), 0, 0)
            self._events.pop(0)
            self.endRemoveRows()

        row = len(self._events)
        self.beginInsertRows(QModelIndex(), row, row)
        self._events.append(event)
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self._events.clear()
        self.endResetModel()


class IncidentQueueTable(QTableView):
    """Live cheat-flag detection queue; double-click a row to seek the canvas to that frame."""

    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = IncidentQueueModel(self)
        self.setModel(self._model)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.doubleClicked.connect(self._on_row_double_clicked)

    def _on_row_double_clicked(self, index: QModelIndex) -> None:
        event = self._model.event_at(index.row())
        if event is not None:
            self.seekRequested.emit(event.frame_id)

    def add_event(self, event: CheatEvent) -> None:
        self._model.add_event(event)
        self.scrollToBottom()

    def clear(self) -> None:
        self._model.clear()

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.core.anti_cheat_pipeline import FrameContext
from src.ui.theme import ACCENT, CANVAS_IDLE_COLOR


class VideoCanvas(QWidget):
    """Letterboxed frame display.

    Holds a reference to the full FrameContext (not just the derived QImage)
    because QImage(Format_BGR888) wraps the backing numpy buffer without
    copying it - dropping the array reference early would let it be freed
    or mutated mid-paint.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ctx: FrameContext | None = None
        self._image: QImage | None = None
        self._idle_text = "No capture device detected — Mount a gameplay recording to begin"

    def set_idle_text(self, text: str) -> None:
        self._idle_text = text
        if self._image is None:
            self.update()

    def set_frame(self, ctx: FrameContext, qimage: QImage) -> None:
        self._ctx = ctx
        self._image = qimage
        self.update()

    def clear_frame(self) -> None:
        self._ctx = None
        self._image = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(CANVAS_IDLE_COLOR))

        if self._image is None:
            painter.setPen(QColor(ACCENT))
            painter.drawText(self.rect(), Qt.AlignCenter, self._idle_text)
            painter.end()
            return

        painter.drawImage(self._letterbox_rect(self._image.size(), self.size()), self._image)
        painter.end()

    def _letterbox_rect(self, image_size: QSize, widget_size: QSize) -> QRect:
        if image_size.width() <= 0 or image_size.height() <= 0:
            return QRect(0, 0, widget_size.width(), widget_size.height())

        scale = min(widget_size.width() / image_size.width(), widget_size.height() / image_size.height())
        render_w = max(1, int(image_size.width() * scale))
        render_h = max(1, int(image_size.height() * scale))
        x = (widget_size.width() - render_w) // 2
        y = (widget_size.height() - render_h) // 2
        return QRect(x, y, render_w, render_h)

from __future__ import annotations

BACKGROUND = "#07090C"
SURFACE = "#0A1012"
PANEL = "#0E1518"
PANEL_ALT = "#131C1F"
BORDER = "#45A19F"
HAIRLINE = "#1B2A2E"
ACCENT = "#2EE6C7"
ACCENT_DIM = "#0F4C4A"
TEXT_PRIMARY = "#E8F1F2"
TEXT_MUTED = "#8AA0A6"
ALERT = "#FF3B5C"
WARNING = "#F5A623"
VOD_ACCENT = "#E8B86D"

CANVAS_IDLE_COLOR = "#0B1020"

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", Arial, sans-serif;
}}

QWidget#ControlBar, QWidget#PlaybackControlsBar {{
    background-color: {SURFACE};
}}

QLabel#TitleLabel {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: bold;
}}

QLabel#StatusLabel {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
    font-weight: bold;
}}

QLabel#EventCountLabel {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

QPushButton {{
    background-color: {PANEL};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {ACCENT_DIM};
}}

QPushButton:checked {{
    background-color: {ACCENT};
    color: {SURFACE};
}}

QPushButton#MountButton {{
    background-color: {SURFACE};
}}

QFrame#ControlGroup {{
    background-color: {PANEL_ALT};
    border: 1px solid {ACCENT_DIM};
    border-radius: 6px;
}}

QLabel#ControlGroupLabel {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: bold;
}}

QFrame#ChipGroup {{
    background-color: transparent;
    border: none;
}}

QFrame#ChipGroup QPushButton {{
    padding: 3px 10px;
    font-size: 10px;
    border-radius: 10px;
}}

QLabel#ModePill {{
    background-color: {PANEL_ALT};
    border: 1px solid {ACCENT_DIM};
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: bold;
    color: {TEXT_MUTED};
}}

QLabel#ModePill[mode="live"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: {BACKGROUND};
}}

QLabel#ModePill[mode="vod"] {{
    background-color: {VOD_ACCENT};
    border: 1px solid {VOD_ACCENT};
    color: {BACKGROUND};
}}

QSplitter::handle {{
    background-color: {HAIRLINE};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QWidget#LeftRail {{
    background-color: {PANEL};
    border-right: 1px solid {HAIRLINE};
}}

QFrame#RailCard {{
    background-color: {PANEL_ALT};
    border: 1px solid {HAIRLINE};
    border-radius: 6px;
}}

QLabel#RailCardTitle {{
    color: {TEXT_MUTED};
    font-size: 9px;
    font-weight: bold;
}}

QLabel#RailCardBody {{
    color: {TEXT_PRIMARY};
    font-size: 11px;
}}

QWidget#IncidentsDrawer {{
    background-color: {PANEL};
    border-left: 1px solid {HAIRLINE};
}}

QLabel#IncidentCollapseLabel {{
    background-color: {PANEL};
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: bold;
}}

QComboBox {{
    background-color: {PANEL};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: bold;
    min-width: 64px;
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {ACCENT};
    selection-background-color: {ACCENT_DIM};
    selection-color: {ACCENT};
    outline: none;
}}

QSlider::groove:horizontal {{
    background: {SURFACE};
    height: 6px;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {BORDER};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QTableView {{
    background-color: {PANEL};
    alternate-background-color: {PANEL_ALT};
    gridline-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    selection-color: {ACCENT};
    font-size: 11px;
}}

QHeaderView::section {{
    background-color: {SURFACE};
    color: {BORDER};
    padding: 4px;
    border: none;
    font-size: 10px;
    font-weight: bold;
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
}}
"""

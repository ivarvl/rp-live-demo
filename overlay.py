"""Draw the classification results onto a frame for the live stream."""

from __future__ import annotations

import cv2
import numpy as np

from inference import Prediction

_FONT = cv2.QT_FONT_NORMAL
_PANEL_BG = (0, 0, 0)
_BAR_BG = (60, 60, 60)
_BAR_FG = (0, 200, 90)
_TEXT = (255, 255, 255)
_DETECT = (0, 255, 120)  # bright green for a confident top-1
_DETECT_BG = (0, 80, 35)  # subtle fill behind the enlarged row

# Layout (pixels)
_PAD = 12
_HEADER_H = 26  # FPS line
_GAP = 24  # space between the label column and the bar
_LABEL_W = 200  # width reserved for label text
_BAR_W = 165
_BAR_X = _PAD + _LABEL_W + _GAP
_PANEL_W = _BAR_X + _BAR_W + _PAD


def _text_w(text: str, scale: float, thickness: int) -> int:
    return cv2.getTextSize(text, _FONT, scale, thickness)[0][0]


def _fit(text: str, max_w: int, scale: float, thickness: int) -> str:
    """Truncate text with an ASCII ellipsis so it fits within ``max_w`` pixels.

    (The Hershey fonts OpenCV ships only render ASCII, so a real '…' would
    come out as garbage — we use '..' instead.)
    """
    if _text_w(text, scale, thickness) <= max_w:
        return text
    while text and _text_w(text + "..", scale, thickness) > max_w:
        text = text[:-1]
    return text + ".." if text else ""


def _fit_label(
    text: str, max_w: int, max_scale: float, thickness: int, min_scale: float = 0.5
) -> tuple[str, float]:
    """Largest font (down to ``min_scale``) at which ``text`` fits ``max_w``.

    Lets short breed names render big while long ones shrink to stay readable;
    only names too long even at ``min_scale`` get truncated. Returns the text
    (possibly ellipsised) and the chosen scale.
    """
    scale = max_scale
    while scale > min_scale and _text_w(text, scale, thickness) > max_w:
        scale = round(scale - 0.05, 2)
    if _text_w(text, scale, thickness) > max_w:
        return _fit(text, max_w, scale, thickness), scale
    return text, scale


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def draw_results(
    frame: np.ndarray,
    preds: list[Prediction],
    cam_fps: float,
    infer_fps: float,
    threshold: float = 0.5,
) -> np.ndarray:
    """Overlay a semi-transparent panel with top-k bars and FPS readouts.

    ``cam_fps`` is the smooth stream rate; ``infer_fps`` is the (lower) rate the
    model actually updates predictions at — the two are decoupled by design.
    When the top-1 score reaches ``threshold`` (0-1) that row is enlarged and
    highlighted so the audience can see it has been detected.
    """

    # Per-row geometry, computed up front so we can size the panel and draw the
    # translucent background in one pass before painting text/bars on top.
    def emphasized(i: int, p: Prediction) -> bool:
        return i == 0 and p.score >= threshold

    rows = []
    y = _PAD + _HEADER_H
    for i, p in enumerate(preds):
        big = emphasized(i, p)
        row_h = 50 if big else 30
        rows.append((p, big, y, row_h))
        y += row_h
    panel_h = y + _PAD

    # Pass 1: translucent panel + per-row detection fill, blended in one shot.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (_PANEL_W, panel_h), _PANEL_BG, -1)
    for _p, big, ry, rh in rows:
        if big:
            cv2.rectangle(
                overlay, (4, ry + 3), (_PANEL_W - 4, ry + rh - 3), _DETECT_BG, -1
            )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Pass 2: opaque text, bars and the detection border.
    cv2.putText(
        frame,
        f"cam {cam_fps:4.1f}   infer {infer_fps:4.1f} FPS",
        (_PAD, _PAD + 12),
        _FONT,
        0.55,
        _TEXT,
        1,
        cv2.LINE_AA,
    )

    for p, big, ry, rh in rows:
        f_pct, thick, bar_h = (0.8, 2, 26) if big else (0.45, 1, 16)
        label_color = _DETECT if big else _TEXT
        bar_fg = _DETECT if big else _BAR_FG

        if big:  # frame the detected row so it reads from across the room
            cv2.rectangle(frame, (3, ry + 2), (_PANEL_W - 3, ry + rh - 2), _DETECT, 2)

        # Label, vertically centred. The detected row shrinks-to-fit so the
        # full breed name stays visible; other rows use a fixed size + ellipsis.
        if big:
            label, f_label = _fit_label(p.label, _LABEL_W, 0.95, thick)
        else:
            f_label = 0.55
            label = _fit(p.label, _LABEL_W, f_label, thick)
        (_, lh), _ = cv2.getTextSize(label, _FONT, f_label, thick)
        cv2.putText(
            frame,
            label,
            (_PAD, ry + (rh + lh) // 2),
            _FONT,
            f_label,
            label_color,
            thick,
            cv2.LINE_AA,
        )

        # Confidence bar.
        bar_y = ry + (rh - bar_h) // 2
        cv2.rectangle(
            frame, (_BAR_X, bar_y), (_BAR_X + _BAR_W, bar_y + bar_h), _BAR_BG, -1
        )
        fill = int(_BAR_W * _clamp01(p.score))
        cv2.rectangle(
            frame, (_BAR_X, bar_y), (_BAR_X + fill, bar_y + bar_h), bar_fg, -1
        )

        # Percentage, right-aligned inside the bar.
        pct = f"{p.score * 100:4.1f}%"
        (pw, ph), _ = cv2.getTextSize(pct, _FONT, f_pct, thick)
        cv2.putText(
            frame,
            pct,
            (_BAR_X + _BAR_W - pw - 6, bar_y + (bar_h + ph) // 2),
            _FONT,
            f_pct,
            _TEXT,
            thick,
            cv2.LINE_AA,
        )

    return frame

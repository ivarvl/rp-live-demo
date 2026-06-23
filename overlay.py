"""Draw the classification results onto a frame for the live stream."""

from __future__ import annotations

import cv2
import numpy as np

from inference import Prediction

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_BG = (0, 0, 0)
_BAR_BG = (60, 60, 60)
_BAR_FG = (0, 200, 90)
_TEXT = (255, 255, 255)


def draw_results(frame: np.ndarray, preds: list[Prediction], fps: float) -> np.ndarray:
    """Overlay a semi-transparent panel with top-k bars and an FPS readout."""
    h, w = frame.shape[:2]
    rows = len(preds)
    pad = 12
    row_h = 28
    panel_w = min(int(w * 0.46), 460)
    panel_h = pad * 2 + row_h * rows + 26

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"{fps:4.1f} FPS", (pad, 20), _FONT, 0.6, _TEXT, 1, cv2.LINE_AA)

    y = 26 + pad
    bar_x = pad + 150
    bar_w = panel_w - bar_x - pad
    for p in preds:
        label = p.label if len(p.label) <= 18 else p.label[:17] + "…"
        cv2.putText(frame, label, (pad, y + 18), _FONT, 0.55, _TEXT, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (bar_x, y + 5), (bar_x + bar_w, y + 21), _BAR_BG, -1)
        fill = int(bar_w * max(0.0, min(1.0, p.score)))
        cv2.rectangle(frame, (bar_x, y + 5), (bar_x + fill, y + 21), _BAR_FG, -1)
        cv2.putText(frame, f"{p.score * 100:4.1f}%", (bar_x + bar_w - 52, y + 18),
                    _FONT, 0.45, _TEXT, 1, cv2.LINE_AA)
        y += row_h

    return frame

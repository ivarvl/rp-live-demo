"""Thin wrapper around an OpenCV V4L2 webcam capture."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0, width: int | None = None,
                 height: int | None = None) -> None:
        # CAP_V4L2 is the right backend for USB webcams on Linux/Raspberry Pi.
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")

        # MJPG lets the webcam deliver higher resolutions/framerates over USB
        # than the default YUYV, which the Pi's USB bus often can't sustain.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep the buffer shallow so we always grab the freshest frame.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

"""MJPEG-over-HTTP server with decoupled capture and inference.

Two background threads share state:

* ``CaptureWorker`` reads the webcam as fast as it can, overlays the most
  recent predictions, encodes a JPEG, and publishes it. This drives the
  stream, so the feed stays smooth at the camera's framerate.
* ``InferenceWorker`` pulls the freshest raw frame whenever it is free, runs
  the ONNX model at whatever (lower) rate it can manage, and updates the
  shared predictions.

Any number of HTTP clients read the latest JPEG via a
multipart/x-mixed-replace stream, which every browser renders inside an <img>.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

from camera import Camera
from inference import Classifier, Prediction
from overlay import draw_results


class SharedState:
    """Hand-off point between the capture and inference threads.

    The capture thread publishes raw frames (by reference, never mutated) and
    the inference thread publishes predictions back. A sequence number lets
    inference block until a genuinely new frame exists, so it always works on
    the freshest frame instead of a backlog.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._preds: list[Prediction] = []
        self._inf_fps = 0.0

    def put_frame(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._frame_seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``last_seq`` is available."""
        with self._cond:
            if self._frame_seq == last_seq:
                self._cond.wait(timeout)
            return self._frame, self._frame_seq

    def set_predictions(self, preds: list[Prediction], inf_fps: float) -> None:
        with self._cond:
            self._preds = preds
            self._inf_fps = inf_fps

    def get_predictions(self) -> tuple[list[Prediction], float]:
        with self._cond:
            return self._preds, self._inf_fps


class FrameBroker:
    """Holds the latest encoded JPEG and wakes up waiting HTTP clients."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def wait_for_next(self, last_seq: int, timeout: float = 5.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


def _ema(prev: float, dt: float) -> float:
    """Smoothed FPS from an inter-frame delta, so the readout doesn't flicker."""
    inst = 1.0 / dt if dt > 0 else 0.0
    return inst if prev == 0 else prev * 0.9 + inst * 0.1


class CaptureWorker(threading.Thread):
    """Camera -> overlay latest predictions -> JPEG -> broker. Drives the stream."""

    def __init__(self, camera: Camera, state: SharedState, broker: FrameBroker,
                 jpeg_quality: int = 80, detect_threshold: float = 0.5,
                 mirror: bool = False) -> None:
        super().__init__(daemon=True, name="capture")
        self.camera = camera
        self.state = state
        self.broker = broker
        self.jpeg_quality = jpeg_quality
        self.detect_threshold = detect_threshold
        self.mirror = mirror
        self._stop_event = threading.Event()
        self.fps = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        last = time.monotonic()
        while not self._stop_event.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Flip horizontally before anything else, so the stream, the overlay
            # text, and the frame inference sees are all consistently mirrored.
            if self.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            self.fps = _ema(self.fps, now - last)
            last = now

            # Hand the raw frame to inference, then draw on a copy so the
            # model never sees the overlay we burn into the streamed image.
            self.state.put_frame(frame)
            preds, inf_fps = self.state.get_predictions()
            annotated = frame.copy()
            draw_results(annotated, preds, self.fps, inf_fps, self.detect_threshold)

            ok, buf = cv2.imencode(".jpg", annotated, encode_params)
            if ok:
                self.broker.publish(buf.tobytes())


class InferenceWorker(threading.Thread):
    """Freshest raw frame -> ONNX model -> shared predictions. Runs independently."""

    def __init__(self, classifier: Classifier, state: SharedState) -> None:
        super().__init__(daemon=True, name="inference")
        self.classifier = classifier
        self.state = state
        self._stop_event = threading.Event()
        self.fps = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        last_seq = 0
        last = time.monotonic()
        while not self._stop_event.is_set():
            frame, seq = self.state.wait_frame(last_seq)
            if frame is None or seq == last_seq:
                continue  # timed out waiting for a new frame
            last_seq = seq

            preds = self.classifier.infer(frame)

            now = time.monotonic()
            self.fps = _ema(self.fps, now - last)
            last = now
            self.state.set_predictions(preds, self.fps)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
  html,body{margin:0;height:100%;background:#000;display:flex;
            align-items:center;justify-content:center}
  img{max-width:100%;max-height:100%}
</style></head>
<body><img src="/stream" alt="live inference feed"></body></html>"""


def create_app(broker: FrameBroker, capture: CaptureWorker,
               inference: InferenceWorker, title: str) -> Flask:
    app = Flask(__name__)
    boundary = "frame"

    @app.route("/")
    def index():
        return render_template_string(_PAGE, title=title)

    @app.route("/healthz")
    def healthz():
        return {
            "ok": capture.is_alive() and inference.is_alive(),
            "cam_fps": round(capture.fps, 1),
            "infer_fps": round(inference.fps, 1),
        }

    @app.route("/stream")
    def stream():
        def generate():
            seq = -1
            while True:
                jpeg, seq = broker.wait_for_next(seq)
                if jpeg is None:
                    continue
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")

        return Response(generate(),
                        mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

    return app

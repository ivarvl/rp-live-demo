"""MJPEG-over-HTTP server.

A background worker continuously pulls frames from the webcam, runs ONNX
inference, annotates the frame, and stores the freshest JPEG. Any number of
HTTP clients can then read that JPEG via a multipart/x-mixed-replace stream,
which every browser renders natively inside an <img> tag.
"""

from __future__ import annotations

import threading
import time

import cv2
from flask import Flask, Response, render_template_string

from camera import Camera
from inference import Classifier
from overlay import draw_results


class FrameBroker:
    """Holds the latest encoded JPEG and wakes up waiting clients."""

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
        """Block until a frame newer than ``last_seq`` is available."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq


class InferenceWorker(threading.Thread):
    def __init__(self, camera: Camera, classifier: Classifier, broker: FrameBroker,
                 jpeg_quality: int = 80) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.classifier = classifier
        self.broker = broker
        self.jpeg_quality = jpeg_quality
        self._stop = threading.Event()
        self.fps = 0.0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        last = time.monotonic()
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            preds = self.classifier.infer(frame)

            now = time.monotonic()
            dt = now - last
            last = now
            # Exponential moving average keeps the readout from flickering.
            inst = 1.0 / dt if dt > 0 else 0.0
            self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1

            draw_results(frame, preds, self.fps)
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                self.broker.publish(buf.tobytes())


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
  html,body{margin:0;height:100%;background:#000;display:flex;
            align-items:center;justify-content:center}
  img{max-width:100%;max-height:100%}
</style></head>
<body><img src="/stream" alt="live inference feed"></body></html>"""


def create_app(broker: FrameBroker, worker: InferenceWorker, title: str) -> Flask:
    app = Flask(__name__)
    boundary = "frame"

    @app.route("/")
    def index():
        return render_template_string(_PAGE, title=title)

    @app.route("/healthz")
    def healthz():
        return {"ok": worker.is_alive(), "fps": round(worker.fps, 1)}

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

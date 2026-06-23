"""Entry point: webcam -> ONNX ViT classifier -> annotated MJPEG stream.

Run on the Raspberry Pi; view from the laptop's browser at
``http://<pi-ip>:8000/``. See README.md for the direct-Ethernet setup.
"""

from __future__ import annotations

import argparse

from waitress import serve

from camera import Camera
from inference import IMAGENET_MEAN, IMAGENET_STD, Classifier, load_labels
from server import (
    CaptureWorker,
    FrameBroker,
    InferenceWorker,
    SharedState,
    create_app,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="path to the .onnx classifier")
    p.add_argument("--labels", help="newline-separated class labels file")

    p.add_argument("--camera", default="0",
                   help="webcam index (e.g. 0) or device path (/dev/video0)")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--mirror", action="store_true",
                   help="flip the feed horizontally (selfie/mirror view)")

    p.add_argument("--input-size", type=int,
                   help="override model input side length (auto-detected otherwise)")
    p.add_argument("--no-normalize", action="store_true",
                   help="skip ImageNet mean/std normalisation")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--detect-threshold", type=float, default=0.5,
                   help="top-1 score (0-1) above which the result is enlarged "
                        "and highlighted as a detection")
    p.add_argument("--threads", type=int, default=4,
                   help="onnxruntime intra-op threads (Pi 5 has 4 cores)")

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--title", default="ViT Live Demo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera

    classifier = Classifier(
        model_path=args.model,
        labels=load_labels(args.labels),
        input_size=args.input_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        normalize=not args.no_normalize,
        num_threads=args.threads,
        topk=args.topk,
    )

    camera = Camera(source, width=args.cam_width, height=args.cam_height)
    broker = FrameBroker()
    state = SharedState()

    # Capture drives the smooth stream; inference runs independently and only
    # updates the predictions the capture loop overlays.
    capture = CaptureWorker(camera, state, broker, jpeg_quality=args.jpeg_quality,
                            detect_threshold=args.detect_threshold, mirror=args.mirror)
    inference = InferenceWorker(classifier, state)
    capture.start()
    inference.start()

    app = create_app(broker, capture, inference, title=args.title)
    import socket
    host_url = f"http://{socket.gethostname()}.local:{args.port}/"
    print(f"Serving live inference feed on port {args.port}")
    print(f"  From the laptop, open: {host_url}  (or http://<pi-ip>:{args.port}/)")
    try:
        # waitress is a production WSGI server; Flask's dev server is single
        # threaded and would stall the stream under load.
        serve(app, host=args.host, port=args.port, threads=8)
    finally:
        capture.stop()
        inference.stop()
        camera.release()


if __name__ == "__main__":
    main()

"""ONNX vision-transformer classification inference.

Wraps an onnxruntime session and turns a raw BGR camera frame into a
ranked list of ``(label, probability)`` pairs. Pre/post-processing is
auto-detected from the model where possible and otherwise configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Standard ImageNet normalisation. Most ViT checkpoints are trained with this.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Prediction:
    label: str
    score: float


@dataclass
class Classifier:
    model_path: str
    labels: list[str] | None = None
    input_size: int | None = None          # square side; auto-detected if None
    layout: str | None = None              # "NCHW" or "NHWC"; auto-detected
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    normalize: bool = True
    num_threads: int | None = None
    topk: int = 5

    session: ort.InferenceSession = field(init=False, repr=False)
    _input_name: str = field(init=False, repr=False)
    _h: int = field(init=False, repr=False)
    _w: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.num_threads:
            opts.intra_op_num_threads = self.num_threads

        # CPU is the only provider on a Raspberry Pi. Listing it explicitly
        # silences onnxruntime's "provider not available" warnings.
        self.session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self._detect_geometry(inp.shape)

    def _detect_geometry(self, shape: list) -> None:
        """Infer layout and spatial size from the model's input shape.

        Dynamic axes show up as strings or None; we fall back to a 224px
        square (the ViT default) and whatever the user supplied.
        """
        dims = [d if isinstance(d, int) else None for d in shape]

        if self.layout is None:
            # [N, C, H, W] has the channel dim at index 1; [N, H, W, C] at -1.
            if len(dims) == 4 and dims[1] == 3:
                self.layout = "NCHW"
            elif len(dims) == 4 and dims[-1] == 3:
                self.layout = "NHWC"
            else:
                self.layout = "NCHW"

        if self.layout == "NCHW":
            h, w = (dims[2], dims[3]) if len(dims) == 4 else (None, None)
        else:
            h, w = (dims[1], dims[2]) if len(dims) == 4 else (None, None)

        side = self.input_size or h or w or 224
        self._h = h or side
        self._w = w or side

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(frame_bgr, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.normalize:
            img = (img - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)

        if self.layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))
        return img[np.newaxis, ...]  # add batch dim

    def infer(self, frame_bgr: np.ndarray) -> list[Prediction]:
        x = self._preprocess(frame_bgr)
        logits = self.session.run(None, {self._input_name: x})[0]
        probs = _softmax(np.asarray(logits).reshape(-1))

        k = min(self.topk, probs.shape[0])
        top = np.argpartition(-probs, k - 1)[:k]
        top = top[np.argsort(-probs[top])]
        return [Prediction(self._label_for(i), float(probs[i])) for i in top]

    def _label_for(self, index: int) -> str:
        if self.labels and 0 <= index < len(self.labels):
            return self.labels[index]
        return f"class {index}"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_labels(path: str | None) -> list[str] | None:
    """Load newline-separated class labels, ignoring blank lines."""
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]

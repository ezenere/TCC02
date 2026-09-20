"""Live hand-gesture recognition from a webcam with one of the project's models.

    python app/webcam_demo.py                       # menu: pick a model, then the camera opens
    python app/webcam_demo.py --model runs/eixo1_resnet50_s0/checkpoints/best.pt
    python app/webcam_demo.py --model runs/eixo1_densenet121_s0/model_int8_fbgemm.pt --device cpu
    python app/webcam_demo.py --source pasta_ou_video_ou_imagem --headless   # no camera / no window

The model is chosen BEFORE the camera starts; to try another one, close and reopen.
Supported artifacts (detected by file name):
  checkpoints/*.pt        PyTorch checkpoint  (GPU if available, else CPU; pruned models included)
  model_int8_fbgemm.pt    TorchScript int8    (CPU only)
  model*.onnx             ONNX Runtime        (CPU)
  trt/*.engine            TensorRT engine     (GPU)

The networks were trained on a square crop around the hand (bbox + 10% margin), so the
app classifies what is inside the green square: put your hand there. There is no
"no gesture" class among the 18 — a low confidence is shown as "?" instead of a label.

Keys:  q / ESC quit   + / - resize the square   m mirror on/off   clique: move the square
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES_18 = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted",
              "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------- models
def discover_models() -> list[Path]:
    runs = ROOT / "runs"
    found = []
    for pattern in ("*/checkpoints/best.pt", "*/model_int8_fbgemm.pt", "*/trt/model_*.engine", "*/model.onnx"):
        found += sorted(runs.glob(pattern))
    return found


def describe(path: Path) -> str:
    run = path.parents[1] if path.parent.name in ("checkpoints", "trt") else path.parent
    kind = ("TensorRT " + path.stem.replace("model_", "") if path.suffix == ".engine" else
            "TorchScript int8 (CPU)" if path.name == "model_int8_fbgemm.pt" else
            "ONNX Runtime (CPU)" if path.suffix == ".onnx" else "PyTorch")
    extra = ""
    m = run / "metrics.json"
    if m.exists():
        d = json.loads(m.read_text())
        extra = f"  teste {100 * d['acc']:.2f}%"
        if d.get("prune_sparsity"):
            extra += f"  poda {int(round(100 * d['prune_sparsity']))}%"
    return f"{run.name:<42} {kind:<24}{extra}"


def choose_model() -> Path:
    models = discover_models()
    if not models:
        raise SystemExit("nenhum modelo encontrado em runs/ — passe --model CAMINHO")
    print("\nModelos disponíveis:\n")
    for i, p in enumerate(models, 1):
        print(f"  [{i:>2}] {describe(p)}")
    while True:
        ans = input(f"\nEscolha o modelo [1-{len(models)}] (Enter = 1): ").strip() or "1"
        if ans.isdigit() and 1 <= int(ans) <= len(models):
            return models[int(ans) - 1]
        print("opção inválida")


def run_classes(path: Path) -> list[str]:
    run = path.parents[1] if path.parent.name in ("checkpoints", "trt") else path.parent
    m = run / "metrics.json"
    if m.exists():
        return json.loads(m.read_text()).get("classes") or CLASSES_18
    return CLASSES_18


class Predictor:
    """Uniform predict(batch NCHW float32 numpy) -> logits numpy, whatever the artifact."""

    def __init__(self, path: Path, device: str):
        self.path, self.kind = path, None
        self.classes = run_classes(path)
        if path.suffix == ".engine":
            from compress.trt_runtime import TRTRunner
            self.kind, self.device = "tensorrt", "cuda"
            runner = TRTRunner(path)
            self._fn = lambda x: runner(torch.from_numpy(x).cuda()).float().cpu().numpy()
        elif path.suffix == ".onnx":
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            self.kind, self.device = "onnxruntime", "cpu"
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            self._fn = lambda x: sess.run(None, {name: x})[0]
        elif path.name == "model_int8_fbgemm.pt":
            self.kind, self.device = "torchscript-int8", "cpu"
            torch.backends.quantized.engine = "x86"
            model = torch.jit.load(str(path), map_location="cpu").eval()
            self._fn = lambda x: model(torch.from_numpy(x)).detach().numpy()
        else:
            from train import build_model
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.classes = ck.get("classes") or self.classes
            self.kind = "pytorch"
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(ck["config"]["model"]["arch"], len(self.classes), pretrained=False)
            model.load_state_dict(ck["model"])
            model.eval().to(self.device)
            half = self.device == "cuda"

            def fn(x):
                t = torch.from_numpy(x).to(self.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
                    return model(t).float().cpu().numpy()
            self._fn = fn

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Square BGR crop -> probabilities. Same geometry as evaluation: 256 then centre 224."""
        img = cv2.resize(crop_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)[16:240, 16:240]
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        logits = self._fn(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))[0]
        e = np.exp(logits - logits.max())
        return e / e.sum()


# ----------------------------------------------------------------------------- frames
def frame_source(source: str):
    """Yields (frame_bgr, name). Camera index, video file, image file or directory of images."""
    p = Path(source)
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_EXT):
            yield cv2.imread(str(f)), f.name
        return
    if p.is_file() and p.suffix.lower() in IMAGE_EXT:
        yield cv2.imread(str(p)), p.name
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir a câmera/vídeo '{source}'. Câmeras: ls /dev/video*")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame, "camera"
    finally:
        cap.release()


def roi_box(w: int, h: int, frac: float, center: tuple[float, float]) -> tuple[int, int, int]:
    side = int(min(w, h) * frac)
    x0 = int(min(max(center[0] * w - side / 2, 0), w - side))
    y0 = int(min(max(center[1] * h - side / 2, 0), h - side))
    return x0, y0, side


def draw(frame, box, probs, classes, threshold, info):
    x0, y0, side = box
    top = np.argsort(probs)[::-1][:3]
    sure = probs[top[0]] >= threshold
    color = (60, 200, 60) if sure else (0, 170, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), color, 2)
    label = f"{classes[top[0]]}  {100 * probs[top[0]]:.0f}%" if sure else "?"
    cv2.rectangle(frame, (x0, max(y0 - 38, 0)), (x0 + side, y0), color, -1)
    cv2.putText(frame, label, (x0 + 8, max(y0 - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    for i, k in enumerate(top):
        y = 30 + 28 * i
        cv2.rectangle(frame, (10, y - 16), (10 + int(220 * probs[k]), y + 4), (60, 200, 60), -1)
        cv2.putText(frame, f"{classes[k]} {100 * probs[k]:.0f}%", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, info, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help="artifact path; omit for the menu")
    ap.add_argument("--list", action="store_true", help="list the available models and exit")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="PyTorch checkpoints only")
    ap.add_argument("--source", default="0", help="camera index, video, image, or directory of images")
    ap.add_argument("--roi", type=float, default=0.6, help="side of the square as a fraction of the shorter frame side")
    ap.add_argument("--threshold", type=float, default=0.6, help="below this confidence the label is '?'")
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA factor over frames (0 = none)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window; print one line per frame")
    args = ap.parse_args()

    if args.list:
        for p in discover_models():
            print(f"{describe(p)}  ->  {p.relative_to(ROOT)}")
        return 0

    path = args.model or choose_model()
    print(f"\ncarregando {path} ...")
    predictor = Predictor(path, args.device)
    print(f"backend: {predictor.kind} em {predictor.device} | {len(predictor.classes)} classes")

    mirror, frac, center = not args.no_mirror, args.roi, [0.5, 0.5]
    ema, last, fps = None, time.perf_counter(), 0.0
    window = "TCC - gestos (q sai, +/- quadrado, m espelho, clique move)"
    frame_wh = [1, 1]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:           # re-centre the square where the user clicked
            center[0], center[1] = x / frame_wh[0], y / frame_wh[1]

    if not args.headless:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)

    for frame, name in frame_source(args.source):
        if frame is None:
            continue
        static = name != "camera"
        if mirror and not static:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_wh[:] = [w, h]
        box = roi_box(w, h, 1.0 if static and args.roi >= 1.0 else frac, tuple(center))
        x0, y0, side = box
        t0 = time.perf_counter()
        probs = predictor(frame[y0:y0 + side, x0:x0 + side])
        ms = 1000 * (time.perf_counter() - t0)
        ema = probs if (ema is None or static or args.smooth <= 0) else args.smooth * ema + (1 - args.smooth) * probs
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now

        if args.headless:
            k = int(np.argmax(ema))
            print(f"{name:<44} -> {predictor.classes[k]:<16} {100 * ema[k]:5.1f}%  ({ms:.1f} ms)")
            continue
        draw(frame, box, ema, predictor.classes, args.threshold,
             f"{path.parent.parent.name if path.parent.name in ('checkpoints', 'trt') else path.parent.name} | "
             f"{predictor.kind}/{predictor.device} | {ms:.1f} ms | {fps:.0f} fps")
        cv2.imshow(window, frame)
        key = cv2.waitKey(0 if static else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            frac = min(frac + 0.05, 1.0)
        if key == ord("-"):
            frac = max(frac - 0.05, 0.2)
        if key == ord("m"):
            mirror = not mirror
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

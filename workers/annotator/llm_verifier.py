"""
Vision LLM label verification.

Stage 3 of the three-stage annotation pipeline.
Crops each annotated region, sends it to a Vision LLM (Qwen-VL-Chat or LLaVA),
and asks: "Is this a <class_name>? Answer YES or NO."

Supports two backends (tried in order):
  1. transformers + Qwen-VL-Chat  (local, requires GPU recommended)
  2. ollama REST API              (local server, e.g. `ollama run llava`)

Falls back to pass-through (accept all) if neither backend is available.
"""
import logging
import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from workers.annotator.yolo_annotator import BoundingBox
from workers.annotator.sam2_refiner import RefinedAnnotation

logger = logging.getLogger(__name__)

# Padding fraction added around each crop before sending to the LLM
_CROP_PADDING = 0.10


@dataclass
class VerifiedAnnotation:
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)
    rejected_count: int = 0
    backend: str = "passthrough"   # "qwen-vl" | "ollama" | "passthrough"


# ─── Crop helper ──────────────────────────────────────────────────────────────

def _crop_region(img_rgb: np.ndarray, box: BoundingBox, padding: float = _CROP_PADDING) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    cx, cy, bw, bh = box.x_center, box.y_center, box.width, box.height
    x1 = max(0, int((cx - bw / 2 - padding * bw) * w))
    y1 = max(0, int((cy - bh / 2 - padding * bh) * h))
    x2 = min(w,  int((cx + bw / 2 + padding * bw) * w))
    y2 = min(h,  int((cy + bh / 2 + padding * bh) * h))
    return img_rgb[y1:y2, x1:x2]


def _crop_to_b64(crop: np.ndarray) -> str:
    """Convert numpy RGB crop to base64-encoded JPEG string."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _is_positive(text: str) -> bool:
    """Parse YES/NO from LLM response (tolerant of extra text)."""
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return True
    if lowered.startswith("no"):
        return False
    # Fallback: look for yes/no anywhere
    return "yes" in lowered and "no" not in lowered


# ─── Backend: Qwen-VL-Chat ────────────────────────────────────────────────────

class _QwenVLBackend:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = "Qwen/Qwen-VL-Chat"
        logger.info(f"Loading {model_name}…")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.name = "qwen-vl"

    def ask(self, class_name: str, crop_b64: str) -> bool:
        import tempfile, os
        from PIL import Image as PILImage

        # Write crop to temp file (Qwen-VL needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(base64.b64decode(crop_b64))
            tmp_path = f.name

        try:
            query = self.tokenizer.from_list_format([
                {"image": tmp_path},
                {"text": f"Is the main subject in this image a {class_name}? Answer YES or NO only."},
            ])
            response, _ = self.model.chat(self.tokenizer, query=query, history=None)
            return _is_positive(response)
        finally:
            os.unlink(tmp_path)


# ─── Backend: Ollama REST ─────────────────────────────────────────────────────

class _OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava"):
        import urllib.request
        # Quick connectivity check
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            pass
        self.base_url = base_url
        self.model = model
        self.name = "ollama"
        logger.info(f"Ollama backend ready ({model} @ {base_url})")

    def ask(self, class_name: str, crop_b64: str) -> bool:
        import json, urllib.request

        payload = json.dumps({
            "model": self.model,
            "prompt": f"Is the main subject in this image a {class_name}? Answer YES or NO only.",
            "images": [crop_b64],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return _is_positive(data.get("response", ""))


# ─── Backend: CLIP zero-shot ───────────────────────────────────────────────────

class _ClipZeroShotBackend:
    """
    CPU-friendly Stage-3 substitute for Qwen-VL/LLaVA: classifies a crop via
    CLIP zero-shot similarity between the crop and a small set of text prompts
    ("a photo of a {class_name}" vs. generic distractor prompts). Used when no
    local Vision-LLM (Qwen-VL, Ollama) is available.
    """

    _DISTRACTORS = ["a photo of a road", "a photo of a person",
                    "a photo of a car", "a blurry background image"]

    def __init__(self, threshold: float = 0.0):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.threshold = threshold
        self.name = "clip-zeroshot"

    def ask(self, class_name: str, crop_b64: str) -> bool:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(base64.b64decode(crop_b64))).convert("RGB")
        image_input = self.preprocess(img).unsqueeze(0)

        prompts = [f"a photo of a {class_name}"] + self._DISTRACTORS
        text_input = self.tokenizer(prompts)

        with self.torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)

        best_idx = int(sims.argmax())
        return best_idx == 0


# ─── Backend loader ───────────────────────────────────────────────────────────

def _load_backend(ollama_url: str, ollama_model: str):
    """Try Qwen-VL, then Ollama, then CLIP zero-shot; return None if all fail."""
    try:
        return _QwenVLBackend()
    except Exception as e:
        logger.debug(f"Qwen-VL unavailable: {e}")

    try:
        return _OllamaBackend(base_url=ollama_url, model=ollama_model)
    except Exception as e:
        logger.debug(f"Ollama unavailable: {e}")

    try:
        return _ClipZeroShotBackend()
    except Exception as e:
        logger.debug(f"CLIP zero-shot unavailable: {e}")

    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def verify_with_llm(
    refined_results: list[RefinedAnnotation],
    confidence_threshold: float = 0.5,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "llava",
    min_crop_px: int = 32,
) -> list[VerifiedAnnotation]:
    """
    Verify each annotated bounding box with a Vision LLM.

    Boxes that the LLM classifies as NOT the expected class are rejected.
    Low-confidence boxes (below confidence_threshold) are always rejected
    without calling the LLM.

    Falls back to pass-through when no LLM backend is available.

    Args:
        refined_results: Output from refine_with_sam2().
        confidence_threshold: Minimum YOLO confidence to keep a box.
        ollama_url: Ollama server base URL.
        ollama_model: Ollama model tag (e.g. "llava", "bakllava").
        min_crop_px: Skip LLM call if crop is smaller than this in either dimension.

    Returns:
        List of VerifiedAnnotation, one per input.
    """
    import cv2

    backend = _load_backend(ollama_url, ollama_model)
    backend_name = backend.name if backend else "passthrough"

    if backend is None:
        logger.warning("No Vision LLM backend available — passing all annotations through")

    verified_results: list[VerifiedAnnotation] = []

    for refined in refined_results:
        if not refined.boxes:
            verified_results.append(VerifiedAnnotation(
                image_path=refined.image_path,
                boxes=[],
                backend=backend_name,
            ))
            continue

        # Confidence pre-filter (no LLM needed)
        kept = [b for b in refined.boxes if b.confidence >= confidence_threshold]
        rejected_by_conf = len(refined.boxes) - len(kept)

        if backend is None:
            verified_results.append(VerifiedAnnotation(
                image_path=refined.image_path,
                boxes=kept,
                rejected_count=rejected_by_conf,
                backend="passthrough",
            ))
            continue

        # Read image for cropping
        img_bgr = cv2.imread(refined.image_path)
        if img_bgr is None:
            verified_results.append(VerifiedAnnotation(
                image_path=refined.image_path,
                boxes=kept,
                rejected_count=rejected_by_conf,
                backend=backend_name,
            ))
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        verified_boxes: list[BoundingBox] = []
        rejected_by_llm = 0

        for box in kept:
            crop = _crop_region(img_rgb, box)
            if crop.shape[0] < min_crop_px or crop.shape[1] < min_crop_px:
                # Too small — trust YOLO
                verified_boxes.append(box)
                continue

            try:
                crop_b64 = _crop_to_b64(crop)
                is_correct = backend.ask(box.class_name, crop_b64)
                if is_correct:
                    verified_boxes.append(box)
                else:
                    rejected_by_llm += 1
                    logger.debug(f"LLM rejected '{box.class_name}' box in {refined.image_path}")
            except Exception as e:
                logger.warning(f"LLM call failed ({e}), keeping box")
                verified_boxes.append(box)

        verified_results.append(VerifiedAnnotation(
            image_path=refined.image_path,
            boxes=verified_boxes,
            rejected_count=rejected_by_conf + rejected_by_llm,
            backend=backend_name,
        ))

    return verified_results

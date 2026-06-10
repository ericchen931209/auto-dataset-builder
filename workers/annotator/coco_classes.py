"""Canonical COCO-80 class names (the vocabulary of the pretrained YOLOv11n
checkpoint used by Stage 1's "yolo11" backend), plus a helper to decide
whether a requested class list is covered by it.
"""

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_COCO_CLASSES_LOWER = {c.lower() for c in COCO_CLASSES}


def select_detector_backend(target_classes: list[str] | None) -> str:
    """
    Pick Stage 1's detector backend based on the requested class names.

    Returns "yolo11" if every requested class is in the COCO-80 vocabulary
    (or no classes are specified), otherwise "yolo_world" — so callers get
    open-vocabulary detection automatically for any class outside COCO-80,
    without needing to choose a backend themselves.
    """
    if not target_classes:
        return "yolo11"
    if all(c.lower() in _COCO_CLASSES_LOWER for c in target_classes):
        return "yolo11"
    return "yolo_world"

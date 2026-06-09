import os
import logging
from pathlib import Path

from PIL import Image
import imagehash

logger = logging.getLogger(__name__)

# Images with pHash distance <= this are considered duplicates
DEFAULT_HASH_THRESHOLD = 8


def remove_duplicates(
    image_dir: str,
    threshold: int = DEFAULT_HASH_THRESHOLD,
) -> dict:
    """
    Scan all images in image_dir, compute pHash for each, and remove duplicates.

    Returns a summary dict:
      {
        "total": int,
        "kept": int,
        "removed": int,
        "removed_files": [str, ...]
      }
    """
    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    hashes: dict[str, str] = {}  # path -> hash string
    removed: list[str] = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            h = imagehash.phash(img)
        except Exception as e:
            logger.warning(f"Cannot hash {path}: {e} — skipping")
            continue

        is_duplicate = False
        for existing_path, existing_hash_str in hashes.items():
            existing_hash = imagehash.hex_to_hash(existing_hash_str)
            if abs(h - existing_hash) <= threshold:
                is_duplicate = True
                logger.debug(f"Duplicate: {path.name} ≈ {existing_path}")
                break

        if is_duplicate:
            os.remove(path)
            removed.append(str(path))
        else:
            hashes[str(path)] = str(h)

    return {
        "total": len(image_paths),
        "kept": len(image_paths) - len(removed),
        "removed": len(removed),
        "removed_files": removed,
    }

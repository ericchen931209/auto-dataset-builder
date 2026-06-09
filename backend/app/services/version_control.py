"""
Dataset version control — snapshot creation and diff between versions.
"""
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class VersionDiff:
    from_tag: str
    to_tag: str
    added: list[str]       # image filenames added in to_tag
    removed: list[str]     # image filenames removed in to_tag
    unchanged: int
    total_from: int
    total_to: int


def _file_hash(path: str) -> str:
    """MD5 of file content — fast enough for per-image checks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(images_dir: str, labels_dir: str) -> dict[str, str]:
    """Return {filename: md5} for all images in the directory."""
    manifest = {}
    img_dir = Path(images_dir)
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            manifest[p.name] = _file_hash(str(p))
    return manifest


def create_snapshot(
    dataset_id: int,
    version_tag: str,
    images_dir: str,
    labels_dir: str,
    snapshots_root: str,
    notes: str = "",
) -> dict:
    """
    Create a versioned snapshot (zip archive) of the current dataset state.

    Snapshot layout inside zip::

        {dataset_id}/{version_tag}/
          images/
          labels/
          manifest.json    (filename → md5 map)
          meta.json        (version metadata)

    Args:
        dataset_id: DB id of the dataset.
        version_tag: Human-readable tag e.g. "v1.2".
        images_dir: Current images directory.
        labels_dir: Current labels directory.
        snapshots_root: Where to store snapshot zips.
        notes: Optional release notes.

    Returns:
        dict with snapshot_path, checksum, total_images, manifest.
    """
    snap_dir = Path(snapshots_root) / str(dataset_id) / version_tag
    snap_dir.mkdir(parents=True, exist_ok=True)

    img_out = snap_dir / "images"
    lbl_out = snap_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)

    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)

    manifest: dict[str, str] = {}
    image_count = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(str(img_path), img_out / img_path.name)
        manifest[img_path.name] = _file_hash(str(img_path))
        image_count += 1

        lbl_src = lbl_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(str(lbl_src), lbl_out / lbl_src.name)

    (snap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (snap_dir / "meta.json").write_text(json.dumps({
        "dataset_id": dataset_id,
        "version_tag": version_tag,
        "total_images": image_count,
        "notes": notes,
        "created_at": datetime.utcnow().isoformat(),
    }, indent=2))

    zip_path = str(snap_dir) + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in snap_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(snap_dir.parent.parent))

    shutil.rmtree(str(snap_dir))   # keep only zip

    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    checksum = h.hexdigest()

    return {
        "snapshot_path": zip_path,
        "checksum": checksum,
        "total_images": image_count,
        "manifest": manifest,
    }


def diff_versions(
    snapshot_a_path: str,
    snapshot_b_path: str,
    tag_a: str,
    tag_b: str,
) -> VersionDiff:
    """
    Compare two snapshot zip files and return what changed.

    Reads manifest.json from each zip to compute the diff without
    extracting all image files.
    """
    def _load_manifest(zip_path: str) -> dict[str, str]:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            manifest_name = next(
                (n for n in names if n.endswith("manifest.json")), None
            )
            if manifest_name is None:
                return {}
            return json.loads(zf.read(manifest_name))

    manifest_a = _load_manifest(snapshot_a_path)
    manifest_b = _load_manifest(snapshot_b_path)

    files_a = set(manifest_a.keys())
    files_b = set(manifest_b.keys())

    added   = sorted(files_b - files_a)
    removed = sorted(files_a - files_b)
    common  = files_a & files_b
    unchanged = sum(1 for f in common if manifest_a[f] == manifest_b[f])

    return VersionDiff(
        from_tag=tag_a,
        to_tag=tag_b,
        added=added,
        removed=removed,
        unchanged=unchanged,
        total_from=len(files_a),
        total_to=len(files_b),
    )

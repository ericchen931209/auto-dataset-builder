"""
YouTube video downloader for dataset construction.

COPYRIGHT NOTICE
─────────────────────────────────────────────────────────────────────────────
Downloaded videos are used SOLELY for extracting still frames as training
data for computer vision models (research / academic purpose).

By default this module only downloads videos with a Creative Commons (CC)
license (license_filter="creativecommons"). Passing license_filter=None
will download ALL videos — only do this if you have verified usage rights
or if the dataset is strictly for private research and will NOT be
redistributed publicly.

References:
  - YouTube Terms of Service: https://www.youtube.com/t/terms
  - Creative Commons: https://creativecommons.org/licenses/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import yt_dlp

logger = logging.getLogger(__name__)

# Allowed license values returned by yt-dlp
CC_LICENSE_KEYWORDS = ("creativecommon", "cc-by", "cc by")


@dataclass
class DownloadResult:
    video_id: str
    title: str
    output_path: str
    duration: float
    license: str
    success: bool
    error: str = ""


def _is_cc_license(license_str: str | None) -> bool:
    if not license_str:
        return False
    ls = license_str.lower()
    return any(kw in ls for kw in CC_LICENSE_KEYWORDS)


def download_videos(
    keywords: list[str],
    output_dir: str,
    max_videos: int = 5,
    resolution: str = "720",
    license_filter: str | None = "creativecommons",
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[DownloadResult]:
    """
    Search YouTube for each keyword and download up to max_videos total.

    Args:
        license_filter: "creativecommons" → only download CC-licensed videos (default).
                        None → download any video (use only for private research).
    Returns:
        List of DownloadResult for successfully downloaded videos.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    per_keyword = max(1, max_videos // max(len(keywords), 1))

    for keyword in keywords:
        if len(results) >= max_videos:
            break

        # yt-dlp supports "ytsearchX:query" syntax
        search_query = f"ytsearch{per_keyword * 3}:{keyword}"  # fetch more to filter

        ydl_opts = {
            "format": f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "merge_output_format": "mp4",
        }

        # Apply CC license filter at the yt-dlp level when possible
        if license_filter == "creativecommons":
            ydl_opts["match_filter"] = yt_dlp.utils.match_filter_func(
                "license ~= 'Creative Commons'"
            )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_list = ydl.extract_info(search_query, download=False)
                entries = info_list.get("entries", [info_list]) if info_list else []

                to_download = []
                for info in entries:
                    if info is None:
                        continue
                    lic = info.get("license", "")

                    # Secondary check: if match_filter didn't catch it, verify manually
                    if license_filter == "creativecommons" and not _is_cc_license(lic):
                        logger.debug(
                            f"Skipping '{info.get('title', '')}' — license: '{lic}'"
                        )
                        continue

                    to_download.append(info)
                    if len(to_download) + len(results) >= max_videos:
                        break

                for info in to_download:
                    if len(results) >= max_videos:
                        break
                    video_id = info.get("id", "unknown")
                    try:
                        ydl.download([info["webpage_url"]])
                    except Exception as e:
                        results.append(DownloadResult(
                            video_id=video_id,
                            title=info.get("title", ""),
                            output_path="",
                            duration=info.get("duration", 0.0),
                            license=info.get("license", ""),
                            success=False,
                            error=str(e),
                        ))
                        continue

                    ext = "mp4"
                    output_path = os.path.join(output_dir, f"{video_id}.{ext}")
                    results.append(DownloadResult(
                        video_id=video_id,
                        title=info.get("title", ""),
                        output_path=output_path,
                        duration=info.get("duration", 0.0),
                        license=info.get("license", ""),
                        success=os.path.exists(output_path),
                    ))

                    if progress_callback:
                        progress_callback(
                            len(results) / max_videos,
                            f"Downloaded: {info.get('title', video_id)} [{info.get('license', 'unknown license')}]"
                        )

        except Exception as e:
            logger.warning(f"YouTube download failed for keyword '{keyword}': {e}")

    skipped_no_cc = sum(1 for r in results if not r.success)
    logger.info(
        f"YouTube: {len([r for r in results if r.success])} downloaded, "
        f"{skipped_no_cc} failed, filter='{license_filter}'"
    )
    return results

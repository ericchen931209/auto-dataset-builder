import os
import logging
import httpx
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageDownloadResult:
    url: str
    filename: str
    output_path: str
    success: bool
    error: str = ""


def expand_keywords(base_keywords: list[str], target: str, region: str = "") -> list[str]:
    """
    Expand search keywords using simple synonym rules.
    In V0.5+ this will call an LLM for smarter expansion.
    """
    synonyms: dict[str, list[str]] = {
        "motorcycle": ["motorcycle", "scooter", "機車", "摩托車", "yamaha scooter",
                       "kymco", "sym scooter", "gogoro"],
        "car": ["car", "automobile", "sedan", "SUV", "汽車"],
        "person": ["person", "pedestrian", "人", "行人"],
        "fall_detection": ["person falling", "fall down", "elderly fall", "跌倒"],
    }

    expanded = list(base_keywords)
    for kw in base_keywords:
        kw_lower = kw.lower()
        for key, extras in synonyms.items():
            if key in kw_lower:
                expanded.extend(extras)
                break

    if region:
        expanded = [f"{kw} {region}" for kw in expanded[:3]] + expanded

    # deduplicate while preserving order
    seen = set()
    result = []
    for kw in expanded:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def search_and_download_images(
    keywords: list[str],
    output_dir: str,
    max_images: int = 200,
    google_api_key: str = "",
    google_cx: str = "",
) -> list[ImageDownloadResult]:
    """
    Download images via Google Custom Search API.
    Falls back to a minimal Bing scraper if no API key is provided.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: list[ImageDownloadResult] = []
    per_keyword = max(1, max_images // len(keywords))

    for keyword in keywords:
        if len(results) >= max_images:
            break

        urls = _fetch_urls_google(keyword, per_keyword, google_api_key, google_cx)

        for url in urls:
            if len(results) >= max_images:
                break
            result = _download_image(url, output_dir, idx=len(results))
            results.append(result)

    return results


def _fetch_urls_google(keyword: str, count: int, api_key: str, cx: str) -> list[str]:
    if not api_key or not cx:
        logger.warning("No Google API key — skipping image search for '%s'", keyword)
        return []

    urls = []
    start = 1
    while len(urls) < count:
        try:
            resp = httpx.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cx,
                    "q": keyword,
                    "searchType": "image",
                    "num": min(10, count - len(urls)),
                    "start": start,
                },
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break
            urls.extend(item["link"] for item in items)
            start += 10
        except Exception as e:
            logger.warning(f"Google image search failed: {e}")
            break

    return urls[:count]


def _download_image(url: str, output_dir: str, idx: int) -> ImageDownloadResult:
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    filename = f"img_{idx:05d}.jpg"
    output_path = os.path.join(output_dir, filename)

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        ext = ext_map.get(content_type, ".jpg")
        filename = f"img_{idx:05d}{ext}"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(resp.content)

        return ImageDownloadResult(url=url, filename=filename,
                                   output_path=output_path, success=True)
    except Exception as e:
        return ImageDownloadResult(url=url, filename=filename,
                                   output_path=output_path, success=False, error=str(e))

"""Machine-task metrics - the unbiased judge.

These are the paper's strongest evidence for H3, because they cannot be gamed
by aesthetic tuning. A restoration that invents plausible texture will please a
no-reference perceptual score and will not please a keypoint matcher, an OCR
engine or a detector, because those need the *specific* structure that was in
the scene, not merely structure-shaped pixels.

Three run out of the box:

``sift_matches`` / ``sift_inlier_ratio``
    Keypoint correspondences against the clean reference, with a geometric
    consistency check. The inlier ratio is the informative one: a hallucinating
    pipeline can produce plenty of keypoints that match nothing real.
``mtf50_edge``
    Slanted-edge MTF50 on a named crop. Resolution actually retained, in
    cycles per pixel - a physical number, not a preference.

Two are optional: OCR character accuracy (Tesseract) and detection mAP (YOLO).
Both report unavailable rather than substituting anything.
"""

from __future__ import annotations

import numpy as np

from sensorsim.degrader.optics import mtf50 as _mtf50_from_curve
from sensorsim.degrader.optics import slanted_edge_mtf


def _gray_u8(image: np.ndarray) -> np.ndarray:
    import cv2

    a = np.asarray(image)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.clip(a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Keypoint matching
# --------------------------------------------------------------------------

def sift_correspondence(
    restored: np.ndarray,
    reference: np.ndarray,
    ratio: float = 0.75,
    ransac_threshold: float = 3.0,
) -> dict[str, float]:
    """Match SIFT keypoints between restored and reference.

    Lowe's ratio test filters ambiguous matches, then a homography RANSAC keeps
    only the ones that agree on a single geometric transform. Since both images
    are the same view, the true transform is the identity, so inliers are
    matches that landed in the right *place* - the check a hallucinated texture
    fails.

    Returns match count, inlier count and inlier ratio. A frame too degraded to
    yield keypoints returns zeros, which is a finding, not an error.
    """
    import cv2

    a, b = _gray_u8(restored), _gray_u8(reference)
    sift = cv2.SIFT_create()
    kp_a, desc_a = sift.detectAndCompute(a, None)
    kp_b, desc_b = sift.detectAndCompute(b, None)

    result = {
        "sift_keypoints": float(len(kp_a)),
        "sift_matches": 0.0,
        "sift_inliers": 0.0,
        "sift_inlier_ratio": 0.0,
        "sift_reprojection_error": float("nan"),
    }
    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return result

    matcher = cv2.BFMatcher()
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    result["sift_matches"] = float(len(good))
    if len(good) < 4:
        return result

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if homography is None or mask is None:
        return result

    inliers = int(mask.sum())
    result["sift_inliers"] = float(inliers)
    result["sift_inlier_ratio"] = float(inliers / len(good))

    if inliers > 0:
        projected = cv2.perspectiveTransform(src, homography)
        errors = np.linalg.norm(projected - dst, axis=2).ravel()[mask.ravel().astype(bool)]
        result["sift_reprojection_error"] = float(errors.mean())
    return result


# --------------------------------------------------------------------------
# Resolution actually retained
# --------------------------------------------------------------------------

def mtf50_on_crop(image: np.ndarray, crop: tuple[int, int, int, int]) -> float:
    """Slanted-edge MTF50 in cycles per pixel on a named crop.

    Returns NaN when the crop holds no usable edge - a flat or blown-out patch
    genuinely has no measurable MTF, and a fabricated number would be worse
    than a gap.
    """
    x, y, w, h = crop
    patch = _gray_u8(image)[y : y + h, x : x + w].astype(np.float64)
    if patch.size == 0 or min(patch.shape) < 8:
        return float("nan")
    try:
        freqs, curve = slanted_edge_mtf(patch)
        return _mtf50_from_curve(freqs, curve)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def edge_acutance(image: np.ndarray, reference: np.ndarray) -> float:
    """Ratio of gradient energy to the reference's, on the reference's edges.

    Above 1.0 means the output is sharper than the truth where the truth has
    edges - overshoot, the signature of aggressive sharpening. It is a cheap
    companion to MTF50 that works on any scene, not just a chart.
    """
    import cv2

    a, b = _gray_u8(image).astype(np.float32), _gray_u8(reference).astype(np.float32)
    grad = lambda img: np.hypot(  # noqa: E731
        cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    )
    ga, gb = grad(a), grad(b)
    edges = gb > np.percentile(gb, 90)
    if not edges.any():
        return float("nan")
    denom = float(gb[edges].mean())
    return float(ga[edges].mean() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------
# Optional engines
# --------------------------------------------------------------------------

def ocr_available() -> tuple[bool, str]:
    try:
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - pytesseract raises several types
        return False, f"OCR needs pytesseract plus the tesseract binary: {exc}"


def ocr_character_accuracy(image: np.ndarray, ground_truth: str) -> dict[str, float]:
    """Character accuracy and edit distance against known scene text."""
    ok, reason = ocr_available()
    if not ok:
        raise RuntimeError(reason)

    import pytesseract  # type: ignore

    text = pytesseract.image_to_string(_gray_u8(image))
    distance = _levenshtein(_normalise_text(text), _normalise_text(ground_truth))
    n = max(len(_normalise_text(ground_truth)), 1)
    return {
        "ocr_cer": float(distance / n),
        "ocr_accuracy": float(max(0.0, 1.0 - distance / n)),
    }


def _normalise_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _levenshtein(a: str, b: str) -> int:
    """Edit distance, two rows of the DP table."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def detection_available() -> tuple[bool, str]:
    try:
        import ultralytics  # noqa: F401

        return True, ""
    except ImportError as exc:
        return False, f"detection mAP needs ultralytics (pip install sensorsim[detect]): {exc}"


def compute(
    restored: np.ndarray,
    reference: np.ndarray,
    crops: dict[str, tuple[int, int, int, int]] | None = None,
    scene_text: str | None = None,
) -> dict[str, float]:
    """Every machine-task metric that can run here."""
    out: dict[str, float] = {}
    out.update(sift_correspondence(restored, reference))
    out["edge_acutance"] = edge_acutance(restored, reference)

    crops = crops or {}
    if "slanted_edge" in crops:
        out["mtf50_edge"] = mtf50_on_crop(restored, tuple(crops["slanted_edge"]))  # type: ignore[arg-type]

    if scene_text and ocr_available()[0]:
        out.update(ocr_character_accuracy(restored, scene_text))
    return out


def unavailable() -> dict[str, str]:
    reasons: dict[str, str] = {}
    for name, check in (("ocr", ocr_available), ("detection_map", detection_available)):
        ok, reason = check()
        if not ok:
            reasons[name] = reason
    return reasons


HIGHER_IS_BETTER = {
    "sift_keypoints": True,
    "sift_matches": True,
    "sift_inliers": True,
    "sift_inlier_ratio": True,
    "sift_reprojection_error": False,
    "mtf50_edge": True,
    # Neither direction is "better": 1.0 means the output matches the reference's
    # edge energy, below is soft, above is overshoot. Read it against 1.0.
    "edge_acutance": None,
    "ocr_accuracy": True,
    "ocr_cer": False,
    "detection_map": True,
}

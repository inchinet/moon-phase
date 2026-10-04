import os
import shutil
import cv2
import numpy as np
from flask import Flask, jsonify, send_from_directory, request, Response
import threading

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

app = Flask(__name__, static_folder=".")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

processing_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "results": [],
    "error": None,
    "input_dir": "",
    "output_dir": "",
    "reject_dir": "",
}


# ── EXIF helpers ──────────────────────────────────────────────────────────────

def get_focal_length(fpath):
    """Return focal length in mm as a float, or None if unavailable."""
    if not HAS_PIEXIF:
        return None
    try:
        exif = piexif.load(fpath)
        fl = exif.get("Exif", {}).get(piexif.ExifIFD.FocalLength)
        if fl and fl[1] != 0:
            return fl[0] / fl[1]
    except Exception:
        pass
    return None


# ── Moon detection ────────────────────────────────────────────────────────────

def sample_background_color(img_bgr, cx, cy, radius):
    """Sample sky colour from a ring just outside the moon edge."""
    h, w = img_bgr.shape[:2]
    sample_cx = max(radius + 20, min(w - radius - 20, cx))
    sample_cy = max(radius + 20, min(h - radius - 20, cy))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - sample_cx) ** 2 + (yy - sample_cy) ** 2)
    ring_mask = (dist >= radius + 20) & (dist <= radius + 120)
    pixels = img_bgr[ring_mask]
    if len(pixels) >= 20:
        return (int(np.median(pixels[:, 0])),
                int(np.median(pixels[:, 1])),
                int(np.median(pixels[:, 2])))
    strip_h = min(80, h // 10)
    sample = img_bgr[:strip_h, :].reshape(-1, 3)
    return (int(np.median(sample[:, 0])),
            int(np.median(sample[:, 1])),
            int(np.median(sample[:, 2])))


def _hough_detect(img_bgr, min_r, max_r):
    """
    Detect moon circle using Canny edges + HoughCircles.
    Returns (cx, cy, r) or None.

    Strategy:
      - Downscale the image for speed (Hough is slow on 4K images)
      - Apply Canny on the grayscale to find edges regardless of brightness
      - HoughCircles with gradient accumulator
      - Return result scaled back to original coordinates
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1200.0 / max(h, w))
    sw = max(1, int(w * scale))
    sh = max(1, int(h * scale))

    small = cv2.resize(img_bgr, (sw, sh))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (9, 9), 2)

    s_min_r = max(20, int(min_r * scale))
    s_max_r = min(int(max(sh, sw) * 0.9), int(max_r * scale))

    # Try with different Canny thresholds to handle varying contrast
    for canny_lo, canny_hi in [(30, 100), (20, 60), (10, 40)]:
        edges = cv2.Canny(blur, canny_lo, canny_hi)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(sh, sw) // 3,
            param1=50,
            param2=15,
            minRadius=s_min_r,
            maxRadius=s_max_r,
        )
        if circles is not None:
            # Pick the circle with the most edge support
            best = _best_supported_circle(circles[0], edges, s_min_r, s_max_r)
            if best is not None:
                scx, scy, sr = best
                return (int(round(scx / scale)),
                        int(round(scy / scale)),
                        int(round(sr / scale)))
    return None


def _best_supported_circle(circles, edges, min_r, max_r):
    """Among Hough candidates, pick the one with best edge support on its circumference."""
    h, w = edges.shape
    best_circ = None
    best_score = -1
    for cx, cy, r in circles:
        if not (min_r <= r <= max_r):
            continue
        # Sample ~200 points on circumference, count how many hit an edge
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        xs = np.clip(np.round(cx + r * np.cos(angles)).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy + r * np.sin(angles)).astype(int), 0, h - 1)
        score = int(np.sum(edges[ys, xs] > 0))
        if score > best_score:
            best_score = score
            best_circ  = (cx, cy, r)
    return best_circ


def _fit_limb_circle(img_bgr, seed, expected_r=None):
    """Refine a circle from the moon's visible limb, including clipped arcs.

    HoughCircles is useful for finding a starting point, but its accumulator can
    be biased by the bright halo or by the moon's internal texture.  This fit
    uses only strong edge pixels near the candidate circumference and repeatedly
    removes outliers.  Because the fit is geometric, the centre can be recovered
    even when part of the disc is outside the source frame.
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(gray, 15, 55)

    sx, sy, sr = [float(v) * scale for v in (*seed[:2], seed[2])]
    target_r = float(expected_r or sr) * scale
    band = max(10.0, target_r * 0.075)
    ys, xs = np.nonzero(edges)
    if len(xs) < 20:
        return None
    distances = np.hypot(xs - sx, ys - sy)
    keep = (distances >= target_r - band) & (distances <= target_r + band)
    points = np.column_stack((xs[keep], ys[keep])).astype(np.float64)
    if len(points) < 20:
        return None

    def algebraic_fit(p):
        # x^2+y^2 + D*x + E*y + F = 0
        A = np.column_stack((p[:, 0], p[:, 1], np.ones(len(p))))
        b = -(p[:, 0] ** 2 + p[:, 1] ** 2)
        coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -coef[0] / 2.0, -coef[1] / 2.0
        radius_sq = cx * cx + cy * cy - coef[2]
        return cx, cy, np.sqrt(max(0.0, radius_sq))

    # Three robust passes. Restrict the radius so unrelated sky edges cannot
    # pull the solution away from the common focal-length scale.
    fit = None
    work = points
    for _ in range(3):
        if len(work) < 12:
            return None
        fit = algebraic_fit(work)
        cx, cy, radius = fit
        residual = np.abs(np.hypot(work[:, 0] - cx, work[:, 1] - cy) - radius)
        limit = max(3.0, min(target_r * 0.035, np.median(residual) * 3.0 + 2.0))
        work = work[residual <= limit]

    if fit is None:
        return None
    cx, cy, radius = fit
    if radius < target_r * 0.85 or radius > target_r * 1.15:
        return None
    # Keep the fitted centre at sub-pixel precision.  Rounding here can leave
    # a visible one-pixel registration error after translation.
    return (cx / scale, cy / scale, radius / scale)


# ── Precise limb fit + stack registration ─────────────────────────────────────

def _moon_mask(gray):
    """Binary mask of the lit moon (largest bright blob) + the blurred gray."""
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blur > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None, blur
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8), blur


def _algebraic_circle(p):
    A = np.column_stack((p[:, 0], p[:, 1], np.ones(len(p))))
    b = -(p[:, 0] ** 2 + p[:, 1] ** 2)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -c[0] / 2.0, -c[1] / 2.0
    return cx, cy, float(np.sqrt(max(cx * cx + cy * cy - c[2], 0.0)))


def detect_disc(img_bgr):
    """Find a (nearly) full moon as the roundest compact bright blob.

    Thresholds the smoothed image at several levels and keeps blobs whose
    contour is convex and fills its minimum enclosing circle (city lights and
    buildings are irregular, so they fail).  The centre is a circle fit to the
    blob's convex hull, which ignores things like a plane crossing the disc.
    Returns dict(cx, cy, r) in full-resolution pixels, or None.
    """
    h, w = img_bgr.shape[:2]
    sc = min(1.0, 1500.0 / max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mx = float(gray.max())
    levels = [otsu] + [mx * f for f in (0.8, 0.65, 0.5, 0.35)] + [40, 50, 60, 80, 110]
    short = min(sh, sw)
    best, best_score = None, 0.0
    for t in levels:
        mask = (gray > t).astype(np.uint8)
        mask[: int(sh * 0.08), :] = 0           # caption text
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < (short * 0.03) ** 2 * np.pi:
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) > short * 0.5 or max(bw, bh) > 1.3 * min(bw, bh):
                continue
            comp = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=len)
            hull = cv2.convexHull(cnt)
            harea = cv2.contourArea(hull)
            (_, _), er = cv2.minEnclosingCircle(hull)
            if harea <= 0 or er <= 0:
                continue
            solidity = area / harea
            fill = harea / (np.pi * er * er)
            if solidity < 0.9 or fill < 0.9:
                continue
            score = solidity * fill * np.sqrt(area)
            if score > best_score:
                best_score = score
                best = hull[:, 0, :].astype(np.float64)
    if best is None:
        return None
    # densify hull points so the fit is not biased toward vertex spacing
    pts = np.vstack([np.linspace(best[k], best[(k + 1) % len(best)], 8, endpoint=False)
                     for k in range(len(best))])
    cx, cy, r = _algebraic_circle(pts)
    return {"cx": (cx + 0.5) / sc - 0.5, "cy": (cy + 0.5) / sc - 0.5, "r": r / sc}


def detect_crescent(img_bgr):
    """Find the moon circle from its bright crescent/limb.

    A white top-hat keeps thin bright structures (the crescent) while
    discarding smooth sky gradients and wide step edges (horizon/hills).
    Text overlays are removed by ignoring the border bands and keeping only
    the biggest blob.  A circle is fitted to the OUTER (convex hull) edge.
    Returns dict(cx, cy, r) in full-resolution pixels, or None.
    """
    h, w = img_bgr.shape[:2]
    sc = min(1.0, 1500.0 / max(h, w))
    small = cv2.resize(img_bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)

    k = max(15, int(min(sh, sw) * 0.14)) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern).astype(np.float32)

    # Ignore caption/text bands
    th[: int(sh * 0.10), :] = 0
    th[int(sh * 0.85):, :] = 0

    # Suppress dark ground (hills/trees) and the bright rim right above it
    p75 = float(np.percentile(gray, 75))
    dark = (gray < 0.3 * p75).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dk = max(5, int(min(sh, sw) * 0.02))
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dk + 1, 2 * dk + 1)))
    th[dark > 0] = 0

    med = float(np.median(th))
    mad = float(np.median(np.abs(th - med))) + 1e-3
    best = None
    for kmul in (12, 8, 5, 3.5):
        thr = med + kmul * mad * 1.4826
        mask = (th > max(thr, 4)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n < 2:
            continue
        # prefer the largest blob that is not tiny (rejects stars/planets)
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        for idx in order[:3]:
            bw, bh = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) < min(sh, sw) * 0.08:
                continue
            comp = (lab == idx).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=len)
            if len(cnt) < 40:
                continue
            pts = cnt[:, 0, :].astype(np.float64)
            hull = cv2.convexHull(cnt)
            hd = np.array([abs(cv2.pointPolygonTest(hull, (float(x), float(y)), True))
                           for x, y in pts])
            seed = pts[hd < 2.5]
            if len(seed) < 20:
                continue
            cx, cy, r = _algebraic_circle(seed)
            for _ in range(4):
                d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
                sel = pts[(d < max(2.0, r * 0.04)) & (hd < 4.0)]
                if len(sel) < 20:
                    break
                cx, cy, r = _algebraic_circle(sel)
            if not (min(sh, sw) * 0.04 < r < min(sh, sw) * 0.35):
                continue
            best = (cx, cy, r)
            break
        if best:
            break
    if best is None:
        return None
    return {"cx": best[0] / sc, "cy": best[1] / sc, "r": best[2] / sc}


def detect_limb(img_bgr):
    """Fit a circle to the moon's bright LIMB only.

    The terminator (day/night line) and any part of the disc clipped by the
    frame edge are excluded, so they cannot drag the centre:
      * limb pixels lie on the convex hull of the lit area (terminator of a
        crescent is recessed),
      * on the limb, brightness falls off radially OUTWARD from the centre;
        on the terminator it falls off INWARD — checked with the gradient.
    Returns dict(cx, cy, r, pts, centroid) or None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask, blur = _moon_mask(gray)
    if mask is None:
        return None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=len)
    pts = cnt[:, 0, :].astype(np.float64)
    on_frame = ((pts[:, 0] > 3) & (pts[:, 0] < w - 4) &
                (pts[:, 1] > 3) & (pts[:, 1] < h - 4))
    pts = pts[on_frame]
    if len(pts) < 50:
        return None

    hull = cv2.convexHull(cnt)
    hull_d = np.array([abs(cv2.pointPolygonTest(hull, (float(x), float(y)), True))
                       for x, y in pts])
    seed = pts[hull_d < 2.0]
    if len(seed) < 30:
        return None
    cx, cy, r = _algebraic_circle(seed)

    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
    nx, ny = -gx[yi, xi], -gy[yi, xi]          # bright -> dark direction
    nn = np.hypot(nx, ny) + 1e-9
    sel = None
    for _ in range(5):
        rx, ry = pts[:, 0] - cx, pts[:, 1] - cy
        rr = np.hypot(rx, ry) + 1e-9
        outward = (nx * rx + ny * ry) / (nn * rr)
        cur = (outward > 0.85) & (np.abs(rr - r) < max(3.0, r * 0.02))
        if cur.sum() < 30:
            break
        sel = cur
        cx, cy, r = _algebraic_circle(pts[sel])
    if sel is None:
        return None

    M = cv2.moments(mask, True)
    centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    return {"cx": cx, "cy": cy, "r": r, "pts": pts[sel], "centroid": centroid}


def fit_center_fixed_r(pts, cx, cy, R):
    """Geometric (Gauss-Newton) centre of a circle whose radius is known.

    Using the shared consensus radius removes the radius/centre ambiguity of a
    partial arc, which otherwise shifts the centre by several pixels.
    """
    for _ in range(30):
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        d = np.hypot(dx, dy) + 1e-9
        J = np.column_stack((-dx / d, -dy / d))
        step, _, _, _ = np.linalg.lstsq(J, -(d - R), rcond=None)
        cx, cy = cx + step[0], cy + step[1]
        if np.hypot(step[0], step[1]) < 1e-4:
            break
    return float(cx), float(cy)


def _affine3(M):
    return np.vstack([np.asarray(M, np.float64), [0.0, 0.0, 1.0]])


def build_matrix(cx, cy, rot_deg, w, h):
    """Rotate by rot_deg about the moon centre, then move it to photo centre."""
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot_deg), 1.0)
    M[0, 2] += w / 2.0 - cx
    M[1, 2] += h / 2.0 - cy
    return M


def refine_stack_alignment(images, mats, w, h, R, iterations=3):
    """Sub-pixel register every frame to the median stack with ECC.

    Optimises rotation + translation on the moon area itself, which is what
    AutoStakkert needs. The mean translation is removed so the limb-fit
    centre stays at the photo centre.
    """
    sc = 0.5
    half = int(R * 1.1)
    x0, x1 = max(0, int(w // 2 - half)), min(w, int(w // 2 + half))
    y0, y1 = max(0, int(h // 2 - half)), min(h, int(h // 2 + half))
    cw, ch = int((x1 - x0) * sc), int((y1 - y0) * sc)
    C = np.array([[sc, 0, -sc * x0], [0, sc, -sc * y0], [0, 0, 1]], np.float64)
    C_inv = np.linalg.inv(C)
    grays = [cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
             for im in images]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    mats = [np.asarray(M, np.float64) for M in mats]

    p0 = np.array([w / 2.0, h / 2.0, 1.0])
    for _ in range(iterations):
        smalls = [cv2.warpAffine(g, (C @ _affine3(M))[:2], (cw, ch),
                                 flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                  for g, M in zip(grays, mats)]
        ref = np.median(np.stack(smalls), axis=0).astype(np.float32)
        corr = []
        for s in smalls:
            W = np.eye(2, 3, dtype=np.float32)
            try:
                _, W = cv2.findTransformECC(ref, s, W, cv2.MOTION_EUCLIDEAN,
                                            crit, None, 5)
            except cv2.error:
                W = np.eye(2, 3, dtype=np.float32)
            Wf = C_inv @ _affine3(W) @ C          # full-res: ref -> frame
            Winv = np.linalg.inv(Wf)              # correction to apply to frame
            rot = np.degrees(np.arctan2(Winv[1, 0], Winv[0, 0]))
            move = (Winv @ p0 - p0)[:2]           # displacement at photo centre
            if np.hypot(move[0], move[1]) > R * 0.1 or abs(rot) > 5:
                Winv = np.eye(3)                  # implausible -> ignore
            corr.append(Winv)
        mean_move = np.mean([(c @ p0 - p0)[:2] for c in corr], axis=0)
        for i, c in enumerate(corr):
            c = c.copy()
            c[:2, 2] -= mean_move
            mats[i] = (c @ _affine3(mats[i]))[:2]
    return mats


def find_moon_center(img_bgr, fixed_r=None):
    """
    Find the true geometric centre of the moon disc.

    If fixed_r is given, search with a tight radius window around it
    (used in pass 2 after consensus radius is known).

    Returns ((cx, cy), radius) or (None, None).
    """
    h, w = img_bgr.shape[:2]
    short = min(h, w)

    if fixed_r is not None:
        # Tight window: ±5% of consensus radius
        margin = max(20, int(fixed_r * 0.05))
        min_r = max(20, fixed_r - margin)
        max_r = fixed_r + margin
    else:
        # Loose window: 10%–90% of shorter dimension
        min_r = max(20, short // 10)
        max_r = int(short * 0.9)

    result = _hough_detect(img_bgr, min_r, max_r)
    if result is not None:
        refined = _fit_limb_circle(img_bgr, result, expected_r=fixed_r)
        cx, cy, r = refined if refined is not None else result
        return (cx, cy), r

    return None, None


def center_image(img, cx, cy, radius, bg_color):
    """Shift image so (cx, cy) -> photo centre. cx/cy may be outside original frame."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, w // 2 - cx], [0, 1, h // 2 - cy]])
    # Replicate the nearest real edge pixels when the translation exposes a
    # small border. This preserves the source sky/background better than
    # painting an artificial flat blue colour, while keeping the exact source
    # dimensions requested by the UI.
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ── Processing thread ─────────────────────────────────────────────────────────

def process_images(input_dir, output_dir, reject_dir, stack_align=True):
    """
    Two-pass processing:
      Pass 1 - detect moon with loose radius range, collect per-image radius.
      Compute consensus_r = median of all detected radii.
      Pass 2 - re-detect moon with radius locked to consensus_r for accuracy,
               then centre every image with the precise consensus radius.

    Pass 1 uses a limb-only fit (terminator / frame edge ignored); pass 2 fits
    the centre with the shared consensus radius.  When stack_align is True
    (same-session frames for AutoStakkert), frames are also de-rotated to a
    common crescent orientation and sub-pixel registered with ECC.
    """
    global processing_state
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith(VALID_EXTS)])
    processing_state["total"]   = len(image_files)
    processing_state["done"]    = 0
    processing_state["results"] = []
    processing_state["error"]   = None

    reference_fl = None

    # ── Pass 1: detect and reject ─────────────────────────────────────────────
    detections = {}   # fname -> {"img": img, "cx": cx, "cy": cy, "r": r, "fl": fl}
    reject_set = set()

    for fname in image_files:
        fpath = os.path.join(input_dir, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                processing_state["results"].append(
                    {"file": fname, "status": "error", "msg": "Cannot read image"})
                processing_state["done"] += 1
                continue

            fl = get_focal_length(fpath)  # informational only; never rejects

            fixed = False
            limb = None
            cres = detect_disc(img) or detect_crescent(img)
            if cres is not None:
                moon_center, radius = (cres["cx"], cres["cy"]), cres["r"]
                fixed = True
            else:
                limb = detect_limb(img)
                if limb is not None:
                    moon_center, radius = (limb["cx"], limb["cy"]), limb["r"]
                else:
                    moon_center, radius = find_moon_center(img)
            if moon_center is None:
                processing_state["results"].append(
                    {"file": fname, "status": "skipped", "msg": "Moon not detected"})
                processing_state["done"] += 1
            else:
                cx, cy = moon_center
                detections[fname] = {"img": img, "cx": cx, "cy": cy,
                                     "r": radius, "fl": fl, "limb": limb,
                                     "fixed": fixed}

        except Exception as e:
            processing_state["results"].append(
                {"file": fname, "status": "error", "msg": str(e)})
            processing_state["done"] += 1

    if not detections:
        processing_state["running"] = False
        return

    # Each photo is processed independently with its own fitted radius.
    radii = {f: float(d["r"]) for f, d in detections.items()}

    # ── Pass 2: centre with own radius (+ optional stack registration) ───────
    names = list(detections.keys())
    h, w = detections[names[0]]["img"].shape[:2]
    centres, angles = {}, {}
    for fname in names:
        det = detections[fname]
        limb = det["limb"]
        if det.get("fixed"):
            cx, cy = det["cx"], det["cy"]
        elif limb is not None:
            cx, cy = fit_center_fixed_r(limb["pts"], limb["cx"], limb["cy"],
                                        radii[fname])
            gx, gy = limb["centroid"]
            angles[fname] = float(np.degrees(np.arctan2(gy - cy, gx - cx)))
        else:
            moon_center2, _ = find_moon_center(det["img"], fixed_r=int(round(radii[fname])))
            cx, cy = moon_center2 if moon_center2 is not None else (det["cx"], det["cy"])
        centres[fname] = (float(cx), float(cy))

    # Pure translation: moon centre -> photo centre. No rotation, no stack ECC.
    rots = {f: 0.0 for f in names}
    mats = [build_matrix(*centres[f], 0.0, w, h) for f in names]

    for fname, M in zip(names, mats):
        det = detections[fname]
        try:
            img = det["img"]
            fl  = det["fl"]
            ih, iw = img.shape[:2]
            cx, cy = centres[fname]
            r = int(round(radii[fname]))

            bg_color = sample_background_color(img, int(cx), int(cy), r)
            if (ih, iw) != (h, w):
                M = build_matrix(cx, cy, rots[fname], iw, ih)
            centered = cv2.warpAffine(img, np.asarray(M, np.float64), (iw, ih),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=bg_color)
            cv2.imwrite(os.path.join(output_dir, fname), centered,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])

            # Where the moon centre landed relative to the source pixel grid
            sx = float(M[0][0] * cx + M[0][1] * cy + M[0][2]) - cx
            sy = float(M[1][0] * cx + M[1][1] * cy + M[1][2]) - cy
            rot = float(np.degrees(np.arctan2(M[1][0], M[0][0])))
            fl_str = f"{fl:.1f}mm" if fl else "N/A"
            processing_state["results"].append({
                "file": fname, "status": "ok",
                "msg": (f"Moon ({cx:.2f},{cy:.2f}) | r={r}px (own) | "
                        f"shift ({sx:.2f},{sy:.2f}) | rot {rot:+.2f}° | fl={fl_str}"),
                "offset": [sx, sy]
            })
        except Exception as e:
            processing_state["results"].append(
                {"file": fname, "status": "error", "msg": str(e)})

        processing_state["done"] += 1

    processing_state["running"] = False


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/start", methods=["POST"])
def start():
    global processing_state
    if processing_state["running"]:
        return jsonify({"error": "Already running"}), 400
    data       = request.get_json(silent=True) or {}
    input_dir  = data.get("input_dir",  "").strip()
    output_dir = data.get("output_dir", "").strip()
    reject_dir = data.get("reject_dir", "").strip()
    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({"error": "Invalid input directory"}), 400
    if not output_dir:
        output_dir = os.path.join(input_dir, "centered")
    if not reject_dir:
        reject_dir = os.path.join(input_dir, "reject")
    stack_align = bool(data.get("stack_align", True))
    processing_state.update(running=True, input_dir=input_dir,
                             output_dir=output_dir, reject_dir=reject_dir)
    threading.Thread(target=process_images,
                     args=(input_dir, output_dir, reject_dir, stack_align),
                     daemon=True).start()
    return jsonify({"status": "started", "input_dir": input_dir,
                    "output_dir": output_dir, "reject_dir": reject_dir})


@app.route("/status")
def status():
    return jsonify(processing_state)


@app.route("/browse")
def browse():
    path = request.args.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Invalid path"}), 400
    entries = []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "path": full})
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403
    img_count = len([f for f in os.listdir(path) if f.lower().endswith(VALID_EXTS)])
    return jsonify({"path": path, "dirs": entries, "img_count": img_count})


@app.route("/images")
def images():
    d = processing_state.get("input_dir", "")
    if not d or not os.path.isdir(d):
        return jsonify([])
    return jsonify(sorted([f for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)]))


@app.route("/images/centered")
def images_centered():
    d = processing_state.get("output_dir", "")
    if not d or not os.path.isdir(d):
        return jsonify([])
    return jsonify(sorted([f for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)]))


@app.route("/images/rejected")
def images_rejected():
    d = processing_state.get("reject_dir", "")
    if not d or not os.path.isdir(d):
        return jsonify([])
    return jsonify(sorted([f for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)]))


def _serve_thumb(path, size=400):
    img = cv2.imread(path)
    if img is None:
        return "Not found", 404
    h, w  = img.shape[:2]
    scale = size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/preview/<path:fname>")
def preview(fname):
    d = processing_state.get("input_dir", "")
    if not d:
        return "No input dir set", 400
    return _serve_thumb(os.path.join(d, fname))


@app.route("/preview_centered/<path:fname>")
def preview_centered(fname):
    d = processing_state.get("output_dir", "")
    if not d:
        return "No output dir set", 400
    return _serve_thumb(os.path.join(d, fname))


@app.route("/preview_rejected/<path:fname>")
def preview_rejected(fname):
    d = processing_state.get("reject_dir", "")
    if not d:
        return "No reject dir set", 400
    return _serve_thumb(os.path.join(d, fname))


if __name__ == "__main__":
    print("Moon Phase Centering Tool running at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)

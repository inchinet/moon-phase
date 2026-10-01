import os
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
        cx, cy, r = result
        return (cx, cy), r

    return None, None


def center_image(img, cx, cy, radius, bg_color):
    """Shift image so (cx, cy) -> photo centre. cx/cy may be outside original frame."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, w // 2 - cx], [0, 1, h // 2 - cy]])
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)


# ── Processing thread ─────────────────────────────────────────────────────────

def process_images(input_dir, output_dir, reject_dir):
    """
    Two-pass processing:
      Pass 1 - detect moon with loose radius range, collect per-image radius.
      Compute consensus_r = median of all detected radii.
      Pass 2 - re-detect moon with radius locked to consensus_r for accuracy,
               then centre every image with the precise consensus radius.
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

            fl = get_focal_length(fpath)
            if fl is not None:
                if reference_fl is None:
                    reference_fl = fl
                elif abs(fl - reference_fl) > 0.5:
                    import shutil
                    shutil.copy2(fpath, os.path.join(reject_dir, fname))
                    processing_state["results"].append({
                        "file": fname, "status": "rejected",
                        "msg": f"Focal length {fl:.1f}mm != {reference_fl:.1f}mm"})
                    reject_set.add(fname)
                    processing_state["done"] += 1
                    continue

            moon_center, radius = find_moon_center(img)
            if moon_center is None:
                processing_state["results"].append(
                    {"file": fname, "status": "skipped", "msg": "Moon not detected"})
                processing_state["done"] += 1
            else:
                cx, cy = moon_center
                detections[fname] = {"img": img, "cx": cx, "cy": cy,
                                     "r": radius, "fl": fl}

        except Exception as e:
            processing_state["results"].append(
                {"file": fname, "status": "error", "msg": str(e)})
            processing_state["done"] += 1

    if not detections:
        processing_state["running"] = False
        return

    # ── Consensus radius (same focal length = same pixel radius) ──────────────
    all_radii   = [d["r"] for d in detections.values()]
    consensus_r = int(round(np.median(all_radii)))

    processing_state["results"].append({
        "file": "__consensus__",
        "status": "info",
        "msg": (f"Consensus radius = {consensus_r}px  "
                f"(individual: {sorted(all_radii)}) across {len(all_radii)} images")
    })

    # ── Pass 2: re-detect with fixed radius, then centre ─────────────────────
    for fname, det in detections.items():
        try:
            img = det["img"]
            fl  = det["fl"]
            h, w = img.shape[:2]

            # Re-detect with tight radius window for accuracy
            moon_center2, r2 = find_moon_center(img, fixed_r=consensus_r)
            if moon_center2 is not None:
                cx, cy = moon_center2
            else:
                # Fallback to pass-1 centre if re-detection fails
                cx, cy = det["cx"], det["cy"]

            r = consensus_r

            bg_color = sample_background_color(img, cx, cy, r)
            centered = center_image(img, cx, cy, r, bg_color)
            cv2.imwrite(os.path.join(output_dir, fname), centered)

            fl_str = f"{fl:.1f}mm" if fl else "N/A"
            processing_state["results"].append({
                "file": fname, "status": "ok",
                "msg": (f"Moon ({cx},{cy}) | r={r}px (consensus) | "
                        f"raw_r={det['r']}px | "
                        f"shift ({w//2-cx},{h//2-cy}) | fl={fl_str}"),
                "offset": [w // 2 - cx, h // 2 - cy]
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
    processing_state.update(running=True, input_dir=input_dir,
                             output_dir=output_dir, reject_dir=reject_dir)
    threading.Thread(target=process_images,
                     args=(input_dir, output_dir, reject_dir),
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

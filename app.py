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

def sample_background_color(img_bgr, cx, cy, radius, sample_margin=60):
    """
    Sample the dominant background color near the edges of the image,
    away from the moon.  Returns a BGR tuple.
    """
    h, w = img_bgr.shape[:2]
    # Build a mask of pixels far from the moon center
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bg_mask = (dist > radius + sample_margin).astype(np.uint8)

    pixels = img_bgr[bg_mask == 1]
    if len(pixels) == 0:
        # Fallback: sample all four corners
        corners = [
            img_bgr[:50, :50].reshape(-1, 3),
            img_bgr[:50, -50:].reshape(-1, 3),
            img_bgr[-50:, :50].reshape(-1, 3),
            img_bgr[-50:, -50:].reshape(-1, 3),
        ]
        pixels = np.vstack(corners)

    # Use median for robustness against outliers
    b, g, r = (int(np.median(pixels[:, 0])),
                int(np.median(pixels[:, 1])),
                int(np.median(pixels[:, 2])))
    return (b, g, r)


def find_moon_center(img_bgr):
    """
    Detect the moon in the photo.

    Strategy:
    1. HSV saturation mask: the moon is grayish/white (low sat) while
       blue sky has high saturation.
    2. Compute mask blob centroid as ground-truth reference center.
    3. Hough Circle Transform, validated against the mask centroid.
    4. Fall back to mask centroid if Hough deviates too far.
    5. Contour moment centroid as last resort.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    s = hsv[:, :, 1]   # saturation
    v = hsv[:, :, 2]   # value / brightness

    # Moon: low saturation + reasonably bright
    low_sat  = (s < 70).astype(np.uint8) * 255
    bright   = (v > 60).astype(np.uint8) * 255
    moon_mask = cv2.bitwise_and(low_sat, bright)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    moon_mask = cv2.morphologyEx(moon_mask, cv2.MORPH_CLOSE, kernel)
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    moon_mask = cv2.morphologyEx(moon_mask, cv2.MORPH_OPEN, kernel_sm)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    min_r = max(30, min(h, w) // 50)
    max_r = int(min(h, w) * 0.8)

    # Compute mask centroid as ground-truth reference
    # This is the most reliable center for any moon phase (full, crescent, etc.)
    mask_cx, mask_cy, mask_r = _largest_blob_centroid(moon_mask, min_r)

    # Attempt 1: Hough on saturation-masked gray
    masked_gray = cv2.bitwise_and(gray, moon_mask)
    blurred = cv2.GaussianBlur(masked_gray, (13, 13), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 4,
        param1=40,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None:
        best = _pick_best_circle(circles[0], mask_cx, mask_cy, mask_r, h, w)
        if best is not None:
            cx, cy, r = best
            return (int(cx), int(cy)), int(r)

    # Attempt 2: Hough on full blurred gray
    blurred_full = cv2.GaussianBlur(gray, (17, 17), 3)
    circles2 = cv2.HoughCircles(
        blurred_full,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(h, w) // 4,
        param1=70,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles2 is not None:
        best = _pick_best_circle(circles2[0], mask_cx, mask_cy, mask_r, h, w)
        if best is not None:
            cx, cy, r = best
            return (int(cx), int(cy)), int(r)

    # Attempt 3: use mask centroid directly (most robust fallback)
    if mask_cx is not None:
        return (int(mask_cx), int(mask_cy)), int(mask_r)

    # Attempt 4: largest circular contour from mask
    contours, _ = cv2.findContours(moon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    min_area = np.pi * min_r ** 2
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        candidates.append((circularity, area, cnt))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = candidates[0][2]
    else:
        chosen = max(contours, key=cv2.contourArea)

    # Use moment centroid: more accurate than minEnclosingCircle center
    # especially for crescent/partial moons
    M_cnt = cv2.moments(chosen)
    if M_cnt["m00"] != 0:
        cx = int(M_cnt["m10"] / M_cnt["m00"])
        cy = int(M_cnt["m01"] / M_cnt["m00"])
    else:
        (cx, cy), _ = cv2.minEnclosingCircle(chosen)
        cx, cy = int(cx), int(cy)
    _, radius = cv2.minEnclosingCircle(chosen)
    return (cx, cy), int(radius)


def _largest_blob_centroid(mask, min_r):
    """
    Find the largest blob in the binary mask and return its
    moment centroid (cx, cy) and approximate radius.
    Returns (None, None, None) if no valid blob found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    min_area = np.pi * min_r ** 2
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None, None, None

    chosen = max(valid, key=cv2.contourArea)
    M = cv2.moments(chosen)
    if M["m00"] == 0:
        return None, None, None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    area = M["m00"]
    radius = np.sqrt(area / np.pi)   # equivalent circle radius
    return cx, cy, radius


def _pick_best_circle(circles, mask_cx, mask_cy, mask_r, h, w):
    """
    From a list of Hough circles, pick the one whose center is closest
    to the mask centroid AND whose radius is plausible.
    Returns (cx, cy, r) or None if no circle passes validation.
    """
    if mask_cx is None:
        # No mask reference: return the first circle if it is inside the image
        cx, cy, r = circles[0]
        if 0 <= cx < w and 0 <= cy < h:
            return (cx, cy, r)
        return None

    # Allow the Hough center to deviate up to 1x mask_r from the mask centroid
    tolerance = max(mask_r * 1.0, 50)

    best = None
    best_dist = float("inf")
    for cx, cy, r in circles:
        dist = np.sqrt((cx - mask_cx) ** 2 + (cy - mask_cy) ** 2)
        if dist < tolerance and dist < best_dist:
            best = (cx, cy, r)
            best_dist = dist

    return best


def center_image(img, cx, cy, radius, bg_color):
    """
    Translate so moon center lands at canvas center.
    Fill the revealed gap with the sampled background color.
    """
    h, w = img.shape[:2]
    dx = w // 2 - cx
    dy = h // 2 - cy
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # borderValue fills the gap with the background color
    return cv2.warpAffine(img, M, (w, h), borderValue=bg_color)


# ── Processing thread ─────────────────────────────────────────────────────────

def process_images(input_dir, output_dir, reject_dir):
    global processing_state
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTS)])
    processing_state["total"] = len(image_files)
    processing_state["done"] = 0
    processing_state["results"] = []
    processing_state["error"] = None

    reference_fl = None   # focal length of first successfully read image

    for fname in image_files:
        fpath = os.path.join(input_dir, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                processing_state["results"].append(
                    {"file": fname, "status": "error", "msg": "Cannot read image"}
                )
                processing_state["done"] += 1
                continue

            # ── Focal-length check ────────────────────────────────────────────
            fl = get_focal_length(fpath)
            if fl is not None:
                if reference_fl is None:
                    reference_fl = fl
                elif abs(fl - reference_fl) > 0.5:
                    # Different focal length → reject
                    import shutil
                    shutil.copy2(fpath, os.path.join(reject_dir, fname))
                    processing_state["results"].append({
                        "file": fname,
                        "status": "rejected",
                        "msg": f"Focal length {fl:.1f}mm ≠ {reference_fl:.1f}mm"
                    })
                    processing_state["done"] += 1
                    continue

            # ── Detect moon ───────────────────────────────────────────────────
            moon_center, radius = find_moon_center(img)

            if moon_center is None:
                processing_state["results"].append(
                    {"file": fname, "status": "skipped", "msg": "Moon not detected"}
                )
            else:
                cx, cy = moon_center
                # Sample background color before shifting
                bg_color = sample_background_color(img, cx, cy, radius)
                centered = center_image(img, cx, cy, radius, bg_color)
                out_path = os.path.join(output_dir, fname)
                cv2.imwrite(out_path, centered)
                h, w = img.shape[:2]
                fl_str = f"{fl:.1f}mm" if fl else "N/A"
                processing_state["results"].append({
                    "file": fname,
                    "status": "ok",
                    "msg": (f"Moon ({cx},{cy}) r={radius}px | "
                            f"shift ({w//2-cx},{h//2-cy}) | fl={fl_str}"),
                    "offset": [w // 2 - cx, h // 2 - cy]
                })
        except Exception as e:
            processing_state["results"].append(
                {"file": fname, "status": "error", "msg": str(e)}
            )

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

    data = request.get_json(silent=True) or {}
    input_dir  = data.get("input_dir", "").strip()
    output_dir = data.get("output_dir", "").strip()
    reject_dir = data.get("reject_dir", "").strip()

    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({"error": "Invalid input directory"}), 400

    if not output_dir:
        output_dir = os.path.join(input_dir, "centered")
    if not reject_dir:
        reject_dir = os.path.join(input_dir, "reject")

    processing_state["running"]    = True
    processing_state["input_dir"]  = input_dir
    processing_state["output_dir"] = output_dir
    processing_state["reject_dir"] = reject_dir

    t = threading.Thread(
        target=process_images,
        args=(input_dir, output_dir, reject_dir),
        daemon=True
    )
    t.start()
    return jsonify({"status": "started", "input_dir": input_dir,
                    "output_dir": output_dir, "reject_dir": reject_dir})


@app.route("/status")
def status():
    return jsonify(processing_state)


@app.route("/browse")
def browse():
    """List subdirectories (and image counts) for a given path."""
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
    input_dir = processing_state.get("input_dir", "")
    if not input_dir or not os.path.isdir(input_dir):
        return jsonify([])
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTS)])
    return jsonify(files)


@app.route("/images/centered")
def images_centered():
    output_dir = processing_state.get("output_dir", "")
    if not output_dir or not os.path.isdir(output_dir):
        return jsonify([])
    files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(VALID_EXTS)])
    return jsonify(files)


@app.route("/images/rejected")
def images_rejected():
    reject_dir = processing_state.get("reject_dir", "")
    if not reject_dir or not os.path.isdir(reject_dir):
        return jsonify([])
    files = sorted([f for f in os.listdir(reject_dir) if f.lower().endswith(VALID_EXTS)])
    return jsonify(files)


def _serve_thumb(path, size=400):
    img = cv2.imread(path)
    if img is None:
        return "Not found", 404
    h, w = img.shape[:2]
    scale = size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/preview/<path:fname>")
def preview(fname):
    input_dir = processing_state.get("input_dir", "")
    if not input_dir:
        return "No input dir set", 400
    return _serve_thumb(os.path.join(input_dir, fname))


@app.route("/preview_centered/<path:fname>")
def preview_centered(fname):
    output_dir = processing_state.get("output_dir", "")
    if not output_dir:
        return "No output dir set", 400
    return _serve_thumb(os.path.join(output_dir, fname))


@app.route("/preview_rejected/<path:fname>")
def preview_rejected(fname):
    reject_dir = processing_state.get("reject_dir", "")
    if not reject_dir:
        return "No reject dir set", 400
    return _serve_thumb(os.path.join(reject_dir, fname))


if __name__ == "__main__":
    print("Moon Phase Centering Tool running at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)

import os
import cv2
import numpy as np
from flask import Flask, jsonify, send_from_directory, request, Response
import threading

app = Flask(__name__, static_folder=".")

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(INPUT_DIR, "centered")
VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

processing_state = {
    "running": False,
    "total": 0,
    "done": 0,
    "results": [],
    "error": None
}


def find_moon_center(img_bgr):
    """
    Detect the moon in blue-sky daytime photos.

    Strategy:
    1. HSV saturation mask — the moon is grayish/white (low sat) while the
       blue sky has high saturation. This isolates the moon blob nicely.
    2. Hough Circle Transform on the masked area for robust circle detection.
    3. Contour circularity fallback if Hough yields nothing.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    s = hsv[:, :, 1]   # saturation
    v = hsv[:, :, 2]   # value / brightness

    # Moon: low saturation + reasonably bright
    # Blue sky: high saturation
    low_sat  = (s < 70).astype(np.uint8) * 255
    bright   = (v > 60).astype(np.uint8) * 255
    moon_mask = cv2.bitwise_and(low_sat, bright)

    # Morphological cleanup to fill holes and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    moon_mask = cv2.morphologyEx(moon_mask, cv2.MORPH_CLOSE, kernel)
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    moon_mask = cv2.morphologyEx(moon_mask, cv2.MORPH_OPEN, kernel_sm)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Attempt 1: Hough circles on saturation-masked gray image ──────────────
    masked_gray = cv2.bitwise_and(gray, moon_mask)
    blurred = cv2.GaussianBlur(masked_gray, (13, 13), 2)

    min_r = max(30, min(h, w) // 50)
    max_r = int(min(h, w) * 0.8) # Increased range for large moons

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 4,
        param1=40,
        param2=30, # Slightly more stringent to avoid noise
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        # Pick the circle with highest accumulated votes
        cx, cy, r = circles[0][0]
        return (int(cx), int(cy)), int(r)

    # ── Attempt 2: Hough circles on full blurred gray ────────────────────────
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
        cx, cy, r = circles2[0][0]
        return (int(cx), int(cy)), int(r)

    # ── Attempt 3: largest circular contour from mask ─────────────────────────
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
        # Prefer most circular; break ties by largest area
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = candidates[0][2]
    else:
        chosen = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(chosen)
    return (int(cx), int(cy)), int(radius)


def center_image(img, cx, cy):
    """Translate the image so the moon center sits at the canvas center."""
    h, w = img.shape[:2]
    dx = w // 2 - cx
    dy = h // 2 - cy
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))


def process_images():
    global processing_state
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(VALID_EXTS)])
    processing_state["total"] = len(image_files)
    processing_state["done"] = 0
    processing_state["results"] = []
    processing_state["error"] = None

    for fname in image_files:
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                processing_state["results"].append(
                    {"file": fname, "status": "error", "msg": "Cannot read image"}
                )
                processing_state["done"] += 1
                continue

            moon_center, radius = find_moon_center(img)

            if moon_center is None:
                processing_state["results"].append(
                    {"file": fname, "status": "skipped", "msg": "Moon not detected"}
                )
            else:
                centered = center_image(img, moon_center[0], moon_center[1])
                out_path = os.path.join(OUTPUT_DIR, fname)
                cv2.imwrite(out_path, centered)
                h, w = img.shape[:2]
                processing_state["results"].append({
                    "file": fname,
                    "status": "ok",
                    "msg": f"Moon at ({moon_center[0]}, {moon_center[1]}), r={radius}px",
                    "offset": [w // 2 - moon_center[0], h // 2 - moon_center[1]]
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
    processing_state["running"] = True
    t = threading.Thread(target=process_images, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    return jsonify(processing_state)


@app.route("/images")
def images():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(VALID_EXTS)])
    return jsonify(files)


@app.route("/images/centered")
def images_centered():
    if not os.path.isdir(OUTPUT_DIR):
        return jsonify([])
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(VALID_EXTS)])
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
    return _serve_thumb(os.path.join(INPUT_DIR, fname))


@app.route("/preview_centered/<path:fname>")
def preview_centered(fname):
    return _serve_thumb(os.path.join(OUTPUT_DIR, fname))


if __name__ == "__main__":
    print("Moon Phase Centering Tool running at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)

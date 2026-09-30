# 🌕 Moon Phase Auto-Centering Tool

A local web utility that automatically detects and centers the moon in a series of moon phase photographs — works for both **nighttime (dark sky)** and **daytime (blue sky)** photos.  
Centered results are saved to a configurable output folder (default: `centered/` under the source path).  
Photos with a different focal length are automatically moved to a `reject/` folder.

![center moon](https://github.com/inchinet/moon-phase/blob/main/screen.png)
---

## 🚀 Quick Start

Double-click **`run.bat`** — it will:
1. Start the Flask server at `http://127.0.0.1:5050`
2. Open your browser automatically

Then in the UI:
1. Set the **Input** folder path (or use **Browse…** to navigate)
2. Optionally set custom **Output** and **Rejected** folder paths
3. Click **⚡ Start Centering**

---

## 📁 File Structure

```
moon-phase/
├── input/              ← Source moon photos (any folder, configurable in UI)
├── centered/           ← Auto-created; centered output images (same filenames)
├── reject/             ← Auto-created; photos with mismatched focal length
├── app.py              ← Flask backend + OpenCV processing logic
├── index.html          ← Web UI (Liquid Glass design)
├── run.bat             ← Launch script (Windows)
└── README.md           ← This file
```

---

## ⚙️ How It Works

### 📁 Folder & Path Selection
The UI lets you specify any source folder on your system via text input or the **Browse…** inline folder navigator. Output and Reject paths default to `<input>/centered` and `<input>/reject` respectively.

### 🔭 Focal Length Matching
- The tool reads the **EXIF focal length** from each photo using `piexif`.
- The **first valid image** sets the reference focal length for the batch.
- Any subsequent image with a focal length that differs by more than **0.5 mm** is **rejected** — copied to the `reject/` folder and logged as 🚫 in the UI.
- If a photo has no EXIF focal length data, the check is skipped and it is processed normally.

### 🌑 Moon Detection
The detection pipeline runs three passes and uses the first that succeeds:

1. **HSV colour segmentation** — Converts to HSV colour space. The moon is grey/white (low saturation); sky is blue (high saturation). A saturation + brightness mask isolates the moon blob.
2. **Morphological cleanup** — Fills holes in the moon disc and removes small noise.
3. **Hough Circle Transform (masked)** — Runs on the saturation-masked grayscale image.
4. **Hough Circle Transform (full image)** — Looser second pass on raw grayscale.
5. **Contour circularity fallback** — Picks the most circular contour from the mask.

### 🎯 Centering & Background Fill
6. **Background sampling** — Median colour of pixels far from the moon is sampled to determine the scene background (e.g. blue sky, black night sky).
7. **Translation** — Image is shifted so the detected centre `(cx, cy)` aligns with the canvas centre.
8. **Gap fill** — The revealed border after shifting is filled with the **sampled background colour** (not a hard black). Blue sky stays blue; dark sky stays dark.
9. **Save** — Centered image written to the output folder with the **original filename**.

All output images keep the **exact same resolution** as the source.

---

## 🖼️ UI Features

| Feature | Details |
|---|---|
| **📁 Folder Setup** | Input / Output / Rejected path fields with inline **Browse…** navigator |
| **📷 Source Photos tab** | Thumbnail gallery of all input images |
| **🎯 Centered Output tab** | Thumbnail gallery of processed results |
| **🚫 Rejected tab** | Thumbnail gallery of focal-length-rejected photos |
| **⇄ Before / After compare** | Click any source photo → lightbox → **Compare** button shows side-by-side |
| **Stats bar** | Total · Processed · Centered OK · Rejected (FL) · Skipped/Error |
| **Processing log** | Live progress bar + per-file status (✅ ok / ⚠️ skipped / 🚫 rejected / ❌ error) |

---

## ⚠️ Restrictions & Known Limitations

| Limitation | Detail |
|---|---|
| **Mixed focal lengths** | Photos with different focal lengths (e.g. mixed zoom settings) are automatically separated into the `reject/` folder. |
| **No EXIF data** | If photos have no EXIF focal length, focal-length filtering is skipped for those images. |
| **Heavy overcast / fog** | If the moon is barely visible behind thick clouds, detection may fail. |
| **One dominant object** | Only the best-matching circle/contour is treated as the moon. Bright lamps or sun glare near the moon can confuse detection. |
| **No zoom/scale change** | The tool only **translates** (shifts) — it does not resize or crop. Output resolution equals input resolution. |
| **Moon near edge** | If the moon is very close to the image border, the shifted version may show a narrow background-filled band. |
| **Supported formats** | `.jpg`, `.jpeg`, `.png` (case-insensitive). RAW files (`.CR2`, `.NEF`, etc.) are **not** supported. |
| **Local only** | The server binds to `127.0.0.1` — not accessible over the network. |
| **No overwrite warning** | Re-running will silently overwrite files already in the output folder. |

---

## 🛠️ Dependencies

Uses the shared virtual environment at `Z:\antigravity\venv`:

- `opencv-python` ≥ 4.10
- `numpy` ≥ 2.2
- `flask`
- `piexif` ≥ 1.1.3 ← new; for EXIF focal length reading

No additional installation needed — `piexif` is already installed in the shared venv.

---

## 📝 Tips

- If a photo is marked **⚠️ skipped (Moon not detected)**, the moon may be very small, heavily obscured, or partially cropped out of frame.
- For **daytime photos**, a clear blue sky gives the best detection results — the algorithm relies on the saturation contrast between the grey moon and the blue sky.
- For **night photos**, ensure the moon is reasonably bright relative to the surroundings (avoid shots dominated by very bright foreground lights).
- **Background fill** matches the sky colour automatically — no manual adjustment needed.
- Centered output images keep the **exact same filename** as the source, preserving the sequence order for timelapse assembly.

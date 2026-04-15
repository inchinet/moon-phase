# 🌕 Moon Phase Auto-Centering Tool

A local web utility that automatically detects and centers the moon in a series of moon phase photographs — works for both **nighttime (dark sky)** and **daytime (blue sky)** photos. Centered results are saved to a `centered/` subfolder.

---

## 🚀 Quick Start

Double-click **`run.bat`** — it will:
1. Start the Flask server at `http://127.0.0.1:5050`
2. Open your browser automatically

Then click **⚡ Start Centering** in the UI.

---

## 📁 File Structure

```
moon-phase/
├── *.JPG / *.jpg       ← Source moon photos (input)
├── centered/           ← Auto-created; output images saved here (same filenames)
├── app.py              ← Flask backend + OpenCV processing logic
├── index.html          ← Web UI (Liquid Glass design)
├── run.bat             ← Launch script (Windows)
└── README.md           ← This file
```

---

## ⚙️ How It Works

The detection pipeline runs three passes and uses the first one that succeeds:

### 🌑 Night photos (dark sky)
The original approach still applies as a fallback — bright pixels stand out strongly against the black sky.

### ☀️ Day photos (blue sky) — new
1. **HSV colour segmentation** — Converts the image to HSV colour space.  
   The blue sky has **high colour saturation**; the moon is grey/white with **low saturation**.  
   A saturation mask isolates the moon blob from the background.
2. **Morphological cleanup** — Fills holes in the moon disc and removes small noise blobs.
3. **Hough Circle Transform (masked)** — Runs on the saturation-masked grayscale image for precise circular detection.
4. **Hough Circle Transform (full image)** — Looser second pass on the raw grayscale if the first yields nothing.
5. **Contour circularity fallback** — Picks the most circular contour from the saturation mask.

### Common final step (both modes)
6. **Translation** — Image is shifted so the detected centre `(cx, cy)` aligns with the canvas centre.
7. **Save** — Centered image written to `centered/<original_filename>`.

---

## 🖼️ UI Features

| Feature | Details |
|---|---|
| **📷 Source Photos tab** | Thumbnail gallery of all input images |
| **🎯 Centered Output tab** | Thumbnail gallery of processed results |
| **⇄ Before / After compare** | Click any source photo → lightbox → **Compare** button shows side-by-side |
| **Processing log** | Live progress bar + per-file status (✅ ok / ⚠️ skipped / ❌ error) |

---

## ⚠️ Restrictions & Known Limitations

| Limitation | Detail |
|---|---|
| **Mixed lighting** | A mix of night and day photos in the same folder works — each image is processed independently. |
| **Heavy overcast / fog** | If the moon is barely visible behind thick clouds, detection may fail regardless of mode. |
| **One dominant object** | Only the best-matching circle/contour is treated as the moon. A very bright lamp or sun glare near the moon can confuse detection. |
| **No zoom/scale change** | The tool only **translates** (shifts) the image — it does not resize or crop. Output resolution equals input resolution. |
| **Moon near edge** | If the moon is very close to the image border, the centered version may show a black band after translation. |
| **Supported formats** | `.jpg`, `.jpeg`, `.png` (case-insensitive). RAW files (`.CR2`, `.NEF`, etc.) are **not** supported. |
| **Local only** | The server binds to `127.0.0.1` — not accessible over the network. |
| **No overwrite warning** | Re-running will silently overwrite files already in `centered/`. |

---

## 🛠️ Dependencies

Uses the shared virtual environment at `Z:\antigravity\venv`:

- `opencv-python` ≥ 4.10
- `numpy` ≥ 2.2
- `flask`

No additional installation needed if the venv is already set up.

---

## 📝 Tips

- If a photo is marked **⚠️ skipped (Moon not detected)**, it usually means the moon is very small, heavily obscured, or heavily cropped to one side.
- For **daytime photos**, ensure the sky is relatively clear and blue — the detection relies on the contrast between low-saturation moon and high-saturation sky.
- For **night photos**, ensure the moon is reasonably bright relative to the surroundings (avoid shots with very bright foreground lights dominating the frame).
- Centered output images keep the **exact same filename** as the source, so the sequence order is preserved.

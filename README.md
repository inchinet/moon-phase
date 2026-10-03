# 🌕 Moon Phase Auto-Centering Tool

A local web utility that automatically detects and centers the moon in a series of moon phase photographs — works for both **nighttime (dark sky)** and **daytime (blue sky)** photos.  
Centered results are saved to a configurable output folder (default: `centered/` under the source path).  
Photos with a different focal length are automatically moved to a `reject/` folder.  
With **Stack align** enabled, same-session frames are also de-rotated and sub-pixel registered so they stack sharply in **AutoStakkert!4**.

![center moon](https://github.com/inchinet/moon-phase/blob/main/screen.png)
---

## 🚀 Quick Start

Double-click **`run.bat`** — it will:
1. Start the Flask server at `http://127.0.0.1:5050`
2. Open your browser automatically

Then in the UI:
1. Set the **Input** folder path (or use **Browse…** to navigate)
2. Optionally set custom **Output** and **Rejected** folder paths
3. Keep **Stack align (AutoStakkert)** ticked for frames from the same session (untick for multi-night phase sequences)
4. Click **⚡ Start Centering**

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
- If a photo has no EXIF focal length data, the check is skipped; a clear mismatch in detected moon radius (> 8 %) is then used as the practical focal-length/zoom check.

### 🌑 Moon Detection (limb-only circle fit)
The detection pipeline finds the true geometric centre of the moon disc — even for a thin **crescent** or when the disc is **partially clipped** at the frame edge:

1. **Moon mask** — Gaussian blur + Otsu threshold; the largest bright blob is the lit moon.
2. **Limb-only edge points** — Only the bright outer **limb** is used. The terminator (day/night line) and the frame edge are rejected:
   - limb pixels lie on the **convex hull** of the lit area (a crescent's terminator is recessed);
   - on the limb, brightness falls off **outward** from the centre, while on the terminator it falls off **inward** — checked with the image gradient;
   - points touching the frame border (clipped disc) are discarded.
3. **Robust circle fit** — Iterative least-squares fit on the limb points gives each photo's centre and radius. (The older Hough + Canny detector is kept as a fallback.)
4. **Consensus radius** — Same focal length ⇒ same moon size, so the **median radius** is applied to every photo. Each centre is then re-fitted with this **fixed radius** (Gauss-Newton), removing the radius/centre ambiguity of a partial arc.

### 🧲 Stack Align for AutoStakkert (optional, default ON)
Hand-held / alt-az shots from one session are not only shifted but also **rotated** relative to each other (test set: ~3.5° between frames ⇒ ~70 px movement at the limb). Centering alone cannot fix that, and AutoStakkert then produces a blurred stack.

5. **De-rotation** — The crescent direction (limb centre → lit-area centroid) is measured per photo; every frame is rotated about the moon centre to the median orientation.
6. **Sub-pixel ECC registration** — Each frame is registered to the median stack with OpenCV `findTransformECC` (rotation + translation, 3 iterations) on the moon area. The mean shift is removed so the moon stays exactly at the photo centre.

Measured residual misalignment on the test set: **7.6 px (old) → ~0.2 px (new)**.

> Untick **Stack align** for multi-night *phase* sequences — the crescent shape and orientation legitimately change between nights, so only centering (no rotation / registration) is applied.

### 🎯 Centering & Background Fill
7. **Background sampling** — Sky colour is sampled from an annular ring of pixels **just outside the moon's edge** (radius+20 to radius+120 px). These pixels are guaranteed to be sky, giving an accurate fill colour even when the moon fills most of the frame.
8. **Sub-pixel transform** — One combined shift (+ rotation in stack mode) moves the fitted centre `(cx, cy)` exactly to the canvas centre, using bicubic interpolation (no integer rounding).
9. **Gap fill** — The revealed border after shifting is filled with the **sampled sky colour**. Blue sky stays blue; dark sky stays dark.
10. **Save** — Centered image written to the output folder with the **original filename** (JPEG quality 98).

All output images keep the **exact same resolution** as the source.

---

## 🖼️ UI Features

| Feature | Details |
|---|---|
| **📁 Folder Setup** | Input / Output / Rejected path fields with inline **Browse…** navigator |
| **🧲 Stack align checkbox** | De-rotate + sub-pixel register same-session frames for AutoStakkert (default on) |
| **📷 Source Photos tab** | Thumbnail gallery of all input images |
| **🎯 Centered Output tab** | Thumbnail gallery of processed results |
| **🚫 Rejected tab** | Thumbnail gallery of focal-length-rejected photos |
| **⇄ Before / After compare** | Click any source photo → lightbox → **Compare** button shows side-by-side |
| **Stats bar** | Total · Processed · Centered OK · Rejected (FL) · Skipped/Error |
| **Processing log** | Live progress bar + per-file status (✅ ok / ⚠️ skipped / 🚫 rejected / ❌ error), incl. sub-pixel shift and rotation applied |

---

## ⚠️ Restrictions & Known Limitations

| Limitation | Detail |
|---|---|
| **Mixed focal lengths** | Photos with different focal lengths (e.g. mixed zoom settings) are automatically separated into the `reject/` folder. |
| **No EXIF data** | If photos have no EXIF focal length, the moon-radius check is used instead. |
| **Heavy overcast / fog** | If the moon is barely visible behind thick clouds, detection may fail. |
| **One dominant object** | The largest bright blob is treated as the moon. Bright lamps or sun glare near the moon can confuse detection. |
| **No zoom/scale change** | The tool only **shifts** (and, in Stack align mode, **rotates**) — it does not resize or crop. Output resolution equals input resolution. |
| **Stack align = same session** | Registration assumes the moon looks the same in all frames. Do not use it for photos taken on different nights. |
| **Moon near edge / large moon** | If the moon disc extends beyond the frame edge, the shifted output will show a background-filled band on the opposite side. The fill colour is sampled from sky pixels near the moon's edge for a natural look. |
| **Supported formats** | `.jpg`, `.jpeg`, `.png` (case-insensitive). RAW files (`.CR2`, `.NEF`, etc.) are **not** supported. |
| **Local only** | The server binds to `127.0.0.1` — not accessible over the network. |
| **No overwrite warning** | Re-running will silently overwrite files already in the output folder. |

---

## 🛠️ Dependencies

Uses the shared virtual environment at `Z:\antigravity\venv`:

- `opencv-python` ≥ 4.10
- `numpy` ≥ 2.2
- `flask`
- `piexif` ≥ 1.1.3 ← for EXIF focal length reading

No additional installation needed — `piexif` is already installed in the shared venv.

---

## 📝 Tips

- If a photo is marked **⚠️ skipped (Moon not detected)**, the moon may be very small, heavily obscured, or its edge arc too faint for the circle fit to converge.
- For **daytime photos**, a clear blue sky gives the best detection results.
- For **night photos**, ensure the moon is reasonably bright relative to the surroundings (avoid shots dominated by very bright foreground lights).
- **Background fill** matches the sky colour automatically — no manual adjustment needed.
- Centered output images keep the **exact same filename** as the source, preserving the sequence order for timelapse assembly.
- For **AutoStakkert!4**: feed the `centered/` folder with Stack align on; the frames are already globally aligned, so AS4's alignment points only need to handle seeing.

## Implementation update

The centre is fitted from the visible lunar **limb** only (terminator and frame-clipped edges excluded), using one consensus moon radius for the whole batch. This is important when the moon is close to or partly outside the image edge: the visible arc can still be used to extrapolate the complete circle centre.

In Stack align mode, frames are additionally de-rotated to a common crescent orientation and sub-pixel registered (ECC) against the median stack, so the output stacks without blur. Photos with a mismatched EXIF focal length are copied to `reject/`; when EXIF data is missing, a substantial mismatch in detected moon radius is used as the practical focal-length/zoom check. Skipped images are logged when no reliable moon circle can be found.

The GUI default input path is `E:\AI\codex\moon-phase\input`. Output defaults to `<input>\centered`, rejected images default to `<input>\reject`, and output files retain their original names and exact source dimensions.

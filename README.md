# 🌙 Moon Phase Auto-Centering Tool (v2)

A local web utility (liquid glass UI) that detects the moon in each photo and **pulls the moon centre to the photo centre**. It works for night and daytime (blue sky) shots, including very thin crescents.
Results go to a configurable output folder (default: `centered/` under the source path), keeping the **original filename and exact size**.

Derived from `moon-phase`, with these changes:
![center moon](https://github.com/inchinet/moon-phase/blob/main/screen.png)

| | moon-phase | moon-phase2 |
|---|---|---|
| Focal-length (EXIF) rejection | yes | **removed** |
| Shared "consensus" moon radius | yes | **removed** – each photo uses its own fitted radius |
| Rotation / stack alignment (ECC) | optional | **removed** – pure shift only |
| Moon detection | limb fit / Hough | **disc detector** (round bright blob) for full/gibbous moons, then **crescent detector** (top-hat), limb/Hough as fallback |

---

## 🚀 Quick Start

Double-click **`run.bat`**. It starts the Flask server and opens the browser.

In the UI:
1. Set the **Input** folder (or use **Browse…**), e.g. `Z:\antigravity\moon-phase2\input`
2. Optionally change the **Output** (default `<input>\centered`) and **Rejected** paths
3. Click **⚡ Start Centering**

---

## 📁 File Structure

```
moon-phase2/
├── input/              ← Source moon photos
├── input/centered/     ← Auto-created; centred output (same filenames)
├── input/reject/       ← Auto-created (no longer used for focal-length rejects)
├── app.py              ← Flask backend + OpenCV logic
├── index.html          ← Web UI
├── run.bat             ← Launcher (Windows)
└── README.md           ← This file
```

---

## ⚙️ How It Works

1. **Read** each photo independently (EXIF focal length is shown if present, never used to reject).
2. **Detect a full/gibbous moon** – the image is thresholded at several levels (including absolute levels for dim orange moons) and the roundest compact bright blob is kept (it must be convex and fill its enclosing circle). City lights, windows and buildings are irregular, so they are rejected. A circle is fitted to the blob's convex hull, so a plane or bird crossing the disc does not shift the centre.
3. **Detect the crescent** – if no round disc is found, a white top-hat filter keeps thin bright structures (the crescent) and drops smooth sky gradients and wide edges. Caption text bands at the top/bottom are ignored, and very dark ground (hills, trees) and its rim are masked out so a horizon is not mistaken for the moon. The biggest valid blob is kept.
4. **Fit the circle** – a circle is fitted to the *outer* (convex) edge of the crescent, giving that photo's own centre `(cx, cy)` and radius. The terminator (inner edge) is ignored.
5. **Fallback** – if neither detector finds anything, the older limb fit, then Hough circles, are tried.
6. **Shift** – the image is translated so `(cx, cy)` lands exactly on the photo centre (bicubic, sub-pixel). No rotation, no scaling.
7. **Fill** – the exposed border is filled with a flat colour sampled from the sky near the moon.
8. **Save** – written with the original filename (JPEG quality 98), same resolution as the source.

Different zoom levels are fine: each photo gets its own radius.

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| **Heavy cloud / fog** | If the crescent is almost invisible, detection may fail and the photo is logged as skipped. |
| **Very faint crescents** | Thin crescents rely on the top-hat threshold; check the result visually. |
| **Bright objects** | Bright blobs (Venus, lamps, glare) near the moon can confuse detection. |
| **Moon near the edge** | The shifted output shows a flat-colour band on the opposite side. |
| **No resize/crop** | Output size equals input size. |
| **Formats** | `.jpg`, `.jpeg`, `.png`. RAW is not supported. |
| **Local only** | Server binds to `127.0.0.1`. |
| **Overwrite** | Re-running silently overwrites files in the output folder. |

---

## 🛠️ Dependencies

Uses the shared venv at `Z:\antigravity\venv`: `opencv-python`, `numpy`, `flask`, `piexif` (optional, for the focal length display only).

---

## 💡 Tips

- Each result line in the log shows the detected centre, radius and the shift applied. Use it to spot a wrong detection.
- Tested on 6 photos with different zooms, thin crescents, a horizon and caption text; all centred.
- The old "Stack align" checkbox in the UI is now ignored, since there is no rotation or registration.

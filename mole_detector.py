import argparse
import os
import cv2
import numpy as np

WIDTH = 800  # every image is resized to this width


def ellipse(size):
    size = max(3, int(size)) | 1          # odd, at least 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def color_distance_map(img, shade):
    # measure distance from normal skin colour.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0]
    background = cv2.GaussianBlur(L, (0, 0), sigmaX=WIDTH * shade)
    lab[:, :, 0] = L - background + background.mean()

    # normal skin colour = median colour of the image
    # the mole is small, so the median is skin.
    skin = np.median(lab.reshape(-1, 3), axis=0)
    diff = np.sqrt(((lab - skin) ** 2).sum(axis=2))

    diff[lab[:, :, 0] > skin[0] - 2] = 0

    return cv2.GaussianBlur(diff, (0, 0), sigmaX=3)


def refine_with_grabcut(img, contour):
    H, W = img.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    pad = int(max(w, h) * 0.5)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    roi = img[y0:y1, x0:x1]

    rough = np.zeros(roi.shape[:2], np.uint8)
    shifted = (contour - np.array([x0, y0])).astype(np.int32)
    cv2.drawContours(rough, [shifted], -1, 1, -1)

    gc = np.full(roi.shape[:2], cv2.GC_BGD, np.uint8)            # far away = background
    near = cv2.dilate(rough, ellipse(max(w, h) * 0.6))
    gc[near > 0] = cv2.GC_PR_BGD                                 # near = probably background
    gc[rough > 0] = cv2.GC_PR_FGD                                # rough mole = probably mole
    core = cv2.erode(rough, ellipse(min(w, h) * 0.3))
    gc[core > 0] = cv2.GC_FGD                                    # centre = surely mole

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return contour                                           # fall back to rough contour

    result = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, ellipse(5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, ellipse(9))

    cnts, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return contour
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.3 * cv2.contourArea(contour):   # GrabCut collapsed
        return contour
    return (best + np.array([x0, y0])).astype(np.int32)


def detect_moles(image_bgr, a):
    # Resize so the same settings work for any photo
    h, w = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (WIDTH, int(h * WIDTH / w)))
    H, W = img.shape[:2]

    diff = color_distance_map(img, a.shade)

    # Otsu on the distance map, with a floor so a mole-free photo gives nothing
    hi = max(float(np.percentile(diff, 99.5)), 1e-6)
    diff8 = np.clip(diff / hi * 255, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 255 * hi, a.min_delta)
    # print(f"  colour-distance threshold: {thr:.1f}  (otsu {otsu / 255 * hi:.1f}, floor {a.min_delta})")

    rough = (diff > thr).astype(np.uint8) * 255
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, ellipse(5))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, ellipse(11))

    # filter candidates
    cnts, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    margin = 3
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a.min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= margin or y <= margin or x + bw >= W - margin or y + bh >= H - margin:
            continue                                   # touches the border = shading, not a mole
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0 or 4 * np.pi * area / perimeter ** 2 < a.circ:
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        strength = cv2.mean(diff, mask=m)[0]
        candidates.append((strength * np.sqrt(area), c))

    moles = []
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)   # strongest first
        best = candidates[0][0]
        for s, c in candidates:
            if a.single and moles:
                break                                        # keep only the strongest one
            if s >= a.rel * best:
                moles.append(c if a.no_grabcut else refine_with_grabcut(img, c))

    # Final mask: white where a mole is, black everywhere else
    final = np.zeros((H, W), np.uint8)
    cv2.drawContours(final, moles, -1, 255, -1)

    # --- Hide everything except the mole(s) ---
    # 1) Mole on a black background
    mole_only = np.zeros_like(img)
    mole_only[final > 0] = img[final > 0]

    # 2) Mole on a transparent background (PNG with alpha channel)
    b, g, r = cv2.split(img)
    cutout = cv2.merge([b, g, r, final])

    # 3) Skin dimmed to 15%, mole shown normally, green outline
    focus = (img * 0.15).astype(np.uint8)
    focus[final > 0] = img[final > 0]
    cv2.drawContours(focus, moles, -1, (0, 255, 0), 2)

    # Boundary on the original photo
    outline = img.copy()
    cv2.drawContours(outline, moles, -1, (0, 255, 0), 2)

    # Translucent red highlight on the original photo
    overlay = img.copy()
    cv2.drawContours(overlay, moles, -1, (0, 0, 255), -1)
    highlight = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    return {"original": img, "distance": diff8, "rough": rough, "mask": final,
            "outline": outline, "highlight": highlight, "count": len(moles),
            "mole_only": mole_only, "cutout": cutout, "focus": focus}


def process_file(path, out_dir, a):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read: {path}")
        return
    print(os.path.basename(path))
    r = detect_moles(img, a)
    name = os.path.splitext(os.path.basename(path))[0]

    # for key in ["outline", "highlight", "mask", "rough", "distance", "mole_only", "focus"]:
    for key in ["outline", "highlight","mole_only", "focus"]:
        cv2.imwrite(os.path.join(out_dir, f"{name}_{key}.jpg"), r[key])

    # transparent cutout must be PNG (JPG cannot store transparency)
#   1  cv2.imwrite(os.path.join(out_dir, f"{name}_cutout.png"), r["cutout"])

    combined = np.hstack([r["original"], r["outline"], r["mole_only"]])
# 2   cv2.imwrite(os.path.join(out_dir, f"{name}_combined.jpg"), combined)
    print(f"  -> {r['count']} mole(s) found")


def main():
    p = argparse.ArgumentParser(description="Mole detection (OpenCV, no AI)")
    p.add_argument("image", nargs="?", help="image path (default: all in images/)")
    p.add_argument("--shade", type=float, default=0.15,
                   help="shading blur size as a fraction of width; must be bigger than the mole")
    p.add_argument("--min-delta", type=float, default=12,
                   help="minimum colour difference from skin; lower = more sensitive")
    p.add_argument("--rel", type=float, default=0.3,
                   help="drop candidates weaker than this fraction of the best one")
    p.add_argument("--min-area", type=int, default=150, help="smallest region in pixels")
    p.add_argument("--circ", type=float, default=0.3, help="min roundness, 0 to 1")
    p.add_argument("--single", action="store_true", help="keep only the strongest mole")
    p.add_argument("--no-grabcut", action="store_true", help="skip the boundary refinement")
    a = p.parse_args()

    os.makedirs("output", exist_ok=True)
    if a.image:
        files = [a.image]
    else:
        files = [os.path.join("images", f) for f in sorted(os.listdir("images"))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        if not files:
            print("No images found in images/")
            return
    for f in files:
        process_file(f, "output", a)
    print("Done. Open the output/ folder.")


if __name__ == "__main__":
    main()
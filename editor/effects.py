"""Turn (frame, label map, objects, background settings) into the final picture.

Used for both the live "Preview" in the editor and for export, so what you see is what you get.

Per object effects:
  none       - nothing (object can still act as foreground for the background effect)
  highlight  - colored overlay + outline
  outline    - outline only
  blur       - blur the object (privacy)
  pixelate   - mosaic the object
  recolor    - change the object's hue to the chosen color, keep shading
  erase      - remove the object (inpaint from surroundings)
  cutout     - only for preview clarity: same as none

Background (everything outside foreground objects):
  keep, remove (transparent), color, blur, grayscale, image
"""

import cv2
import numpy as np


def hex_to_rgb(h, default=(255, 79, 160)):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    except (AttributeError, ValueError):
        return default


def soft_mask(binary, grow=0, feather=0):
    """bool mask -> float32 alpha in [0,1] after grow/shrink (px) and edge feather (px)."""
    m = binary.astype(np.uint8)
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1, 2 * abs(grow) + 1))
        m = cv2.dilate(m, k) if grow > 0 else cv2.erode(m, k)
    a = m.astype(np.float32)
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    return a


def _blur(img, strength):
    s = max(1, int(strength))
    return cv2.GaussianBlur(img, (0, 0), s)


def _pixelate(img, block):
    h, w = img.shape[:2]
    b = max(2, int(block))
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _recolor(img, rgb):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    target = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    hsv[..., 0] = target[0]
    hsv[..., 1] = np.maximum(hsv[..., 1], target[1] // 2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _mix(base, layer, alpha):
    a = alpha[..., None]
    return base * (1 - a) + layer * a


def _cover(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (max(w, round(iw * s)), max(h, round(ih * s))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y : y + h, x : x + w]


def checkerboard(h, w, size=16):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // size) + (xx // size)) % 2).astype(np.float32)
    return np.repeat((200 + 55 * c)[..., None], 3, axis=2)


def object_alphas(labels, objects, neighbors=None):
    """{obj_id: soft alpha}. `neighbors` = other frames' label maps for temporal smoothing."""
    out = {}
    for o in objects:
        if not o.get("visible", True):
            continue
        eff = o.get("effect", {})
        grow, feather = int(eff.get("grow", 0)), float(eff.get("feather", 2))
        a = soft_mask(labels == o["id"], grow, feather)
        if neighbors:
            for nb in neighbors:
                a = a + soft_mask(nb == o["id"], grow, feather)
            a = a / (1 + len(neighbors))
        out[o["id"]] = a
    return out


def render(frame, labels, objects, background, bg_image=None, neighbors=None, preview=True):
    """frame: RGB uint8. labels: uint8 label map (same size) or None.

    Returns RGB uint8, or RGBA uint8 when background mode is 'remove' and preview=False.
    """
    h, w = frame.shape[:2]
    img = frame.astype(np.float32)
    if labels is None:
        labels = np.zeros((h, w), np.uint8)
    elif labels.shape != (h, w):
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        if neighbors:
            neighbors = [cv2.resize(nb, (w, h), interpolation=cv2.INTER_NEAREST) for nb in neighbors]

    alphas = object_alphas(labels, objects, neighbors)
    fg_ids = [o["id"] for o in objects if o.get("foreground", True) and o["id"] in alphas]
    fg = np.zeros((h, w), np.float32)
    for oid in fg_ids:
        fg = np.maximum(fg, alphas[oid])

    mode = (background or {}).get("mode", "keep")
    out_alpha = None
    if mode != "keep" and fg_ids:
        strength = (background or {}).get("strength", 25)
        if mode == "remove":
            if preview:
                img = _mix(checkerboard(h, w), img, fg)
            else:
                out_alpha = fg
        elif mode == "color":
            bg = np.empty_like(img)
            bg[:] = hex_to_rgb(background.get("color"), (0, 177, 64))
            img = _mix(bg, img, fg)
        elif mode == "blur":
            # normalized blur of background pixels only, so the object's colors don't bleed out as a halo
            w_bg = (1 - fg)[..., None]
            num = _blur(img * w_bg, strength)
            den = _blur(np.repeat(w_bg, 3, axis=2), strength)
            bg_blur = np.where(den > 1e-3, num / np.maximum(den, 1e-3), _blur(img, strength))
            img = _mix(bg_blur, img, fg)
        elif mode == "grayscale":
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            img = _mix(np.repeat(g[..., None], 3, axis=2), img, fg)
        elif mode == "image" and bg_image is not None:
            img = _mix(_cover(bg_image, w, h).astype(np.float32), img, fg)

    for o in objects:
        a = alphas.get(o["id"])
        if a is None or not a.any():
            continue
        eff = o.get("effect", {})
        kind = eff.get("type", "highlight")
        rgb = hex_to_rgb(eff.get("color") or o.get("color"))
        opacity = float(eff.get("opacity", 0.5))
        strength = float(eff.get("strength", 15))
        if kind == "highlight":
            layer = np.empty_like(img)
            layer[:] = rgb
            img = _mix(img, layer, a * opacity)
        elif kind == "blur":
            img = _mix(img, _blur(img, strength), a)
        elif kind == "pixelate":
            img = _mix(img, _pixelate(img.astype(np.uint8), strength).astype(np.float32), a)
        elif kind == "recolor":
            recolored = _recolor(np.clip(img, 0, 255).astype(np.uint8), rgb).astype(np.float32)
            img = _mix(img, recolored, a * min(1.0, opacity * 2))  # opacity >= 0.5 means full recolor
        elif kind == "erase":
            hole = (a > 0.3).astype(np.uint8) * 255
            hole = cv2.dilate(hole, np.ones((7, 7), np.uint8))
            filled = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), hole, 5, cv2.INPAINT_TELEA)
            img = _mix(img, filled.astype(np.float32), np.maximum(a, hole / 255.0))
        if kind in ("highlight", "outline"):
            thick = max(1, int(eff.get("outline", 2)))
            contours, _ = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tmp = np.clip(img, 0, 255).astype(np.uint8)
            cv2.drawContours(tmp, contours, -1, rgb if kind == "outline" else (255, 255, 255), thick, cv2.LINE_AA)
            img = tmp.astype(np.float32)

    out = np.clip(img, 0, 255).astype(np.uint8)
    if out_alpha is not None:
        return np.dstack([out, (out_alpha * 255).astype(np.uint8)])
    return out

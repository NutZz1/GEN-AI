"""Morph animations and multi-parent blend canvases."""
import io

import numpy as np
from PIL import Image, ImageDraw

from avatarblend import decode_np


def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)  # smooth start/stop


def morph_frames(model, keyframes, frames_per_step=16):
    """Decode a smooth path through a list of latents (A -> B, or a whole lineage)."""
    zs = []
    for za, zb in zip(keyframes[:-1], keyframes[1:]):
        zs += [(1 - _ease(t)) * za + _ease(t) * zb for t in np.linspace(0, 1, frames_per_step, endpoint=False)]
    zs.append(keyframes[-1])
    return decode_np(model, np.stack(zs).astype(np.float32))


def gif_bytes(frames, size=256, ms=60, hold=12, pingpong=True, captions=None):
    """Animated GIF (loops). `hold` repeats the first/last frame; `captions` = {frame_index: text}."""
    ims = [Image.fromarray(f).resize((size, size), Image.LANCZOS) for f in frames]
    if captions:
        for i, text in captions.items():
            d = ImageDraw.Draw(ims[i])
            d.rectangle((0, size - 18, size, size), fill="white")
            d.text((6, size - 16), text, fill="black")
    seq = [ims[0]] * hold + ims + [ims[-1]] * hold
    if pingpong:
        seq += ims[::-1]
    buf = io.BytesIO()
    seq[0].save(buf, format="GIF", save_all=True, append_images=seq[1:], duration=ms, loop=0, disposal=2)
    return buf.getvalue()


def triangle_weights(steps):
    """Barycentric lattice rows: row r has r+1 points (corner 0 at the top)."""
    rows = []
    for r in range(steps + 1):
        rows.append([np.array([steps - r, r - j, j], dtype=float) / steps for j in range(r + 1)])
    return rows


def square_weights(steps):
    """Bilinear grid over 4 corners (top-left, top-right, bottom-left, bottom-right)."""
    rows = []
    for i in range(steps + 1):
        v = i / steps
        rows.append([np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])
                     for u in np.linspace(0, 1, steps + 1)])
    return rows


def canvas(model, parent_z, parent_imgs, steps=5, cell=96):
    """Decode every weighted mix of 3 (triangle) or 4 (square) parents into one image.
    Corner cells show the parents' own reconstructions, outlined."""
    k = len(parent_z)
    rows = triangle_weights(steps) if k == 3 else square_weights(steps)
    ws = np.array([w for r in rows for w in r])
    imgs = iter(decode_np(model, (ws @ np.stack(parent_z)).astype(np.float32)))
    n_cols = max(len(r) for r in rows)
    pad = 6
    W, H = n_cols * (cell + pad), len(rows) * (cell + pad)
    out = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(out)
    for ri, r in enumerate(rows):
        x0 = (W - len(r) * (cell + pad)) // 2  # centre the row -> triangle shape
        for ci, w in enumerate(r):
            x, y = x0 + ci * (cell + pad), ri * (cell + pad)
            out.paste(Image.fromarray(next(imgs)).resize((cell, cell), Image.LANCZOS), (x, y))
            if w.max() > 0.999:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(214, 42, 42), width=3)
                draw.text((x + 4, y + 2), "ABCD"[int(w.argmax())], fill=(214, 42, 42))
    return out

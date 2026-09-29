"""Cartoon Set 10k (extracted folder or .tgz) -> data/cartoonset64.npz.

Each 500x500 RGBA avatar is composited on white, center-cropped to the region the
face and hair occupy, and resized to --size (64 or 128). The 18 attribute labels from the
matching .csv are stored alongside, plus a fixed 80/10/10 train/val/test split.

Dataset: Google Cartoon Set (CC-BY-4.0), https://google.github.io/cartoonset/
Mirror used for download: https://huggingface.co/datasets/cgarciae/cartoonset
"""
import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
TGZ = ROOT / "data" / "cartoonset10k.tgz"
# extracted archive: data/cartoonset10k/cartoonset10k/*.png + *.csv
FOLDER_CANDIDATES = [ROOT / "data" / "cartoonset10k" / "cartoonset10k", ROOT / "data" / "cartoonset10k"]
CROP = (40, 40, 460, 460)  # left, top, right, bottom in the 500x500 source


def process_png(raw, size):
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB").crop(CROP)
    return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def parse_csv(raw):
    names, vals, nvar = [], [], []
    for line in raw.decode().strip().splitlines():
        name, v, n = [s.strip().strip('"') for s in line.split(",")]
        names.append(name)
        vals.append(int(v))
        nvar.append(int(n))
    return names, vals, nvar


def iter_files():
    """Yield (filename, bytes) from the extracted folder if present, else stream the .tgz."""
    for folder in FOLDER_CANDIDATES:
        if folder.is_dir() and any(folder.glob("*.png")):
            print(f"reading extracted folder {folder}")
            for f in sorted(folder.iterdir()):
                if f.suffix in (".png", ".csv"):
                    yield f.name, f.read_bytes()
            return
    print(f"reading archive {TGZ}")
    with tarfile.open(TGZ, "r|gz") as t:
        for m in t:
            if m.isfile():
                yield m.name.rsplit("/", 1)[-1], t.extractfile(m).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, choices=[64, 128])
    size = ap.parse_args().size
    out = ROOT / "data" / f"cartoonset{size}.npz"
    imgs, attrs = {}, {}
    names = nvar = None
    for fname, raw in iter_files():
        stem, ext = fname.rsplit(".", 1)
        if ext == "png":
            imgs[stem] = process_png(raw, size)
            if len(imgs) % 1000 == 0:
                print(f"{len(imgs)} images", flush=True)
        elif ext == "csv":
            names, vals, nvar = parse_csv(raw)
            attrs[stem] = vals

    keys = sorted(set(imgs) & set(attrs))
    rng = np.random.default_rng(0)
    split = np.zeros(len(keys), dtype=np.int8)
    perm = rng.permutation(len(keys))
    n_val = n_test = len(keys) // 10
    split[perm[:n_val]] = 1
    split[perm[n_val:n_val + n_test]] = 2

    np.savez_compressed(
        out,
        images=np.stack([imgs[k] for k in keys]),
        attrs=np.array([attrs[k] for k in keys], dtype=np.int16),
        attr_names=np.array(names),
        attr_nvariants=np.array(nvar, dtype=np.int16),
        ids=np.array(keys),
        split=split,
    )
    print(f"saved {len(keys)} avatars -> {out}")


if __name__ == "__main__":
    main()

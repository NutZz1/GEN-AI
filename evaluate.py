"""Feasibility evaluation for AvatarBlend. Writes figures + metrics to results/.

1. Test-set reconstruction (BCE, KL, MSE, PSNR).
2. Figures: reconstructions, prior samples, latent vs pixel blends, trait mixing, editing, training curves.
3. Semantic checks with an auxiliary attribute classifier trained on real avatars (never seen by the VAE):
   - reconstructions keep their attributes?
   - whole-avatar blends move attributes from A to B as alpha falls? (proposal claim)
   - trait-selective blend: swapped trait follows its donor while the others stay? (new feature)
   - attribute editing: does the edited trait hit the requested variant? (new feature)

    python evaluate.py                          # default model (see avatarblend.PREFERRED)
    python evaluate.py --ckpt vae128_acai.pt    # a specific checkpoint -> results/vae128_acai/

4. Blend realism: Frechet distance (FID-style) between judge-classifier features of real test avatars and
   of decoded 50/50 blends / random prior samples. Lower = closer to real avatars.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avatarblend import (CKPT_DIR, ROOT, decode_np, default_ckpt, describe_device, get_device, load_dataset,
                         load_model, to_tensor, to_uint8, vae_loss)
from traits import LABELS, build

# fixed head order of the cached judge classifier (checkpoints/attr_classifier.pt)
TRAITS = ["glasses", "hair_color", "face_color", "facial_hair", "hair", "face_shape", "eye_color"]

OUT = ROOT / "results"
ALPHAS = [0.75, 0.5, 0.25]
torch.manual_seed(0)
DEVICE = get_device()


class AttrClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        blk = lambda i, o: [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
        self.body = nn.Sequential(*blk(3, 48), *blk(48, 96), *blk(96, 192), *blk(192, 256),
                                  nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(0.3))
        self.heads = nn.ModuleList(nn.Linear(1024, n) for n in n_classes)

    def forward(self, x):
        h = self.body(F.interpolate(x, size=64, mode="area") if x.shape[-1] != 64 else x)
        return [head(h) for head in self.heads]


def train_classifier(X, Y, n_classes, epochs=25):
    clf = AttrClassifier(n_classes).to(X.device)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=X.device)
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            loss = sum(F.cross_entropy(o, Y[b, k]) for k, o in enumerate(clf(X[b])))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        print(f"  classifier epoch {ep + 1}/{epochs} loss {loss.item():.3f}", flush=True)
    return clf.eval()


@torch.no_grad()
def predict(clf, X):
    """-> list (per trait) of (N, n_cls) probabilities."""
    outs = [[] for _ in clf.heads]
    for i in range(0, len(X), 256):
        for k, o in enumerate(clf(X[i:i + 256])):
            outs[k].append(F.softmax(o, 1))
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def judge_features(clf, X):
    """256-d globally pooled features of the judge classifier (for the Frechet distance)."""
    body = clf.body[:-3]  # up to the last conv block, before pooling/flatten/dropout
    out = []
    for i in range(0, len(X), 256):
        x = X[i:i + 256]
        x = F.interpolate(x, size=64, mode="area") if x.shape[-1] != 64 else x
        out.append(body(x).mean((2, 3)).double().cpu())
    return torch.cat(out).numpy()


def frechet(f1, f2):
    """Frechet distance between Gaussians fitted to two feature sets (as in FID)."""
    m1, m2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    ev, U = np.linalg.eigh(s1)
    r1 = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(r1 @ s2 @ r1), 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def dec(model, z_np):
    """numpy latents -> float image tensor on DEVICE (batched)."""
    return torch.cat([model.decode(torch.as_tensor(z_np[i:i + 256], device=DEVICE))
                      for i in range(0, len(z_np), 256)])


def grid(rows, titles, path, row_labels=None):
    r, c = len(rows), len(rows[0])
    fig, axes = plt.subplots(r, c, figsize=(c * 1.35, r * 1.45))
    axes = np.atleast_2d(axes)
    for i in range(r):
        for j in range(c):
            ax = axes[i, j]
            ax.imshow(rows[i][j])
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0 and titles:
                ax.set_title(titles[j], fontsize=7)
        if row_labels:
            axes[i, 0].set_ylabel(row_labels[i], fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint file name in checkpoints/ (default: preferred)")
    ckpt = ap.parse_args().ckpt
    ckpt = CKPT_DIR / ckpt if ckpt else default_ckpt()
    model = load_model(ckpt, device=DEVICE)
    S = model.img_size
    out_dir = OUT / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {describe_device(DEVICE)}  model: {ckpt.name}")
    data = load_dataset(S)
    names = list(data["attr_names"])
    cols = [names.index(a) for a in TRAITS]
    n_classes = [int(data["attr_nvariants"][c]) for c in cols]
    split = data["split"]
    te = np.nonzero(split == 2)[0]
    X = to_tensor(data["images"]).to(DEVICE)
    Y = torch.from_numpy(data["attrs"][:, cols].astype(np.int64)).to(DEVICE)
    metrics = {"model": ckpt.name, "img_size": S, "latent_dim": model.latent_dim}

    # ---- 1. reconstruction on held-out test set
    Xte, Yte = X[te], Y[te]
    mu_t, logvar_t = model.encode(Xte)
    rec = model.decode(mu_t)
    _, bce, kl = vae_loss(rec, Xte, mu_t, logvar_t)
    mse = F.mse_loss(rec, Xte, reduction="none").mean((1, 2, 3))
    metrics.update(test_bce_per_image=bce.item(), test_kl_per_image=kl.item(),
                   test_recon_mse=mse.mean().item(), test_recon_psnr_db=(10 * torch.log10(1 / mse)).mean().item(),
                   active_latent_dims=int((mu_t.var(0) > 0.01).sum()))

    # ---- trait space (fit on train split latents only)
    traits, mu = build(model, data)
    lab = lambda i: traits.labels_from_attrs(data["attrs"][i], names)
    metrics["lda_variant_inference_acc_test"] = {t: float(np.mean([traits.infer_labels(mu[i])[t] == lab(i)[t] for i in te]))
                                                 for t in traits.traits}
    swap_one = lambda a, b, t: traits.selective_blend(mu[a], lab(a), mu[b], lab(b), {s: float(s == t) for s in traits.traits}, 0.0)[0]

    # ---- 2. figures
    idx = torch.arange(10, device=DEVICE)
    grid([to_uint8(Xte[idx]), to_uint8(rec[idx])], None, out_dir / "reconstructions.png", ["original", "VAE recon"])
    samples = to_uint8(model.decode(torch.randn(20, model.latent_dim, device=DEVICE)))
    grid([samples[:10], samples[10:]], None, out_dir / "prior_samples.png")

    rng = np.random.default_rng(1)
    pairs = rng.integers(0, len(te), (600, 2))
    pairs = te[pairs[pairs[:, 0] != pairs[:, 1]]]  # dataset indices
    rows, labels = [], []
    for k in range(4):
        a, b = pairs[k]
        lat = decode_np(model, np.stack([al * mu[a] + (1 - al) * mu[b] for al in ALPHAS]))
        pa, pb = data["images"][a], data["images"][b]
        pix = [(al * pa.astype(float) + (1 - al) * pb.astype(float)).astype(np.uint8) for al in ALPHAS]
        rows += [[pa, *lat, pb], [pa, *pix, pb]]
        labels += [f"pair {k + 1}: latent", "pixel blend"]
    grid(rows, ["Parent A"] + [f"a={a}" for a in ALPHAS] + ["Parent B"], out_dir / "blend_latent_vs_pixel.png", labels)

    a, b = pairs[5]
    strip = decode_np(model, np.stack([mu[a] * (1 - t) + mu[b] * t for t in np.linspace(0, 1, 10)]))
    grid([strip], None, out_dir / "interpolation_strip.png")

    # trait mixer figure: base = B, take one trait at a time from A
    rows, labels = [], []
    for k in range(3):
        a, b = pairs[10 + k]
        outs = [decode_np(model, swap_one(a, b, t))[0] for t in traits.traits]
        rows.append([data["images"][a], *outs, data["images"][b]])
        labels.append(f"pair {k + 1}")
    grid(rows, ["Parent A (donor)"] + [f"{LABELS[t]}\nfrom A" for t in traits.traits] + ["Parent B (base)"],
         out_dir / "trait_mixer.png", labels)

    # editor figure: sweep hair colour and glasses variants on one avatar
    subj = te[3]
    rows = []
    for t in ["hair_color", "glasses", "face_color"]:
        n = traits.n_variants(t)
        rows.append([data["images"][subj]] + list(decode_np(model, np.stack(
            [traits.edit(mu[subj], lab(subj), t, v, 1.0)[0] for v in range(min(n, 8))]))))
    grid(rows, ["original"] + [f"variant {v}" for v in range(8)], out_dir / "attribute_editing.png",
         ["hair colour", "glasses", "skin tone"])

    hist_path = CKPT_DIR / f"history{model.name[3:]}.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        ep = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        for key in ("train", "val"):
            ax[0].plot(ep, [h[key][1] for h in hist], label=key)
            ax[1].plot(ep, [h[key][2] for h in hist], label=key)
        ax[0].set_title("reconstruction (BCE / image)"), ax[1].set_title("KL / image")
        for x in ax:
            x.set_xlabel("epoch"), x.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=110)
        plt.close(fig)

    # ---- 3. semantic evaluation with auxiliary classifier
    clf_path = CKPT_DIR / "attr_classifier.pt"  # judge: trained on real 64px-equivalent images only
    if clf_path.exists():
        clf = AttrClassifier(n_classes).to(DEVICE)
        clf.load_state_dict(torch.load(clf_path, map_location=DEVICE))
        clf.eval()
    else:
        with torch.enable_grad():
            print("training auxiliary attribute classifier on real train images...")
            clf = train_classifier(X[np.nonzero(split == 0)[0]], Y[np.nonzero(split == 0)[0]], n_classes)
        torch.save(clf.state_dict(), clf_path)
    acc = lambda P, T: {t: (p.argmax(1) == T[:, i]).float().mean().item() for i, (t, p) in enumerate(zip(TRAITS, P))}
    metrics["classifier_acc_real_test"] = acc(predict(clf, Xte), Yte)
    metrics["classifier_acc_vae_recon"] = acc(predict(clf, rec), Yte)

    A, B = pairs[:, 0], pairs[:, 1]
    YA, YB = Y[A], Y[B]
    # (a) whole-avatar blends
    blend_stats = {}
    for al in ALPHAS:
        P = predict(clf, dec(model, al * mu[A] + (1 - al) * mu[B]))
        Ppix = predict(clf, al * X[A] + (1 - al) * X[B])
        row = {}
        for i, t in enumerate(TRAITS):
            diff = YA[:, i] != YB[:, i]
            pred = P[i].argmax(1)[diff]
            row[t] = {"matches_A": (pred == YA[diff, i]).float().mean().item(),
                      "matches_B": (pred == YB[diff, i]).float().mean().item(),
                      "confidence_latent": P[i].max(1).values.mean().item(),
                      "confidence_pixel": Ppix[i].max(1).values.mean().item()}
        blend_stats[str(al)] = row
    metrics["whole_blend_attribute_transfer"] = blend_stats

    # (b) trait-selective: take trait t from A, everything else from B
    swap = {}
    for i, t in enumerate(TRAITS):
        z = np.stack([swap_one(a, b, t) for a, b in pairs])
        P = predict(clf, dec(model, z))
        diff = YA[:, i] != YB[:, i]
        others = [j for j in range(len(TRAITS)) if j != i]
        swap[t] = {
            "swapped_trait_matches_donor_A": (P[i].argmax(1)[diff] == YA[diff, i]).float().mean().item(),
            "other_traits_keep_B": float(np.mean([(P[j].argmax(1) == YB[:, j]).float().mean().item() for j in others])),
        }
    metrics["trait_selective_swap"] = swap
    # reference: other traits' accuracy on a plain reconstruction of B (upper bound for "keep B")
    PB = predict(clf, dec(model, mu[B]))
    metrics["recon_B_accuracy_reference"] = {t: (PB[i].argmax(1) == YB[:, i]).float().mean().item() for i, t in enumerate(TRAITS)}

    # (c) editing: set trait t to a random target variant
    edit = {}
    subj = pairs[:300, 0]
    for i, t in enumerate(TRAITS):
        n = traits.n_variants(t)
        tgt = rng.integers(0, n, len(subj))
        z = np.stack([traits.edit(mu[s], lab(s), t, int(v), 1.0)[0] for s, v in zip(subj, tgt)])
        P = predict(clf, dec(model, z))
        tgt_t = torch.as_tensor(tgt, device=DEVICE)
        others = [j for j in range(len(TRAITS)) if j != i]
        edit[t] = {"hit_target_variant": (P[i].argmax(1) == tgt_t).float().mean().item(),
                   "chance": 1 / n,
                   "other_traits_preserved": float(np.mean([(P[j].argmax(1) == Y[subj, j]).float().mean().item() for j in others]))}
    metrics["attribute_editing"] = edit

    # (d) blend realism: Frechet distance to real test avatars in judge-feature space
    f_real = judge_features(clf, Xte)
    f_train = judge_features(clf, X[np.nonzero(split == 0)[0][:1000]])
    fd = {"real_train_vs_real_test (floor)": frechet(f_train, f_real),
          "vae_recon": frechet(judge_features(clf, rec), f_real),
          "prior_samples": frechet(judge_features(clf, model.decode(torch.randn(1000, model.latent_dim, device=DEVICE))), f_real)}
    for al in (0.5, 0.25):
        fd[f"latent_blend_a{al}"] = frechet(judge_features(clf, dec(model, al * mu[A] + (1 - al) * mu[B])), f_real)
    fd["pixel_blend_a0.5"] = frechet(judge_features(clf, 0.5 * X[A] + 0.5 * X[B]), f_real)
    metrics["frechet_distance_to_real"] = fd
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()

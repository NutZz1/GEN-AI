"""Train the AvatarBlend convolutional VAE.

    python train.py --size 64  --epochs 40                                   # proposal baseline
    python train.py --size 128 --epochs 50 --latent 256 --batch 64 --perc 500   # sharper, GPU

Fine-tune an existing model with ACAI interpolation regularisation (cleaner in-between blends):
    python train.py --size 128 --latent 256 --batch 64 --perc 500 --init checkpoints/vae128.pt \
                    --epochs 20 --lr 3e-4 --acai 2000 --tag _acai
The same command with --acai 0 --tag _ft is the control run (extra training only).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avatarblend import (ROOT, ConvVAE, VGGPerceptual, ckpt_path, describe_device, get_device,
                         load_dataset, to_uint8, vae_loss)


class Critic(nn.Module):
    """ACAI critic (Berthelot et al., 2018): predicts the mixing coefficient alpha of a decoded
    interpolation, and 0 for real data. The VAE is trained to make it output 0 on interpolations,
    i.e. to make in-between decodes indistinguishable from real avatars."""

    def __init__(self, img_size, base=32):
        super().__init__()
        n_down = int(np.log2(img_size // 4))
        layers, c_in = [], 3
        for i in range(n_down):
            c = min(base * 2 ** i, 512)
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.LeakyReLU(0.2)]
            c_in = c
        self.net = nn.Sequential(*layers, nn.Conv2d(c_in, 1, 4))

    def forward(self, x):
        return self.net(x).flatten(1).mean(1)


def batches(X_u8, idx, batch, shuffle, device):
    """Yield float [0,1] NCHW batches from a (pinned, CPU) uint8 NHWC tensor."""
    if shuffle:
        idx = idx[torch.randperm(len(idx))]
    for i in range(0, len(idx), batch):
        xb = X_u8[idx[i:i + batch]].to(device, non_blocking=True)
        yield xb.permute(0, 3, 1, 2).float().div_(255.0)


def run_epoch(model, X, idx, args, beta, device, perc=None, opt=None, critic=None, opt_c=None):
    train = opt is not None
    model.train(train)
    tot, n = np.zeros(6), 0
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.set_grad_enabled(train):
        for x in batches(X, idx, args.batch if train else 256, train, device):
            recon, mu, logvar = model(x)
            loss, bce, kl = vae_loss(recon, x, mu, logvar, beta)
            p = adv = loss_c = torch.zeros((), device=device)
            if perc is not None:
                with torch.autocast(**amp):
                    p = perc(recon, x).float()
                loss = loss + args.perc * p
            if train and critic is not None:
                # decode random interpolations between batch members, alpha in [0, 0.5]
                alpha = 0.5 * torch.rand(x.size(0), device=device)
                perm = torch.randperm(x.size(0), device=device)
                x_int = model.decode(alpha[:, None] * mu + (1 - alpha[:, None]) * mu[perm])
                with torch.autocast(**amp):
                    adv = critic(x_int).float().pow(2).mean()
                loss = loss + args.acai * adv
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if train and critic is not None:
                with torch.autocast(**amp):
                    c_int = critic(x_int.detach()).float()
                    c_real = critic(0.2 * x + 0.8 * recon.detach()).float()  # ACAI regulariser, gamma=0.2
                loss_c = F.mse_loss(c_int, alpha) + c_real.pow(2).mean()
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()
            tot += np.array([loss.item(), bce.item(), kl.item(), p.item(), adv.item(), loss_c.item()]) * x.size(0)
            n += x.size(0)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, choices=[64, 128])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--perc", type=float, default=0.0, help="VGG perceptual loss weight (0 = off)")
    ap.add_argument("--acai", type=float, default=0.0, help="ACAI interpolation-critic weight (0 = off)")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. _acai -> vae128_acai.pt")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="force cuda / cpu (default: auto-detect)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data = load_dataset(args.size)
    X = torch.from_numpy(data["images"])  # uint8 on CPU; batches are moved to the GPU on the fly
    if device.type == "cuda":
        X = X.pin_memory()
    split = torch.from_numpy(data["split"])
    tr_idx, va_idx = torch.nonzero(split == 0).squeeze(1), torch.nonzero(split == 1).squeeze(1)
    print(f"{args.size}px  train={len(tr_idx)} val={len(va_idx)} test={int((split == 2).sum())}")

    config = {"latent_dim": args.latent, "img_size": args.size}
    model = ConvVAE(**config).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"initialised from {args.init}")
    perc = VGGPerceptual().to(device) if args.perc > 0 else None
    critic = Critic(args.size).to(device) if args.acai > 0 else None
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4) if critic else None
    print(f"VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
          f"{'  + VGG16 perceptual x' + str(args.perc) if perc else ''}{'  + ACAI critic x' + str(args.acai) if critic else ''}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out = ckpt_path(args.size, args.tag)
    out.parent.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    fixed = X[va_idx[:8]].to(device).permute(0, 3, 1, 2).float() / 255
    history, best = [], float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        # KL warm-up keeps early training from collapsing the latent (not needed when fine-tuning)
        beta = args.beta if args.init else args.beta * min(1.0, ep / 5)
        tr = run_epoch(model, X, tr_idx, args, beta, device, perc, opt, critic, opt_c)
        va = run_epoch(model, X, va_idx, args, args.beta, device, perc)
        sched.step()
        history.append({"epoch": ep, "train": tr.tolist(), "val": va.tolist()})
        print(f"ep {ep:3d} | train {tr[0]:8.1f} (bce {tr[1]:8.1f} kl {tr[2]:6.1f} perc {tr[3]:6.3f}"
              f"{f' adv {tr[4]:.4f} critic {tr[5]:.4f}' if critic else ''}) "
              f"| val {va[0]:8.1f} (bce {va[1]:8.1f} kl {va[2]:6.1f}) | {time.time() - t0:5.1f}s", flush=True)
        # fine-tunes keep the final weights (the ACAI term trades a little reconstruction for better blends)
        if args.init or va[0] < best:
            best = min(best, va[0])
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "val": va.tolist(),
                        "args": vars(args)}, out)
        if ep % 5 == 0 or ep == args.epochs:  # progress snapshot: originals over reconstructions
            model.eval()
            with torch.no_grad():
                rec = model.decode(model.encode(fixed)[0])
            from PIL import Image
            grid = np.concatenate([np.concatenate(list(to_uint8(t)), 1) for t in (fixed, rec)], 0)
            Image.fromarray(grid).save(ROOT / "results" / f"train_progress_{args.size}{args.tag}.png")

    (out.parent / f"history{args.size}{args.tag}.json").write_text(json.dumps(history, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

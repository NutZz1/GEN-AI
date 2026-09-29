"""AvatarBlend core: convolutional VAE, losses, device + data helpers, latent blending (PyTorch)."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def data_path(size):
    return ROOT / "data" / f"cartoonset{size}.npz"


def ckpt_path(size, tag=""):
    return CKPT_DIR / f"vae{size}{tag}.pt"


# model used by the app / evaluate.py when none is given; first existing one wins
PREFERRED = ["vae128.pt", "vae64.pt"]


def default_ckpt():
    """AVATARBLEND_CKPT env var if set, else the first existing checkpoint in PREFERRED."""
    env = os.environ.get("AVATARBLEND_CKPT")
    if env:
        return Path(env) if Path(env).is_absolute() else CKPT_DIR / env
    for name in PREFERRED:
        if (CKPT_DIR / name).exists():
            return CKPT_DIR / name
    raise FileNotFoundError("no trained checkpoint in checkpoints/ - run train.py first")


# ---------- device ----------

def get_device(prefer=None):
    """'cuda' if an NVIDIA GPU is usable, else 'mps' (Apple), else 'cpu'. `prefer` forces one."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "cpu" and "+cpu" in torch.__version__:
        return "cpu (this PyTorch build is CPU-only, see README for GPU setup)"
    return device.type


# ---------- model ----------

class ConvVAE(nn.Module):
    """RGB img_size x img_size -> latent (mu, logvar) -> RGB, DCGAN-style.

    Encoder: stride-2 convs down to 4x4 (64px: 4 layers, 128px: 5 layers).
    Decoder: mirror image with transposed convs, sigmoid output in [0, 1].
    """

    def __init__(self, latent_dim=128, img_size=64, base=32):
        super().__init__()
        self.latent_dim, self.img_size = latent_dim, img_size
        n_down = int(np.log2(img_size // 4))
        ch = [min(base * 2 ** i, 512) for i in range(n_down)]
        layers, c_in = [], 3
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2)]
            c_in = c
        self.encoder = nn.Sequential(*layers, nn.Flatten())
        feat = ch[-1] * 16
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, feat)
        rev = ch[::-1]
        dec = []
        for c_a, c_b in zip(rev[:-1], rev[1:]):
            dec += [nn.ConvTranspose2d(c_a, c_b, 4, 2, 1), nn.BatchNorm2d(c_b), nn.ReLU()]
        dec += [nn.ConvTranspose2d(rev[-1], 3, 4, 2, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)
        self._ch_last = ch[-1]

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, self._ch_last, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


class VGGPerceptual(nn.Module):
    """Frozen VGG16 feature distance (relu1_2, relu2_2, relu3_3) - sharper than pixel loss alone."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([feats[:4], feats[4:9], feats[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, recon, x):
        a, b = (recon - self.mean) / self.std, (x - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            a, b = s(a), s(b)
            loss = loss + F.mse_loss(a, b, reduction="none").mean((1, 2, 3)).sum()
        return loss / x.size(0)


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Per-image BCE reconstruction + beta * KL(q(z|x) || N(0, I))."""
    n = x.size(0)
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
    return bce + beta * kl, bce, kl


# ---------- data helpers ----------

def load_dataset(size=64):
    d = np.load(data_path(size), allow_pickle=False)
    return {k: d[k] for k in d.files}


def to_tensor(imgs_uint8):
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(imgs_uint8)).permute(0, 3, 1, 2).float().div(255.0)


def to_uint8(t):
    """(N,3,H,W) float in [0,1] -> (N,H,W,3) uint8."""
    return (t.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


# ---------- inference ----------

def load_model(ckpt=None, device=None):
    device = device or get_device()
    ckpt = Path(ckpt or default_ckpt())
    state = torch.load(ckpt, map_location=device)
    model = ConvVAE(**state["config"]).to(device)
    model.name = ckpt.stem  # e.g. "vae128_acai" - keys the trait model and result files
    model.load_state_dict(state["model"])
    model.eval()
    return model


def model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def encode_mu(model, imgs_uint8, batch=500):
    """uint8 images -> latent means as a float32 numpy array."""
    dev = model_device(model)
    return torch.cat([model.encode(to_tensor(imgs_uint8[i:i + batch]).to(dev))[0].cpu()
                      for i in range(0, len(imgs_uint8), batch)]).numpy()


@torch.no_grad()
def decode_np(model, z):
    """numpy latent(s) (D,) or (N, D) -> uint8 images (N, H, W, 3)."""
    z = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32, device=model_device(model))
    return to_uint8(model.decode(z))


def blend(model, img_a, img_b, alphas=(0.25, 0.5, 0.75)):
    """Encode two uint8 avatars, mix latent means, decode.

    Returns uint8 images for z = alpha * z_A + (1 - alpha) * z_B, one per alpha.
    """
    z_a, z_b = encode_mu(model, np.stack([img_a, img_b]))
    return decode_np(model, np.stack([a * z_a + (1 - a) * z_b for a in alphas]))


def pixel_blend(img_a, img_b, alpha):
    """Naive baseline: raw pixel alpha-blend (the 'double exposure')."""
    return (alpha * img_a.astype(np.float32) + (1 - alpha) * img_b.astype(np.float32)).round().astype(np.uint8)

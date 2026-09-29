# AvatarBlend Studio

Convolutional VAE (PyTorch) trained on Google Cartoon Set, extended from plain latent blending
(`x* = Decoder(a·z_A + (1-a)·z_B)`) into a small generative "studio":

| Tab | What it does | How |
|---|---|---|
| **Blend** | whole-avatar mixes at 75/50/25% + custom ratio, vs naive pixel blend | linear latent interpolation |
| **Trait Mixer** | hair from A, face shape from B, glasses from A, ... | per-trait latent subspaces (LDA on encoder means, orthogonalised) |
| **Avatar Editor** | change one trait of any avatar (e.g. hair colour -> variant 3) | latent arithmetic toward the target variant's mean |
| **Genetics Lab** | breed litters, pick favourites, evolve generations, family tree | trait-genes inherited from one parent (or blended) + Gaussian mutation |

## Results (held-out test avatars, `results/metrics_{64,128}.json`)
Judged by an independent attribute classifier trained on real avatars (99-100% accurate on real images,
86% on eye colour). Figures in `results/` are from the 128px model.

| | 64px baseline (CPU, 40 ep) | 128px + perceptual (GPU, 50 ep) |
|---|---|---|
| Reconstruction PSNR | 26.9 dB | 28.4 dB |
| Reconstruction keeps trait (hairstyle / glasses / facial hair) | 64% / 60% / 19% | **98% / 90% / 91%** |
| Trait Mixer: trait taken from A (hairstyle / face shape / glasses) | 59% / 39% / 37% | **98% / 89% / 77%** |
| Trait Mixer: other traits stay B's | 49-62% | **79-89%** |
| Editor: hits requested variant (hairstyle, 1 of 111 / hair colour) | 53% / 84% | **96% / 89%** |

Whole-avatar blending behaves as the proposal predicts: at a=0.75 the result has A's hairstyle 91% of the time
(B's 0%), at a=0.25 B's 93% (A's 0%). Weak spots: 50/50 blends of very different hairstyles are messy
(use the Trait Mixer instead), and skin tone (11 close shades) is only kept ~60-70% of the time.

Design note: an orthogonal-subspace (LDA projection) trait model was tried first and failed
(hairstyle swap 1%); the additive label-offset model in `traits.py` replaced it.

## Files
| file | purpose |
|---|---|
| `prepare_data.py --size 64/128` | Cartoon Set (extracted folder or .tgz) -> `data/cartoonset{size}.npz` |
| `avatarblend.py` | `ConvVAE` (64 or 128px), BCE+KL loss, VGG16 perceptual loss, device + blend helpers |
| `train.py` | trains the VAE -> `checkpoints/vae{size}.pt` |
| `traits.py` | fits the trait subspaces -> `checkpoints/traits{size}.npz`; selective blend / edit / crossover |
| `genetics.py` | lineage bookkeeping + family-tree rendering |
| `evaluate.py` | metrics + figures in `results/` (incl. auxiliary-classifier checks of blending, trait swap, editing) |
| `app.py` | Streamlit UI (uses 128px model if trained, else 64px) |
| `run_gpu.sh` | 128px GPU training + evaluation inside WSL |

## GPU setup (this machine: RTX 4060 Laptop)
Windows **Smart App Control** blocks the CUDA build of PyTorch's DLLs on this PC
(`WinError 4551 ... An Application Control policy has blocked this file`), so the Windows Python
keeps the CPU build. GPU training runs inside **WSL2 Ubuntu**, which uses the same NVIDIA driver:

```bash
wsl -d Ubuntu -- bash -lc "python3 -m venv ~/avatarblend-venv && ~/avatarblend-venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 && ~/avatarblend-venv/bin/pip install numpy pillow matplotlib streamlit"
wsl -d Ubuntu -- bash /mnt/c/Users/sreep/OneDrive/Desktop/GEN-AI/run_gpu.sh
```

All scripts auto-detect the device (`cuda` > `mps` > `cpu`); force one with `train.py --device cpu`.
On a machine without Smart App Control, `pip install torch --index-url https://download.pytorch.org/whl/cu130` works directly.

## Run (CPU / Windows)
```bash
pip install -r requirements.txt
python prepare_data.py --size 64
python train.py --size 64 --epochs 40
python evaluate.py
streamlit run app.py
```

Dataset: Google Cartoon Set (CC-BY-4.0), downloaded from the Hugging Face mirror
`cgarciae/cartoonset` because the original Google Cloud bucket now returns 403.

"""Trait "genes" in the VAE latent space - an additive attribute model.

Cartoon Set assigns every trait independently at random, so an avatar's latent decomposes as

    z  =  mu  +  sum_t offset_t[variant_t]  +  residual

where offset_t[k] = (mean latent of all training avatars with variant k of trait t) - mu, and the
residual is whatever the labels don't name. Every avatar carries its variant labels ("genotype").
This gives three operations with a single mechanism (label-aware latent arithmetic):

  * trait-selective blend - each trait's offset from A, B or a mix; residual mixed separately
  * attribute editing     - replace one trait's offset by the target variant's offset
  * genetic crossover     - each gene (trait) inherited from one parent or blended,
                            + point mutations (gene flips to a random variant) + latent noise

For avatars without labels (random samples from the prior) the variants are inferred with a
per-trait linear discriminant (LDA) on the latent.

Measured on held-out pairs with an independent attribute classifier (evaluate.py): taking one trait
from A reaches 59% (hairstyle) / 79% (hair colour) vs a 64% / 86% reconstruction ceiling, while an
orthogonal-subspace projection approach reached only 1% / 41%.

    python traits.py            # fit and save checkpoints/traits{size}.npz for the current model
"""
import json

import numpy as np

from avatarblend import CKPT_DIR, encode_mu, load_dataset, load_model

TRAITS = ["hair", "hair_color", "face_shape", "face_color", "glasses", "glasses_color", "facial_hair", "eye_color"]
LABELS = {"glasses": "Glasses", "hair_color": "Hair colour", "face_color": "Skin tone",
          "facial_hair": "Facial hair", "hair": "Hairstyle", "face_shape": "Face shape",
          "eye_color": "Eye colour", "glasses_color": "Glasses colour"}


class TraitSpace:
    def __init__(self, mu, offsets, lda_w, lda_b, protos):
        self.mu = mu                # (D,) global mean latent
        self.offsets = offsets      # trait -> (n_variants, D)
        self.lda_w = lda_w          # trait -> (n_variants, D) linear discriminant weights
        self.lda_b = lda_b          # trait -> (n_variants,)
        self.protos = protos        # trait -> (n_variants,) dataset index of a typical example
        self.traits = list(offsets)

    def n_variants(self, t):
        return len(self.offsets[t])

    # ----- fitting / io -----
    @classmethod
    def fit(cls, mu_all, attrs, attr_names, train_mask, reg=1e-2):
        names = list(attr_names)
        Z = mu_all[train_mask].astype(np.float64)
        g = Z.mean(0)
        D = Z.shape[1]
        offsets, lda_w, lda_b, protos = {}, {}, {}, {}
        for t in TRAITS:
            y_all = attrs[:, names.index(t)]
            y = y_all[train_mask]
            n = int(y_all.max()) + 1
            means = np.stack([Z[y == c].mean(0) if (y == c).any() else g for c in range(n)])
            offsets[t] = (means - g).astype(np.float32)
            # shared-covariance LDA for inferring the variant of unlabelled latents
            Sw = sum((Z[y == c] - means[c]).T @ (Z[y == c] - means[c]) for c in range(n) if (y == c).any())
            Sw = Sw / len(Z) + reg * np.eye(D)
            Sinv_m = np.linalg.solve(Sw, means.T).T
            prior = np.log(np.bincount(y, minlength=n) + 1.0)
            lda_w[t] = Sinv_m.astype(np.float32)
            lda_b[t] = (-0.5 * (Sinv_m * means).sum(1) + prior).astype(np.float32)
            # prototype = the avatar closest to its variant's mean latent (for UI thumbnails)
            pr = []
            for c in range(n):
                ids = np.nonzero(y_all == c)[0]
                pr.append(int(ids[np.argmin(np.linalg.norm(mu_all[ids] - means[c], axis=1))]) if len(ids) else 0)
            protos[t] = np.array(pr)
        return cls(g.astype(np.float32), offsets, lda_w, lda_b, protos)

    def save(self, path):
        arrs = {"mu": self.mu}
        for t in self.traits:
            arrs.update({f"off_{t}": self.offsets[t], f"w_{t}": self.lda_w[t], f"b_{t}": self.lda_b[t],
                         f"protos_{t}": self.protos[t]})
        np.savez(path, traits=np.array(self.traits), **arrs)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        ts = [str(t) for t in d["traits"]]
        return cls(d["mu"], {t: d[f"off_{t}"] for t in ts}, {t: d[f"w_{t}"] for t in ts},
                   {t: d[f"b_{t}"] for t in ts}, {t: d[f"protos_{t}"] for t in ts})

    # ----- genotype helpers -----
    def labels_from_attrs(self, attr_row, attr_names):
        names = list(attr_names)
        return {t: int(attr_row[names.index(t)]) for t in self.traits}

    def infer_labels(self, z):
        return {t: int(np.argmax(self.lda_w[t] @ z + self.lda_b[t])) for t in self.traits}

    def trait_part(self, labels, weights=None, other=None):
        """sum_t offset_t; with `weights`/`other`, trait t mixes w*offset(labels) + (1-w)*offset(other)."""
        s = np.zeros_like(self.mu)
        for t in self.traits:
            w = 1.0 if weights is None else weights.get(t, 0.5)
            s = s + w * self.offsets[t][labels[t]]
            if w < 1.0:
                s = s + (1 - w) * self.offsets[t][other[t]]
        return s

    def residual(self, z, labels):
        return z - self.mu - self.trait_part(labels)

    # ----- operations -----
    def selective_blend(self, z_a, lab_a, z_b, lab_b, weights, rest_weight=0.5):
        """weights[t] in [0,1] = share of parent A for trait t (1 = A's variant, 0 = B's).
        Returns (z, labels); labels[t] follows the majority parent."""
        res = rest_weight * self.residual(z_a, lab_a) + (1 - rest_weight) * self.residual(z_b, lab_b)
        z = self.mu + self.trait_part(lab_a, weights, lab_b) + res
        labels = {t: lab_a[t] if weights.get(t, 0.5) >= 0.5 else lab_b[t] for t in self.traits}
        return z.astype(np.float32), labels

    def edit(self, z, labels, trait, target_variant, strength=1.0):
        """Swap `trait`'s offset toward `target_variant` (1.0 = fully, >1 exaggerates, <0 moves away)."""
        z = z + strength * (self.offsets[trait][target_variant] - self.offsets[trait][labels[trait]])
        labels = dict(labels)
        if strength >= 0.5:
            labels[trait] = int(target_variant)
        return z.astype(np.float32), labels

    def crossover(self, z_a, lab_a, z_b, lab_b, rng, noise=0.3, blend_prob=0.2, point_mutation=0.05):
        """Child (z, labels, genotype). genotype[t] in {'A', 'B', 'mix', 'mut'}."""
        weights, geno = {}, {}
        for t in self.traits:
            if rng.random() < blend_prob:
                weights[t], geno[t] = float(rng.uniform(0.3, 0.7)), "mix"
            else:
                from_a = rng.random() < 0.5
                weights[t], geno[t] = float(from_a), "A" if from_a else "B"
        z, labels = self.selective_blend(z_a, lab_a, z_b, lab_b, weights, float(rng.uniform(0.25, 0.75)))
        for t in self.traits:  # point mutation: gene jumps to a random variant
            if rng.random() < point_mutation:
                z, labels = self.edit(z, labels, t, int(rng.integers(self.n_variants(t))))
                geno[t] = "mut"
        z = z + noise * rng.standard_normal(z.shape).astype(np.float32)
        return z.astype(np.float32), labels, geno


def traits_path(model):
    return CKPT_DIR / f"traits_{model.name}.npz"


def build(model=None, data=None):
    model = model or load_model()
    data = data or load_dataset(model.img_size)
    mu = encode_mu(model, data["images"])
    ts = TraitSpace.fit(mu, data["attrs"], data["attr_names"], data["split"] == 0)
    ts.save(traits_path(model))
    return ts, mu


if __name__ == "__main__":
    model = load_model()
    print(f"model: {model.name}")
    data = load_dataset(model.img_size)
    ts, mu = build(model, data)
    te = data["split"] == 2
    names = list(data["attr_names"])
    acc = {t: float(np.mean(np.array([ts.infer_labels(z)[t] for z in mu[te]]) == data["attrs"][te, names.index(t)]))
           for t in ts.traits}
    print("LDA variant inference accuracy on test latents:", json.dumps({t: round(a, 3) for t, a in acc.items()}))
    print("saved", traits_path(model))

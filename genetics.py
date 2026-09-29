"""Avatar Genetics Lab: breeding generations of avatars and drawing their family tree."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GENE_COLORS = {"A": "#2a78d6", "B": "#e8743b", "mix": "#8a8a8a", "mut": "#1baa6e"}


class Lineage:
    """All avatars ever created in a session.
    Node: id, gen, parents, z, labels (trait variants), genotype (gene source per trait), img."""

    def __init__(self):
        self.nodes = {}
        self._next = 0

    def add(self, z, labels, img, gen, parents=(), genotype=None, name=None):
        nid = self._next
        self._next += 1
        self.nodes[nid] = {"id": nid, "gen": gen, "parents": tuple(parents), "z": z, "labels": labels, "img": img,
                           "genotype": genotype or {}, "name": name or f"G{gen}-{nid}"}
        return nid

    def breed(self, pa, pb, traits, decode, n_children, rng, noise, blend_prob, point_mutation):
        a, b = self.nodes[pa], self.nodes[pb]
        gen = max(a["gen"], b["gen"]) + 1
        kids = [traits.crossover(a["z"], a["labels"], b["z"], b["labels"], rng, noise, blend_prob, point_mutation)
                for _ in range(n_children)]
        imgs = decode(np.stack([k[0] for k in kids]))
        return [self.add(z, lab, img, gen, (pa, pb), g) for (z, lab, g), img in zip(kids, imgs)]

    def ancestors(self, nid):
        seen, stack = set(), [nid]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(self.nodes[n]["parents"])
        return seen


def family_tree_png(lineage, focus_ids, traits_order, labels):
    """Draw the ancestry of `focus_ids` as a PNG: one row per generation, lines parent -> child,
    and a coloured gene bar under each child showing which parent each trait came from."""
    keep = set()
    for f in focus_ids:
        keep |= lineage.ancestors(f)
    nodes = [lineage.nodes[i] for i in sorted(keep)]
    gens = sorted({n["gen"] for n in nodes})
    rows = {g: [n for n in nodes if n["gen"] == g] for g in gens}
    width = max(len(r) for r in rows.values())
    pos = {}
    for gi, g in enumerate(gens):
        r = rows[g]
        for j, n in enumerate(r):
            pos[n["id"]] = ((j + 0.5) * width * 1.3 / len(r), -gi * 1.6)

    fig, ax = plt.subplots(figsize=(max(4, 2.0 * width), 1.9 * len(gens) + 0.6))
    for n in nodes:
        for p in n["parents"]:
            if p in pos:
                (x0, y0), (x1, y1) = pos[p], pos[n["id"]]
                ax.plot([x0, x1], [y0 - 0.5, y1 + 0.5], color="#9a9a9a", lw=1, zorder=0)
    for n in nodes:
        x, y = pos[n["id"]]
        ax.imshow(n["img"], extent=(x - 0.5, x + 0.5, y - 0.5, y + 0.5), zorder=2)
        edge = "#d62a2a" if n["id"] in focus_ids else "#cccccc"
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec=edge, lw=2 if edge != "#cccccc" else 0.8, zorder=3))
        ax.text(x, y + 0.56, n["name"], ha="center", va="bottom", fontsize=7)
        if n["genotype"]:
            w = 1.0 / len(traits_order)
            for k, t in enumerate(traits_order):
                ax.add_patch(plt.Rectangle((x - 0.5 + k * w, y - 0.68), w, 0.12,
                                           color=GENE_COLORS[n["genotype"][t]], zorder=3))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GENE_COLORS.values()]
    ax.legend(handles, ["gene from parent 1", "gene from parent 2", "blended gene", "point mutation"], loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False,
              title="bar order: " + ", ".join(labels[t] for t in traits_order), title_fontsize=6)
    ax.set_xlim(-0.2, width * 1.3 + 0.2)
    ax.set_ylim(-1.6 * (len(gens) - 1) - 0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

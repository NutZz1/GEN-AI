"""AvatarBlend Streamlit interface.

    streamlit run app.py
"""
import io

import numpy as np
import streamlit as st
from PIL import Image

from avatarblend import (decode_np, describe_device, encode_mu, get_device, load_dataset, load_model,
                         model_device, pixel_blend)
from genetics import GENE_COLORS, Lineage, family_tree_png
from traits import LABELS, TraitSpace, build, traits_path

GALLERY_SIZE = 24
ATTR_MENU = ["face_shape", "hair", "hair_color", "face_color", "eye_color", "glasses", "facial_hair"]
DISPLAY = 176


@st.cache_resource
def resources():
    model = load_model(device=get_device())
    data = load_dataset(model.img_size)
    if traits_path(model).exists():
        traits, mu = TraitSpace.load(traits_path(model)), encode_mu(model, data["images"])
    else:
        traits, mu = build(model, data)
    # farthest-point sampling in latent space -> visually diverse gallery
    gallery = [0]
    d = np.linalg.norm(mu - mu[0], axis=1)
    for _ in range(GALLERY_SIZE - 1):
        gallery.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(mu - mu[gallery[-1]], axis=1))
    return model, data, traits, mu, gallery


def big(img, size=DISPLAY):
    return Image.fromarray(img).resize((size, size), Image.LANCZOS)


def png_bytes(img):
    buf = io.BytesIO()
    big(img, 256).save(buf, format="PNG")
    return buf.getvalue()


def show(col, img, caption, key=None):
    col.image(big(img), caption=caption)
    if key:
        col.download_button("Download", png_bytes(img), f"{key}.png", "image/png", key=f"dl_{key}")


# ---------------- parent selection ----------------

def pick_gallery(label, data, gallery, key, default):
    choice = st.selectbox(f"{label}: gallery #", list(range(1, len(gallery) + 1)), index=default, key=key)
    idx = gallery[choice - 1]
    return idx, f"gallery #{choice}"


def pick_attributes(label, data, key, seed):
    names = list(data["attr_names"])
    rng = np.random.default_rng(seed)
    wanted = {}
    cols = st.columns(2)
    for i, a in enumerate(ATTR_MENU):
        n = int(data["attr_nvariants"][names.index(a)])
        wanted[a] = cols[i % 2].selectbox(LABELS.get(a, a), list(range(n)), index=int(rng.integers(n)),
                                          key=f"{key}_{a}")
    cols_idx = [names.index(a) for a in ATTR_MENU]
    score = (data["attrs"][:, cols_idx] == np.array([wanted[a] for a in ATTR_MENU])).sum(1)
    best = int(score.argmax())
    return best, f"closest match ({score[best]}/{len(ATTR_MENU)})"


def choose_parents(data, gallery, mode):
    if mode == "Gallery selection":
        with st.expander("Gallery", expanded=True):
            gcols = st.columns(12)
            for i, gi in enumerate(gallery):
                gcols[i % 12].image(big(data["images"][gi], 80), caption=f"#{i + 1}")
        c1, c2 = st.columns(2)
        with c1:
            ia, da = pick_gallery("Parent A", data, gallery, "ga", 6)
        with c2:
            ib, db = pick_gallery("Parent B", data, gallery, "gb", 18)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Parent A attributes**")
            ia, da = pick_attributes("A", data, "aa", 1)
        with c2:
            st.markdown("**Parent B attributes**")
            ib, db = pick_attributes("B", data, "ab", 2)
    return ia, da, ib, db


# ---------------- tabs ----------------

def tab_blend(model, A, B, da, db):
    st.markdown("Whole-avatar latent interpolation: `x* = Decoder(a·z_A + (1-a)·z_B)`")
    extra = st.slider("Extra custom ratio (% of A)", 0, 100, 60, 5)
    show_pixel = st.checkbox("Compare with naive pixel blend", value=False)
    alphas = [0.75, 0.5, 0.25, extra / 100]
    hybrids = decode_np(model, np.stack([a * A["z"] + (1 - a) * B["z"] for a in alphas]))
    cols = st.columns(6)
    show(cols[0], A["img"], f"Parent A - {da}")
    for j, (a, h) in enumerate(zip(alphas, hybrids)):
        show(cols[j + 1], h, f"{round(a * 100)}% A / {round((1 - a) * 100)}% B", f"blend_{round(a * 100)}A")
    show(cols[5], B["img"], f"Parent B - {db}")
    if show_pixel:
        st.caption("Naive pixel alpha-blend at the same ratios (note the ghosting):")
        pc = st.columns(6)
        for j, a in enumerate(alphas):
            pc[j + 1].image(big(pixel_blend(A["img"], B["img"], a)), caption=f"pixel {round(a * 100)}% A")


def tab_trait_mixer(model, traits, A, B):
    st.markdown("Pick **which parent each trait comes from**, e.g. hairstyle from A and face shape from B. "
                "Each avatar is modelled as `mean + sum of trait offsets + residual` in latent space, so single "
                "traits can be swapped by latent arithmetic.")
    c_ctrl, c_out = st.columns([1.1, 1])
    for t in traits.traits + ["rest"]:
        st.session_state.setdefault(f"tm_{t}", 50)
    with c_ctrl:
        b1, b2, b3 = st.columns(3)
        if b1.button("All from A"):
            for t in traits.traits + ["rest"]:
                st.session_state[f"tm_{t}"] = 100
        if b2.button("All from B"):
            for t in traits.traits + ["rest"]:
                st.session_state[f"tm_{t}"] = 0
        if b3.button("Surprise me"):
            rng = np.random.default_rng()
            for t in traits.traits + ["rest"]:
                st.session_state[f"tm_{t}"] = int(rng.choice([0, 100]))
        weights = {t: st.slider(f"{LABELS[t]}  (B ← → A)", 0, 100, step=25, key=f"tm_{t}") / 100
                   for t in traits.traits}
        rest = st.slider("Everything else  (B ← → A)", 0, 100, step=25, key="tm_rest") / 100
    z, labels = traits.selective_blend(A["z"], A["labels"], B["z"], B["labels"], weights, rest)
    st.session_state["mixer"] = {"z": z, "labels": labels}
    with c_out:
        out = decode_np(model, z)[0]
        cols = st.columns(3)
        show(cols[0], A["img"], "Parent A")
        show(cols[1], out, "Trait mix", "trait_mix")
        show(cols[2], B["img"], "Parent B")
        st.caption(", ".join(f"{LABELS[t]}: {'A' if w == 1 else 'B' if w == 0 else f'{int(w * 100)}% A'}"
                             for t, w in weights.items()))


def tab_editor(model, traits, data, subjects):
    st.markdown("Edit any avatar with **latent arithmetic**: swap one trait's offset for another "
                "variant's, leaving the other traits untouched. The thumbnail shows the target variant.")
    name = st.radio("Avatar to edit", list(subjects), horizontal=True, key="ed_subject")
    z, labels = subjects[name]["z"], subjects[name]["labels"]
    z0 = z
    for t in traits.traits:
        n = traits.n_variants(t)
        c1, c2, c3 = st.columns([1.2, 1.2, 0.5])
        target = c1.selectbox(f"{LABELS[t]} (now variant {labels[t]}) →", ["keep"] + list(range(n)), key=f"ed_t_{t}")
        if target != "keep":
            strength = c2.slider(f"{LABELS[t]} strength", -1.0, 2.0, 1.0, 0.1, key=f"ed_s_{t}")
            c3.image(big(data["images"][traits.protos[t][target]], 64))
            z, labels = traits.edit(z, labels, t, int(target), strength)
    before, after = decode_np(model, np.stack([z0, z]))
    cols = st.columns(4)
    show(cols[1], before, f"{name} (original)")
    show(cols[2], after, "Edited", "edited")


def tab_genetics(model, traits, A, B):
    st.markdown("Breed avatars like organisms. Each child inherits every **trait-gene** from one parent "
                "(sometimes a blend of both), with occasional **point mutations** (a gene jumps to a random "
                "variant) plus small random variation. Pick favourites and evolve new generations.")
    restart = st.button("Restart with current Parent A & B as founders")
    if restart or "lineage" not in st.session_state:
        lin = Lineage()
        fa = lin.add(A["z"], A["labels"], A["img"], 0, name="Founder A")
        fb = lin.add(B["z"], B["labels"], B["img"], 0, name="Founder B")
        st.session_state.update(lineage=lin, pair=[fa, fb], litter=[], rng=np.random.default_rng())
    lin, rng = st.session_state["lineage"], st.session_state["rng"]

    c = st.columns(5)
    n_children = c[0].slider("Children per litter", 4, 12, 8, 2)
    point_mut = c[1].slider("Point-mutation rate", 0.0, 0.5, 0.05, 0.05)
    noise = c[2].slider("Random variation", 0.0, 1.0, 0.3, 0.05)
    blend_prob = c[3].slider("Blended-gene probability", 0.0, 1.0, 0.2, 0.05)
    if c[4].button("Add a wild avatar", help="decode a random latent from the VAE prior; its genes are inferred"):
        z = (0.8 * rng.standard_normal(model.latent_dim)).astype(np.float32)
        lin.add(z, traits.infer_labels(z), decode_np(model, z)[0], 0, name=f"Wild-{lin._next}")

    options = {n["name"]: nid for nid, n in lin.nodes.items()}
    pair_names = st.multiselect("Parents for the next litter (choose 2)", list(options),
                                default=[lin.nodes[i]["name"] for i in st.session_state["pair"] if i in lin.nodes],
                                max_selections=2)
    if st.button("Breed litter", type="primary", disabled=len(pair_names) != 2):
        pa, pb = options[pair_names[0]], options[pair_names[1]]
        st.session_state["pair"] = [pa, pb]
        st.session_state["litter"] = lin.breed(pa, pb, traits, lambda z: decode_np(model, z),
                                               n_children, rng, noise, blend_prob, point_mut)

    litter = st.session_state["litter"]
    if litter:
        pa, pb = lin.nodes[litter[0]]["parents"]
        legend = " / ".join(f"<span style='color:{GENE_COLORS[k]}'>{v}</span>" for k, v in
                            [("A", "parent 1"), ("B", "parent 2"), ("mix", "blend"), ("mut", "mutation")])
        st.markdown(f"**Litter of {lin.nodes[pa]['name']} × {lin.nodes[pb]['name']}** (genes: {legend})",
                    unsafe_allow_html=True)
        cols = st.columns(min(len(litter), 6))
        for k, nid in enumerate(litter):
            n = lin.nodes[nid]
            col = cols[k % len(cols)]
            col.image(big(n["img"], 128), caption=n["name"])
            genes = "".join(f"<span title='{LABELS[t]}: {n['genotype'][t]}' style='display:inline-block;width:12px;"
                            f"height:8px;background:{GENE_COLORS[n['genotype'][t]]}'></span>" for t in traits.traits)
            col.markdown(genes, unsafe_allow_html=True)
            col.download_button("Download", png_bytes(n["img"]), f"{n['name']}.png", "image/png", key=f"dlg_{nid}")
        st.caption("Pick two favourites in 'Parents for the next litter' to evolve the next generation.")

        st.subheader("Family tree")
        tree = family_tree_png(lin, set(litter), traits.traits, LABELS)
        st.image(tree)
        st.download_button("Download family tree", tree, "avatar_family_tree.png", "image/png")


def main():
    st.set_page_config(page_title="AvatarBlend", layout="wide")
    st.title("AvatarBlend Studio")
    model, data, traits, mu, gallery = resources()
    st.sidebar.caption(f"Model: {model.img_size}px VAE, {model.latent_dim}-d latent")
    st.sidebar.caption(f"Running on: {describe_device(model_device(model))}")
    mode = st.sidebar.radio("Choose parents by", ["Gallery selection", "Attribute construction"])

    ia, da, ib, db = choose_parents(data, gallery, mode)
    A, B = ({"z": mu[i], "img": data["images"][i],
             "labels": traits.labels_from_attrs(data["attrs"][i], data["attr_names"])} for i in (ia, ib))

    t1, t2, t3, t4 = st.tabs(["Blend", "Trait Mixer", "Avatar Editor", "Genetics Lab"])
    with t1:
        tab_blend(model, A, B, da, db)
    with t2:
        tab_trait_mixer(model, traits, A, B)
    with t3:
        subjects = {"Parent A": A, "Parent B": B}
        if "mixer" in st.session_state:
            subjects["Trait-mix result"] = st.session_state["mixer"]
        tab_editor(model, traits, data, subjects)
    with t4:
        tab_genetics(model, traits, A, B)


if __name__ == "__main__":
    main()

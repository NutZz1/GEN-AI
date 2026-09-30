# Dataset card

## Source

Google **Cartoon Set** (10k variant, CC-BY-4.0): https://google.github.io/cartoonset/
Downloaded from the Hugging Face mirror `cgarciae/cartoonset` because the original Google Cloud
Storage bucket now returns 403.

- 10,000 cartoon avatar PNGs (500x500 RGBA), each with an 18-attribute CSV label (hairstyle, hair
  colour, face shape, skin tone, glasses shape/colour, facial hair, eye colour, eyebrow shape, ...).
- Every attribute is assigned independently at random per avatar (stated by the dataset authors and
  the assumption `traits.py`'s additive latent model relies on).

## Preprocessing (`prepare_data.py`)

1. Composite the RGBA image onto white, convert to RGB.
2. Center-crop to `(40, 40, 460, 460)` - the region the face and hair actually occupy in the 500x500
   canvas, cutting the unused margin.
3. Resize to 64px or 128px with Lanczos resampling.
4. Fixed 80/10/10 train/val/test split via `np.random.default_rng(0).permutation`, so every run of
   `prepare_data.py` reproduces the same split.
5. Save `images` (uint8), `attrs` (int16 variant ids), `attr_names`, `attr_nvariants`, `ids`, `split`
   to `data/cartoonset{64,128}.npz`.

## The editable trait model

`traits.py`'s additive latent model (`z = mu + sum_t offset_t[variant] + residual`) is fit on 8 of
the 18 dataset attributes - the ones users actually reason about when describing an avatar:

| trait | dataset attribute | variants |
|---|---|---|
| Hairstyle | `hair` | 111 |
| Hair colour | `hair_color` | 10 |
| Face shape | `face_shape` | 7 |
| Skin tone | `face_color` | 11 |
| Glasses | `glasses` | 12 (11 = none) |
| Glasses colour | `glasses_color` | 7 |
| Facial hair | `facial_hair` | 15 |
| Eye colour | `eye_color` | 5 |

The other 10 attributes (`eye_angle`, `eye_lashes`, `eye_lid`, `chin_length`, `eyebrow_weight`,
`eyebrow_shape`, `eyebrow_thickness`, `eye_slant`, `eyebrow_width`, `eye_eyebrow_distance`) exist in
the raw CSVs but are not part of the additive model - they are absorbed into the residual and are not
individually editable or agent-addressable. This is a scope decision, not an oversight: the ones
above are the traits `text2avatar.py`'s original keyword parser (and therefore an ordinary user's
vocabulary) actually names.

## Knowledge layer (`knowledge/`)

- `knowledge/build_ontology.py` - moves the curated word -> variant-id vocabulary (originally in
  `text2avatar.py`) into `knowledge/trait_ontology.json`, cross-referenced against the *real* variant
  counts read from `data/cartoonset128.npz`'s `attr_nvariants` (not guessed). `text2avatar.py` and
  `agent/prompts.py` both read this one file, so the rule-based parser and the LLM agent's system
  prompt can never drift out of sync.
  - Coverage: 111/111 hairstyles, 10/10 hair colours, 11/12 glasses (the "no glasses" variant has no
    descriptive word, by definition), 7/7 glasses colours, 10/15 facial-hair variants, 11/11 skin
    tones, 5/5 eye colours have at least one natural-language tag. Eye colours were **measured, not
    guessed**: Cartoon Set attributes are independent, so with glasses absent each eye-colour
    variant's mean image differs from the global mean only at the iris; the mean RGB there gives
    0 = brown, 1 = blue, 2 = green, 3 = grey, 4 = black (checked visually in a rendered grid).
    `face_shape` has **no** curated vocabulary (never inspected/measured) - the ontology records its
    variant count only, and the agent falls back to random/LDA-inferred choices for it.
- `knowledge/build_prompt_testset.py` - generates `knowledge/prompt_testset.jsonl`, ~109 natural
  language avatar requests with gold trait constraints, used by `evaluate_agent.py` to compare the
  rule-based parser against the LLM parser. `easy`/`compositional`/`negation` prompts are built from
  ontology words and their gold labels are computed by running `text2avatar.parse()` itself (so they
  are correct by construction, not hand-guessed). `vague` (persona descriptions with no ontology
  keywords, e.g. "a friendly librarian") and `code_mixed` (Tanglish/Manglish, e.g. "romba curly hair
  venum") items have **hand-written gold constraints** - this repo author's best-effort reading of
  the phrase, not a user study, and deliberately sparse (only the trait(s) a reasonable reader would
  infer are scored). `blend_edit` items test agent routing (generate vs. blend vs. edit vs. breed)
  rather than trait parsing. 30 prompts (spread across all categories) are held out as the `test`
  split; the rest are `dev`.

## Known biases / limitations

- Only 11 skin-tone variants, and `evaluate.py`'s attribute classifier keeps skin tone right only
  ~60-70% of the time in blends/edits (see `results/vae128/metrics.json`) - the closest shades are
  hard for both the VAE and the judge classifier to tell apart.
- The Cartoon Set vocabulary itself has no notion of age, gender expression beyond what's implied by
  hairstyle/facial-hair combinations, or ethnicity beyond skin tone - "vague" persona prompts that
  invoke those ideas (see `knowledge/build_prompt_testset.py`) can only be answered through weak,
  stereotype-adjacent proxies (e.g. "grandma" -> grey/white hair). This is a real limitation of the
  underlying dataset, not just the parser, and is called out again in the README's Limitations
  section.
- The code-mixed test prompts were constructed by this repo's author for realistic Tanglish/Manglish
  phrasing (English trait nouns + Tamil/Malayalam function words), not sourced from real users or
  verified by a native-speaker linguist - treat their gold labels as a reasonable approximation, and
  the category's *relative* rule-parser-vs-LLM gap as the meaningful signal, not the absolute numbers.

# AvatarBlend Studio — Simple Explainer

A plain-English companion to `README.md`, for anyone (including future-you) who wants the "what and
why" without reading code.

## What is this project?

An app that creates cartoon avatar faces — like a Bitmoji or game character creator, but powered by AI
instead of manually clicking through hairstyle/eye-color/glasses menus.

It has two layers:

1. **A generative AI model (the "artist")** — a neural network trained on thousands of cartoon faces.
   It learned what a face is "made of" internally, as a short list of numbers. Once it knows that, it
   can:
   - **Blend** two faces into one
   - **Mix traits** — hair from one face, glasses from another
   - **Edit** one trait on a face (change hair colour) without touching anything else
   - **Breed** two faces into "children" with inherited traits, like genetics

2. **An AI agent (the "assistant") sitting on top** — type what you want ("curly red hair with
   glasses") or upload a photo, and it:
   - Understands your request
   - Tells the artist what to generate
   - **Checks its own work** with a separate AI ("did it actually get the red hair right?")
   - Fixes it and tries again if not
   - Explains what it did in plain English

## Why is this a "Gen AI project"?

Because the core thing it does — **inventing a new face that never existed** by combining/interpolating
what it learned from real examples — is generative AI. It's not classifying or detecting faces, it's
*creating* new ones. The chat layer adds the other big pillar of modern Gen AI: an **LLM-powered agent**
that plans, acts, and verifies its own output instead of just answering a question.

## Real-world use case

A mini version of what's behind:

- **Bitmoji / Memoji / game character creators** — "describe yourself, get an avatar" instead of
  manually tweaking 20 sliders
- **Profile picture generators for apps** that don't want to store real photos (privacy — no selfie
  needed, or an optional one that gets turned into a cartoon instead of stored as-is)
- **Kids' apps / anonymous platforms** wanting a personal-feeling identity without a real face on record
- The **"verify then fix"** idea (an AI checks its own output with a second, independent AI) is the same
  pattern used in serious production AI systems today — e.g. a coding AI running tests on the code it
  just wrote, or a customer-service bot double-checking facts before answering. Arguably the most
  transferable idea in the whole project, beyond "cartoon face maker."

## Core generative concept: VAE, not GAN

The heart of this project is a **VAE (Variational Autoencoder)** — not a GAN. The difference, simply:

- **GAN**: two networks fight each other — a "forger" makes fake images, a "detective" tries to catch
  them. Great image quality, but the internal representation is messy and hard to control precisely.
- **VAE**: one network learns to *compress* a face down to a short list of numbers, another learns to
  *reconstruct* a face from those numbers. Slightly blurrier than a GAN, but the numbers it produces are
  smooth and well-organised — meaning you can do math on them (blend, add, subtract) and get sensible
  results. That's exactly what this app needs, since blending/mixing/editing are all just arithmetic on
  these numbers.

## Every Gen AI concept actually used, and where

| Concept | Where | Plain explanation |
|---|---|---|
| VAE (encoder + decoder) | `avatarblend.py` | Compresses a face to ~128–256 numbers, reconstructs a face from those numbers |
| Latent space interpolation | Blend tab | Averaging two faces' number-lists and decoding the average → a face "in between" |
| Latent arithmetic / attribute vectors | `traits.py` | Like "king − man + woman = queen" for word embeddings, but for faces: "this face's numbers + (glasses direction) = same face wearing glasses" |
| Perceptual loss (pretrained VGG16 as a judge) | `avatarblend.py` training | Compares images through a pretrained vision model's "eyes" instead of pixel-by-pixel, for sharper results |
| Adversarial training (GAN-adjacent) | `train.py`'s ACAI critic | An optional add-on: a small "critic" network tries to catch fake-looking blends, forcing cleaner interpolations — a mini-GAN bolted onto the VAE for one job |
| Classical ML for interpretability (LDA) | `traits.py` | Figures out which direction in the latent space means "hairstyle," "glasses," etc. |
| Genetic algorithm | `genetics.py` | Not a neural network — classic AI concept (crossover + mutation) applied to the latent numbers to "breed" new faces |
| LLM (Large Language Model) | `agent/` | Claude, or a local model (Llama via Ollama), reads your typed request and understands it |
| Prompt engineering | `agent/prompts.py` | Carefully written instructions + examples telling the LLM exactly what format to answer in and what vocabulary it's allowed to use |
| Grounding / knowledge injection | trait ontology fed into the prompt | The LLM only ever sees the *real* valid trait options from the actual dataset, so it can't invent things that don't exist |
| Structured output (JSON) + validation | `agent/schemas.py` | The LLM must answer in a strict, checkable format; if it messes up, it's asked to fix it, or the system falls back to simpler logic |
| Agentic AI / tool use (function calling) | `agent/tools.py`, `agent/agent.py` | The LLM decides *which function to run* (generate/blend/edit/breed) — this is what turns a chatbot into an "agent" |
| Generate → Verify → Self-correct loop | `agent/agent.py` | A second, independent AI (a classifier) checks whether the generated face actually matches the request, and retries if not |

**One-line summary:** a VAE does the actual image generation (the "hands"), an LLM agent understands
requests and directs/checks that generation (the "brain") — two different generative AI paradigms
working together.

## How does "upload a photo → avatar" work? (Not an LLM, not an API)

```
Your photo
   |
1. Crop to a square, resize to 128x128        (plain image processing, no AI)
   |
2. Feed into the "judge classifier"            (a small CNN - NOT the LLM, NOT an API)
   -> guesses: hairstyle, hair colour, skin tone, glasses, facial hair, face shape, eye colour
   |
3. Those guesses become "constraints"          (same format as if you'd typed a sentence)
   |
4. Feed constraints into the VAE                (the generative model - builds the cartoon face)
   |
5. The same judge classifier checks the result: does the avatar actually have the guessed
   traits? If not, regenerate and try again (self-correct)
   |
Cartoon avatar
```

The LLM's job in this project is turning **English sentences** into structured trait data. A photo
isn't a sentence, so instead a **separate, much simpler neural network** (a plain image classifier,
`AttrClassifier` in `evaluate.py`) does the equivalent job for images — it looks at pixels and predicts
trait labels directly. It's not Claude, not any hosted API, not Ollama — just a small CNN trained
earlier in this project, running entirely on your own machine
(`checkpoints/attr_classifier.pt`). The photo is never saved to disk and never sent anywhere.

The LLM only gets involved *afterwards*, if you type a follow-up chat message like "make the hair
darker" on the avatar the photo produced — that follow-up text goes through the LLM (or the rule-based
fallback) exactly like any other chat message.

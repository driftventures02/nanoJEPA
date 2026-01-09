# NanoJEPA

A minimal implementation of a **Joint-Embedding Predictive Architecture (JEPA)** for text, inspired by Yann LeCun's vision of self-supervised learning.

---

## 🎯 Why This Exists

I believe Yann LeCun is right: **LLMs aren't the path forward**. 

Current language models are incredibly capable pattern matchers, but they don't truly "understand" meaning. They predict the next token based on statistical patterns, not by building an internal model of the world (in high dimensional vector space). LeCun's JEPA architecture offers an alternative: learn to predict **abstract representations** rather than raw outputs. It also happens to be much faster with lower compute requirements than traditional LLMs.

The core insight is that meaning should be captured in a way that works across modalities—text, images, video, audio. A sentence describing a red ball and an image of a red ball should share similar representations. JEPA attempts to learn these unified, semantic representations.

### This Project

I wanted to see if I could recreate Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) but as a JEPA version. This was built in one afternoon, so there's probably a lot wrong and a lot that could be changed but I decided to put it up anyway in case someone had a similar idea.

**Important caveats:**
- I didn't follow the findings and architecture in [LLM-JEPA](https://arxiv.org/abs/2410.10054) as that relies on adapting an existing LLM, I wanted to try building a purely representational version from scratch
- Instead, I adapted concepts from [I-JEPA](https://arxiv.org/abs/2301.08243) and [V-JEPA](https://arxiv.org/abs/2404.15244) to fit text
- I relied more on MLPs rather than Transformers for the predictor to keep things simple and trainable on my Macbook M4
- The next time I have time, I'll try implementing a mini version of the LLM-JEPA architecture (without EMA, using dual representation double pass, basically training a LoRA on top of an existing model)

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NanoJEPA                                 │
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │ STUDENT ENCODER │         │ TEACHER ENCODER │ (EMA updated) │
│  │   (Learnable)   │         │    (Frozen)     │               │
│  └────────┬────────┘         └────────┬────────┘               │
│           │                           │                         │
│           ▼                           ▼                         │
│      z_context                   z_target (truth)               │
│           │                           │                         │
│           ▼                           │                         │
│  ┌─────────────────┐                  │                         │
│  │    PREDICTOR    │                  │                         │
│  │  (MLP + Norm)   │                  │                         │
│  └────────┬────────┘                  │                         │
│           │                           │                         │
│           ▼                           ▼                         │
│      z_predicted ─────────────► MSE Loss                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Student Encoder** | Encodes context sentences → latent vector `z` |
| **Teacher Encoder** | Encodes target sentences → stable target (EMA of student) |
| **Predictor** | Transforms context vector → predicts target vector |
| **Decoder** | (Stage 2) Translates vectors back to text |

---

## 📁 Project Structure

```
nanojepa/
├── model.py        # NanoJEPA architecture (Student/Teacher/Predictor/Decoder)
├── train.py        # Stage 1: Train the "Brain" (encoder + predictor)
├── train_probe.py  # Stage 2: Train the "Mouth" (decoder) on frozen brain
├── generate.py     # Text generation using trained model
├── data.py         # TinyStories data loading with sentence boundaries
└── __init__.py

# Utilities
├── diagnose.py     # Verify if the model is actually learning
├── test_generate.py # Comprehensive generation test suite
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
# or
pip install torch tiktoken tqdm
```

### 2. Train the Brain (Stage 1)

This trains the JEPA to predict future representations. No text generation yet.

```bash
uv run python -m nanojepa.train \
    --run-name my-jepa-brain \
    --max-samples 32000 \
    --epochs 3 \
    --sentences-per-block 1 \
    --ema-decay 0.996
```

### 3. Train the Decoder Probe (Stage 2)

This teaches a decoder to translate the brain's vectors into text.

```bash
uv run python -m nanojepa.train_probe \
    --run my-jepa-brain \
    --epochs 4 \
    --max-samples 32000
```

### 4. Generate Text

```bash
uv run python -m nanojepa.generate \
    --run my-jepa-brain \
    --model-file probed_model.pt \
    --text "The little girl"
```

---

## 🔬 Diagnostics & Testing

### Diagnose Model Health

Run comprehensive diagnostics to verify the model is actually learning:

```bash
uv run python diagnose.py --run big-jepa-brain
```

This tests:
- **Vector Differentiation**: Do different prompts produce different vectors?
- **Semantic Clustering**: Do similar prompts cluster together?
- **Decoder Dependency**: Does the decoder actually use the JEPA vector?
- **Vector Health**: Are norms and statistics reasonable?

### Test Generation Quality

Run multiple prompts through the model and analyze outputs:

```bash
uv run python test_generate.py --run big-jepa-brain --model-file probed_model.pt
```

This runs 18 different prompts and gives you:
- `PASS`: Coherent output
- `WARN`: Has issues (repetition, too short)
- `FAIL`: Error occurred

Add `--quiet` for summary only.

---

## 🔧 How It Works

### The JEPA Training Loop

1. **Input**: Context sentences (e.g., "The cat sat on the mat.")
2. **Target**: Next sentences (e.g., "It was very comfortable.")
3. **Student encodes context** → `z_context` (pooled vector)
4. **Predictor predicts target** → `z_predicted`
5. **Teacher encodes target** → `z_target` (stable EMA)
6. **Loss** = MSE(`z_predicted`, `z_target`) + Repulsion Loss

### Why Two-Stage Training?

If you train the decoder simultaneously with the brain, **Posterior Collapse** occurs:
- The decoder ignores the JEPA vector entirely
- It becomes a standalone language model
- The "thinking" component learns nothing useful

**Solution**: Freeze the brain, then train the decoder separately.

---

## ⚖️ Tradeoffs We Made (Text vs. Images)

JEPA was designed for images (I-JEPA). Adapting it to text required significant compromises:

### 1. Pooling to Single Vector

| I-JEPA (Images) | NanoJEPA (Text) |
|-----------------|-----------------|
| Predicts patches (64+ vectors) | Pools to **1 vector** |
| Preserves spatial structure | Loses word order |
| Can do masked prediction | Can only do next-chunk |

**Why**: Variable-length text makes sequence-to-sequence prediction harder. Pooling simplifies the architecture but creates "smoothie" representations.

**Impact**: Model learns "vibes" (sentiment, topic) rather than grammar.

### 2. Sentence-Based Chunking

| Standard LLM | NanoJEPA |
|--------------|----------|
| Fixed token windows | Sentence boundaries |
| May split mid-sentence | Always complete sentences |

**Why**: JEPA needs semantically coherent chunks. A sentence is a natural unit of meaning.

**Impact**: More padding, variable-length handling, but cleaner representations.

### 3. Repulsion Loss

JEPA can collapse (all vectors become identical). We add:

```python
neg_dist = MSE(z_predicted, z_random_other_story)
repulsion_loss = ReLU(margin - neg_dist)
```

This pushes predicted vectors **away** from random targets.

### 4. Word Dropout in Decoder

To prevent the decoder from ignoring the JEPA vector:

```python
# Randomly replace 40% of input tokens with noise
mask = torch.rand(...) < 0.4
dec_input[mask] = 0
```

This forces the decoder to rely on the thought vector.

### 5. EMA Teacher (The Ghost)

The Teacher is an exponential moving average of the Student:

```python
Teacher = 0.996 * Teacher + 0.004 * Student
```

This provides stable targets and prevents collapse.

---

## 📊 How to Know If It's Working

### During Brain Training (Stage 1)

| Metric | Good | Bad |
|--------|------|-----|
| `cosine_sim` | 0.5 → 0.8 over training | Stuck at 0 or 1 |
| `z_hat_norm` | ~15-20 (stable) | Exploding (100+) or zero |
| `latent_loss` | Decreasing | Stuck or increasing |
| `repulsion_loss` | Near 0 (after warmup) | Always high |

### During Generation

| Output | Diagnosis |
|--------|-----------|
| Coherent sentences | Working! 🎉 |
| Repetition (`the the the`) | Reduce temperature, increase repetition penalty |
| Random words (`housefork`) | Top-K filtering broken, or brain didn't learn |
| Only periods (`......`) | Posterior collapse in decoder |
| Empty output | Start token issue |

---

## 🔬 Hyperparameters

### Brain Training

| Param | Default | Notes |
|-------|---------|-------|
| `--ema-decay` | 0.996 | Higher = more stable, slower learning |
| `--embed-dim` | 256 | Larger = more capacity, slower |
| `--sentences-per-block` | 1 | More = longer context, harder task |
| `--grad-clip` | 0.5 | Prevents exploding gradients |
| `--lr` | 3e-4 | With OneCycleLR warmup |

### Generation

| Param | Default | Notes |
|-------|---------|-------|
| Temperature | 0.7 | Lower = more deterministic |
| Top-K | 50 | Prunes unlikely tokens |
| Repetition Penalty | 1.5 | Penalizes repeated tokens |

---

## 📈 Example Runs

### `pure-jepa-brain`

Brain-only training, no decoder. Use for analyzing representations.

### `big-jepa-brain`

Full pipeline with probed decoder. Use for text generation.

---

## Common Issues

### "Vector norms exploding to 100+"

Add `LayerNorm` after encoder and predictor (already done in current code).

### "Cosine similarity stuck at 1.0"

Model collapsed. Increase repulsion margin or check EMA decay.

### "Decoder outputs periods only"

Posterior collapse. Increase word dropout or train longer.

### "Generation repeats words"

Lower temperature, increase repetition penalty, use Top-K sampling.

### "Missing decoder keys when loading"

You're loading `best_model.pt` (brain only) but need `probed_model.pt` (brain + decoder). Run `train_probe.py` first.

---

## Future Work

- Implement the actual [LLM-JEPA](https://arxiv.org/abs/2410.10054) architecture
- Try dual representation with double pass (no EMA)
- Train a LoRA adapter on top of an existing LLM instead of from scratch
- Add block masking instead of next-chunk prediction
- Experiment with sequence-to-sequence prediction (no pooling)

---

## References

- [A Path Towards Autonomous AI](https://openreview.net/pdf?id=BZ5a1r-kVsf) - LeCun's JEPA paper
- [I-JEPA](https://arxiv.org/abs/2301.08243) - Image JEPA
- [V-JEPA](https://arxiv.org/abs/2404.15244) - Video JEPA
- [LLM-JEPA](https://arxiv.org/abs/2410.10054) - JEPA for Language Models
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy's minimal GPT

---

## License

MIT

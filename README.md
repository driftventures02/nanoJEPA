# NanoJEPA

A tiny implementation of a Joint-Embedding Predictive Architecture (JEPA) for text, inspired by LeCun's vision.

It learns to predict "future thoughts" (vectors) rather than next tokens, and then uses a separate decoder probe to translate those thoughts back into English.

## Installation

```bash
# Using uv (Recommended)
uv sync

# Or pip
pip install torch tiktoken tqdm datasets
```

## How to Run

NanoJEPA uses a **Two-Stage Training** process to prevent "Posterior Collapse" (where the decoder ignores the thought vectors).

### 1. Train the Brain (Stage 1)
Trains the Encoder and Predictor to learn the "Physics of Concepts" in latent space. No decoding happens here.

```bash
uv run python -m nanojepa.train \
    --run-name big-jepa-brain \
    --max-samples 32000 \
    --epochs 3 \
    --sentences-per-block 1 \
    --ema-decay 0.996 \
    --pred-depth 4
```

### 2. Train the Probe (Stage 2)
Freezes the Brain and trains a Decoder ("Mouth") to translate the frozen thoughts into English. Uses noise injection to make the decoder robust.

```bash
uv run python -m nanojepa.train_probe \
    --run big-jepa-brain \
    --epochs 4 \
    --max-samples 32000
```

### 3. Generate Text
Uses the full chain (Context -> Brain -> Predicted Thought -> Mouth -> Text) to write a story.

```bash
uv run python -m nanojepa.generate \
    --run big-jepa-brain \
    --model-file probed_model.pt \
    --text "Once upon a time"
```

## Architecture

*   **Student Encoder:** Reads text, outputs latent vector $z$.
*   **Predictor:** Takes $z$, predicts next latent vector $z'$.
*   **Teacher Encoder:** (EMA of Student) Provides stable targets.
*   **Decoder (Probe):** Trained separately to map $z'$ back to text.


"""
Stage 2: Train the Decoder "Probe" on a Frozen JEPA Brain

This script takes a pre-trained JEPA brain (from train.py) and teaches
a decoder to translate its thought vectors into text.

Why two stages?
    If you train the decoder simultaneously with the brain, "Posterior Collapse"
    occurs: the decoder ignores the JEPA vector entirely and becomes a standalone
    language model. The brain learns nothing useful.
    
    By freezing the brain first, we force the decoder to actually read the vectors.

Key techniques:
    - Brain is FROZEN (no gradient updates)
    - Word Dropout: randomly mask decoder inputs so it can't just copy
    - Noise Injection: add noise to vectors so decoder learns robustness
    - Ground Truth vectors (not predictions): "Training Wheels" mode

Usage:
    uv run python -m nanojepa.train_probe \
        --run my-jepa-brain \
        --epochs 4 \
        --max-samples 32000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from pathlib import Path

from .model import NanoJEPA, JEPAConfig
from .data import get_dataloaders


TOKENIZER = tiktoken.get_encoding("gpt2")


def debug_generate(model, device, prompt="Once upon a time"):
    """
    Generate text during training to verify decoder is learning.
    
    Uses the student encoder directly (not predictor) to test if the
    decoder can translate "perfect" thought vectors into text.
    """
    model.eval()
    try:
        tokens = TOKENIZER.encode(prompt)
        ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Use student encoder for "ground truth" vector
            # This matches training where we use teacher's ground truth
            z_truth = model.encode_student(ctx_idx)
            memory = z_truth.unsqueeze(1)
            
            # Start with BOS token
            curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)
            
            out_tokens = []
            for _ in range(20):
                T = curr_seq.shape[1]
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                tgt_emb = model.embedding(curr_seq) + model.pos_embedding(pos)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
                
                dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = model.lm_head(dec_out)
                
                # Temperature sampling (avoids repetition loops)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == 50256:
                    out_tokens.append("<EOT>")
                    break
                
                out_tokens.append(TOKENIZER.decode([next_token.item()]))
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
                
        print(f"\n[PROBE DEBUG] '{prompt}' -> '{''.join(out_tokens)}'")
        
    except Exception as e:
        print(f"Debug failed: {e}")
    finally:
        model.train()


def train_probe(run_name: str, epochs: int = 3, max_samples: int = 12000):
    """
    Train the decoder probe on a frozen JEPA brain.
    
    Args:
        run_name: Name of the run folder containing best_model.pt
        epochs: Number of training epochs
        max_samples: Maximum training samples to use
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ------------------------------------------------------------------
    # 1. Load the Pre-Trained Brain
    # ------------------------------------------------------------------
    
    path = Path("runs") / run_name / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Brain not found at {path}! Run train.py first.")
        
    checkpoint = torch.load(path, map_location=device)
    saved_config = checkpoint["config"]
    
    # Create model with decoder enabled (encoder_only=False)
    config = JEPAConfig(
        vocab_size=saved_config.get("vocab_size", 50257),
        embed_dim=saved_config["embed_dim"],
        enc_layers=saved_config["enc_layers"],
        enc_heads=saved_config["enc_heads"],
        pred_depth=saved_config["pred_depth"],
        block_size=saved_config["block_size"],
        dropout=saved_config["dropout"],
        encoder_only=False  # Enable decoder
    )
    
    model = NanoJEPA(config).to(device)
    
    # Load brain weights (strict=False because we're adding decoder)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # ------------------------------------------------------------------
    # 2. FREEZE THE BRAIN
    # ------------------------------------------------------------------
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only decoder and language model head
    if hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = True
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Brain frozen. Training {trainable:,} / {total:,} parameters (decoder only)")
    
    # ------------------------------------------------------------------
    # 3. Load Data
    # ------------------------------------------------------------------
    
    train_loader, _ = get_dataloaders(
        block_size=config.block_size,
        batch_size=32,
        sentences_per_block=saved_config.get("sentences_per_block", 1),
        max_samples=max_samples,
        subset_ratio=1.0
    )
    
    # ------------------------------------------------------------------
    # 4. Optimizer (only for decoder parameters)
    # ------------------------------------------------------------------
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3  # Higher LR for probe training
    )
    
    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (context, target) in enumerate(pbar):
            context, target = context.to(device), target.to(device)
            
            # A. Get "Ground Truth" vector from frozen Teacher
            # This is "Training Wheels" mode - decoder learns to read
            # perfect vectors before trying to read predicted ones
            with torch.no_grad():
                z_target_truth = model.encode_teacher(target)
            
            # B. Noise Injection
            # Add small noise to simulate the imperfect predictions
            # the decoder will receive at inference time
            noise_level = 0.05
            noise = torch.randn_like(z_target_truth) * noise_level
            memory = (z_target_truth + noise).unsqueeze(1)  # [Batch, 1, Dim]
            
            # C. Prepare decoder inputs (shifted for autoregression)
            dec_in = target[:, :-1]
            dec_label = target[:, 1:]
            
            # D. Word Dropout
            # Randomly replace tokens to prevent decoder from just copying
            # Forces reliance on the thought vector
            dropout_prob = 0.2
            mask = torch.rand(dec_in.shape, device=device) < dropout_prob
            mask = mask & (dec_in != 50256)  # Don't drop BOS/pad
            dec_in_dropped = dec_in.clone()
            dec_in_dropped[mask] = 0  # Replace with token "!"
            
            # E. Forward pass
            T_dec = dec_in.shape[1]
            pos = torch.arange(0, T_dec, dtype=torch.long, device=device)
            tgt_emb = model.embedding(dec_in_dropped) + model.pos_embedding(pos)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
            
            dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = model.lm_head(dec_out)
            
            # F. Loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, 50257),
                dec_label.reshape(-1),
                ignore_index=50256
            )
            
            # G. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(probe_loss=f"{loss.item():.4f}")
            
            # Debug generation every 50 batches
            if batch_idx % 50 == 0:
                debug_generate(model, device)
    
    # ------------------------------------------------------------------
    # 6. Save the Probed Model
    # ------------------------------------------------------------------
    
    save_path = Path("runs") / run_name / "probed_model.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict()
    }, save_path)
    
    print(f"\nProbe training complete!")
    print(f"Saved to: {save_path}")
    print(f"\nTo generate text:")
    print(f"  uv run python -m nanojepa.generate --run {run_name} --model-file probed_model.pt --text \"Your prompt\"")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NanoJEPA Decoder Probe (Stage 2)")
    parser.add_argument("--run", type=str, required=True, help="Run name with trained brain")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--max-samples", type=int, default=12000, help="Max training samples")
    args = parser.parse_args()
    
    train_probe(args.run, args.epochs, args.max_samples)

"""
Train a decoder probe on frozen JEPA representations.

The JEPA brain is FROZEN. Only the decoder learns.
This teaches the decoder to "speak" what the brain "thinks."
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import NanoJEPA, JEPAConfig
from .data import get_dataloaders, decode, get_tokenizer


class Decoder(nn.Module):
    """
    Autoregressive decoder that translates JEPA vectors to tokens.
    
    Takes "memory" (JEPA vectors) and generates tokens autoregressively.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens, memory, word_dropout: float = 0.0):
        """
        Args:
            tokens: [Batch, Seq] - Target tokens (shifted for teacher forcing)
            memory: [Batch, MemSeq, Dim] - JEPA vectors to condition on
            word_dropout: Probability of dropping input tokens (prevents ignoring memory)
        
        Returns:
            logits: [Batch, Seq, Vocab]
        """
        B, T = tokens.shape
        device = tokens.device
        
        # Apply word dropout to prevent decoder from ignoring memory
        if word_dropout > 0 and self.training:
            mask = torch.rand(tokens.shape, device=device) < word_dropout
            tokens = tokens.clone()
            tokens[mask] = 0  # Replace with token 0 (usually '!')
        
        # Embeddings
        pos = torch.arange(T, dtype=torch.long, device=device)
        x = self.embedding(tokens) + self.pos_embedding(pos)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        
        # Decode
        out = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        logits = self.lm_head(out)
        
        return logits


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_probe(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load the trained JEPA brain
    brain_path = Path("runs") / args.brain_run / "best_model.pt"
    if not brain_path.exists():
        raise FileNotFoundError(f"Brain not found: {brain_path}")
    
    print(f"Loading JEPA brain from {brain_path}...")
    checkpoint = torch.load(brain_path, map_location=device, weights_only=False)
    brain_config = checkpoint["config"]
    
    # Recreate brain
    jepa_config = JEPAConfig(
        vocab_size=brain_config["vocab_size"],
        embed_dim=brain_config["embed_dim"],
        enc_layers=brain_config["enc_layers"],
        enc_heads=brain_config["enc_heads"],
        pred_depth=brain_config["pred_depth"],
        block_size=brain_config["block_size"],
        dropout=brain_config["dropout"],
        mask_ratio=brain_config["mask_ratio"],
    )
    brain = NanoJEPA(jepa_config).to(device)
    brain.load_state_dict(checkpoint["model_state_dict"])
    
    # 2. FREEZE THE BRAIN
    for param in brain.parameters():
        param.requires_grad = False
    brain.eval()
    print("Brain frozen.")
    
    # 3. Create decoder
    decoder = Decoder(
        vocab_size=brain_config["vocab_size"],
        embed_dim=brain_config["embed_dim"],
        num_layers=4,
        num_heads=brain_config["enc_heads"],
        dropout=0.1,
        max_len=brain_config["block_size"]
    ).to(device)
    
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {n_params:,}")
    
    # 4. Data
    train_loader, val_loader = get_dataloaders(
        block_size=brain_config["block_size"],
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    # 5. Optimizer (ONLY decoder params)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    
    # 6. Training loop
    run_name = args.run_name or f"probe-{args.brain_run}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        decoder.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            # Get JEPA vectors (no gradients!)
            with torch.no_grad():
                # Encode the full sequence
                z = brain.encode(batch)  # [Batch, Seq, Dim]
                # Apply predictor to get "predicted" vectors
                z_pred = brain.predict(z)  # [Batch, Seq, Dim]
            
            # Use z_pred as memory for decoder
            # Decoder input: [BOS, t1, t2, ...] -> predict [t1, t2, t3, ...]
            # For simplicity, use same sequence shifted
            dec_input = batch[:, :-1]
            dec_target = batch[:, 1:]
            
            # Forward decoder with word dropout
            logits = decoder(dec_input, memory=z_pred, word_dropout=0.3)
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, brain_config["vocab_size"]),
                dec_target.reshape(-1),
                ignore_index=50256  # Ignore padding
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")
        
        # Validation
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = brain.encode(batch)
                z_pred = brain.predict(z)
                
                dec_input = batch[:, :-1]
                dec_target = batch[:, 1:]
                
                logits = decoder(dec_input, memory=z_pred, word_dropout=0.0)
                loss = F.cross_entropy(
                    logits.reshape(-1, brain_config["vocab_size"]),
                    dec_target.reshape(-1),
                    ignore_index=50256
                )
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"  val_loss={val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "decoder_state_dict": decoder.state_dict(),
                "brain_run": args.brain_run,
                "brain_config": brain_config,
            }, log_dir / "decoder.pt")
            print(f"  -> Saved best decoder (val_loss={val_loss:.4f})")
        
        # Debug generation
        if epoch % 1 == 0:
            debug_generate(brain, decoder, device)
    
    print(f"\nProbe training complete. Best val_loss: {best_val_loss:.4f}")


def debug_generate(brain, decoder, device, prompt="Once upon a time"):
    """Generate a sample to see if it's working."""
    brain.eval()
    decoder.eval()
    
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(prompt)
    
    # Pad to reasonable length
    block_size = 64
    if len(tokens) < block_size:
        tokens = tokens + [50256] * (block_size - len(tokens))
    
    x = torch.tensor([tokens[:block_size]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Get JEPA vectors
        z = brain.encode(x)
        z_pred = brain.predict(z)
        
        # Autoregressive generation
        generated = [50256]  # Start token
        for _ in range(50):
            dec_input = torch.tensor([generated], dtype=torch.long, device=device)
            logits = decoder(dec_input, memory=z_pred)
            
            # Sample from last position
            probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 50256:
                break
            generated.append(next_token)
    
    output = tokenizer.decode(generated[1:])  # Skip start token
    print(f"\n[DEBUG GENERATION]")
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train decoder probe on JEPA")
    parser.add_argument("--brain-run", type=str, required=True, help="Name of the JEPA brain run to use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    
    train_probe(args)


if __name__ == "__main__":
    main()

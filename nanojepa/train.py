"""
Stage 1: Train the JEPA "Brain" (Encoder + Predictor)

This script trains the representation learning components of NanoJEPA.
The decoder is disabled - we only train the encoder to predict future
abstract representations, not words.

After training, run train_probe.py (Stage 2) to teach a decoder to
translate these representations into text.

Usage:
    uv run python -m nanojepa.train \
        --run-name my-jepa-brain \
        --max-samples 32000 \
        --epochs 3 \
        --ema-decay 0.996
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import tiktoken
import torch
from tqdm import tqdm

from .model import NanoJEPA, JEPAConfig
from .data import get_dataloaders


TOKENIZER = tiktoken.get_encoding("gpt2")


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gradient_norms(model: torch.nn.Module) -> dict:
    """
    Get gradient norms for key components.
    
    Useful for debugging training - if norms explode, increase grad_clip.
    If norms vanish, learning rate might be too low.
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()

    # Aggregate by component
    enc_norm = sum(v for k, v in norms.items() if "student_encoder" in k)
    pred_norm = sum(v for k, v in norms.items() if "predictor" in k)
    emb_norm = sum(v for k, v in norms.items() if "embedding" in k)

    return {
        "grad_norm_encoder": enc_norm,
        "grad_norm_predictor": pred_norm,
        "grad_norm_embedding": emb_norm,
        "grad_norm_total": sum(norms.values()),
    }


def debug_brain(model, device, prompt="Once upon a time"):
    """
    Print vector statistics for debugging.
    
    Healthy signs:
        - Norm: ~15-20 (with LayerNorm)
        - Std: ~1.0
        - Values spread across positive and negative
    
    Bad signs:
        - Norm → 0: Collapse
        - Norm → 100+: Exploding (need more grad_clip or LayerNorm)
        - All values identical: Collapse
    """
    model.eval()
    try:
        tokens = TOKENIZER.encode(prompt)
        ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            z = model.encode_student(ctx_idx)
            z_pred = model.predictor(z)
            
            print("\n" + "="*40)
            print("[BRAIN DEBUG]")
            print(f"Input Text: '{prompt}'")
            
            for name, vec in [("Encoded (z)", z), ("Predicted (z_pred)", z_pred)]:
                v = vec[0]  # First item in batch
                print(f"\n{name}:")
                print(f"  Norm: {v.norm().item():.4f}")
                print(f"  Mean: {v.mean().item():.4f} | Std: {v.std().item():.4f}")
                print(f"  Min:  {v.min().item():.4f} | Max: {v.max().item():.4f}")
                vals = ", ".join([f"{x:.3f}" for x in v[:5].tolist()])
                print(f"  Sample: [{vals}, ...]")
            print("="*40 + "\n")
                
    except Exception as e:
        print(f"Brain Debug Failed: {e}")
    finally:
        model.train()


class Logger:
    """Simple CSV + console logger for training metrics."""

    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = log_dir / f"{run_name}.csv"
        self.config_path = log_dir / f"{run_name}_config.json"

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None
        self.step = 0

    def log_config(self, config: dict):
        """Save config to JSON for reproducibility."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {self.config_path}")

    def log(self, metrics: dict):
        """Log metrics to CSV file."""
        metrics["step"] = self.step

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()

        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        self.step += 1

    def close(self):
        self.csv_file.close()
        print(f"Logs saved to {self.csv_path}")


@torch.no_grad()
def evaluate(model: NanoJEPA, val_loader, device: torch.device) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_latent_loss = 0.0
    n_batches = 0

    for context, target in val_loader:
        context, target = context.to(device), target.to(device)
        _, losses = model(context, target)

        total_loss += losses["total_loss"].item()
        total_latent_loss += losses["latent_loss"].item()
        n_batches += 1

    model.train()
    return {
        "val_loss": total_loss / n_batches,
        "val_latent_loss": total_latent_loss / n_batches,
    }


def train(config: dict):
    """
    Main training loop for the JEPA brain.
    
    Key components:
        1. OneCycleLR scheduler with warmup (prevents cold start explosion)
        2. Gradient clipping (prevents training instability)
        3. EMA teacher updates (prevents representation collapse)
    """
    device = get_device()
    print(f"Using device: {device}")

    # Setup logging
    run_name = config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / run_name
    logger = Logger(log_dir, run_name)
    logger.log_config(config)

    # Create model (encoder_only=True for brain-only training)
    model_config = JEPAConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        enc_layers=config["enc_layers"],
        enc_heads=config["enc_heads"],
        pred_depth=config["pred_depth"],
        block_size=config["block_size"],
        dropout=config["dropout"],
        encoder_only=True,
    )
    model = NanoJEPA(model_config).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Load data
    train_loader, val_loader = get_dataloaders(
        block_size=config["block_size"],
        batch_size=config["batch_size"],
        stride=config["stride"],
        subset_ratio=config["subset_ratio"],
        sentences_per_block=config["sentences_per_block"],
        max_samples=config["max_samples"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler: OneCycleLR
    # Includes warmup (30% of training) to prevent cold start explosion which I experienced
    # Then anneals down to near-zero for stable convergence
    total_steps = config["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.3,          # Warmup for first 30%
        div_factor=25,          # Start at lr/25
        final_div_factor=1000   # End at almost zero
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch_idx, (context, target) in enumerate(pbar):
            context, target = context.to(device), target.to(device)

            # Forward pass
            _, losses = model(context, target)
            loss = losses["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Get gradient norms before clipping (for debugging)
            grad_norms = get_gradient_norms(model)

            # Gradient clipping (prevents instability)
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            
            # CRITICAL: Update the Ghost Teacher via EMA
            model.update_teacher(decay=config["ema_decay"])
            
            # Step the scheduler
            scheduler.step()

            # Logging
            epoch_loss += loss.item()

            if global_step % config["log_interval"] == 0:
                logger.log({
                    "train_loss": loss.item(),
                    "train_latent_loss": losses["latent_loss"].item(),
                    "cosine_sim": losses["cosine_sim"].item(),
                    "z_hat_norm": losses["z_hat_norm"].item(),
                    "z_true_norm": losses["z_true_norm"].item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    **grad_norms,
                })

            # Print debug info periodically
            if global_step % config["sample_interval"] == 0:
                debug_brain(model, device)
                print(f"  cosine_sim={losses['cosine_sim'].item():.4f}, "
                      f"z_hat_norm={losses['z_hat_norm'].item():.2f}, "
                      f"z_true_norm={losses['z_true_norm'].item():.2f}")
                print(f"  grad_norms: enc={grad_norms['grad_norm_encoder']:.4f}, "
                      f"pred={grad_norms['grad_norm_predictor']:.4f}")

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lat=f"{losses['latent_loss'].item():.4f}",
                cos=f"{losses['cosine_sim'].item():.3f}",
            )
            global_step += 1

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={avg_epoch_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = log_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    logger.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Logs: {log_dir}")
    print(f"\nNext step: Train the decoder probe:")
    print(f"  uv run python -m nanojepa.train_probe --run {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Train NanoJEPA Brain (Stage 1)")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    
    # Model architecture
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--enc-heads", type=int, default=4)
    parser.add_argument("--pred-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Data params
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--subset-ratio", type=float, default=0.1)
    parser.add_argument("--sentences-per-block", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--run-name", type=str, default=None)
    
    args = parser.parse_args()

    config = {
        "vocab_size": 50257,
        "embed_dim": args.embed_dim,
        "enc_layers": args.enc_layers,
        "enc_heads": args.enc_heads,
        "pred_depth": args.pred_depth,
        "block_size": args.block_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "log_interval": args.log_interval,
        "sample_interval": args.sample_interval,
        "run_name": args.run_name,
        "subset_ratio": args.subset_ratio,
        "sentences_per_block": args.sentences_per_block,
        "max_samples": args.max_samples,
        "ema_decay": args.ema_decay,
    }

    train(config)


if __name__ == "__main__":
    main()

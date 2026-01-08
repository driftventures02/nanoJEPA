"""Training script for NanoJEPA encoder-predictor on TinyStories."""

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


# Global tokenizer for decoding samples
TOKENIZER = tiktoken.get_encoding("gpt2")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gradient_norms(model: torch.nn.Module) -> dict:
    """Get gradient norms for key components."""
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
    Sanity check for the Brain (JEPA).
    Prints vector stats to ensure no collapse (all zeros) or explosion.
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
            
            # Print Stats
            for name, vec in [("Encoded (z)", z), ("Predicted (z_pred)", z_pred)]:
                v = vec[0] # Take first item in batch
                print(f"\n{name}:")
                print(f"  Norm: {v.norm().item():.4f}")
                print(f"  Mean: {v.mean().item():.4f} | Std: {v.std().item():.4f}")
                print(f"  Min:  {v.min().item():.4f} | Max: {v.max().item():.4f}")
                # Print first 5 values formatted nicely
                vals = ", ".join([f"{x:.3f}" for x in v[:5].tolist()])
                print(f"  Sample: [{vals}, ...]")
            print("="*40 + "\n")
                
    except Exception as e:
        print(f"Brain Debug Failed: {e}")
    finally:
        model.train()


class Logger:
    """Simple CSV + console logger."""

    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = log_dir / f"{run_name}.csv"
        self.config_path = log_dir / f"{run_name}_config.json"

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None
        self.step = 0

    def log_config(self, config: dict):
        """Save config to JSON."""
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {self.config_path}")

    def log(self, metrics: dict):
        """Log metrics to CSV."""
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
    """Main training loop."""
    device = get_device()
    print(f"Using device: {device}")

    # Setup logging
    run_name = config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / run_name
    logger = Logger(log_dir, run_name)
    logger.log_config(config)

    # Create model
    model_config = JEPAConfig(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        enc_layers=config["enc_layers"],
        enc_heads=config["enc_heads"],
        pred_depth=config["pred_depth"],
        block_size=config["block_size"],
        dropout=config["dropout"],
        encoder_only=True, # Always train brain-only in this script
    )
    model = NanoJEPA(model_config).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Get data
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

    # Learning rate scheduler: OneCycleLR (Warmup + Decay)
    # This prevents the "Cold Start" explosion.
    total_steps = config["epochs"] * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.3,   # Spend first 30% of time warming up
        div_factor=25,   # Start at lr / 25
        final_div_factor=1000 # End at almost zero
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_latent_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch_idx, (context, target) in enumerate(pbar):
            context, target = context.to(device), target.to(device)

            # Forward pass
            _, losses = model(context, target)
            loss = losses["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Get gradient norms before clipping
            grad_norms = get_gradient_norms(model)

            # Gradient clipping
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            
            # CRITICAL: Update the Ghost Teacher
            model.update_teacher(decay=config["ema_decay"])
            
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            epoch_latent_loss += losses["latent_loss"].item()

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

            # Print sample every N steps
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
        # avg_epoch_latent = epoch_latent_loss / len(train_loader) # Unused

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


def main():
    parser = argparse.ArgumentParser(description="Train NanoJEPA")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence block size")
    parser.add_argument("--stride", type=int, default=64, help="Stride between samples")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--enc-layers", type=int, default=4, help="Encoder layers")
    parser.add_argument("--enc-heads", type=int, default=4, help="Encoder attention heads")
    parser.add_argument("--pred-depth", type=int, default=2, help="Predictor MLP depth")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=0.5, help="Gradient clipping")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--sample-interval", type=int, default=100, help="Print sample every N steps")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logs")
    parser.add_argument("--subset-ratio", type=float, default=0.1, help="Ratio of TinyStories to use (0.0-1.0)")
    parser.add_argument("--sentences-per-block", type=int, default=3, help="Number of sentences per context/target block")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use (for quick testing)")
    parser.add_argument("--ema-decay", type=float, default=0.996, help="EMA decay rate for teacher")
    args = parser.parse_args()

    config = {
        "vocab_size": 50257,  # GPT-2 vocab
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

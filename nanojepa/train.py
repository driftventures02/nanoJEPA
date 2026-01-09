"""
Training script for NanoJEPA with block masking.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from .model import NanoJEPA, JEPAConfig
from .data import get_dataloaders, decode, get_tokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gradient_norms(model: torch.nn.Module) -> dict:
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()

    enc_norm = sum(v for k, v in norms.items() if "student_encoder" in k)
    pred_norm = sum(v for k, v in norms.items() if "predictor" in k)

    return {
        "grad_norm_encoder": enc_norm,
        "grad_norm_predictor": pred_norm,
        "grad_norm_total": sum(norms.values()),
    }


def debug_masking(model, batch, device):
    """
    Visualize how masking works on a sample.
    Shows: Original text, Masked positions, What model sees.
    """
    model.eval()
    try:
        sample = batch[0:1].to(device)  # Take first sample
        B, T = sample.shape
        
        # Create mask
        mask = model.create_block_mask(B, T, device)
        
        # Find mask boundaries
        mask_positions = mask[0].nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            mask_start = mask_positions[0].item()
            mask_end = mask_positions[-1].item() + 1
        else:
            mask_start = mask_end = 0
        
        # Decode
        original_text = decode(sample[0])
        
        # Create "masked" version for visualization
        masked_tokens = sample[0].clone()
        masked_tokens[mask[0]] = 50256  # Replace with EOT for visual
        masked_text = decode(masked_tokens)
        
        print("\n" + "="*60)
        print("[MASKING DEBUG]")
        print(f"Sequence length: {T}, Masked: {mask.sum().item()} tokens ({mask_start}:{mask_end})")
        print("="*60)
        print(f"\nORIGINAL:\n{original_text[:200]}...")
        print(f"\nMASKED (positions {mask_start}-{mask_end} hidden):\n{masked_text[:200]}...")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Debug failed: {e}")
    finally:
        model.train()


class Logger:
    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / f"{run_name}.csv"
        self.config_path = log_dir / f"{run_name}_config.json"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None
        self.step = 0

    def log_config(self, config: dict):
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log(self, metrics: dict):
        metrics["step"] = self.step
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        self.step += 1

    def close(self):
        self.csv_file.close()


@torch.no_grad()
def evaluate(model: NanoJEPA, val_loader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = batch.to(device)
        loss, _ = model(batch)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return {"val_loss": total_loss / max(n_batches, 1)}


def train(config: dict):
    device = get_device()
    print(f"Using device: {device}")

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
        mask_ratio=config["mask_ratio"],
    )
    model = NanoJEPA(model_config).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Get data
    train_loader, val_loader = get_dataloaders(
        block_size=config["block_size"],
        batch_size=config["batch_size"],
        max_samples=config["max_samples"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    total_steps = config["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )

    # Training
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch in pbar:
            batch = batch.to(device)

            # Forward (mask is created internally)
            loss, info = model(batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            grad_norms = get_gradient_norms(model)

            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()
            model.update_teacher(decay=config["ema_decay"])
            scheduler.step()

            epoch_loss += loss.item()

            # Logging
            if global_step % config["log_interval"] == 0:
                logger.log({
                    "train_loss": loss.item(),
                    "pos_loss": info["pos_loss"].item(),
                    "repulsion_loss": info["repulsion_loss"].item(),
                    "cosine_sim": info["cosine_sim"].item(),
                    "z_pred_norm": info["z_pred_norm"].item(),
                    "z_teacher_norm": info["z_teacher_norm"].item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    **grad_norms,
                })

            # Debug visualization
            if global_step % config["sample_interval"] == 0 and global_step > 0:
                debug_masking(model, batch, device)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cos=f"{info['cosine_sim'].item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            global_step += 1

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]

        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = log_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, checkpoint_path)
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    logger.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train NanoJEPA")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--pred-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.05, help="Fraction of sequence to mask")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    args = parser.parse_args()

    config = {
        "vocab_size": 50257,
        "embed_dim": args.embed_dim,
        "enc_layers": args.enc_layers,
        "enc_heads": args.enc_heads,
        "pred_depth": args.pred_depth,
        "block_size": args.block_size,
        "dropout": args.dropout,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "log_interval": args.log_interval,
        "sample_interval": args.sample_interval,
        "run_name": args.run_name,
        "max_samples": args.max_samples,
        "ema_decay": args.ema_decay,
    }

    train(config)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from tqdm import tqdm
from pathlib import Path
from .model import NanoJEPA, JEPAConfig
from .data import get_dataloaders

TOKENIZER = tiktoken.get_encoding("gpt2")

def debug_generate(model, device, prompt="Once upon a time"):
    """
    Sanity check: Force the model to generate from a vector during training.
    """
    model.eval()
    try:
        tokens = TOKENIZER.encode(prompt)
        ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # 1. Brain Step (Cheat Mode: Use Encoder directly to test Decoder translation)
            # We want to see if the Decoder can speak a "perfect thought" derived from the prompt.
            z_truth = model.encode_student(ctx_idx)
            memory = z_truth.unsqueeze(1)
            
            # 2. Mouth Step
            curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)
            
            out_tokens = []
            for _ in range(20):
                T = curr_seq.shape[1]
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                tgt_emb = model.embedding(curr_seq) + model.pos_embedding(pos)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
                
                dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = model.lm_head(dec_out)
                
                # Use Sampling (Temperature) to prevent loops
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

def train_probe(run_name, epochs=3, max_samples=12000):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load the Pre-Trained Brain (JEPA)
    path = Path("runs") / run_name / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Run {run_name} not found at {path}!")
        
    checkpoint = torch.load(path, map_location=device)
    saved_config = checkpoint["config"]
    
    # Force encoder_only=False so we create a Decoder this time
    config = JEPAConfig(
        vocab_size=saved_config.get("vocab_size", 50257),
        embed_dim=saved_config["embed_dim"],
        enc_layers=saved_config["enc_layers"],
        enc_heads=saved_config["enc_heads"],
        pred_depth=saved_config["pred_depth"],
        block_size=saved_config["block_size"],
        dropout=saved_config["dropout"],
        encoder_only=False 
    )
    
    model = NanoJEPA(config).to(device)
    # Load weights (Strict=False because we are adding a decoder that wasn't there before)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # 2. FREEZE THE BRAIN (Encoder + Predictor)
    # We only want to train the "Mouth" (Decoder)
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only Decoder and Head
    if hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            param.requires_grad = True
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
    print("Brain frozen. Training Decoder Probe only.")
    
    # 3. Data
    train_loader, _ = get_dataloaders(
        block_size=config.block_size,
        batch_size=32,
        sentences_per_block=saved_config.get("sentences_per_block", 1), # Default to 1 if missing
        max_samples=max_samples, # Use the argument
        subset_ratio=1.0 # Use all of max_samples
    )
    
    # 4. Optimizer (Only for Decoder)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3 # Higher LR because we are just training the probe
    )
    
    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}")
        for batch_idx, (context, target) in enumerate(pbar):
            context, target = context.to(device), target.to(device)
            
            # A. Get Fixed Thought Vector (No Gradients)
            with torch.no_grad():
                # NEW: Ground Truth Target Vector (Training Wheels)
                z_target_truth = model.encode_teacher(target)
            
            # --- VITAL: ADD NOISE ---
            # Simulate the "fuzziness" of the JEPA Predictor.
            # Without this, the Probe will be a "Snob" and reject the JEPA's output.
            noise_level = 0.05
            noise = torch.randn_like(z_target_truth) * noise_level
            memory = (z_target_truth + noise).unsqueeze(1)
            # ------------------------
            
            # B. Train Decoder to speak this vector
            dec_in = target[:, :-1]
            dec_label = target[:, 1:]
            
            # Create Memory (The Thought Vector)
            # memory = z_target_truth.unsqueeze(1) # [Batch, 1, Dim] (Replaced by noise injection above)
            
            # Masking
            T_dec = dec_in.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
            pos = torch.arange(0, T_dec, dtype=torch.long, device=device)
            
            # Forward Decoder
            # Word Dropout
            if True: 
                dropout_prob = 0.2
                mask = torch.rand(dec_in.shape, device=device) < dropout_prob
                mask = mask & (dec_in != 50256) # Don't drop BOS
                dec_in_dropped = dec_in.clone()
                dec_in_dropped[mask] = 0
                tgt_emb = model.embedding(dec_in_dropped) + model.pos_embedding(pos)
            else:
                tgt_emb = model.embedding(dec_in) + model.pos_embedding(pos)

            dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = model.lm_head(dec_out)
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, 50257), 
                dec_label.reshape(-1),
                ignore_index=50256
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(probe_loss=f"{loss.item():.4f}")
            
            if batch_idx % 50 == 0:
                debug_generate(model, device)
            
    # Save the Probed Model
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict()
    }, Path("runs") / run_name / "probed_model.pt")
    print(f"Probe training done. Saved to runs/{run_name}/probed_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=12000)
    args = parser.parse_args()
    train_probe(args.run, args.epochs, args.max_samples)

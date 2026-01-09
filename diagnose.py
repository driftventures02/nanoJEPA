#!/usr/bin/env python3
"""
NanoJEPA Diagnostic Tool

Verifies whether the JEPA brain is actually learning meaningful representations.
Run this after training to check model health before investing time in generation.

Tests:
    1. Vector Health Check - Are norms/stats reasonable?
    2. Vector Differentiation - Do different prompts produce different vectors?
    3. Semantic Clustering - Do similar prompts cluster together?
    4. Decoder Dependency - Does the decoder actually use the JEPA vector?

Usage:
    uv run python diagnose.py --run big-jepa-brain
    uv run python diagnose.py --run big-jepa-brain --model-file probed_model.pt

Interpretation:
    - Vector norms ~15-20: Healthy (LayerNorm is working)
    - Vector norms → 0: Collapse (EMA decay too low or training issue)
    - Vector norms → 100+: Explosion (need more grad_clip)
    - High cosine similarity between ALL prompts: Collapse
    - Decoder outputs same text for real/random vectors: Posterior collapse
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

from nanojepa.model import NanoJEPA, JEPAConfig


device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")


def load_model(run_name: str, model_file: str = "probed_model.pt") -> NanoJEPA:
    """Load a trained NanoJEPA model."""
    path = Path("runs") / run_name / model_file
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    
    if isinstance(saved_config, JEPAConfig):
        config = saved_config
        config.encoder_only = False
    else:
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode_prompt(model: NanoJEPA, prompt: str):
    """Encode a prompt and return both encoded and predicted vectors."""
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        z = model.encode_student(idx)
        z_pred = model.predictor(z)
    return z, z_pred


def test_vector_stats(model: NanoJEPA):
    """
    Test 1: Basic vector health check.
    
    What to look for:
        - Norm ~15-20: Healthy with LayerNorm
        - Std ~1.0: Good distribution
        - Norm → 0: Collapse
        - Norm → 100+: Explosion
    """
    print("\n" + "="*60)
    print("TEST 1: Vector Health Check")
    print("="*60)
    
    prompts = ["Once upon a time", "The happy girl", "A big dog"]
    
    for p in prompts:
        z_enc, z_pred = encode_prompt(model, p)
        print(f"\n'{p}':")
        print(f"  Encoded: norm={z_enc.norm().item():.2f}, mean={z_enc.mean().item():.4f}, std={z_enc.std().item():.4f}")
        print(f"  Predicted: norm={z_pred.norm().item():.2f}, mean={z_pred.mean().item():.4f}, std={z_pred.std().item():.4f}")
        
        # Diagnose
        if z_pred.norm().item() < 1.0:
            print("  ⚠ Vector norm too low - possible collapse!")
        elif z_pred.std().item() < 0.1:
            print("  ⚠ Vector std too low - possible collapse!")
        else:
            print("  ✓ Vector looks healthy")


def test_vector_differentiation(model: NanoJEPA):
    """
    Test 2: Do different prompts produce different vectors?
    
    If all vectors are nearly identical (cosine sim > 0.95),
    the model has collapsed and isn't learning.
    """
    print("\n" + "="*60)
    print("TEST 2: Vector Differentiation")
    print("="*60)
    
    prompts = [
        "The happy girl",
        "The sad boy",
        "The angry dog",
        "Once upon a time",
        "The scary monster",
    ]
    
    vectors = []
    for p in prompts:
        _, z_pred = encode_prompt(model, p)
        vectors.append(z_pred)
    
    # Print similarity matrix
    print("\nCosine Similarity Matrix:")
    print(f"{'':20}", end="")
    for i in range(len(prompts)):
        print(f"{i:8}", end="")
    print()
    
    for i, p1 in enumerate(prompts):
        print(f"{p1[:18]:20}", end="")
        for j in range(len(prompts)):
            sim = F.cosine_similarity(vectors[i], vectors[j], dim=-1).item()
            print(f"{sim:8.3f}", end="")
        print()
    
    # Calculate average off-diagonal similarity
    total_sim = 0
    count = 0
    for i in range(len(prompts)):
        for j in range(len(prompts)):
            if i != j:
                sim = F.cosine_similarity(vectors[i], vectors[j], dim=-1).item()
                total_sim += sim
                count += 1
    avg_sim = total_sim / count
    
    print(f"\nAverage similarity between DIFFERENT prompts: {avg_sim:.3f}")
    if avg_sim > 0.95:
        print("⚠ WARNING: Vectors are too similar! Possible collapse.")
    elif avg_sim < 0.3:
        print("✓ GOOD: Vectors are nicely differentiated.")
    else:
        print("? MIXED: Some differentiation, but could be better.")


def test_semantic_clustering(model: NanoJEPA):
    """
    Test 3: Do semantically similar prompts cluster together?
    
    If within-group similarity > between-group similarity,
    the model is learning semantic structure.
    """
    print("\n" + "="*60)
    print("TEST 3: Semantic Clustering")
    print("="*60)
    
    # Groups of semantically similar prompts
    groups = {
        "Happy": ["The happy girl", "She was very happy", "They laughed and played"],
        "Sad": ["The sad boy", "He was crying", "She felt very sad"],
        "Action": ["He ran fast", "She jumped high", "They played ball"],
    }
    
    group_vectors = {}
    for name, prompts in groups.items():
        vecs = []
        for p in prompts:
            _, z_pred = encode_prompt(model, p)
            vecs.append(z_pred)
        group_vectors[name] = torch.stack(vecs)
    
    print("\nWithin-group vs Between-group similarity:")
    
    for name, vecs in group_vectors.items():
        # Within-group similarity
        within_sim = 0
        count = 0
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within_sim += F.cosine_similarity(vecs[i], vecs[j], dim=-1).item()
                count += 1
        within_sim /= max(count, 1)
        
        # Between-group similarity
        between_sim = 0
        count = 0
        for other_name, other_vecs in group_vectors.items():
            if other_name != name:
                for v1 in vecs:
                    for v2 in other_vecs:
                        between_sim += F.cosine_similarity(v1, v2, dim=-1).item()
                        count += 1
        between_sim /= max(count, 1)
        
        diff = within_sim - between_sim
        status = "✓" if diff > 0 else "⚠"
        print(f"  {status} {name:10}: within={within_sim:.3f}, between={between_sim:.3f}, diff={diff:+.3f}")
    
    print("\nIf within > between (diff positive), the model is learning semantic clusters!")


def test_decoder_dependency(model: NanoJEPA):
    """
    Test 4: Does the decoder actually use the JEPA vector?
    
    We generate text from a real vector vs a random vector.
    If outputs are identical, the decoder is ignoring the vector
    (posterior collapse).
    """
    print("\n" + "="*60)
    print("TEST 4: Decoder Dependency")
    print("="*60)
    
    # Check if model has decoder
    if not hasattr(model, 'decoder'):
        print("\n⚠ Model has no decoder. Run with probed_model.pt")
        return
    
    prompt = "The little girl"
    _, z_pred_real = encode_prompt(model, prompt)
    z_pred_random = torch.randn_like(z_pred_real)
    
    def generate_from_vector(z_pred, max_tokens=20):
        memory = z_pred.unsqueeze(1)
        curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)
        
        for _ in range(max_tokens):
            T = curr_seq.shape[1]
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            tgt_emb = model.embedding(curr_seq) + model.pos_embedding(pos)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(device)
            
            with torch.no_grad():
                dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = model.lm_head(dec_out)
            
            probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == 50256:
                break
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
        
        return tokenizer.decode(curr_seq[0, 1:].tolist())
    
    print(f"\nPrompt: '{prompt}'")
    
    output_real = generate_from_vector(z_pred_real)
    output_random = generate_from_vector(z_pred_random)
    
    print(f"\nWith REAL vector:   '{output_real}'")
    print(f"With RANDOM vector: '{output_random}'")
    
    if output_real == output_random:
        print("\n⚠ WARNING: Outputs are IDENTICAL! Decoder is ignoring the vector.")
    else:
        print("\n✓ GOOD: Outputs are different. Decoder is using the vector.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NanoJEPA model")
    parser.add_argument("--run", type=str, required=True, help="Run name (folder in runs/)")
    parser.add_argument("--model-file", type=str, default="probed_model.pt", help="Model file to load")
    args = parser.parse_args()
    
    print(f"Loading model from runs/{args.run}/{args.model_file}...")
    
    try:
        model = load_model(args.run, args.model_file)
    except FileNotFoundError:
        print(f"\n✗ Could not find runs/{args.run}/{args.model_file}")
        print("  If testing brain-only, try: --model-file best_model.pt")
        print("  If testing generation, run train_probe.py first")
        return
    
    test_vector_stats(model)
    test_vector_differentiation(model)
    test_semantic_clustering(model)
    test_decoder_dependency(model)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

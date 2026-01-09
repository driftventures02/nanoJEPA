#!/usr/bin/env python3
"""
Comprehensive test suite for NanoJEPA text generation.

Runs multiple prompts through the model and analyzes the outputs
to determine if the model is actually learning something meaningful.

Usage:
    uv run python test_generate.py --run big-jepa-brain --model-file probed_model.pt
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from collections import Counter

from nanojepa.model import NanoJEPA, JEPAConfig

# Test prompts covering different scenarios
TEST_PROMPTS = [
    # Standard story starters
    "Once upon a time",
    "The little girl",
    "The boy was playing",
    "Mom said",
    
    # Emotional contexts
    "She was very happy because",
    "He felt sad when",
    "The dog was scared of",
    
    # Action-oriented
    "They went to the",
    "He ran to the",
    "She picked up the",
    
    # Dialogue starters
    '"Hello," said the',
    '"Can I have" asked the',
    
    # Descriptive
    "The big red",
    "A tiny little",
    "The old house",
    
    # Cause and effect
    "Because it was raining",
    "After eating lunch",
    "When the sun came up",
]

def load_model(run_name, model_file):
    """Load the trained model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    path = Path("runs") / run_name / model_file
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    print(f"Loading {path}...")
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
    
    return model, device


def generate_one(model, prompt, device, max_tokens=30, temperature=0.7, top_k=50, rep_penalty=1.5):
    """Generate text from a single prompt."""
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(prompt)
    ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        z_context = model.encode_student(ctx_idx)
        z_future = model.predictor(z_context)
        
        curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)
        memory = z_future.unsqueeze(1)
        
        generated_tokens = []
        
        for _ in range(max_tokens):
            T_tgt = curr_seq.shape[1]
            pos_tgt = torch.arange(0, T_tgt, dtype=torch.long, device=device)
            tgt_emb = model.embedding(curr_seq) + model.pos_embedding(pos_tgt)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T_tgt).to(device)
            
            dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = model.lm_head(dec_out)
            next_logits = logits[:, -1, :].clone()
            
            # Repetition penalty
            for token_id in set(curr_seq[0].tolist()):
                next_logits[0, token_id] -= rep_penalty
            
            # Temperature
            next_logits = next_logits / temperature
            
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == 50256:
                break
            
            generated_tokens.append(next_token.item())
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
    
    return tokenizer.decode(generated_tokens)


def analyze_output(text):
    """Analyze generated text for quality metrics."""
    words = text.split()
    
    metrics = {
        "length": len(words),
        "unique_words": len(set(words)),
        "has_punctuation": any(c in text for c in ".!?,"),
        "has_repetition": False,
        "is_coherent": True,
    }
    
    # Check for word repetition (same word 3+ times in a row)
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            metrics["has_repetition"] = True
            break
    
    # Check for character spam
    if len(text) > 0:
        char_counts = Counter(text)
        most_common_char, count = char_counts.most_common(1)[0]
        if count / len(text) > 0.3 and most_common_char in ".!?":
            metrics["is_coherent"] = False
    
    # Very short output is suspicious
    if len(words) < 3:
        metrics["is_coherent"] = False
    
    return metrics


def run_tests(model, device, verbose=True):
    """Run all test prompts and analyze results."""
    results = []
    
    print("\n" + "="*70)
    print("NANOJEPA GENERATION TEST SUITE")
    print("="*70)
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        try:
            output = generate_one(model, prompt, device)
            metrics = analyze_output(output)
            
            result = {
                "prompt": prompt,
                "output": output,
                "metrics": metrics,
                "status": "PASS" if metrics["is_coherent"] and not metrics["has_repetition"] else "WARN"
            }
            
            if verbose:
                status_icon = "✓" if result["status"] == "PASS" else "⚠"
                print(f"\n[{i}/{len(TEST_PROMPTS)}] {status_icon} {prompt}")
                print(f"    → {output[:100]}{'...' if len(output) > 100 else ''}")
                if result["status"] == "WARN":
                    issues = []
                    if metrics["has_repetition"]: issues.append("repetition")
                    if not metrics["is_coherent"]: issues.append("incoherent")
                    print(f"    ⚠ Issues: {', '.join(issues)}")
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "prompt": prompt,
                "output": "",
                "metrics": {},
                "status": "FAIL",
                "error": str(e)
            })
            if verbose:
                print(f"\n[{i}/{len(TEST_PROMPTS)}] ✗ {prompt}")
                print(f"    Error: {e}")
    
    return results


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    
    print(f"\nResults: {passed}/{total} PASS, {warned} WARN, {failed} FAIL")
    
    # Aggregate metrics
    coherent_results = [r for r in results if r["metrics"].get("is_coherent", False)]
    if coherent_results:
        avg_length = sum(r["metrics"]["length"] for r in coherent_results) / len(coherent_results)
        avg_unique = sum(r["metrics"]["unique_words"] for r in coherent_results) / len(coherent_results)
        print(f"Average output length: {avg_length:.1f} words")
        print(f"Average unique words: {avg_unique:.1f}")
    
    # Diagnosis
    print("\n" + "-"*40)
    print("DIAGNOSIS")
    print("-"*40)
    
    if passed >= total * 0.8:
        print("✓ Model is generating coherent text!")
        print("  The JEPA brain has learned meaningful representations.")
    elif passed >= total * 0.5:
        print("⚠ Model is partially working.")
        print("  Consider: more training, different hyperparameters, or more data.")
    else:
        print("✗ Model is struggling.")
        print("  Possible issues:")
        
        # Check for specific problems
        all_outputs = " ".join(r["output"] for r in results)
        
        if all_outputs.count(".") > len(all_outputs) * 0.3:
            print("  - Posterior collapse (outputs mostly periods)")
            print("    → Train probe longer, increase word dropout")
        
        if any("and and" in r["output"] or "the the" in r["output"] for r in results):
            print("  - Repetition loops")
            print("    → Increase repetition penalty, lower temperature")
        
        if sum(r["metrics"].get("length", 0) for r in results) < len(results) * 3:
            print("  - Very short outputs")
            print("    → Check start token handling, train longer")


def main():
    parser = argparse.ArgumentParser(description="Test NanoJEPA generation")
    parser.add_argument("--run", type=str, required=True, help="Run name (folder in runs/)")
    parser.add_argument("--model-file", type=str, default="probed_model.pt", help="Model file")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    args = parser.parse_args()
    
    model, device = load_model(args.run, args.model_file)
    results = run_tests(model, device, verbose=not args.quiet)
    print_summary(results)


if __name__ == "__main__":
    main()


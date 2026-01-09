"""
Text Generation from a Trained NanoJEPA Model

This script demonstrates the JEPA generation process:
    1. Encode the prompt into a thought vector (Student Encoder)
    2. Predict the "future thought" vector (Predictor)
    3. Decode that thought into text (Decoder)

The key insight: generation is driven by abstract thought vectors,
not just pattern matching on the input words.

Usage:
    uv run python -m nanojepa.generate \
        --run my-jepa-brain \
        --model-file probed_model.pt \
        --text "Once upon a time"
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

from nanojepa.model import NanoJEPA, JEPAConfig


device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")


def load_model(run_name: str, model_file: str = "best_model.pt") -> NanoJEPA:
    """
    Load a trained NanoJEPA model.
    
    Args:
        run_name: Name of the run folder in runs/
        model_file: Which checkpoint to load (best_model.pt or probed_model.pt)
        
    Returns:
        Loaded NanoJEPA model in eval mode
    """
    path = Path("runs") / run_name / model_file
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")
    
    print(f"Loading {path}...")
    
    # weights_only=False needed to load custom JEPAConfig class
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    
    # Handle both JEPAConfig objects and dicts
    if isinstance(saved_config, JEPAConfig):
        config = saved_config
        config.encoder_only = False  # Need decoder for generation
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


def generate(model: NanoJEPA, prompt: str, max_new_tokens: int = 50):
    """
    Generate text continuation from a prompt.
    
    Process:
        1. Encode prompt → context vector (what we understand)
        2. Predict future → thought vector (what we expect to happen)
        3. Decode thought → text (translate abstract to words)
    
    Sampling strategies to avoid common issues:
        - Temperature (0.7): Balance between coherence and creativity
        - Repetition Penalty: Subtract from logits of recently seen tokens
        - Top-K Filtering (50): Only sample from top 50 tokens
    """
    tokens = tokenizer.encode(prompt)
    ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    
    with torch.no_grad():
        # ------------------------------------------------------------------
        # JEPA THINKING: Predict abstract future thought
        # ------------------------------------------------------------------
        
        # Encode current context
        z_current = model.encode_student(ctx_idx)
        
        # Predict future thought (the "meaning" of what comes next)
        z_future = model.predictor(z_current)
        
        print(f"Thinking... (Predicted Vector Norm: {z_future.norm().item():.2f})")
        print("-" * 40)
        print("Prediction: ", end="", flush=True)

        # ------------------------------------------------------------------
        # DECODER: Translate thought vector into words
        # ------------------------------------------------------------------
        
        # Memory for cross-attention
        memory = z_future.unsqueeze(1)  # [1, 1, embed_dim]
        
        # Start with BOS token (GPT-2 uses EOT=50256 as BOS)
        curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)

        for _ in range(max_new_tokens):
            # Prepare decoder input
            T_tgt = curr_seq.shape[1]
            pos_tgt = torch.arange(0, T_tgt, dtype=torch.long, device=device)
            tgt_emb = model.embedding(curr_seq) + model.pos_embedding(pos_tgt)
            
            # Causal mask
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T_tgt).to(device)
            
            # Decode
            dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = model.lm_head(dec_out)
            next_token_logits = logits[:, -1, :].clone()

            # ----------------------------------------------------------
            # SAMPLING STRATEGIES
            # ----------------------------------------------------------
            
            # 1. Repetition Penalty
            # Subtract from logits of tokens we've already generated
            for token_id in set(curr_seq[0].tolist()):
                next_token_logits[0, token_id] -= 1.5

            # 2. Temperature Scaling
            temperature = 0.6 # Seemed to work best
            scaled_logits = next_token_logits / temperature

            # 3. Top-K Filtering
            # Only consider the top 50 most likely tokens (I haven't tried anything else, top k helped reduce repetition)
            top_k = 50
            v, _ = torch.topk(scaled_logits, top_k)
            scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')

            # 4. Sample from distribution (lots of repetition otherwise)
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop on EOT
            if next_token.item() == 50256:
                print(" <EOT>", end="")
                break
                
            # Print and continue
            word = tokenizer.decode([next_token.item()])
            print(word, end="", flush=True)
            
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
            
        print("\n" + "-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Generate text with NanoJEPA")
    parser.add_argument("--run", type=str, required=True,
                        help="Run name (folder in runs/)")
    parser.add_argument("--model-file", type=str, default="best_model.pt",
                        help="Model file (use probed_model.pt for generation)")
    parser.add_argument("--text", type=str, default="Once upon a time",
                        help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    try:
        model = load_model(args.run, args.model_file)
        generate(model, args.text, args.max_tokens)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNote: For text generation, you need a probed model.")
        print("Run train_probe.py first, then use --model-file probed_model.pt")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

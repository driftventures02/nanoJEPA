"""
Generate text using trained JEPA brain + decoder probe.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NanoJEPA, JEPAConfig
from .train_probe import Decoder
from .data import get_tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_models(brain_run: str, decoder_run: str = None):
    """Load JEPA brain and decoder."""
    device = get_device()
    
    # Load brain
    brain_path = Path("runs") / brain_run / "best_model.pt"
    print(f"Loading brain from {brain_path}...")
    brain_checkpoint = torch.load(brain_path, map_location=device, weights_only=False)
    brain_config = brain_checkpoint["config"]
    
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
    brain.load_state_dict(brain_checkpoint["model_state_dict"])
    brain.eval()
    
    # Load decoder
    decoder_run = decoder_run or f"probe-{brain_run}"
    decoder_path = Path("runs") / decoder_run / "decoder.pt"
    print(f"Loading decoder from {decoder_path}...")
    decoder_checkpoint = torch.load(decoder_path, map_location=device, weights_only=False)
    
    decoder = Decoder(
        vocab_size=brain_config["vocab_size"],
        embed_dim=brain_config["embed_dim"],
        num_layers=4,
        num_heads=brain_config["enc_heads"],
        dropout=0.0,  # No dropout for inference
        max_len=brain_config["block_size"]
    ).to(device)
    decoder.load_state_dict(decoder_checkpoint["decoder_state_dict"])
    decoder.eval()
    
    return brain, decoder, brain_config, device


def generate(
    brain: NanoJEPA,
    decoder: Decoder,
    prompt: str,
    device: torch.device,
    block_size: int = 128,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generate text from a prompt.
    
    The key insight: JEPA is trained to predict MASKED regions.
    So we must give it: [prompt tokens] + [MASK MASK MASK ...]
    Then the brain PREDICTS what goes in those mask slots.
    The decoder translates those predictions to text.
    """
    tokenizer = get_tokenizer()
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)
    
    # Build input: [prompt] + [MASK tokens for the rest]
    # The MASK positions are where the brain will PREDICT
    num_masks = block_size - prompt_len
    if num_masks <= 0:
        # Prompt is too long, truncate
        prompt_tokens = prompt_tokens[:block_size // 2]
        prompt_len = len(prompt_tokens)
        num_masks = block_size - prompt_len
    
    # Create input tensor (we'll replace mask positions with mask_token embedding later)
    # For now, use dummy tokens - the model.forward will handle masking
    x = torch.tensor([prompt_tokens + [0] * num_masks], dtype=torch.long, device=device)
    
    # Create mask: True for positions we want to predict (after prompt)
    mask = torch.zeros(1, block_size, dtype=torch.bool, device=device)
    mask[0, prompt_len:] = True  # Mask everything after the prompt
    
    with torch.no_grad():
        # Encode with masking
        # We need to manually apply the mask token
        B, T = x.shape
        pos = torch.arange(T, dtype=torch.long, device=device)
        
        # Get embeddings
        emb = brain.embedding(x) + brain.pos_embedding(pos)
        
        # Replace masked positions with the learnable mask token
        mask_expanded = mask.unsqueeze(-1)  # [1, T, 1]
        mask_tokens = brain.mask_token.expand(B, T, -1)
        emb = torch.where(mask_expanded, mask_tokens, emb)
        
        # Encode through student encoder
        z = brain.student_encoder(emb)
        z = brain.student_norm(z)
        
        # Predict (this is where the magic happens!)
        z_pred = brain.predictor(z)
        z_pred = brain.predictor_norm(z_pred)
        
        # The z_pred now contains:
        # - Encoded vectors for prompt positions
        # - PREDICTED vectors for mask positions (the future!)
        
        print(f"Prompt: {prompt_len} tokens, Predicting: {num_masks} future vectors")
        
        # Start generation using the predicted vectors as memory
        generated = [50256]  # BOS token
        
        for _ in range(max_new_tokens):
            # Decoder input
            dec_input = torch.tensor([generated], dtype=torch.long, device=device)
            logits = decoder(dec_input, memory=z_pred)
            
            # Get logits for last position
            next_logits = logits[0, -1, :].clone()
            
            # Repetition penalty
            for token_id in set(generated):
                next_logits[token_id] /= repetition_penalty
            
            # Temperature
            next_logits = next_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_value = values[-1]
                next_logits[next_logits < min_value] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 50256:  # EOT
                break
            
            generated.append(next_token)
    
    # Decode (skip BOS)
    output = tokenizer.decode(generated[1:])
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate text with JEPA")
    parser.add_argument("--brain-run", type=str, required=True, help="JEPA brain run name")
    parser.add_argument("--decoder-run", type=str, default=None, help="Decoder run name (default: probe-{brain-run})")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty")
    args = parser.parse_args()
    
    brain, decoder, config, device = load_models(args.brain_run, args.decoder_run)
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    output = generate(
        brain=brain,
        decoder=decoder,
        prompt=args.prompt,
        device=device,
        block_size=config["block_size"],
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    print(f"\nGenerated:\n{output}")


if __name__ == "__main__":
    main()

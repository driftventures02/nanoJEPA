import torch
import torch.nn.functional as F
import tiktoken
import argparse
from pathlib import Path
from nanojepa.model import NanoJEPA, JEPAConfig

# Setup device
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

def load_model(run_name, model_file="best_model.pt"):
    """Loads the best model from a specific training run."""
    # 1. Find the checkpoint
    path = Path("runs") / run_name / model_file
    if not path.exists():
        raise FileNotFoundError(f"Could not find model at {path}")
    
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    
    # 2. Rebuild the exact architecture from the config
    if isinstance(saved_config, JEPAConfig):
        config = saved_config
        config.encoder_only = False # Force decoder creation
    else:
        # It's a dict
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

def generate(model, prompt, max_new_tokens=50):
    """
    1. Encodes the prompt into a Context Vector.
    2. Predicts the Future Thought Vector.
    3. Decodes that Future Vector into text.
    """
    # A. Encode the Prompt (Context)
    # Note: We need to pad to block_size because our model expects it due to Positional Embeddings
    # But for a simple test, we'll just feed the raw tokens and let the model handle it 
    # (our masking logic in encode_student handles variable lengths well enough if not batched awkwardly)
    tokens = tokenizer.encode(prompt)
    ctx_idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    
    with torch.no_grad():
        # B. THE JEPA STEP: Predict the abstract future
        # 1. Get current thought
        z_current = model.encode_student(ctx_idx)
        # 2. Predict future thought
        z_future = model.predictor(z_current)
        
        print(f"Thinking... (Predicted Vector Norm: {z_future.norm().item():.2f})")
        print("-" * 40)
        print("Prediction: ", end="", flush=True)

        # C. THE DECODER STEP: Translate vector to words
        # We start with an empty sequence. The decoder will rely purely on z_future initially.
        # However, TransformerDecoder needs a 'tgt' input (what has been generated so far).
        # We usually start with a Start Token. GPT-2 doesn't have a specific BOS, so we can use EOT (50256)
        # or just start with the first generated token being implied.
        
        # We'll use EOT (50256) as the start token.
        curr_seq = torch.tensor([[50256]], dtype=torch.long).to(device)
        
        # We need to expand the thought vector to match the decoder's expected input
        # The decoder treats this as the "Memory" (like the output of an encoder in a std transformer)
        memory = z_future.unsqueeze(1) # Shape: [1, 1, embed_dim]

        for _ in range(max_new_tokens):
            # 1. Prepare input for decoder
            dec_input = curr_seq

            # 2. Masking (Standard Causal Mask for auto-regressive generation)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(dec_input.shape[1]).to(device)
            
            # 3. Run Decoder
            # It tries to predict the next token based on 'dec_input' (what we said so far)
            # AND 'memory' (the JEPA's predicted thought vector)
            # Note: We need to create embeddings for dec_input inside the model usually, 
            # but our model.decoder expects embeddings + pos_encodings.
            
            # Wait, model.decoder expects TENSOR inputs (embeddings), not token indices?
            # Let's check model.py... 
            # forward() does: tgt_emb = self.embedding(target_idx) + self.pos_embedding(...)
            # So we need to do that manually here since we are calling model.decoder directly.
            
            T_tgt = dec_input.shape[1]
            pos_tgt = torch.arange(0, T_tgt, dtype=torch.long, device=device)
            tgt_emb = model.embedding(dec_input) + model.pos_embedding(pos_tgt)
            
            dec_out = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            
            # 4. Project to Vocabulary
            logits = model.lm_head(dec_out)
            next_token_logits = logits[:, -1, :] 

            # 2. Strict Repetition Penalty (Kill the "and and and" and "needs needs")
            for token_id in set(curr_seq[0].tolist()):
                count = (curr_seq[0] == token_id).sum().item()
                if count > 0:
                    # Using subtraction for robust penalty (works for negative logits too)
                    # User suggested division, but subtraction is safer:
                    next_token_logits[0, token_id] -= 1.5

            # 3. Temperature (0.7 is the sweet spot for coherence)
            temperature = 0.7
            scaled_logits = next_token_logits / temperature

            # --- CRITICAL FIX: Top-K Filtering ---
            # This deletes the weird words like "coursepen" or "thankedaters"
            # It forces the model to pick from the top 50 valid words.
            top_k = 50
            v, _ = torch.topk(scaled_logits, top_k)
            scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
            # -------------------------------------

            # 4. Sample
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 6. Stop if we hit EOT (and it's not the very first start token we forced)
            if next_token.item() == 50256:
                print(" <EOT>", end="")
                break
                
            # 7. Print and Append
            word = tokenizer.decode([next_token.item()])
            print(word, end="", flush=True)
            
            curr_seq = torch.cat([curr_seq, next_token], dim=1)
            
        print("\n" + "-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Name of the folder in 'runs/' (e.g. tiny-test-layernorm)")
    parser.add_argument("--model-file", type=str, default="best_model.pt", help="Filename of the model checkpoint (e.g. probed_model.pt)")
    parser.add_argument("--text", type=str, default="Once upon a time", help="The start of the story")
    args = parser.parse_args()

    try:
        model = load_model(args.run, args.model_file)
        generate(model, args.text)
    except Exception as e:
        print(f"\nError: {e}")


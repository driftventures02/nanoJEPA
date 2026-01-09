"""
Debug script to visualize the full data pipeline:
1. Story-aware chunking (no crossing EOT boundaries)
2. Random block masking
"""
from nanojepa.data import (
    get_dataloaders, 
    decode, 
    create_random_mask,
    EOT_TOKEN
)

def main():
    print("="*70)
    print("NANOJEPA DATA PIPELINE DEBUG")
    print("="*70)
    
    # Load data
    train_loader, _ = get_dataloaders(
        block_size=128,
        batch_size=4,
        max_samples=2000
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")  # [4, 128]
    
    # Verify no EOT tokens (story boundaries respected)
    eot_count = (batch == EOT_TOKEN).sum().item()
    print(f"EOT tokens in batch: {eot_count} (should be 0 for clean stories)")
    
    # Show a few samples
    for i in range(min(2, len(batch))):
        sample = batch[i]
        
        print("\n" + "="*70)
        print(f"SAMPLE {i+1}")
        print("="*70)
        
        # Original text
        text = decode(sample)
        print(f"\nORIGINAL TEXT ({len(sample)} tokens):")
        print(text[:300] + "..." if len(text) > 300 else text)
        
        # Create random mask
        mask, start, end = create_random_mask(len(sample), mask_ratio=0.3)
        mask_len = end - start
        
        print(f"\n--- MASK REGION: tokens {start} to {end} ({mask_len} tokens, {mask_len/len(sample)*100:.1f}%) ---")
        
        # Show what's visible (before and after the mask)
        before_text = decode(sample[:start])
        after_text = decode(sample[end:])
        hidden_text = decode(sample[start:end])
        
        print(f"\nWHAT MODEL SEES:")
        print(f"  BEFORE: \"{before_text[-100:]}\"" if len(before_text) > 100 else f"  BEFORE: \"{before_text}\"")
        print(f"  [MASK] <-- Learnable embedding vector (NOT <|endoftext|>!)")
        print(f"  AFTER:  \"{after_text[:100]}\"" if len(after_text) > 100 else f"  AFTER:  \"{after_text}\"")
        
        print(f"\nWHAT MODEL MUST PREDICT (hidden):")
        print(f"  \"{hidden_text}\"")

    print("\n" + "="*70)
    print("NOTE: The [MASK] is a LEARNABLE EMBEDDING VECTOR")
    print("      NOT the <|endoftext|> token!")
    print("      The model learns what [MASK] means during training.")
    print("="*70)

if __name__ == "__main__":
    main()

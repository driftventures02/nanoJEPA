import os
import re
import urllib.request
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Use the validation set (~19MB) as our "nano" training set
DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
DATA_DIR = Path(__file__).parent.parent / "data"

def download_tinystories() -> str:
    """Download TinyStories-valid.txt if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / "TinyStories-valid.txt"

    if not filepath.exists():
        print(f"Downloading TinyStories to {filepath}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(DATA_URL, filepath)
        print("Done.")

    return str(filepath)

class SentenceJEPADataset(Dataset):
    """
    Dataset that respects sentence boundaries.
    
    Structure:
    - Context: N sentences
    - Target:  Next N sentences
    """
    def __init__(self, text_data: str, tokenizer, block_size: int = 128, sentences_per_block: int = 3):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 1. Clean and Split by Sentence
        # Split by .!? followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text_data)
        self.sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        
        print(f"Found {len(self.sentences)} sentences.")
        
        # 2. Create Samples (Sliding window of Sentences)
        self.samples = []
        N = sentences_per_block
        
        # Stride of 1 sentence
        for i in range(0, len(self.sentences) - 2*N, 1):
             ctx_text = " ".join(self.sentences[i : i+N])
             tgt_text = " ".join(self.sentences[i+N : i+2*N])
             self.samples.append((ctx_text, tgt_text))
             
        print(f"Created {len(self.samples)} context-target pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx_text, tgt_text = self.samples[idx]
        
        # Tokenize
        ctx_tokens = self.tokenizer.encode(ctx_text, allowed_special={'<|endoftext|>'})
        tgt_tokens = self.tokenizer.encode(tgt_text, allowed_special={'<|endoftext|>'})
        
        # === THE FIX: Add Start Token to Target ===
        # We prepend 50256 to the target so the model learns:
        # 50256 (Start) -> First Word
        tgt_tokens = [50256] + tgt_tokens 
        
        # Pad / Truncate
        ctx_tensor = self._pad_truncate(ctx_tokens)
        tgt_tensor = self._pad_truncate(tgt_tokens)
        
        return ctx_tensor, tgt_tensor

    def _pad_truncate(self, tokens):
        if len(tokens) > self.block_size:
            return torch.tensor(tokens[:self.block_size], dtype=torch.long)
        else:
            # Using 50256 (eot) as padding token
            padding = [50256] * (self.block_size - len(tokens))
            return torch.tensor(tokens + padding, dtype=torch.long)

def get_dataloaders(
    block_size: int = 128,
    batch_size: int = 32,
    train_split: float = 0.9,
    stride: int = 64, # Ignored now
    subset_ratio: float = 0.1,
    sentences_per_block: int = 1,
    max_samples: int = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders for TinyStories.
    """
    filepath = download_tinystories()
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Take subset of TEXT first
    split_char = int(len(full_text) * subset_ratio)
    text_subset = full_text[:split_char]
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create Full Dataset
    dataset = SentenceJEPADataset(text_subset, tokenizer, block_size, sentences_per_block)
    
    if max_samples:
        print(f"Limiting dataset to {max_samples} samples.")
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

# --- Debug Tool ---
if __name__ == "__main__":
    # Quick test to verify data loading
    print("Running Data Debug...")
    train_loader, _ = get_dataloaders(subset_ratio=0.01, batch_size=2)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n--- Sample Batch ---")
    for ctx, tgt in train_loader:
        print(f"Context Shape: {ctx.shape}")
        print(f"Target Shape: {tgt.shape}")
        
        print("\n[Sample 1]")
        # Decode and manually strip padding token (50256) which often renders as <|endoftext|>
        ctx_tokens = [t for t in ctx[0].tolist() if t != 50256]
        tgt_tokens = [t for t in tgt[0].tolist() if t != 50256]
        
        print("Context:", tokenizer.decode(ctx_tokens))
        print("-" * 20)
        print("Target:", tokenizer.decode(tgt_tokens))
        break

"""
Data Loading for NanoJEPA

Loads TinyStories dataset and creates context-target pairs for training.

Key Design Decisions:
    - Sentence-based chunking: Respects natural language boundaries
    - Context → Target pairs: Model learns to predict next sentences
    - BOS token prepended: Teaches model how to start generation
    - Padding with EOT (50256): Standard GPT-2 end-of-text token

TinyStories is used because it's:
    - Small enough to train quickly (~19MB validation set)
    - Simple language (children's stories)
    - Good for testing representation learning
"""

import os
import re
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# TinyStories validation set (~19MB) - small enough for quick experiments
DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_tinystories() -> str:
    """
    Download TinyStories-valid.txt if not already present.
    
    Returns:
        Path to the downloaded file
    """
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
    Dataset that creates context-target pairs from sentence boundaries.
    
    Structure:
        - Context: N consecutive sentences
        - Target: Next N sentences (the "future" to predict)
    
    This ensures the model learns to predict coherent semantic units,
    not arbitrary token spans that might cut mid-sentence.
    """
    
    def __init__(self, text_data: str, tokenizer, block_size: int = 128, sentences_per_block: int = 3):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Split text by sentence boundaries (. ! ? followed by whitespace)
        raw_sentences = re.split(r'(?<=[.!?])\s+', text_data)
        
        # Filter out very short sentences (likely artifacts)
        self.sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        print(f"Found {len(self.sentences)} sentences.")
        
        # Create context-target pairs with sliding window
        self.samples = []
        N = sentences_per_block
        
        for i in range(0, len(self.sentences) - 2*N):
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
        
        # Prepend BOS token (50256) to target
        # This teaches the model: [BOS] → First Word
        # Without this, the model doesn't know how to start generation
        tgt_tokens = [50256] + tgt_tokens
        
        # Pad or truncate to block_size
        ctx_tensor = self._pad_truncate(ctx_tokens)
        tgt_tensor = self._pad_truncate(tgt_tokens)
        
        return ctx_tensor, tgt_tensor

    def _pad_truncate(self, tokens: list) -> torch.Tensor:
        """Pad (with EOT=50256) or truncate tokens to block_size."""
        if len(tokens) > self.block_size:
            return torch.tensor(tokens[:self.block_size], dtype=torch.long)
        else:
            padding = [50256] * (self.block_size - len(tokens))
            return torch.tensor(tokens + padding, dtype=torch.long)


def get_dataloaders(
    block_size: int = 128,
    batch_size: int = 32,
    train_split: float = 0.9,
    stride: int = 64,  # Currently unused, kept for API compatibility
    subset_ratio: float = 0.1,
    sentences_per_block: int = 1,
    max_samples: int = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for TinyStories.
    
    Args:
        block_size: Maximum sequence length
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest is validation)
        stride: Unused (kept for compatibility)
        subset_ratio: Fraction of text to use (0.1 = 10%)
        sentences_per_block: Number of sentences per context/target
        max_samples: Cap on total samples (for quick testing)
        num_workers: DataLoader workers
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Download data if needed
    filepath = download_tinystories()
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    # Take subset of text (for faster experiments)
    split_char = int(len(full_text) * subset_ratio)
    text_subset = full_text[:split_char]
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = SentenceJEPADataset(text_subset, tokenizer, block_size, sentences_per_block)
    
    # Optionally limit total samples
    if max_samples and max_samples < len(dataset):
        print(f"Limiting dataset to {max_samples} samples.")
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Train/val split
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


if __name__ == "__main__":
    """Quick test to verify data loading works."""
    print("Running Data Debug...")
    train_loader, _ = get_dataloaders(subset_ratio=0.01, batch_size=2)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n--- Sample Batch ---")
    for ctx, tgt in train_loader:
        print(f"Context Shape: {ctx.shape}")
        print(f"Target Shape: {tgt.shape}")
        
        print("\n[Sample 1]")
        # Decode, stripping padding tokens
        ctx_tokens = [t for t in ctx[0].tolist() if t != 50256]
        tgt_tokens = [t for t in tgt[0].tolist() if t != 50256]
        
        print("Context:", tokenizer.decode(ctx_tokens))
        print("-" * 20)
        print("Target:", tokenizer.decode(tgt_tokens))
        break

"""
Data utilities for NanoJEPA.

Provides modular functions for:
- Downloading data
- Tokenizing text
- Splitting into stories
- Creating training chunks
- Block masking
"""

import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
DATA_DIR = Path(__file__).parent.parent / "data"
EOT_TOKEN = 50256  # GPT-2 <|endoftext|>

# Global tokenizer (lazy loaded)
_TOKENIZER = None

def get_tokenizer():
    """Get the GPT-2 tokenizer (cached)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("gpt2")
    return _TOKENIZER


def download_tinystories() -> Path:
    """Download TinyStories if not present. Returns filepath."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / "TinyStories-valid.txt"

    if not filepath.exists():
        print(f"Downloading TinyStories to {filepath}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(DATA_URL, filepath)
        print("Done.")

    return filepath


def load_text(filepath: Path) -> str:
    """Load raw text from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> List[int]:
    """Tokenize text using GPT-2 tokenizer."""
    tokenizer = get_tokenizer()
    return tokenizer.encode(text, allowed_special={'<|endoftext|>'})


def decode(tokens: List[int]) -> str:
    """Decode tokens back to text."""
    tokenizer = get_tokenizer()
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens)


def split_into_stories(tokens: List[int], min_length: int = 64) -> List[List[int]]:
    """
    Split token stream into individual stories using EOT token.
    
    Returns list of stories, each story is a list of token IDs.
    Only returns stories with at least min_length tokens.
    """
    stories = []
    current_story = []
    
    for token in tokens:
        if token == EOT_TOKEN:
            if len(current_story) >= min_length:
                stories.append(current_story)
            current_story = []
        else:
            current_story.append(token)
    
    # Handle last story if it doesn't end with EOT
    if len(current_story) >= min_length:
        stories.append(current_story)
    
    return stories


def create_chunks(stories: List[List[int]], block_size: int = 128, stride: int = None) -> List[np.ndarray]:
    """
    Create fixed-size chunks from stories.
    
    Each chunk is guaranteed to be from a single story.
    Uses sliding window with given stride (default: block_size // 2).
    """
    if stride is None:
        stride = block_size // 2
    
    chunks = []
    for story in stories:
        if len(story) >= block_size:
            for start in range(0, len(story) - block_size + 1, stride):
                chunk = np.array(story[start : start + block_size], dtype=np.int64)
                chunks.append(chunk)
    
    return chunks


def create_random_mask(seq_len: int, mask_ratio: float = 0.3, min_mask_len: int = 8) -> Tuple[torch.Tensor, int, int]:
    """
    Create a random contiguous block mask.
    
    Returns:
        mask: Boolean tensor [seq_len], True = masked
        mask_start: Start index of mask
        mask_end: End index of mask (exclusive)
    """
    mask_len = max(min_mask_len, int(seq_len * mask_ratio))
    
    # Random start position
    max_start = seq_len - mask_len
    mask_start = torch.randint(0, max_start + 1, (1,)).item()
    mask_end = mask_start + mask_len
    
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[mask_start:mask_end] = True
    
    return mask, mask_start, mask_end


# --- Dataset Class ---

class StoryDataset(Dataset):
    """Simple dataset wrapping pre-computed chunks."""
    
    def __init__(self, chunks: List[np.ndarray]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.from_numpy(self.chunks[idx])


# --- Main Entry Point ---

def get_dataloaders(
    block_size: int = 128,
    batch_size: int = 64,
    train_split: float = 0.9,
    max_samples: int = None,
    num_workers: int = 0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.
    
    Uses all the utility functions above.
    """
    # 1. Download
    filepath = download_tinystories()
    
    # 2. Load & Tokenize
    print("Loading and tokenizing...")
    text = load_text(filepath)
    tokens = tokenize(text)
    print(f"Total tokens: {len(tokens):,}")
    
    # 3. Split into stories
    print("Splitting into stories...")
    stories = split_into_stories(tokens, min_length=block_size)
    print(f"Found {len(stories)} stories (>= {block_size} tokens)")
    
    # 4. Limit if requested
    if max_samples:
        max_stories = max(1, max_samples // 2)  # ~2 chunks per story
        stories = stories[:max_stories]
        print(f"Limited to {len(stories)} stories")
    
    # 5. Split stories into train/val
    split_idx = int(len(stories) * train_split)
    train_stories = stories[:split_idx]
    val_stories = stories[split_idx:]
    
    # 6. Create chunks
    train_chunks = create_chunks(train_stories, block_size)
    val_chunks = create_chunks(val_stories, block_size)
    print(f"Train chunks: {len(train_chunks):,}, Val chunks: {len(val_chunks):,}")
    
    # 7. Create datasets and loaders
    train_dataset = StoryDataset(train_chunks)
    val_dataset = StoryDataset(val_chunks)
    
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


# --- Debug ---
if __name__ == "__main__":
    print("="*60)
    print("DATA UTILITIES DEBUG")
    print("="*60)
    
    # Test each function
    filepath = download_tinystories()
    print(f"\n1. Downloaded: {filepath}")
    
    text = load_text(filepath)
    print(f"2. Loaded text: {len(text):,} chars")
    
    tokens = tokenize(text[:10000])  # Just first 10k chars for speed
    print(f"3. Tokenized: {len(tokens)} tokens")
    print(f"   Sample: {decode(tokens[:20])}...")
    
    stories = split_into_stories(tokens, min_length=32)
    print(f"4. Split into {len(stories)} stories")
    if stories:
        print(f"   First story: {decode(stories[0][:30])}...")
    
    chunks = create_chunks(stories, block_size=64)
    print(f"5. Created {len(chunks)} chunks")
    
    # Test masking
    mask, start, end = create_random_mask(64, mask_ratio=0.3)
    print(f"\n6. Random mask: positions {start} to {end} ({mask.sum().item()} tokens)")
    print(f"   Mask: {mask.int().tolist()[:20]}...")
    
    # Visual demo of masking
    if chunks:
        sample = chunks[0]
        text_before = decode(sample)
        
        masked_sample = sample.copy()
        masked_sample[start:end] = EOT_TOKEN  # Replace with EOT for visualization
        text_after = decode(masked_sample)
        
        print(f"\n7. Masking demo:")
        print(f"   Original: {text_before[:100]}...")
        print(f"   Masked:   {text_after[:100]}...")

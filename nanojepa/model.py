"""
NanoJEPA Model - Block Masking Architecture

The model:
1. Takes a sequence of tokens
2. Randomly masks a contiguous block
3. Predicts the masked vectors from the unmasked context

This is similar to I-JEPA but for text.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JEPAConfig:
    vocab_size: int = 50257
    embed_dim: int = 512
    enc_layers: int = 4
    enc_heads: int = 8
    pred_depth: int = 4
    block_size: int = 128
    dropout: float = 0.1
    mask_ratio: float = 0.1  # Fraction of sequence to mask


class NanoJEPA(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------------
        # 1. STUDENT COMPONENTS (Gradient Updated)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)
        
        # Learnable MASK token (replaces masked positions for student)
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, 
            nhead=config.enc_heads,
            dim_feedforward=config.embed_dim * 4, 
            dropout=config.dropout,
            batch_first=True, 
            norm_first=True
        )
        self.student_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.enc_layers)
        self.student_norm = nn.LayerNorm(config.embed_dim)

        # Predictor: Refines the context to predict masked regions
        pred_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.enc_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.predictor = nn.TransformerEncoder(pred_layer, num_layers=config.pred_depth)
        self.predictor_norm = nn.LayerNorm(config.embed_dim)

        # ----------------------------------------------------------------------
        # 2. TEACHER COMPONENTS (EMA Updated)
        # ----------------------------------------------------------------------
        self.teacher_embedding = copy.deepcopy(self.embedding)
        self.teacher_pos_embedding = copy.deepcopy(self.pos_embedding)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_norm = copy.deepcopy(self.student_norm)

        # Freeze Teacher
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_pos_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_norm.parameters(): p.requires_grad = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    @torch.no_grad()
    def update_teacher(self, decay=0.996):
        """EMA update: Teacher = decay * Teacher + (1 - decay) * Student"""
        for t, s in zip(self.teacher_embedding.parameters(), self.embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        for t, s in zip(self.teacher_pos_embedding.parameters(), self.pos_embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        for t, s in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        for t, s in zip(self.teacher_norm.parameters(), self.student_norm.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))

    def create_block_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """
        Create random contiguous block masks for a batch.
        
        Returns:
            mask: [Batch, Seq] boolean tensor, True = masked
        """
        # Minimum 2 tokens, but respect the ratio
        mask_len = max(2, int(seq_len * self.config.mask_ratio))
        max_start = seq_len - mask_len
        
        # Random start for each sample in batch
        starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
        
        # Create mask
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, Seq]
        starts_expanded = starts.unsqueeze(1)  # [Batch, 1]
        ends_expanded = starts_expanded + mask_len
        
        mask = (positions >= starts_expanded) & (positions < ends_expanded)  # [Batch, Seq]
        
        return mask

    def forward(self, x, mask=None):
        """
        Forward pass with block masking.
        
        Args:
            x: [Batch, Seq] token indices
            mask: Optional [Batch, Seq] boolean mask (True = masked)
                  If None, creates random block mask
        
        Returns:
            loss: Scalar tensor
            info: Dict with debug metrics
        """
        device = x.device
        B, T = x.shape
        
        # Create mask if not provided
        if mask is None:
            mask = self.create_block_mask(B, T, device)
        
        # --- 1. TEACHER PATH (sees everything, provides targets) ---
        with torch.no_grad():
            pos = torch.arange(T, dtype=torch.long, device=device)
            teacher_emb = self.teacher_embedding(x) + self.teacher_pos_embedding(pos)
            z_teacher = self.teacher_encoder(teacher_emb)
            z_teacher = self.teacher_norm(z_teacher)  # [Batch, Seq, Dim]

        # --- 2. STUDENT PATH (sees masked input) ---
        pos = torch.arange(T, dtype=torch.long, device=device)
        student_emb = self.embedding(x) + self.pos_embedding(pos)
        
        # Replace masked positions with learnable MASK token
        mask_expanded = mask.unsqueeze(-1)  # [Batch, Seq, 1]
        mask_tokens = self.mask_token.expand(B, T, -1)  # [Batch, Seq, Dim]
        student_emb = torch.where(mask_expanded, mask_tokens, student_emb)
        
        z_student = self.student_encoder(student_emb)
        z_student = self.student_norm(z_student)  # [Batch, Seq, Dim]

        # --- 3. PREDICTOR (refines student output) ---
        z_pred = self.predictor(z_student)
        z_pred = self.predictor_norm(z_pred)  # [Batch, Seq, Dim]

        # --- 4. LOSS (only on masked positions) ---
        # Extract only the masked vectors
        z_pred_masked = z_pred[mask]      # [N_masked, Dim]
        z_teacher_masked = z_teacher[mask]  # [N_masked, Dim]
        
        # Positive loss (attraction)
        pos_loss = F.mse_loss(z_pred_masked, z_teacher_masked)
        
        # Repulsion loss (push away from wrong answers)
        z_neg = torch.roll(z_teacher_masked, shifts=1, dims=0)
        neg_dist = F.mse_loss(z_pred_masked, z_neg)
        margin = 1.0
        repulsion_loss = F.relu(margin - neg_dist)
        
        total_loss = pos_loss + repulsion_loss

        # Debug info
        info = {
            "loss": total_loss,
            "pos_loss": pos_loss,
            "repulsion_loss": repulsion_loss,
            "z_pred_norm": z_pred_masked.norm(dim=-1).mean(),
            "z_teacher_norm": z_teacher_masked.norm(dim=-1).mean(),
            "cosine_sim": F.cosine_similarity(z_pred_masked, z_teacher_masked, dim=-1).mean(),
            "mask_count": mask.sum().item(),
        }

        return total_loss, info

    # --- Utility methods for inference ---
    
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens to vectors (using student encoder)."""
        device = tokens.device
        T = tokens.shape[1]
        pos = torch.arange(T, dtype=torch.long, device=device)
        emb = self.embedding(tokens) + self.pos_embedding(pos)
        z = self.student_encoder(emb)
        return self.student_norm(z)
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Apply predictor to encoded vectors."""
        z_pred = self.predictor(z)
        return self.predictor_norm(z_pred)

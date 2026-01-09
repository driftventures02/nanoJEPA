"""
NanoJEPA Model Architecture

A minimal Joint-Embedding Predictive Architecture (JEPA) for text.
Based on Yann LeCun's JEPA concept, adapted from I-JEPA/V-JEPA for language.

Key Components:
    - Student Encoder: Learns via gradient descent
    - Teacher Encoder: Updated via EMA (Exponential Moving Average) of Student
    - Predictor: MLP that predicts future representations from context
    - Decoder: (Optional) Translates thought vectors back to text

The EMA mechanism prevents representation collapse without needing a decoder
or contrastive negatives - the Teacher provides stable targets that the
Student learns to predict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dataclasses import dataclass


@dataclass
class JEPAConfig:
    """Configuration for NanoJEPA model."""
    vocab_size: int = 50257      # GPT-2 vocabulary size
    embed_dim: int = 256         # Dimension of embeddings and hidden states
    
    enc_layers: int = 4          # Number of Transformer encoder layers
    enc_heads: int = 4           # Number of attention heads in encoder
    pred_depth: int = 2          # Number of layers in the predictor MLP
    dec_layers: int = 4          # Number of Transformer decoder layers
    dec_heads: int = 4           # Number of attention heads in decoder
    
    block_size: int = 128        # Maximum sequence length
    dropout: float = 0.1         # Dropout probability
    
    # If True, skip decoder (pure representation learning mode)
    encoder_only: bool = False


class NanoJEPA(nn.Module):
    """
    A minimal JEPA for text.
    
    Architecture:
        1. Student Path: Context → Student Encoder → Predictor → Predicted Vector
        2. Teacher Path: Target → Teacher Encoder → True Vector (stable, EMA)
        3. Loss: MSE(Predicted, True) + Repulsion from random negatives
    
    The Teacher is a "ghost" of the Student - it follows the Student's weights
    with a delay via EMA. This prevents collapse (all vectors becoming identical)
    because the Teacher provides stable targets even as the Student updates.
    """
    
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # STUDENT COMPONENTS (Updated via gradient descent)
        # ------------------------------------------------------------------
        
        # Token and position embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.enc_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable training)
        )
        self.student_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.enc_layers)
        
        # LayerNorm for output vectors - prevents "Volume War" (exploding norms), though a bit more variance might be ideal so concepts are more different
        self.student_norm = nn.LayerNorm(config.embed_dim)

        # Predictor MLP: transforms context vector → predicted target vector
        # This is intentionally simpler than the encoder (asymmetric design)
        # to encourage the encoder to learn rich representations
        self.predictor = self._build_predictor(config)

        # ------------------------------------------------------------------
        # TEACHER COMPONENTS (Updated via EMA - "The Ghost")
        # ------------------------------------------------------------------
        
        # Deep copy student components to initialize teacher
        self.teacher_embedding = copy.deepcopy(self.embedding)
        self.teacher_pos_embedding = copy.deepcopy(self.pos_embedding)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_norm = copy.deepcopy(self.student_norm)

        # CRITICAL: Freeze teacher gradients permanently
        # Teacher only updates via EMA, never via backprop
        for p in self.teacher_embedding.parameters():
            p.requires_grad = False
        for p in self.teacher_pos_embedding.parameters():
            p.requires_grad = False
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_norm.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # DECODER (Optional - for probing/text generation)
        # ------------------------------------------------------------------
        
        if not config.encoder_only:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=config.embed_dim,
                nhead=config.dec_heads,
                dim_feedforward=config.embed_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.dec_layers)
            self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_predictor(self, config: JEPAConfig) -> nn.Sequential:
        """
        Build the predictor MLP.
        
        The predictor is intentionally simpler than the encoder (asymmetric).
        This forces the encoder to learn rich representations rather than
        letting the predictor do all the work.
        """
        layers = []
        
        # Input expansion
        layers.extend([
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU()
        ])
        
        # Hidden layers (depth-2 means just input->output, depth-3 adds one hidden, etc.)
        for _ in range(max(0, config.pred_depth - 2)):
            layers.extend([
                nn.Linear(config.embed_dim * 2, config.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            
        # Output projection with normalization
        layers.extend([
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.embed_dim)  # Stabilizes output magnitude
        ])
        
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        """Initialize weights using small normal distribution."""
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
    def update_teacher(self, decay: float = 0.996):
        """
        Update teacher weights via Exponential Moving Average.
        
        Teacher = decay * Teacher + (1 - decay) * Student
        
        This is the "magic sauce" that prevents collapse:
        - High decay (0.99-0.999): Teacher moves slowly, very stable targets
        - Low decay (0.9): Teacher follows Student quickly, can cause collapse
        
        Args:
            decay: EMA coefficient. Higher = slower teacher updates.
        """
        # Update embeddings
        for t, s in zip(self.teacher_embedding.parameters(), self.embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        for t, s in zip(self.teacher_pos_embedding.parameters(), self.pos_embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        
        # Update encoder
        for t, s in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
            
        # Update norm
        for t, s in zip(self.teacher_norm.parameters(), self.student_norm.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))

    def encode_student(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens using the student encoder.
        
        Uses masked mean pooling to handle variable-length sequences
        with padding tokens (50256 = GPT-2 EOT used as pad).
        
        Args:
            token_idx: [Batch, Seq] token indices
            
        Returns:
            [Batch, embed_dim] pooled representation
        """
        device = token_idx.device
        T = token_idx.shape[1]
        
        # Create padding mask (True where padding)
        is_pad = (token_idx == 50256)
        
        # Mask for mean pooling (1.0 for real tokens, 0.0 for padding)
        mask_float = (~is_pad).float().unsqueeze(-1)
        
        # Embed tokens + positions
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.embedding(token_idx) + self.pos_embedding(pos)
        
        # Encode (mask prevents attention to padding)
        enc_out = self.student_encoder(x, src_key_padding_mask=is_pad)
        
        # Masked mean pooling: average only non-padding tokens
        z_sum = (enc_out * mask_float).sum(dim=1)
        z_count = mask_float.sum(dim=1).clamp(min=1.0)
        z = z_sum / z_count
        
        return self.student_norm(z)

    def encode_teacher(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens using the teacher encoder (frozen, EMA-updated).
        
        Same architecture as student, but weights are updated via EMA
        rather than gradient descent.
        """
        device = token_idx.device
        T = token_idx.shape[1]
        
        is_pad = (token_idx == 50256)
        mask_float = (~is_pad).float().unsqueeze(-1)
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.teacher_embedding(token_idx) + self.teacher_pos_embedding(pos)
        
        enc_out = self.teacher_encoder(x, src_key_padding_mask=is_pad)
        
        z_sum = (enc_out * mask_float).sum(dim=1)
        z_count = mask_float.sum(dim=1).clamp(min=1.0)
        z = z_sum / z_count
        
        return self.teacher_norm(z)

    def forward(self, context_idx: torch.Tensor, target_idx: torch.Tensor = None):
        """
        Forward pass for training or inference.
        
        Training mode (target_idx provided):
            1. Student encodes context → predicts target vector
            2. Teacher encodes target → provides stable target vector
            3. Loss = MSE(predicted, target) + repulsion from negatives
            4. (Optional) Decoder loss for text generation capability
        
        Inference mode (target_idx=None):
            Returns predicted vector for generation.
        
        Args:
            context_idx: [Batch, Seq] context token indices
            target_idx: [Batch, Seq] target token indices (optional)
            
        Returns:
            z_hat_target: [Batch, embed_dim] predicted target vector
            losses: dict of loss components (if training)
        """
        device = context_idx.device
        
        # STUDENT PATH: encode context and predict target
        z_context = self.encode_student(context_idx)
        z_hat_target = self.predictor(z_context)

        losses = {}
        
        if target_idx is not None:
            # TEACHER PATH: encode target (no gradients)
            with torch.no_grad():
                z_true_target = self.encode_teacher(target_idx)

            # ----------------------------------------------------------
            # JEPA LOSS: Attraction + Repulsion
            # ----------------------------------------------------------
            
            # Attraction: pull predicted vector toward true target
            pos_loss = F.mse_loss(z_hat_target, z_true_target)
            
            # Repulsion: push away from random other targets (prevents collapse)
            # We roll the batch to get "negative" samples
            z_random_target = torch.roll(z_true_target, shifts=1, dims=0)
            neg_dist = F.mse_loss(z_hat_target, z_random_target)
            
            # Hinge loss: only penalize if too close to negatives
            # With LayerNorm, vectors have elements ~N(0,1), so expected
            # MSE between random vectors is ~2.0. Margin of 1.0 is safe.
            margin = 1.0
            repulsion_loss = F.relu(margin - neg_dist)
            
            latent_loss = pos_loss + repulsion_loss
            
            losses = {
                "latent_loss": latent_loss,
                "pos_loss": pos_loss,
                "repulsion_loss": repulsion_loss,
                "z_hat_norm": z_hat_target.norm(dim=-1).mean(),
                "z_true_norm": z_true_target.norm(dim=-1).mean(),
                "cosine_sim": F.cosine_similarity(z_hat_target, z_true_target, dim=-1).mean(),
            }

            # ----------------------------------------------------------
            # DECODER LOSS (if enabled)
            # ----------------------------------------------------------
            
            if not self.config.encoder_only:
                # IMPORTANT: Detach so decoder gradients don't affect JEPA
                # The JEPA should learn from meaning, not from word prediction
                dec_input = z_hat_target.detach()
                memory = dec_input.unsqueeze(1)  # [Batch, 1, Dim]
                
                # Shift inputs for teacher forcing:
                # Input:  [t_0, t_1, ..., t_{N-1}]
                # Target: [t_1, t_2, ..., t_N]
                dec_input_idx = target_idx[:, :-1].clone()
                dec_target_idx = target_idx[:, 1:]
                
                # WORD DROPOUT: Randomly mask decoder inputs
                # This prevents "posterior collapse" where the decoder
                # ignores the JEPA vector and just uses autoregressive context
                if self.training:
                    prob = 0.4  # Drop 40% of tokens
                    mask = torch.rand(dec_input_idx.shape, device=device) < prob
                    mask = mask & (dec_input_idx != 50256)  # Don't drop BOS/pad
                    dec_input_idx[mask] = 0  # Replace with token 0 ("!")
                
                T_dec = dec_input_idx.shape[1]
                pos_dec = torch.arange(0, T_dec, dtype=torch.long, device=device)
                tgt_emb = self.embedding(dec_input_idx) + self.pos_embedding(pos_dec)
                
                # Causal mask for autoregressive decoding
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
                
                dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = self.lm_head(dec_out)
                
                # Cross entropy loss (ignore padding in targets)
                gen_loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    dec_target_idx.reshape(-1),
                    ignore_index=50256
                )
                
                losses["gen_loss"] = gen_loss
                losses["total_loss"] = latent_loss + gen_loss
            else:
                losses["total_loss"] = latent_loss

        return z_hat_target, losses

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class JEPAConfig:
    vocab_size: int = 50257
    embed_dim: int = 256
    
    # Architecture dimensions
    enc_layers: int = 4
    enc_heads: int = 4
    pred_depth: int = 2
    dec_layers: int = 4
    dec_heads: int = 4
    
    block_size: int = 128
    dropout: float = 0.1
    encoder_only: bool = False # If True, skips the decoder (pure representation learning)

class NanoJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------------
        # 1. STUDENT COMPONENTS (Gradient Updated)
        # ----------------------------------------------------------------------
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.enc_heads,
            dim_feedforward=config.embed_dim * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.student_encoder = nn.TransformerEncoder(enc_layer, num_layers=config.enc_layers)
        
        # Stabilization: LayerNorm for Thought Vectors
        self.student_norm = nn.LayerNorm(config.embed_dim)

        # The Predictor (The Brain) - Dynamic Depth
        layers = []
        # Input expansion
        layers.extend([
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.GELU()
        ])
        
        # Hidden layers (keeping dim*2 width)
        # Default pred_depth=2 means Input->Hidden->Output (2 layers of weights? No, standard is layer count)
        # If depth=2, we want Linear->Act->Linear.
        # So we add (depth - 2) hidden blocks.
        for _ in range(max(0, config.pred_depth - 2)):
            layers.extend([
                nn.Linear(config.embed_dim * 2, config.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            
        # Output projection
        layers.extend([
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.embed_dim)
        ])
        
        self.predictor = nn.Sequential(*layers)

        # ----------------------------------------------------------------------
        # 2. TEACHER COMPONENTS (EMA Updated - "The Ghost")
        # ----------------------------------------------------------------------
        # Deep copy the student components to initialize the teacher
        self.teacher_embedding = copy.deepcopy(self.embedding)
        self.teacher_pos_embedding = copy.deepcopy(self.pos_embedding)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_norm = copy.deepcopy(self.student_norm)

        # Freeze Teacher Gradients PERMANENTLY
        for p in self.teacher_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_pos_embedding.parameters(): p.requires_grad = False
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        for p in self.teacher_norm.parameters(): p.requires_grad = False

        # ----------------------------------------------------------------------
        # 3. DECODER (Auxiliary / Probing)
        # ----------------------------------------------------------------------
        if not config.encoder_only:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=config.embed_dim, nhead=config.dec_heads,
                dim_feedforward=config.embed_dim * 4, dropout=config.dropout,
                batch_first=True, norm_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.dec_layers)
            self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    @torch.no_grad()
    def update_teacher(self, decay=0.996):
        """
        Updates the teacher weights using Exponential Moving Average.
        Teacher = decay * Teacher + (1 - decay) * Student
        """
        # Embeddings
        for t, s in zip(self.teacher_embedding.parameters(), self.embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        for t, s in zip(self.teacher_pos_embedding.parameters(), self.pos_embedding.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
        
        # Encoder
        for t, s in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))
            
        # Norm
        for t, s in zip(self.teacher_norm.parameters(), self.student_norm.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=(1 - decay))

    def encode_student(self, token_idx):
        device = token_idx.device
        T = token_idx.shape[1]
        
        # 1. Create Masks
        # Padding token is 50256 (GPT-2 eot)
        is_pad = (token_idx == 50256)
        # Mask for Mean Pooling (1.0 for real, 0.0 for pad)
        mask_float = (~is_pad).float().unsqueeze(-1) 
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.embedding(token_idx) + self.pos_embedding(pos)
        
        # 2. Pass Mask to Encoder (Prevent attention to padding)
        # src_key_padding_mask expects True for padding positions
        enc_out = self.student_encoder(x, src_key_padding_mask=is_pad)
        
        # 3. Masked Mean Pooling (Prevent averaging padding)
        # Sum valid vectors / Count valid vectors
        z_sum = (enc_out * mask_float).sum(dim=1)
        z_count = mask_float.sum(dim=1).clamp(min=1.0) # Avoid divide by zero
        
        z = z_sum / z_count
        
        # 4. LayerNorm
        return self.student_norm(z)

    def encode_teacher(self, token_idx):
        device = token_idx.device
        T = token_idx.shape[1]
        
        # 1. Create Masks
        is_pad = (token_idx == 50256)
        mask_float = (~is_pad).float().unsqueeze(-1)
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.teacher_embedding(token_idx) + self.teacher_pos_embedding(pos)
        
        # 2. Pass Mask to Encoder
        enc_out = self.teacher_encoder(x, src_key_padding_mask=is_pad)
        
        # 3. Masked Mean Pooling
        z_sum = (enc_out * mask_float).sum(dim=1)
        z_count = mask_float.sum(dim=1).clamp(min=1.0)
        
        z = z_sum / z_count
        
        # 4. LayerNorm
        return self.teacher_norm(z)

    def forward(self, context_idx, target_idx=None):
        device = context_idx.device
        
        # --- 1. STUDENT PATH (The Guesser) ---
        z_context = self.encode_student(context_idx)
        z_hat_target = self.predictor(z_context) # Predictor now has LayerNorm at end

        losses = {}
        if target_idx is not None:
            # --- 2. TEACHER PATH (The Truth) ---
            # Get the "Real" thought from the stable Teacher
            with torch.no_grad():
                z_true_target = self.encode_teacher(target_idx)

            # --- 3. JEPA LOSS (Meaning + Repulsion) ---
            # Using LayerNormed vectors, we don't need F.normalize anymore
            # but we still want the vectors to be somewhat close in magnitude for MSE to make sense
            
            # A. Attraction (Positive Pair)
            pos_loss = F.mse_loss(z_hat_target, z_true_target)
            
            # B. Repulsion (Negative Pair)
            # Shuffle targets to get "Random Other Stories" from the same batch
            z_random_target = torch.roll(z_true_target, shifts=1, dims=0)
            
            # Calculate distance to negative sample
            neg_dist = F.mse_loss(z_hat_target, z_random_target)
            
            # Hinge Loss: We want neg_dist > margin
            # Margin might need adjustment since we are using LayerNorm (norm ~ sqrt(d))
            # d=256, sqrt(256)=16. Expected norm is around 16.
            # MSE between two random vectors of norm 16:
            # E[|x-y|^2] = E[|x|^2] + E[|y|^2] = 16^2 + 16^2 ?? No, LayerNorm makes elements N(0,1).
            # MSE is mean squared error per element.
            # If elements are N(0,1), diff is N(0,2), squared diff is ChiSq(1)*2. Mean is 2.
            # So expected MSE between random vectors is 2.0.
            # Margin of 1.0 is safe.
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

            # --- 4. DECODER LOSS (Auxiliary) ---
            if not self.config.encoder_only:
                # Detach so decoder gradients don't mess up the JEPA "thinking"
                dec_input = z_hat_target.detach()
                memory = dec_input.unsqueeze(1) # [Batch, 1, Dim]
                
                # --- FIX: SHIFT INPUTS FOR TEACHER FORCING ---
                # Input to Decoder:  [t_0, t_1, ..., t_{N-1}]
                # Target for Loss:   [t_1, t_2, ..., t_N]
                dec_input_idx = target_idx[:, :-1].clone()
                dec_target_idx = target_idx[:, 1:]
                
                # === WORD DROPOUT (Prevent Posterior Collapse) ===
                if self.training:
                    # Drop 40% of tokens to force reliance on thought vector
                    prob = 0.4
                    mask = torch.rand(dec_input_idx.shape, device=device) < prob
                    
                    # Don't drop 50256 (BOS/EOT/PAD)
                    mask = mask & (dec_input_idx != 50256)
                    
                    # Replace with 0 (token '!')
                    dec_input_idx[mask] = 0
                # =================================================
                
                T_dec = dec_input_idx.shape[1]
                pos_dec = torch.arange(0, T_dec, dtype=torch.long, device=device)
                
                # Embed shifted (and potentially noisy) input
                tgt_emb = self.embedding(dec_input_idx) + self.pos_embedding(pos_dec)
                
                # Causal Mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
                
                # Forward Decoder
                dec_out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                logits = self.lm_head(dec_out)
                
                # Cross Entropy on SHIFTED targets
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

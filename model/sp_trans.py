import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
from .model_utils import TimestepEmbedder, modulate
class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network for transformer blocks."""
    def __init__(self, hidden_dim: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h = num_heads
        self.d = hidden_dim // num_heads
        self.dropout_p = dropout                  # only used during training
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:
        """
        The multi-head attention module.

        Args:
            q: (B, L, D) the query tensor of attention
            k: (B, S, E) the key tensor of attention
            v: (B, S, E) the value tensor of attention
            mask: (B, L, S) the mask tensor of attention

        Returns:
            (B, L, dim) output
        """
        B = q.shape[0]
        q = self.q_proj(q).view(B, -1, self.h, self.d).transpose(1, 2)  # (B,h,L,d)
        k = self.k_proj(k).view(B, -1, self.h, self.d).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.h, self.d).transpose(1, 2)

        # PyTorch chooses Flash automatically when shapes/dtypes allow.
        # The context manager ensures we **only** use Flash for this block.
        # with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, -1, self.h * self.d)  # (B,L,H)
        return self.output_proj(out)

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer that processes multi-view features."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional embeddings
        x = x + pos_emb
        if time_emb is not None:
            x = x + time_emb

        # Reformulate mask for attention: from [B, seq_len] to [B, 1, 1, seq_len]
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]

        # Self attention
        attn_out = self.self_attn(x, x, x, attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class TransformerDecoderLayerWithadaLNZero(nn.Module):
    """Transformer Decoder Layer with AdaLN for injecting timestamp information."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim, bias=True)
        )
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                query_embed: torch.Tensor, pos_emb: torch.Tensor,
                conditions: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add query embeddings
        x = x + query_embed
        
        # prepare adaLN modulation
        shift_msa, scale_msa, gate_msa, \
        shift_mca, scale_mca, gate_mca, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditions).chunk(9, dim=1)

        attn_self_mask = None
        if self_mask is not None:
            if len(self_mask.shape) == 2:
                # Reformulate self_mask for attention: [B, seq_len] -> [B, 1, 1, seq_len]
                attn_self_mask = self_mask.unsqueeze(1).unsqueeze(2)
            elif len(self_mask.shape) == 3:
                # Reformulate self_mask for attention: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
                attn_self_mask = self_mask.unsqueeze(1)

        # Self attention
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.dropout(self.self_attn(x_mod, x_mod, x_mod, attn_self_mask))
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        
        # Add positional embeddings to memory for cross attention
        memory_with_pos = memory + pos_emb

        # Reformulate cross_mask for attention: [B, seq_len] -> [B, 1, 1, seq_len]
        attn_cross_mask = None
        if cross_mask is not None:
            attn_cross_mask = cross_mask.unsqueeze(1).unsqueeze(2)

        # Cross attention
        x_mod = modulate(self.norm2(x), shift_mca, scale_mca)
        cross_attn_out = self.dropout(self.cross_attn(x_mod, memory_with_pos, memory_with_pos, attn_cross_mask))
        x = x + gate_mca.unsqueeze(1) * cross_attn_out
        
        # Feed forward
        x = x + gate_mlp.unsqueeze(1) * self.dropout(self.feed_forward(modulate(self.norm3(x), shift_mlp, scale_mlp)))
        
        return x


class SPGenTransformer(nn.Module):
    """Transformer used by SPGen."""
    def __init__(self, hidden_dim: int, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, num_heads: int = 8, return_intermediate: bool = True, dropout: float = 0.1,
                 cfg=None):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
    

        self.return_intermediate = return_intermediate
        # Transformer layers
        if num_encoder_layers > 0:
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder_layers = None
            
        # Decoder layers
        assert num_decoder_layers > 0, "Number of decoder layers must be greater than 0"
        if len(self.encoder_layers) > 0:
            self.t_embedder = TimestepEmbedder(hidden_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithadaLNZero(hidden_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the transformer parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for layer in self.decoder_layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)
    def forward(self, multi_view_feats: torch.Tensor, mask: torch.Tensor,
                query_embed: Optional[torch.Tensor], pos_embed: torch.Tensor, timesteps: Optional[torch.Tensor] = None,
                self_attn_mask: Optional[torch.Tensor] = None) \
                    -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass of the SPGen transformer.
        
        Args:
            multi_view_feats (torch.Tensor): Multi-view features [B, N, C, h, w]
            mask (torch.Tensor): Mask for the input features [B, N, h, w]
            query_embed (torch.Tensor): Query embeddings [Q, C]
            pos_embed (torch.Tensor): Position embeddings [B, N, C, h, w], same as multi_view_feats
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output features from the transformer [num_decoder_layers or 1, B, Q, C]
                                             and encoded memory features
        """
        B, N, C, h, w = multi_view_feats.shape
        
        # query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        
        # Reshape to [B, N*h*w, C]
        src = multi_view_feats.permute(0, 1, 3, 4, 2).reshape(B, N * h * w, C).contiguous() # [B, N*h*w, C]
        pos_embed = pos_embed.permute(0, 1, 3, 4, 2).reshape(B, N * h * w, C).contiguous() # [B, N*h*w, C]
        mask = mask.reshape(B, N * h * w).contiguous()  # [B, N*h*w]
        
        # Encode multi-view features with positional embeddings
        memory = src
        if self.encoder_layers is not None:
            time_embed = self.t_embedder(timesteps.view(B * N))
            time_embed = time_embed.view(B, N, 1, 1, C).repeat(1, 1, h, w, 1).reshape(B, N * h * w, C).contiguous()
            for encoder_layer in self.encoder_layers:
                memory = encoder_layer(memory, pos_embed, mask, time_embed)
        else:
            # If no encoder layers, just add positional embeddings to memory
            memory = memory + pos_embed
        
        conditions = self.t_embedder(timesteps[:, -1])
        # Initialize decoder output with zeros
        assert query_embed is not None, "query_embed must be provided"
                
        output = torch.zeros_like(query_embed)
        intermediate = []
        
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, query_embed, pos_embed, conditions, self_mask=self_attn_mask, cross_mask=mask)
            if self.return_intermediate:
                intermediate.append(output)
        
        # Final output
        if self.return_intermediate:
            output = torch.stack(intermediate)
        else:
            output = output.unsqueeze(0)
        
        return {'output': output}
        # Final output shape: [num_decoder_layers, B, Q, C]
        # Note: The output shape is [num_decoder_layers, B, Q, C] if return_intermediate is True
        # Otherwise, it will be [B, Q, C] after the last decoder layer.

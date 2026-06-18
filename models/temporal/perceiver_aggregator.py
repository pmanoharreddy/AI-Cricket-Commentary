"""
models/temporal/perceiver_aggregator.py

Perceiver-like Temporal Aggregator — adapted from MatchTime (MatchVoice)
Reference: Rao et al., EMNLP 2024 | Jaegle et al., ICML 2021

Architecture:
    Visual features (n_frames × 512) 
        → Cross-Attention with 32 learnable queries
        → Self-Attention
        → Feed-Forward
        (×2 layers)
        → Temporal features (32 × 512)
"""

import torch
import torch.nn as nn
from einops import rearrange


class PerceiverLayer(nn.Module):
    """
    Single Perceiver decoder layer.
    Learnable queries attend (cross-attention) to visual features,
    then self-attend among themselves.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # --- Cross-Attention: queries attend to visual features ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)

        # --- Self-Attention: queries attend to each other ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(hidden_dim)

        # --- Feed-Forward ---
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries:         (B, num_queries, hidden_dim)
            visual_features: (B, n_frames,   hidden_dim)
        Returns:
            queries:         (B, num_queries, hidden_dim)
        """
        # Cross-attention: queries (Q) attend to visual features (K, V)
        attn_out, _ = self.cross_attn(
            query=queries,
            key=visual_features,
            value=visual_features,
        )
        queries = self.norm_cross(queries + attn_out)  # residual + norm

        # Self-attention
        self_out, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm_self(queries + self_out)

        # Feed-forward
        ff_out = self.ff(queries)
        queries = self.norm_ff(queries + ff_out)

        return queries


class PerceiverTemporalAggregator(nn.Module):
    """
    Perceiver-like temporal aggregator exactly as described in MatchTime/MatchVoice.

    Parameters (from paper):
        num_layers  = 2   (transformer decoder layers)
        num_queries = 32  (learnable queries)
        hidden_dim  = 512 (matches CLIP ViT-B/32 output)

    Input:  visual features  (B, n_frames, input_dim)
    Output: temporal features (B, num_queries, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = 512,      # CLIP ViT-B/32 output dim
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_queries: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Project visual features to hidden_dim if different from input_dim
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Learnable queries — initialized randomly, trained end-to-end
        self.learnable_queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )

        # Positional encoding for visual features (optional but helpful)
        # We use a simple learned positional embedding up to 512 frames
        self.pos_embedding = nn.Embedding(512, hidden_dim)

        # Stack of Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverLayer(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, n_frames, input_dim)
                             Raw CLIP features for each frame in the clip

        Returns:
            temporal_features: (B, num_queries, hidden_dim)
                               Compressed temporal representation
        """
        B, n_frames, _ = visual_features.shape

        # Project to hidden_dim
        x = self.input_proj(visual_features)  # (B, n_frames, hidden_dim)

        # Add positional encoding to visual features
        positions = torch.arange(n_frames, device=x.device)
        pos_emb = self.pos_embedding(positions)  # (n_frames, hidden_dim)
        x = x + pos_emb.unsqueeze(0)            # (B, n_frames, hidden_dim)

        # Expand learnable queries for batch
        queries = self.learnable_queries.expand(B, -1, -1)  # (B, num_queries, hidden_dim)

        # Pass through Perceiver layers
        for layer in self.layers:
            queries = layer(queries=queries, visual_features=x)

        # Final norm
        temporal_features = self.output_norm(queries)  # (B, num_queries, hidden_dim)

        return temporal_features


class MLPProjection(nn.Module):
    """
    MLP Projection Layer — maps temporal features to LLM prefix tokens.
    
    Input:  (B, num_queries, hidden_dim)   e.g. (B, 32, 512)
    Output: (B, num_queries, llm_dim)      e.g. (B, 32, 4096) for LLaMA-3-8B
                                                or (B, 32, 3584) for Qwen2.5-7B
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 2048,
        output_dim: int = 3584,   # Qwen2.5-7B hidden size
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: (B, num_queries, input_dim)
        Returns:
            prefix_tokens: (B, num_queries, output_dim)
        """
        return self.projection(temporal_features)


# ---- Quick sanity test ----
if __name__ == "__main__":
    B = 2          # batch size
    n_frames = 16  # 8 seconds × 2 FPS
    input_dim = 512  # CLIP ViT-B/32

    # Simulate CLIP visual features
    visual_features = torch.randn(B, n_frames, input_dim)

    # Temporal Aggregator
    aggregator = PerceiverTemporalAggregator(
        input_dim=512,
        hidden_dim=512,
        num_layers=2,
        num_queries=32,
        num_heads=8,
    )
    temporal_features = aggregator(visual_features)
    print(f"Visual features:   {visual_features.shape}")    # (2, 16, 512)
    print(f"Temporal features: {temporal_features.shape}")  # (2, 32, 512)

    # MLP Projection
    projection = MLPProjection(input_dim=512, hidden_dim=2048, output_dim=3584)
    prefix_tokens = projection(temporal_features)
    print(f"Prefix tokens:     {prefix_tokens.shape}")      # (2, 32, 3584)
    print("✅ Temporal aggregator + MLP projection working correctly.")

"""
GPT from Scratch - Lesson 5: Transformer Block
===============================================

Now we combine all the pieces we've learned into the TRANSFORMER BLOCK.

WHAT WE'VE BUILT SO FAR:
  1. Token + Position Embeddings
  2. Multi-Head Self-Attention

TODAY'S PLACE IN PIPELINE:
  Input embeddings → [Transformer Block × N] → Final representation

WHAT WE'LL BUILD:
  1. Feed-Forward Network (FFN) with nn.Linear
  2. Layer Normalization
  3. Residual Connections
  4. Complete Transformer Block
  5. Stack of Transformer Blocks

GPT vs Transformer:
  - GPT uses MASKED self-attention (causal - can't see future)
  - GPT is decoder-only (no encoder, no cross-attention)
  - GPT predicts next token (autoregressive)

NOTE: This lesson uses PyTorch to show LEARNABLE parameters (nn.Linear)

MATRIX DIMENSIONS WE'LL COVER:
==============================
- Input:             (batch, seq_len, d_model)
- FFN hidden:        (batch, seq_len, d_ff) where d_ff = 4 × d_model
- LayerNorm output:  (batch, seq_len, d_model)
- Transformer Block: (batch, seq_len, d_model) → same shape!

LEARNABLE PARAMETERS:
=====================
- FFN: fc1 (d_model, d_ff), fc2 (d_ff, d_model)
- LayerNorm: weight (d_model,), bias (d_model,)
- MultiHeadAttention: W_q, W_k, W_v, W_o (each d_model, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# COMPONENT 1: Feed-Forward Network (FFN) with nn.Linear
# =============================================================================

class FeedForwardNetwork(nn.Module):
    """
    The position-wise Feed-Forward Network in each transformer block.

    THE BIG QUESTION: WHY DO WE NEED THIS?
    ───────────────────────────────────────
    After self-attention, every token has gathered information from other tokens.
    BUT attention alone is just a weighted SUM. It's a linear operation (roughly).
    We need NON-LINEARITY to make the model powerful.

    THE PROBLEM WITH ATTENTION ALONE:
    ─────────────────────────────────
    Self-attention computes: weighted_sum(values)
    This is essentially: output = Σ attention_weight_i × value_i
    Which is a LINEAR combination of values.

    A network with ONLY linear layers is just ONE linear layer, no matter how deep.
    It can only learn linear relationships. But language is HIGHLY NON-LINEAR!

    THE SOLUTION: Feed-Forward Network (FFN)
    ───────────────────────────────────────
    The FFN adds TWO things:
    1. NON-LINEARITY (via ReLU activation)
    2. LEARNABLE TRANSFORMATION (two nn.Linear layers with ReLU in between)

    ARCHITECTURE:
    ─────────────
    Input:  (batch, seq_len, d_model)
      ↓
    Linear 1: d_model → d_ff       (expand to higher dimension)
      ↓
    ReLU activation                  ← THIS adds non-linearity!
      ↓
    Linear 2: d_ff → d_model       (project back)
      ↓
    Output: (batch, seq_len, d_model)

    DIMENSIONS (GPT-2 Small):
      d_model = 768, d_ff = 3072  (4× expansion)

    WHY d_ff > d_model?
    ───────────────────
    Expanding to a higher dimension gives the network more capacity
    to learn complex transformations, then projects back down.
    It's like a bottleneck layer.

    THE INTUITION:
    ──────────────
    Think of the FFN as the "thinking" part of each layer:
      - Self-attention: "What information do I need from other tokens?"
      - FFN: "Now let me PROCESS that information using what I've learned."

    Analogy: In a team project:
      - Self-attention = gathering input from teammates
      - FFN = your individual thinking/synthesis process

    NOTE: This is "position-wise" because the SAME FFN is applied to
    every position independently. Position 0 and Position 1 both use
    the exact same weights.

    LEARNABLE PARAMETERS:
    - W1 = nn.Linear(d_model, d_ff) - First linear projection
    - W2 = nn.Linear(d_ff, d_model) - Second linear projection
    - b1, b2 - Biases (automatically included in nn.Linear)

    MATRIX DIMENSIONS THROUGH FFN:
    ==============================
    INPUT:
      x: (batch, seq_len, d_model)
      Example: (2, 10, 768) - 2 batches, 10 tokens, 768-dim embeddings

    STEP 1 - FIRST LINEAR (fc1):
      fc1.weight: (d_ff, d_model) = (3072, 768)
      fc1.bias: (d_ff,) = (3072,)
      
      hidden = fc1(x)
      (batch, seq_len, d_model) → (batch, seq_len, d_ff)
      Example: (2, 10, 768) → (2, 10, 3072)
      
      Matrix operation: x @ fc1.weight.T + fc1.bias
      (2, 10, 768) @ (768, 3072) → (2, 10, 3072)

    STEP 2 - ReLU ACTIVATION:
      hidden = ReLU(hidden)
      (batch, seq_len, d_ff) → (batch, seq_len, d_ff)
      Example: (2, 10, 3072) → (2, 10, 3072)
      
      Element-wise operation: max(0, x)
      No shape change, just applies non-linearity

    STEP 3 - SECOND LINEAR (fc2):
      fc2.weight: (d_model, d_ff) = (768, 3072)
      fc2.bias: (d_model,) = (768,)
      
      output = fc2(hidden)
      (batch, seq_len, d_ff) → (batch, seq_len, d_model)
      Example: (2, 10, 3072) → (2, 10, 768)
      
      Matrix operation: hidden @ fc2.weight.T + fc2.bias
      (2, 10, 3072) @ (3072, 768) → (2, 10, 768)

    OUTPUT:
      Same shape as input: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize the Feed-Forward Network with LEARNABLE parameters.

        Args:
            d_model: Dimension of input/output embeddings
            d_ff: Hidden dimension of feed-forward network (typically 4× d_model)

        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        - self.fc1 = nn.Linear(d_model, d_ff) - First linear layer
        - self.fc2 = nn.Linear(d_ff, d_model) - Second linear layer

        These use Xavier/Glorot initialization by default.
        They will be UPDATED via backpropagation during training!
        """
        super().__init__()

        # LEARNABLE linear layers - PyTorch initializes these automatically!
        # fc1.weight: (d_ff, d_model), fc1.bias: (d_ff,)
        self.fc1 = nn.Linear(d_model, d_ff)  # Expand: d_model → d_ff
        # fc2.weight: (d_model, d_ff), fc2.bias: (d_model,)
        self.fc2 = nn.Linear(d_ff, d_model)  # Project back: d_ff → d_model

        print(f"Feed-Forward Network initialized:")
        print(f"  d_model = {d_model}, d_ff = {d_ff}")
        print(f"  Architecture: d_model → d_ff → d_model")
        print(f"  Non-linearity: ReLU activation")
        print()
        print(f"  LEARNABLE PARAMETERS:")
        print(f"    fc1: weight {self.fc1.weight.shape} + bias {self.fc1.bias.shape}")
        print(f"    fc2: weight {self.fc2.weight.shape} + bias {self.fc2.bias.shape}")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total learnable parameters: {total_params:,}")

    def forward(self, x):
        """
        Forward pass through the FFN.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
               Example: (2, 10, 768) - 2 batches, 10 tokens, 768-dim embeddings

        Returns:
            Output tensor, same shape as input (batch, seq_len, d_model)

        DIMENSION FLOW:
        ===============
        Input: (batch, seq_len, d_model)
             ↓
        fc1: (batch, seq_len, d_model) @ (d_ff, d_model).T → (batch, seq_len, d_ff)
             ↓
        ReLU: Element-wise, same shape (batch, seq_len, d_ff)
             ↓
        fc2: (batch, seq_len, d_ff) @ (d_model, d_ff).T → (batch, seq_len, d_model)
             ↓
        Output: (batch, seq_len, d_model)
        """
        # Step 1: Linear 1 - Expand to higher dimension
        # x: (batch, seq_len, d_model)
        # fc1.weight: (d_ff, d_model)
        # hidden: (batch, seq_len, d_ff)
        hidden = self.fc1(x)  # (batch, seq_len, d_ff)

        # Step 2: ReLU activation (non-linearity!)
        # hidden: (batch, seq_len, d_ff)
        # Element-wise: max(0, hidden)
        hidden = F.relu(hidden)  # (batch, seq_len, d_ff)

        # Step 3: Linear 2 - Project back to d_model
        # hidden: (batch, seq_len, d_ff)
        # fc2.weight: (d_model, d_ff)
        # output: (batch, seq_len, d_model)
        output = self.fc2(hidden)  # (batch, seq_len, d_model)

        return output


# =============================================================================
# COMPONENT 2: Layer Normalization
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization: Normalize across the feature dimension.

    WHY LAYER NORM INSTEAD OF BATCH NORM?
    ──────────────────────────────────────
    1. Works well with small batch sizes (even batch_size=1)
    2. Normalizes per-sample, not across batch
    3. More stable for sequence models where seq_len varies

    THE FORMULA:
    ────────────
    For each sample independently:
      μ = mean(features)
      σ² = variance(features)
      normed = (features - μ) / sqrt(σ² + ε)
      output = γ * normed + β

    Where γ (weight) and β (bias) are learnable parameters.

    WHY NORMALIZE?
    ──────────────
    Layer normalization:
    1. Stabilizes training (prevents exploding/vanishing gradients)
    2. Makes optimization easier (smoother loss landscape)
    3. Allows higher learning rates
    4. Reduces sensitivity to initialization

    In GPT, we use Pre-LayerNorm architecture (more stable for deep models):
      x → LayerNorm → Sublayer → x + sublayer_output

    The original Transformer used Post-LayerNorm:
      x → Sublayer → LayerNorm(x + sublayer_output)

    Pre-LayerNorm is more stable for training deep transformers.

    LEARNABLE PARAMETERS:
    - weight (γ): Scale factor, shape (d_model,)
    - bias (β): Shift factor, shape (d_model,)

    MATRIX DIMENSIONS:
    ==================
    INPUT:
      x: (batch, seq_len, d_model)
      Example: (2, 10, 768)

    STEP 1 - COMPUTE STATISTICS:
      mean: (batch, seq_len, 1) - mean over d_model dimension
      var: (batch, seq_len, 1) - variance over d_model dimension
      
      For each position, compute mean and variance across all d_model features.

    STEP 2 - NORMALIZE:
      x_norm = (x - mean) / sqrt(var + ε)
      (batch, seq_len, d_model) → (batch, seq_len, d_model)
      
      Each feature is normalized to have mean≈0, std≈1.

    STEP 3 - SCALE AND SHIFT:
      weight: (d_model,) - learnable scale (γ)
      bias: (d_model,) - learnable shift (β)
      
      output = weight * x_norm + bias
      Broadcasting: (d_model,) applied to (batch, seq_len, d_model)
      
      This allows the network to undo normalization if beneficial.

    OUTPUT:
      Same shape as input: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, eps=1e-5):
        """
        Initialize Layer Normalization.

        Args:
            d_model: Dimension of features to normalize
            eps: Small constant for numerical stability

        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        - self.weight = nn.Parameter(torch.ones(d_model)) - Scale (γ)
        - self.bias = nn.Parameter(torch.zeros(d_model)) - Shift (β)

        These allow the network to UN-normalize if needed!
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # LEARNABLE parameters: weight (scale) and bias (shift)
        # These let the network undo normalization if beneficial
        # weight: (d_model,), bias: (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

        print(f"LayerNorm initialized:")
        print(f"  d_model = {d_model}")
        print(f"  LEARNABLE PARAMETERS:")
        print(f"    weight (γ): {self.weight.shape}")
        print(f"    bias (β): {self.bias.shape}")

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
               Example: (2, 10, 768)

        Returns:
            Normalized tensor, same shape as input (batch, seq_len, d_model)

        DIMENSION FLOW:
        ===============
        Input: (batch, seq_len, d_model)
             ↓
        mean = x.mean(dim=-1, keepdim=True)
             ↓
        mean: (batch, seq_len, 1) - mean over d_model
             ↓
        var = x.var(dim=-1, keepdim=True)
             ↓
        var: (batch, seq_len, 1) - variance over d_model
             ↓
        x_norm = (x - mean) / sqrt(var + eps)
             ↓
        x_norm: (batch, seq_len, d_model) - normalized
             ↓
        output = weight * x_norm + bias
             ↓
        Broadcasting: (d_model,) * (batch, seq_len, d_model) → (batch, seq_len, d_model)
             ↓
        Output: (batch, seq_len, d_model)
        """
        # Step 1: Compute mean and variance over the last dimension (d_model)
        # x: (batch, seq_len, d_model)
        # mean: (batch, seq_len, 1) - keepdim keeps the dimension
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        
        # var: (batch, seq_len, 1) - variance over d_model
        var = x.var(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # Step 2: Normalize
        # x_norm: (batch, seq_len, d_model)
        # (x - mean): Broadcasting (batch, seq_len, 1) to (batch, seq_len, d_model)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (batch, seq_len, d_model)

        # Step 3: Scale and shift with learnable parameters
        # weight: (d_model,), bias: (d_model,)
        # Broadcasting: (d_model,) applied to (batch, seq_len, d_model)
        return self.weight * x_norm + self.bias  # (batch, seq_len, d_model)


# =============================================================================
# COMPONENT 3: Residual Connection
# =============================================================================

# In the Transformer, residual connections are simple:
#   output = input + sublayer(input)
# But they're ALWAYS followed by LayerNorm:
#   output = LayerNorm(input + sublayer_output)
#
# ORIGINAL PAPER (Post-LayerNorm):
#   x → Sublayer → LayerNorm(x + sublayer_output)
#
# MODERN PRACTICE (Pre-LayerNorm - more stable for deep models):
#   x → LayerNorm → Sublayer → x + sublayer_output
#
# GPT-2 and later models use Pre-LayerNorm for better training stability.


# =============================================================================
# MAIN: Transformer Block (GPT Block)
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A complete Transformer (GPT) Block.

    This is the core building block of GPT. Stack N of these to build
    a deep transformer model.

    ARCHITECTURE (Pre-LayerNorm - modern practice):
    ────────────────────────────────────────────────
    Input: (batch, seq_len, d_model)
      ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm (pre-attention)               │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Multi-Head Self-Attention               │
    │   - MASKED (causal - can't see future)  │
    │   - Each token attends to past tokens   │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual Connection: x + attention      │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm (pre-FFN)                     │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Feed-Forward Network                    │
    │   - Applied position-wise               │
    │   - d_model → d_ff → d_model            │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual Connection: x + FFN            │
    └─────────────────────────────────────────┘
          ↓
    Output: (batch, seq_len, d_model)

    THE ORDER (Pre-LayerNorm - modern practice):
      x → LayerNorm → Sublayer → x + sublayer_output

    WHY PRE-LAYERNORM?
    ──────────────────
    Pre-LayerNorm is more stable for training deep transformers because:
    1. Gradients flow better through residual connections
    2. Normalization happens before transformation (better conditioning)
    3. Works well with very deep models (GPT-3 has 96 layers!)

    GPT NOTE:
    ─────────
    GPT uses MASKED self-attention. The mask prevents tokens from
    attending to future positions. This is crucial for autoregressive
    generation (predicting one token at a time).

    LEARNABLE PARAMETERS in this block:
    - MultiHeadAttention: W_q, W_k, W_v, W_o (4 × d_model²)
    - FeedForwardNetwork: fc1, fc2 weights (2 × d_model × d_ff)
    - LayerNorm: weight, bias (2 × d_model × 2 layers)

    MATRIX DIMENSIONS THROUGH TRANSFORMER BLOCK:
    ============================================
    
    INPUT:
      x: (batch, seq_len, d_model)
      Example: (2, 10, 768)

    SUB LAYER 1 - ATTENTION PATH:
    ─────────────────────────────
    Step 1 - Pre-LayerNorm:
      ln1_out = self.ln1(x)
      (batch, seq_len, d_model) → (batch, seq_len, d_model)
      
    Step 2 - Multi-Head Attention:
      attn_out = self.attention(ln1_out)
      (batch, seq_len, d_model) → (batch, seq_len, d_model)
      See MultiHeadAttention for detailed dimension flow.
      
    Step 3 - Residual Connection:
      x = x + attn_out
      (batch, seq_len, d_model) + (batch, seq_len, d_model) → (batch, seq_len, d_model)

    SUB LAYER 2 - FFN PATH:
    ───────────────────────
    Step 4 - Pre-LayerNorm:
      ln2_out = self.ln2(x)
      (batch, seq_len, d_model) → (batch, seq_len, d_model)
      
    Step 5 - Feed-Forward Network:
      ffn_out = self.ffn(ln2_out)
      (batch, seq_len, d_model) → (batch, seq_len, d_ff) → (batch, seq_len, d_model)
      See FeedForwardNetwork for detailed dimension flow.
      
    Step 6 - Residual Connection:
      x = x + ffn_out
      (batch, seq_len, d_model) + (batch, seq_len, d_model) → (batch, seq_len, d_model)

    OUTPUT:
      Same shape as input: (batch, seq_len, d_model)
      But now each token has rich contextual representation!
    """

    def __init__(self, d_model, n_heads, d_ff):
        """
        Initialize a transformer block with LEARNABLE parameters.

        Args:
            d_model: Dimension of input/output embeddings
            n_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network

        LEARNABLE PARAMETERS:
        - LayerNorm ln1: weight (d_model,), bias (d_model,)
        - LayerNorm ln2: weight (d_model,), bias (d_model,)
        - MultiHeadAttention: W_q, W_k, W_v, W_o (each d_model, d_model)
        - FeedForwardNetwork: fc1 (d_model, d_ff), fc2 (d_ff, d_model)
        """
        super().__init__()

        # Layer normalization (before each sublayer - Pre-LayerNorm)
        # LEARNABLE: weight (d_model,), bias (d_model,)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # Multi-Head Self-Attention (masked)
        # LEARNABLE: W_q, W_k, W_v, W_o (each d_model, d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Feed-Forward Network
        # LEARNABLE: fc1 (d_model, d_ff), fc2 (d_ff, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        print(f"\nTransformerBlock initialized:")
        print(f"  d_model = {d_model}, n_heads = {n_heads}, d_ff = {d_ff}")

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
               Example: (2, 10, 768)

        Returns:
            Output tensor, shape (batch, seq_len, d_model)

        DIMENSION FLOW THROUGH BLOCK:
        =============================
        
        SUB LAYER 1 - ATTENTION:
        ────────────────────────
        x: (batch, seq_len, d_model)
             ↓
        ln1_out = self.ln1(x)
             ↓
        ln1_out: (batch, seq_len, d_model)
             ↓
        attn_out = self.attention(ln1_out)
             ↓
        attn_out: (batch, seq_len, d_model)
             ↓
        x = x + attn_out  (residual)
             ↓
        x: (batch, seq_len, d_model)
        
        SUB LAYER 2 - FFN:
        ──────────────────
        x: (batch, seq_len, d_model)
             ↓
        ln2_out = self.ln2(x)
             ↓
        ln2_out: (batch, seq_len, d_model)
             ↓
        ffn_out = self.ffn(ln2_out)
             ↓
        ffn_out: (batch, seq_len, d_model)
             ↓
        x = x + ffn_out  (residual)
             ↓
        Output: (batch, seq_len, d_model)
        """
        # ---- Sublayer 1: Multi-Head Self-Attention ----
        # Pre-LayerNorm: normalize BEFORE attention
        # x: (batch, seq_len, d_model)
        # ln1_out: (batch, seq_len, d_model)
        ln1_out = self.ln1(x)

        # Masked self-attention (causal)
        # ln1_out: (batch, seq_len, d_model)
        # attn_out: (batch, seq_len, d_model)
        attn_out = self.attention(ln1_out)

        # Residual connection
        # x + attn_out: (batch, seq_len, d_model) + (batch, seq_len, d_model)
        x = x + attn_out

        # ---- Sublayer 2: Feed-Forward Network ----
        # Pre-LayerNorm: normalize BEFORE FFN
        # x: (batch, seq_len, d_model)
        # ln2_out: (batch, seq_len, d_model)
        ln2_out = self.ln2(x)

        # FFN transformation
        # ln2_out: (batch, seq_len, d_model)
        # ffn_out: (batch, seq_len, d_model)
        ffn_out = self.ffn(ln2_out)

        # Residual connection
        # x + ffn_out: (batch, seq_len, d_model) + (batch, seq_len, d_model)
        x = x + ffn_out

        return x


# =============================================================================
# ALTERNATIVE: Post-LayerNorm Architecture (original paper)
# =============================================================================

class TransformerBlockPostLN(nn.Module):
    """
    Transformer Block with Post-LayerNorm (original paper style).

    This is provided for comparison and educational purposes.
    Modern practice uses Pre-LayerNorm (TransformerBlock above).

    ARCHITECTURE (Post-LayerNorm - original paper):
    ───────────────────────────────────────────────
    Input: (batch, seq_len, d_model)
      ↓
    ┌─────────────────────────────────────────┐
    │ Multi-Head Self-Attention               │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual: x + attention                 │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm (after residual)              │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Feed-Forward Network                    │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ Residual: x + FFN                       │
    └─────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────┐
    │ LayerNorm (after residual)              │
    └─────────────────────────────────────────┘
          ↓
    Output: (batch, seq_len, d_model)

    THE ORDER (Post-LayerNorm - original paper):
      x → Sublayer → LayerNorm(x + sublayer_output)
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        # Sublayers
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer normalization (after each sublayer - Post-LayerNorm)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        """Forward pass with Post-LayerNorm."""
        # Attention sublayer
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.ln1(x)  # Post-LayerNorm

        # FFN sublayer
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ln2(x)  # Post-LayerNorm

        return x


# =============================================================================
# STACK OF TRANSFORMER BLOCKS
# =============================================================================

class TransformerBlockStack(nn.Module):
    """
    Stack of N transformer blocks.

    Each block builds on top of the previous one:
      - Block 1: Direct context from input tokens
      - Block 2: Context from context (2-hop relationships)
      - Block 3: Context from context from context (3-hop)
      - ...and so on

    Deeper layers capture more abstract relationships.

    GPT-2 Configurations (CORRECTED):
    ─────────────────────────────────
    | Config | d_model | n_heads | d_ff | n_blocks | params |
    |--------|---------|---------|------|----------|--------|
    | Small  |   768   |   12    | 3072 |    12    |  124M  |
    | Medium |  1024   |   16    | 4096 |    24    |  350M  |
    | Large  |  1280   |   20    | 5120 |    36    |  774M  |
    | XL     |  1600   |   25    | 6400 |    48    | 1558M  |

    MATRIX DIMENSIONS THROUGH STACK:
    ================================
    INPUT:
      x: (batch, seq_len, d_model)
      Example: (2, 10, 768)

    BLOCK 1:
      Input:  (batch, seq_len, d_model)
      Output: (batch, seq_len, d_model)
      
    BLOCK 2:
      Input:  (batch, seq_len, d_model)  ← Output of Block 1
      Output: (batch, seq_len, d_model)
      
    BLOCK 3:
      Input:  (batch, seq_len, d_model)  ← Output of Block 2
      Output: (batch, seq_len, d_model)
      
    ...and so on for all N blocks.

    OUTPUT:
      Same shape as input: (batch, seq_len, d_model)
      But now with deep contextual understanding!
    """

    def __init__(self, d_model, n_heads, d_ff, n_blocks):
        """
        Initialize stack of transformer blocks.

        Args:
            d_model: Dimension of embeddings
            n_heads: Number of attention heads
            d_ff: Hidden dimension of FFN
            n_blocks: Number of transformer blocks

        LEARNABLE PARAMETERS:
        - Each TransformerBlock has:
          - LayerNorm: 2 × d_model parameters (weight + bias) × 2 norms
          - MultiHeadAttention: 4 × d_model² parameters
          - FeedForwardNetwork: 2 × d_model × d_ff parameters
        - Total: n_blocks × (4 × d_model + 4 × d_model² + 2 × d_model × d_ff)
        """
        super().__init__()
        self.n_blocks = n_blocks

        # Create transformer blocks
        # Each block: TransformerBlock(d_model, n_heads, d_ff)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Output tensor, shape (batch, seq_len, d_model)

        DIMENSION FLOW:
        ===============
        x: (batch, seq_len, d_model)
             ↓
        Block 0: (batch, seq_len, d_model) → (batch, seq_len, d_model)
             ↓
        Block 1: (batch, seq_len, d_model) → (batch, seq_len, d_model)
             ↓
        ... (repeats for n_blocks)
             ↓
        Block n-1: (batch, seq_len, d_model) → (batch, seq_len, d_model)
             ↓
        Output: (batch, seq_len, d_model)
        """
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_feed_forward_network():
    """Show how the Feed-Forward Network works with nn.Linear."""
    print("=" * 70)
    print("FEED-FORWARD NETWORK DEMO (with nn.Linear)")
    print("=" * 70)
    print()

    d_model = 16
    d_ff = 64  # Expanded dimension (4×)
    batch_size = 1
    seq_len = 3

    ffn = FeedForwardNetwork(d_model, d_ff)

    print(f"\nLEARNABLE PARAMETERS:")
    for name, param in ffn.named_parameters():
        print(f"  {name}: {param.shape}")

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    # Show intermediate shapes
    hidden = ffn.fc1(x)
    print(f"After fc1 (Linear 1): {hidden.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_ff={d_ff})")
    print()

    hidden = F.relu(hidden)
    print(f"After ReLU: {hidden.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_ff={d_ff})")
    print()

    output = ffn.fc2(hidden)
    print(f"After fc2 (Linear 2): {output.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print(f"Architecture:")
    print(f"  fc1: nn.Linear({d_model}, {d_ff})")
    print(f"  ReLU activation")
    print(f"  fc2: nn.Linear({d_ff}, {d_model})")
    print()

    print(f"Key property: Position-wise!")
    print(f"  Same weights applied to every position.")
    print()


def demonstrate_layer_norm():
    """Show how Layer Normalization works."""
    print("=" * 70)
    print("LAYER NORMALIZATION DEMO")
    print("=" * 70)
    print()

    d_model = 4
    batch_size = 1
    seq_len = 3

    layer_norm = LayerNorm(d_model)

    print(f"\nLEARNABLE PARAMETERS:")
    for name, param in layer_norm.named_parameters():
        print(f"  {name}: {param.shape}")

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    # Show what gets normalized
    print("LayerNorm normalizes ACROSS the feature dimension:")
    for s in range(seq_len):
        vals = x[0, s].tolist()
        mean = x[0, s].mean().item()
        std = x[0, s].std().item()
        print(f"  Position [{s}]: {[f'{v:.4f}' for v in vals]}")
        print(f"    mean={mean:.4f}, std={std:.4f}")
    print()

    output = layer_norm(x)
    print("After LayerNorm:")
    for s in range(seq_len):
        vals = output[0, s].tolist()
        mean = output[0, s].mean().item()
        std = output[0, s].std().item()
        print(f"  Position [{s}]: {[f'{v:.4f}' for v in vals]}")
        print(f"    mean≈{mean:.6f}, std≈{std:.4f}")
    print()
    print("✓ All positions now have mean≈0 and std≈1")


def demonstrate_transformer_block():
    """Show a complete transformer block in action."""
    print("=" * 70)
    print("TRANSFORMER BLOCK DEMO")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 256
    batch_size = 1
    seq_len = 5

    block = TransformerBlock(d_model, n_heads, d_ff)

    print(f"\nTotal learnable parameters in this block:")
    total_params = sum(p.numel() for p in block.parameters())
    print(f"  {total_params:,} parameters")
    print()

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print("Processing through transformer block:")
    print(f"  1. LayerNorm (pre-attention)")
    print(f"  2. Masked Self-Attention: each token attends to past tokens")
    print(f"  3. Residual connection")
    print(f"  4. LayerNorm (pre-FFN)")
    print(f"  5. Feed-Forward: position-wise processing")
    print(f"  6. Residual connection")
    print()

    output = block(x)
    print(f"Output shape: {output.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    print("Key observations:")
    print("  ✓ Input and output have the SAME shape")
    print("  ✓ Each position now has CONTEXT from past positions")
    print("  ✓ Causal mask prevents looking ahead (autoregressive)")
    print()


def demonstrate_full_transformer():
    """Show a stack of transformer blocks."""
    print("=" * 70)
    print("FULL TRANSFORMER (Stack of N Blocks)")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_ff = 256
    n_blocks = 3
    batch_size = 1
    seq_len = 4

    transformer = TransformerBlockStack(d_model, n_heads, d_ff, n_blocks)

    print(f"\nTotal learnable parameters:")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"  {total_params:,} parameters across {n_blocks} blocks")
    print()

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Transformer: {n_blocks} stacked blocks")
    print(f"Input shape: {x.shape}")
    print()

    for i in range(n_blocks):
        x = transformer.blocks[i](x)
        print(f"After block {i+1}: {x.shape}")

    print()
    print(f"Final output: {x.shape}")
    print()
    print("Each block builds on top of the previous one:")
    print("  Block 1: Direct context from input tokens")
    print("  Block 2: Context from context (2-hop relationships)")
    print("  Block 3: Context from context from context (3-hop)")
    print("  ...and so on → deeper blocks capture more abstract relationships")
    print()


def show_transformer_parameters():
    """Show parameter counts for different configurations."""
    print("=" * 70)
    print("TRANSFORMER BLOCK PARAMETER COUNTS")
    print("=" * 70)
    print()

    configs = [
        ("Tiny", 64, 4, 128, 2),
        ("Small", 256, 8, 512, 4),
        ("Base", 512, 8, 2048, 6),
        ("Big", 1024, 16, 4096, 6),
    ]

    print(f"{'Config':<10} {'d_model':<10} {'heads':<8} {'d_ff':<10} {'blocks':<8} {'params':<12}")
    print("-" * 64)

    for name, d_model, n_heads, d_ff, n_blocks in configs:
        # Approximate params per block
        attention_params = 4 * d_model * d_model
        ffn_params = 2 * d_model * d_ff
        layernorm_params = 4 * d_model
        block_params = attention_params + ffn_params + layernorm_params
        total_params = n_blocks * block_params
        print(f"{name:<10} {d_model:<10} {n_heads:<8} {d_ff:<10} {n_blocks:<8} {total_params/1e6:.2f}M")

    print()
    print("GPT-2 Small:  12 blocks, d_model=768, n_heads=12 → ~124M params")
    print("GPT-2 Medium: 24 blocks, d_model=1024, n_heads=16 → ~350M params")
    print("GPT-2 Large:  36 blocks, d_model=1280, n_heads=20 → ~774M params")
    print("GPT-2 XL:     48 blocks, d_model=1600, n_heads=25 → ~1.5B params")
    print()


def compare_pre_post_layernorm():
    """Compare Pre-LayerNorm vs Post-LayerNorm architectures."""
    print("=" * 70)
    print("PRE-LAYERNORM vs POST-LAYERNORM COMPARISON")
    print("=" * 70)
    print()

    print("Pre-LayerNorm (Modern Practice - GPT-2/3):")
    print("  x → LayerNorm → Sublayer → x + sublayer_output")
    print()
    print("  Pros:")
    print("    ✓ More stable for deep models")
    print("    ✓ Better gradient flow")
    print("    ✓ Works with very deep transformers (100+ layers)")
    print()
    print("  Cons:")
    print("    • Slightly different from original paper")
    print()

    print("-" * 70)
    print()

    print("Post-LayerNorm (Original Transformer Paper):")
    print("  x → Sublayer → LayerNorm(x + sublayer_output)")
    print()
    print("  Pros:")
    print("    ✓ Matches original paper exactly")
    print("    ✓ Good for shallow models (6-12 layers)")
    print()
    print("  Cons:")
    print("    • Less stable for very deep models")
    print("    • May need learning rate warmup")
    print()

    print("-" * 70)
    print()
    print("Recommendation: Use Pre-LayerNorm for GPT-style models!")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run all demonstrations."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "GPT FROM SCRATCH - LESSON 5" + " " * 24 + "║")
    print("║" + " " * 20 + "Transformer Block" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Feed-Forward Network
    demonstrate_feed_forward_network()
    print()

    # Demo 2: Layer Normalization
    demonstrate_layer_norm()
    print()

    # Demo 3: Transformer Block
    demonstrate_transformer_block()
    print()

    # Demo 4: Full Transformer
    demonstrate_full_transformer()
    print()

    # Parameter counts
    show_transformer_parameters()
    print()

    # Pre vs Post LayerNorm comparison
    compare_pre_post_layernorm()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 5:")
    print("-" * 70)
    print("✓ Feed-Forward Network: nn.Linear(d_model, d_ff) → ReLU → nn.Linear(d_ff, d_model)")
    print("✓ Layer Normalization: nn.Parameter(weight, bias) for learnable scale/shift")
    print("✓ Residual Connection: x + sublayer(x)")
    print("✓ Transformer Block: LN → Attention → +x → LN → FFN → +x")
    print("✓ Stack N transformer blocks for deeper processing")
    print("✓ Pre-LayerNorm is more stable for deep GPT models")
    print()
    print("LEARNABLE PARAMETERS in TransformerBlock:")
    print("  - Attention: W_q, W_k, W_v, W_o (nn.Linear layers)")
    print("  - FFN: fc1, fc2 (nn.Linear layers)")
    print("  - LayerNorm: weight, bias (nn.Parameter)")
    print()
    print("TRANSFORMER BLOCK FORMULA (Pre-LayerNorm):")
    print("  x → LayerNorm(x) → MultiHeadAttention → x + attn")
    print("  → LayerNorm(x) → FFN → x + ffn")
    print()
    print("NEXT: Complete GPT Model (embeddings + transformer stack + output)")
    print("Run: python 06_gpt_architecture.py")
    print("=" * 70)
    print()


# Import MultiHeadAttention for the demos
# We define a minimal version here for standalone execution
class MultiHeadAttention(nn.Module):
    """Minimal MultiHeadAttention for standalone execution of this file."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.register_buffer('causal_mask', None)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch, seq_len, self.d_model)

    def forward(self, x, use_causal_mask=True):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        if use_causal_mask:
            if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
                self.causal_mask = mask.unsqueeze(0).unsqueeze(0)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]

            d_k = Q_heads.shape[-1]
            scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)
            scores = scores + mask
            attn = F.softmax(scores, dim=-1)
        else:
            d_k = Q_heads.shape[-1]
            scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)
            attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, V_heads)
        output = self._combine_heads(output)
        output = self.W_o(output)

        return output


if __name__ == "__main__":
    main()
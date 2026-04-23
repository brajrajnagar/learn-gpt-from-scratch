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
  1. Feed-Forward Network (FFN)
  2. Layer Normalization
  3. Residual Connections
  4. Complete Transformer Block
  5. Stack of Transformer Blocks

GPT vs Transformer:
  - GPT uses MASKED self-attention (causal - can't see future)
  - GPT is decoder-only (no encoder, no cross-attention)
  - GPT predicts next token (autoregressive)
"""

import math
import numpy as np
import importlib

# Import MultiHeadAttention from lesson 3
# We import it at the module level for clean code
try:
    attention_module = importlib.import_module('03_attention')
    MultiHeadAttention = attention_module.MultiHeadAttention
except ImportError:
    # Fallback if running standalone - define minimal version below
    MultiHeadAttention = None


# =============================================================================
# COMPONENT 1: Feed-Forward Network (FFN)
# =============================================================================

class FeedForwardNetwork:
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
    2. LEARNED TRANSFORMATION (two linear layers with ReLU in between)

    ARCHITECTURE:
    ─────────────
    Input:  (seq_len, d_model)
      ↓
    Linear 1: d_model → d_ff       (expand to higher dimension)
      ↓
    ReLU activation                  ← THIS adds non-linearity!
      ↓
    Linear 2: d_ff → d_model       (project back)
      ↓
    Output: (seq_len, d_model)

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
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize the Feed-Forward Network.

        Args:
            d_model: Dimension of input/output embeddings
            d_ff: Hidden dimension of feed-forward network (typically 4× d_model)
        """
        np.random.seed(42)

        # He initialization for better gradient flow
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        Forward pass through the FFN.

        Args:
            x: Input tensor, shape (seq_len, d_model)

        Returns:
            Output tensor, shape (seq_len, d_model)
        """
        # Linear 1: d_model → d_ff
        hidden = np.dot(x, self.W1) + self.b1

        # ReLU activation (non-linearity!)
        hidden = np.maximum(0, hidden)

        # Linear 2: d_ff → d_model
        output = np.dot(hidden, self.W2) + self.b2

        return output


# =============================================================================
# COMPONENT 2: Layer Normalization
# =============================================================================

class LayerNorm:
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

    Where γ (gamma) and β (beta) are learnable parameters.

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
    """

    def __init__(self, d_model, eps=1e-5):
        """
        Initialize Layer Normalization.

        Args:
            d_model: Dimension of features to normalize
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters: gamma (scale) and beta (shift)
        # These let the network UN-normalize if needed
        self.gamma = np.ones(d_model)  # Scale
        self.beta = np.zeros(d_model)  # Shift

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x: Input tensor, shape (..., d_model)

        Returns:
            Normalized tensor, same shape as input
        """
        # Compute mean and variance over the last dimension (d_model)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


# =============================================================================
# COMPONENT 3: Residual Connection
# =============================================================================

# In the Transformer, residual connections are simple:
#   output = input + sublayer(input)
# But they're ALWAYS followed by LayerNorm:
#   output = LayerNorm(input + sublayer(input))
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

class TransformerBlock:
    """
    A complete Transformer (GPT) Block.

    This is the core building block of GPT. Stack N of these to build
    a deep transformer model.

    ARCHITECTURE (Pre-LayerNorm - modern practice):
    ────────────────────────────────────────────────
    Input: (seq_len, d_model)
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
    │ Residual Connection: x + attention    │
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
    │ Residual Connection: x + FFN          │
    └─────────────────────────────────────────┘
          ↓
    Output: (seq_len, d_model)

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
    """

    def __init__(self, d_model, n_heads, d_ff):
        """
        Initialize a transformer block.

        Args:
            d_model: Dimension of input/output embeddings
            n_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
        """
        # Layer normalization (before each sublayer - Pre-LayerNorm)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        # Multi-Head Self-Attention (masked)
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor, shape (seq_len, d_model)

        Returns:
            Output tensor, shape (seq_len, d_model)
        """
        # ---- Sublayer 1: Multi-Head Self-Attention ----
        # Pre-LayerNorm: normalize BEFORE attention
        ln1_out = self.ln1.forward(x)

        # Masked self-attention (causal)
        # MultiHeadAttention.forward() returns (output, attention_weights)
        attn_result = self.attention.forward(ln1_out, use_causal_mask=True)
        attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result

        # Residual connection
        x = x + attn_out

        # ---- Sublayer 2: Feed-Forward Network ----
        # Pre-LayerNorm: normalize BEFORE FFN
        ln2_out = self.ln2.forward(x)

        # FFN transformation
        ffn_out = self.ffn.forward(ln2_out)

        # Residual connection
        x = x + ffn_out

        return x


# Fallback MultiHeadAttention class for when import fails
# This is defined at the end of the file after all other classes


# =============================================================================
# ALTERNATIVE: Post-LayerNorm Architecture (original paper)
# =============================================================================

class TransformerBlockPostLN:
    """
    Transformer Block with Post-LayerNorm (original paper style).

    This is provided for comparison and educational purposes.
    Modern practice uses Pre-LayerNorm (TransformerBlock above).

    ARCHITECTURE (Post-LayerNorm - original paper):
    ───────────────────────────────────────────────
    Input: (seq_len, d_model)
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
    Output: (seq_len, d_model)

    THE ORDER (Post-LayerNorm - original paper):
      x → Sublayer → LayerNorm(x + sublayer_output)
    """

    def __init__(self, d_model, n_heads, d_ff):
        # Sublayers
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer normalization (after each sublayer - Post-LayerNorm)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        """Forward pass with Post-LayerNorm."""
        # Attention sublayer
        attn_out = self.attention.forward(x, use_causal_mask=True)
        x = x + attn_out
        x = self.ln1.forward(x)  # Post-LayerNorm

        # FFN sublayer
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x = self.ln2.forward(x)  # Post-LayerNorm

        return x


# =============================================================================
# STACK OF TRANSFORMER BLOCKS
# =============================================================================

class TransformerBlockStack:
    """
    Stack of N transformer blocks.

    Each block builds on top of the previous one:
      - Block 1: Direct context from input tokens
      - Block 2: Context from context (2-hop relationships)
      - Block 3: Context from context from context (3-hop)
      - ...and so on

    Deeper layers capture more abstract relationships.

    GPT-2 Configurations:
    ─────────────────────
    | Config | d_model | n_heads | d_ff | n_blocks |
    |--------|---------|---------|------|----------|
    | Small  |   768   |   12    | 3072 |    12    |
    | Medium |   768   |   12    | 3072 |    24    |
    | Large  |  1024   |   16    | 4096 |    36    |
    | XL     |  16384  |  100    | 65536|    48    |
    """

    def __init__(self, d_model, n_heads, d_ff, n_blocks):
        """
        Initialize stack of transformer blocks.

        Args:
            d_model: Dimension of embeddings
            n_heads: Number of attention heads
            d_ff: Hidden dimension of FFN
            n_blocks: Number of transformer blocks
        """
        self.n_blocks = n_blocks
        self.blocks = []

        for i in range(n_blocks):
            block = TransformerBlock(d_model, n_heads, d_ff)
            self.blocks.append(block)
            print(f"  Block {i+1}/{n_blocks} created")

    def forward(self, x):
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor, shape (seq_len, d_model)

        Returns:
            Output tensor, shape (seq_len, d_model)
        """
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
        return x


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_feed_forward_network():
    """Show how the Feed-Forward Network works."""
    print("=" * 70)
    print("FEED-FORWARD NETWORK DEMO")
    print("=" * 70)
    print()

    d_model = 16
    d_ff = 64  # Expanded dimension (4×)
    seq_len = 3

    ffn = FeedForwardNetwork(d_model, d_ff)

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"  (seq_len={seq_len}, d_model={d_model})")
    print()

    output = ffn.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"  (seq_len={seq_len}, d_model={d_model})")
    print()

    print(f"Architecture:")
    print(f"  Linear 1: {d_model} → {d_ff}")
    print(f"  ReLU")
    print(f"  Linear 2: {d_ff} → {d_model}")
    print()

    print(f"Key property: Position-wise!")
    print(f"  Position 0 and Position 1 use the EXACT SAME weights.")
    print(f"  The FFN processes each position independently.")
    print()


def demonstrate_layer_norm():
    """Show how Layer Normalization works."""
    print("=" * 70)
    print("LAYER NORMALIZATION DEMO")
    print("=" * 70)
    print()

    seq_len = 3
    d_model = 4

    layer_norm = LayerNorm(d_model)

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"  (seq_len={seq_len}, d_model={d_model})")
    print()

    # Show what gets normalized
    print("LayerNorm normalizes ACROSS the feature dimension:")
    for s in range(seq_len):
        vals = x[s].tolist()
        mean = x[s].mean()
        std = x[s].std()
        print(f"  Position [{s}]: {[f'{v:.4f}' for v in vals]}")
        print(f"    mean={mean:.4f}, std={std:.4f}")
    print()

    output = layer_norm.forward(x)
    print("After LayerNorm:")
    for s in range(seq_len):
        vals = output[s].tolist()
        mean = output[s].mean()
        std = output[s].std()
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
    seq_len = 5

    block = TransformerBlock(d_model, n_heads, d_ff)

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    print(f"Input shape: {x.shape}")
    print(f"  (seq_len={seq_len}, d_model={d_model})")
    print()

    print("Processing through transformer block:")
    print(f"  1. LayerNorm (pre-attention)")
    print(f"  2. Masked Self-Attention: each token attends to past tokens")
    print(f"  3. Residual connection")
    print(f"  4. LayerNorm (pre-FFN)")
    print(f"  5. Feed-Forward: position-wise processing")
    print(f"  6. Residual connection")
    print()

    output = block.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"  (seq_len={seq_len}, d_model={d_model})")
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
    seq_len = 4

    transformer = TransformerBlockStack(d_model, n_heads, d_ff, n_blocks)

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    print(f"Transformer: {n_blocks} stacked blocks")
    print(f"Input shape: {x.shape}")
    print()

    for i in range(n_blocks):
        x = transformer.blocks[i].forward(x)
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

    print(f"{'Config':<10} {'d_model':<10} {'heads':<8} {'d_ff':<10} {'blocks':<8}")
    print("-" * 54)

    for name, d_model, n_heads, d_ff, n_blocks in configs:
        # Approximate params per block
        # Attention: 4 × d_model² (Q, K, V, output projections)
        # FFN: 2 × d_model × d_ff (two linear layers)
        # LayerNorm: 2 × d_model (gamma, beta) × 2 (two norms)
        attention_params = 4 * d_model * d_model
        ffn_params = 2 * d_model * d_ff
        layernorm_params = 4 * d_model
        block_params = attention_params + ffn_params + layernorm_params
        total_params = n_blocks * block_params
        print(f"{name:<10} {d_model:<10} {n_heads:<8} {d_ff:<10} {n_blocks:<8} {total_params/1e6:.2f}M params")

    print()
    print("GPT-2 Small:  12 blocks, d_model=768 → ~124M params")
    print("GPT-2 Medium: 24 blocks, d_model=768 → ~248M params")
    print("GPT-2 Large:  36 blocks, d_model=1024 → ~762M params")
    print("GPT-2 XL:     48 blocks, d_model=16384 → ~1.5B params")
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
    print("✓ Feed-Forward Network: d_model → d_ff → d_model (position-wise)")
    print("✓ Layer Normalization: normalize per-sample across features")
    print("✓ Residual Connection: x + sublayer(x)")
    print("✓ Transformer Block: LN → Attention → +x → LN → FFN → +x")
    print("✓ Stack N transformer blocks for deeper processing")
    print("✓ Pre-LayerNorm is more stable for deep GPT models")
    print()
    print("TRANSFORMER BLOCK FORMULA (Pre-LayerNorm):")
    print("  x → LayerNorm(x) → MultiHeadAttention → x + attn")
    print("  → LayerNorm(x) → FFN → x + ffn")
    print()
    print("NEXT: Complete GPT Model (embeddings + transformer stack + output)")
    print("Run: python 06_gpt_model.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
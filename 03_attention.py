"""
GPT from Scratch - Lesson 3: Self-Attention
============================================

This is the CORE mechanism that makes GPT work.

WHAT WE'VE BUILT SO FAR:
  Raw Text → Token IDs → Token Embeddings + Position Embeddings → Input

TODAY'S PLACE IN PIPELINE:
  Input embeddings → SELF-ATTENTION → Contextualized representations

WHAT WE'LL BUILD:
  1. Scaled Dot-Product Attention (the fundamental operation)
  2. Multi-Head Attention (multiple attention "perspectives")

GPT vs Transformer:
  - GPT uses MASKED self-attention (causal - can't see future)
  - Transformer encoder uses full self-attention (can see all positions)
  - GPT is decoder-only (predicts next token)
"""

import math
import numpy as np


# =============================================================================
# STEP 1: Scaled Dot-Product Attention
# =============================================================================

class ScaledDotProductAttention:
    """
    The fundamental attention operation.

    COMPUTE:
    ─────────
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    WHERE:
    ──────
    Q (Query):    What am I looking for?  shape: (seq_q, d_k)
    K (Key):      What do I contain?      shape: (seq_k, d_k)
    V (Value):    What info do I carry?   shape: (seq_k, d_v)

    STEPS:
    ──────
    1. Compute similarity: scores = Q @ K^T / sqrt(d_k)
       shape: (seq_q, seq_k)
    2. Apply softmax: attention_weights = softmax(scores, axis=-1)
       Each row sums to 1.0 → these are "attention weights"
    3. Weighted sum: output = attention_weights @ V
       shape: (seq_q, d_v)

    WHY sqrt(d_k)?
    ──────────────
    When d_k is large, the dot products Q@K^T can have large variance.
    Large values → softmax becomes very peaked (near one-hot) → gradients
    vanish. Dividing by sqrt(d_k) keeps values in a good range.

    This is why it's called "SCALE"d dot-product attention.

    GPT NOTE:
    ─────────
    In GPT, we apply a causal mask to prevent positions from attending
    to future positions. This is crucial for autoregressive generation.
    """

    def __init__(self):
        pass

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for scaled dot-product attention.

        Args:
            Q: Query tensor, shape (seq_q, d_k)
            K: Key tensor, shape (seq_k, d_k)
            V: Value tensor, shape (seq_k, d_v)
            mask: Optional mask tensor, shape (seq_q, seq_k)
                  Where mask[i,j] = -1e9 means "don't attend"

        Returns:
            output: Attention output, shape (seq_q, d_v)
            attention_weights: Softmax weights, shape (seq_q, seq_k)
        """
        d_k = Q.shape[-1]

        # Step 1: Compute raw attention scores
        # Q: (seq_q, d_k), K^T: (d_k, seq_k) → scores: (seq_q, seq_k)
        scores = np.dot(Q, K.T) / math.sqrt(d_k)

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores + mask

        # Step 2: Softmax → attention weights
        # Each row sums to 1.0
        attention_weights = softmax(scores)

        # Step 3: Weighted sum of values
        # weights: (seq_q, seq_k), V: (seq_k, d_v) → output: (seq_q, d_v)
        output = np.dot(attention_weights, V)

        return output, attention_weights


# =============================================================================
# STEP 2: Multi-Head Attention
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention: Run attention multiple times in parallel.

    WHY MULTI-HEAD?
    ───────────────
    Think of each "head" as having different glasses:
      Head 1  → sees subject-verb relationships
      Head 2  → sees pronoun-antecedent relationships
      Head 3  → sees adjective-noun relationships
      Head 4  → sees temporal/causal relationships

    Each head learns DIFFERENT attention patterns. Then we combine them.

    THE PROCESS:
    ────────────
    1. Project Q, K, V into lower-dimensional spaces (d_model → n_heads × d_k)
    2. Split into multiple heads (each head operates in d_k dimensions)
    3. Apply scaled dot-product attention to EACH head
    4. Concatenate all heads
    5. Project back to d_model

    DIMENSIONS:
    ───────────
    Input:  (seq_len, d_model)
    Where:  d_model = n_heads × d_k

    Step 1: Linear projection → (seq_len, d_model)  [3 copies: Q, K, V]
    Step 2: Reshape → (seq_len, n_heads, d_k)
    Step 3: Transpose → (n_heads, seq_len, d_k)
    Step 4: Attention → (n_heads, seq_len, d_k)
    Step 5: Transpose → (seq_len, n_heads, d_k)
    Step 6: Reshape → (seq_len, n_heads × d_k) = (seq_len, d_model)
    Step 7: Linear project → (seq_len, d_model)

    GPT NOTE:
    ─────────
    GPT uses MASKED multi-head attention in the decoder.
    The mask prevents positions from attending to future positions.
    """

    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Dimension of input/output embeddings
            n_heads: Number of attention heads

        WHY d_k = d_model // n_heads?
        ─────────────────────────────
        The MULTI-HEAD attention splits the d_model dimensions ACROSS the heads.

        Think of it like splitting a deck of cards:
          Total cards = d_model = 64
          Players = n_heads = 4
          Cards per player = d_k = 64 / 4 = 16

        Each head operates independently in its own d_k-dimensional space.
        Then we concatenate all heads back together: n_heads × d_k = d_model

        EXAMPLE with d_model=64, n_heads=4:
          d_k = 64 // 4 = 16
          Head 1: operates in dimensions [0:16]
          Head 2: operates in dimensions [16:32]
          Head 3: operates in dimensions [32:48]
          Head 4: operates in dimensions [48:64]

        After attention, we concatenate: [head1, head2, head3, head4] → 4×16 = 64
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Same as d_k in standard attention

        # Linear projections for Q, K, V and output
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        # Scaled dot-product attention module
        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x):
        """
        Split the last dimension into (n_heads, d_k) and transpose.

        Input:  (seq_len, d_model)
        Output: (n_heads, seq_len, d_k)

        This allows each head to operate independently.
        """
        seq_len = x.shape[0]
        # Reshape: (seq_len, d_model) → (seq_len, n_heads, d_k)
        x = x.reshape(seq_len, self.n_heads, self.d_k)
        # Transpose: (seq_len, n_heads, d_k) → (n_heads, seq_len, d_k)
        return x.transpose(1, 0, 2)

    def _combine_heads(self, x):
        """
        Merge the heads back together.

        Input:  (n_heads, seq_len, d_k)
        Output: (seq_len, d_model)

        This concatenates all heads and prepares for output projection.
        """
        # Transpose: (n_heads, seq_len, d_k) → (seq_len, n_heads, d_k)
        x = x.transpose(1, 0, 2)
        # Reshape: (seq_len, n_heads, d_k) → (seq_len, n_heads × d_k)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)

    def forward(self, x, use_causal_mask=True):
        """
        Forward pass for multi-head attention.

        Args:
            x: Input embeddings, shape (seq_len, d_model)
            use_causal_mask: Whether to apply causal mask (for GPT)

        Returns:
            output: shape (seq_len, d_model)
            attention_weights: shape (n_heads, seq_len, seq_len)
        """
        seq_len = x.shape[0]

        # Step 1: Linear projections
        Q = np.dot(x, self.W_q)  # (seq_len, d_model)
        K = np.dot(x, self.W_k)  # (seq_len, d_model)
        V = np.dot(x, self.W_v)  # (seq_len, d_model)

        # Step 2: Split into multiple heads
        Q_heads = self._split_heads(Q)  # (n_heads, seq_len, d_k)
        K_heads = self._split_heads(K)  # (n_heads, seq_len, d_k)
        V_heads = self._split_heads(V)  # (n_heads, seq_len, d_k)

        # Create causal mask if needed
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
        else:
            mask = None

        # Step 3 & 4: Apply attention to each head
        head_outputs = []
        attention_weights_list = []
        for head_idx in range(self.n_heads):
            Q_head = Q_heads[head_idx]  # (seq_len, d_k)
            K_head = K_heads[head_idx]  # (seq_len, d_k)
            V_head = V_heads[head_idx]  # (seq_len, d_k)

            # Scaled dot-product attention
            output, attn_weights = self.attention.forward(Q_head, K_head, V_head, mask)
            head_outputs.append(output)
            attention_weights_list.append(attn_weights)

        # Step 5: Concatenate heads
        combined = np.stack(head_outputs, axis=0)  # (n_heads, seq_len, d_k)
        combined = self._combine_heads(combined)   # (seq_len, d_model)

        # Step 6: Output projection
        output = np.dot(combined, self.W_o)  # (seq_len, d_model)

        # Stack attention weights for visualization
        attention_weights = np.stack(attention_weights_list, axis=0)

        return output, attention_weights


# =============================================================================
# Helper Functions
# =============================================================================

def softmax(x):
    """
    Numerically stable softmax.

    Converts logits to probabilities that sum to 1.0.

    Args:
        x: Input array (can be 1D or 2D)

    Returns:
        Softmax output (probabilities that sum to 1)
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    """
    Create causal (triangular) mask for autoregressive generation.

    GPT uses this mask to prevent positions from attending to FUTURE positions.

    Visual representation for seq_len=4:
    ```
    Position 0: [0,  -1e9, -1e9, -1e9]  ← can only see itself
    Position 1: [0,  0,    -1e9, -1e9]  ← can see 0 and 1
    Position 2: [0,  0,    0,    -1e9]  ← can see 0, 1, 2
    Position 3: [0,  0,    0,    0   ]  ← can see all
    ```

    Args:
        seq_len: Sequence length

    Returns:
        Mask matrix where future positions have -1e9
    """
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9
    return mask


# =============================================================================
# Demonstration Functions
# =============================================================================

def demonstrate_scaled_dot_product_attention():
    """Show how scaled dot-product attention works step by step."""
    print("=" * 70)
    print("SCALED DOT-PRODUCT ATTENTION DEMO")
    print("=" * 70)
    print()

    seq_len = 4
    d_k = 8

    # Create simple Q, K, V
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    print(f"Input shapes:")
    print(f"  Q (queries):  {Q.shape}")
    print(f"  K (keys):     {K.shape}")
    print(f"  V (values):   {V.shape}")
    print()

    # Step 1: Compute scores
    scores = np.dot(Q, K.T) / math.sqrt(d_k)
    print(f"Step 1: Raw attention scores = Q @ K^T / sqrt(d_k)")
    print(f"  Shape: {scores.shape}")
    print(f"  Values:\n{scores}")
    print()

    # Step 2: Softmax
    attention_weights = softmax(scores)
    print(f"Step 2: Softmax → attention weights")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  Values:\n{attention_weights}")
    print(f"  Row sums: {attention_weights.sum(axis=-1)}")
    print()

    # Step 3: Weighted sum
    output = np.dot(attention_weights, V)
    print(f"Step 3: Weighted sum = weights @ V")
    print(f"  Shape: {output.shape}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 70)
    words = ["[BOS]", "the", "cat", "sat"]
    print(f"  Words: {words}")
    print()
    for i, word in enumerate(words):
        print(f"  '{word}' attends to:")
        weights_i = attention_weights[i]
        for j, (w, wt) in enumerate(zip(words, weights_i)):
            bar = "█" * int(wt * 40)
            print(f"    → {w:8s}: {wt:.3f} {bar}")
    print()


def demonstrate_multi_head_attention():
    """Show how multi-head attention works."""
    print("=" * 70)
    print("MULTI-HEAD ATTENTION DEMO")
    print("=" * 70)
    print()

    d_model = 64
    n_heads = 4
    d_k = d_model // n_heads

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    seq_len = 5

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    print(f"Configuration:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_k = d_model / n_heads = {d_k}")
    print()

    print(f"Input shape:")
    print(f"  x: {x.shape}")
    print()

    output, attention_weights = mha.forward(x, use_causal_mask=True)

    print(f"Output shapes:")
    print(f"  output:            {output.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    print(f"    (n_heads, seq_len, seq_len)")
    print()

    print(f"Attention weights for head 0 (causal - lower triangular):")
    print(f"  {attention_weights[0]}")
    print()

    print(f"Each head learns DIFFERENT attention patterns:")
    for h in range(n_heads):
        weights = attention_weights[h]
        # For each position, show which position gets most attention
        max_positions = np.argmax(weights, axis=-1)
        print(f"  Head {h}: max attention positions = {max_positions.tolist()}")
    print()


def demonstrate_causal_masking():
    """Show causal masking for GPT (no looking ahead!)."""
    print("=" * 70)
    print("CAUSAL MASKING (GPT - No Looking Ahead!)")
    print("=" * 70)
    print()

    print("In GPT, we must prevent tokens from attending to")
    print("FUTURE positions. This is crucial for autoregressive generation.")
    print()

    seq_len = 5

    # Create causal mask
    mask = create_causal_mask(seq_len)
    print(f"Causal mask (lower triangular):")
    print(f"  0 = visible, -1e9 = masked (future)")
    print(f"  {mask}")
    print()

    print("  Rows = query (what I am)")
    print("  Cols = key (what I can attend to)")
    print()

    words = ["[BOS]", "the", "cat", "sat", "on"]
    print("Interpretation:")
    for i in range(len(words)):
        attend_to = [words[j] for j in range(i + 1)]
        print(f"  '{words[i]}' (pos {i}) can attend to: {attend_to}")
    print()

    # Demo with actual attention
    d_model = 32
    n_heads = 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    output, attn = mha.forward(x, use_causal_mask=True)

    print("With causal mask, attention weights are:")
    for i in range(seq_len):
        weights = attn[0, i]
        print(f"  Position {i} ('{words[i]}'): {[f'{w:.3f}' for w in weights]}")
    print()

    print("Notice: Future positions (right of diagonal) have ~0 attention!")


def demonstrate_attention_visualization():
    """Create a visual representation of attention patterns."""
    print("=" * 70)
    print("ATTENTION PATTERN VISUALIZATION")
    print("=" * 70)
    print()

    # Simple example with 6 tokens
    seq_len = 6
    d_model = 64
    n_heads = 4

    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    # Create input
    np.random.seed(42)
    x = np.random.randn(seq_len, d_model)

    output, attn = mha.forward(x, use_causal_mask=True)

    words = ["[BOS]", "the", "cat", "sat", "on", "mat"]

    print(f"Self-attention patterns for: {' '.join(words)}")
    print(f"(Causal mask applied - can only see past positions)")
    print()

    for h in range(n_heads):
        print(f"Head {h}:")
        for i in range(seq_len):
            weights = attn[h, i]
            # Show which positions get most attention
            max_idx = np.argmax(weights)
            marker = "←" if i == max_idx else "  "
            bar = "█" * int(weights[max_idx] * 30)
            print(f"  {marker} {words[i]:8s} → {words[max_idx]:8s} ({weights[max_idx]:.2%}) {bar}")
        print()


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run all demonstrations."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "GPT FROM SCRATCH - LESSON 3" + " " * 24 + "║")
    print("║" + " " * 21 + "Self-Attention Mechanism" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Demo 1: Scaled dot-product attention
    demonstrate_scaled_dot_product_attention()
    print()

    # Demo 2: Multi-head attention
    demonstrate_multi_head_attention()
    print()

    # Demo 3: Causal masking
    demonstrate_causal_masking()
    print()

    # Demo 4: Attention visualization
    demonstrate_attention_visualization()
    print()

    print("=" * 70)
    print("SUMMARY OF LESSON 3:")
    print("-" * 70)
    print("✓ ScaledDotProductAttention: softmax(Q@K^T/sqrt(d_k)) @ V")
    print("✓ MultiHeadAttention: Multiple attention heads in parallel")
    print("✓ Each head learns different attention patterns")
    print("✓ Causal mask: Prevents GPT from looking ahead (autoregressive)")
    print()
    print("Shapes:")
    print("  Input:  (seq_len, d_model)")
    print("  Output: (seq_len, d_model)")
    print()
    print("KEY FORMULAS:")
    print("  scores = Q @ K^T / sqrt(d_k)")
    print("  attention = softmax(scores) @ V")
    print("  multi_head = concat(head_1, ..., head_n) @ W_O")
    print()
    print("NEXT: Transformer Block (attention + feed-forward + residual + layernorm)")
    print("Run: python 05_transformer_block.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
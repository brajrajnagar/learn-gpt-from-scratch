"""
=============================================================================
LESSON 4: Multi-Head Attention - Multiple Perspectives
=============================================================================

Building on Lesson 3's self-attention, we now use MULTIPLE attention
"heads" that each learn different types of relationships.

REAL-WORLD ANALOGY: Team of Experts
====================================

Imagine analyzing a legal document with a team of experts:
- Lawyer: Focuses on legal terminology and obligations
- Accountant: Focuses on financial terms and numbers
- Linguist: Focuses on grammar and sentence structure
- Historian: Focuses on dates and historical context

Each expert looks at the SAME text but pays attention to DIFFERENT things!

Multi-head attention does exactly this - multiple "heads" each learn
to focus on different aspects of the input.

MATRIX DIMENSIONS WE'LL COVER:
==============================
- Input:             (batch, seq_len, d_model)
- Q, K, V (per head): (batch, n_heads, seq_len, d_k)
- Attention scores:   (batch, n_heads, seq_len, seq_len)
- Combined output:    (batch, seq_len, d_model)
- Final projection:   (batch, seq_len, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# STEP 1: Why Multiple Heads?
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Why Multiple Attention Heads?")
print("="*70)

print("""
THE LIMITATION OF SINGLE ATTENTION:
====================================

In Lesson 3, we used ONE attention mechanism.
Every word had ONE way to attend to other words.

EXAMPLE: "The cat sat on the mat"

With single attention, "mat" gets ONE attention pattern.
But "mat" has MULTIPLE relationships:
  1. Grammatical: object of preposition "on"
  2. Semantic: where cats sit
  3. Referential: "the mat" (definite article)
  4. Positional: end of sentence

ONE head can't capture ALL of these!

MULTI-HEAD SOLUTION:
====================

Use MULTIPLE heads, each learning DIFFERENT patterns:

Head 1: Grammar expert
  → "mat" attends to "on" (preposition-object)

Head 2: Meaning expert
  → "mat" attends to "cat" (semantic role)

Head 3: Reference expert
  → "mat" attends to "the" (article-noun)

Head 4: Position expert
  → "mat" attends to sentence structure

COMBINED = Rich, multi-dimensional understanding!
""")

# =============================================================================
# STEP 2: Splitting Embeddings Among Heads
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Splitting Work Among Heads")
print("="*70)

print("""
HOW WE SPLIT:
=============

Input embeddings: (batch, seq_len, d_model)

We split d_model dimensions among n_heads:
  d_k = d_model // n_heads (dimensions per head)

EXAMPLE: d_model = 64, n_heads = 4
  d_k = 64 // 4 = 16
  
  Head 0: processes dimensions 0-15
  Head 1: processes dimensions 16-31
  Head 2: processes dimensions 32-47
  Head 3: processes dimensions 48-63

WHY SPLIT?
==========

1. EFFICIENCY: Each head processes smaller vectors
   1 head × 64-dim = O(64²) = 4096 ops
   4 heads × 16-dim = 4 × O(16²) = 1024 ops

2. DIVERSITY: Each head can specialize
   → Different attention patterns

3. CAPACITY: More total expressiveness
   Same parameters, richer representations
""")


def split_heads(x, n_heads, d_k):
    """
    Split embeddings among attention heads.
    
    MATRIX DIMENSIONS:
    ==================
    Input:  x: (batch, seq_len, d_model)
    Output: (batch, n_heads, seq_len, d_k)
    
    Steps:
      1. Reshape: (batch, seq_len, n_heads, d_k)
      2. Transpose: (batch, n_heads, seq_len, d_k)
    
    Args:
        x: Input tensor
        n_heads: Number of heads
        d_k: Dimensions per head
    
    Returns:
        Split tensor
    """
    batch, seq_len, d_model = x.shape
    # Reshape: (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
    x = x.view(batch, seq_len, n_heads, d_k)
    # Transpose: (batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
    return x.transpose(1, 2)


def combine_heads(x, n_heads, d_k):
    """
    Combine outputs from multiple heads.
    
    MATRIX DIMENSIONS:
    ==================
    Input:  x: (batch, n_heads, seq_len, d_k)
    Output: (batch, seq_len, d_model) where d_model = n_heads * d_k
    
    Steps:
      1. Transpose: (batch, seq_len, n_heads, d_k)
      2. Reshape: (batch, seq_len, n_heads * d_k)
    
    Args:
        x: Input tensor from split heads
        n_heads: Number of heads
        d_k: Dimensions per head
    
    Returns:
        Combined tensor
    """
    batch, n_heads, seq_len, d_k = x.shape
    # Transpose: (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
    x = x.transpose(1, 2)
    # Reshape: (batch, seq_len, n_heads, d_k) → (batch, seq_len, n_heads * d_k)
    return x.contiguous().view(batch, seq_len, n_heads * d_k)


# =============================================================================
# STEP 3: Multi-Head Attention Implementation
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Multi-Head Attention Implementation")
print("="*70)

print("""
COMPLETE FLOW:
==============

1. Create Q, K, V from input (using linear projections)
2. Split Q, K, V into multiple heads
3. Compute attention for each head (in parallel)
4. Combine head outputs
5. Final linear projection

LEARNABLE PARAMETERS:
=====================
- W_Q: (d_model, d_model) - Query projection
- W_K: (d_model, d_model) - Key projection
- W_V: (d_model, d_model) - Value projection
- W_O: (d_model, d_model) - Output projection

Total: 4 × d_model² parameters
""")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention (same as Lesson 3).
    
    MATRIX DIMENSIONS:
    ==================
    Q, K, V: (batch, n_heads, seq_len, d_k)
    mask: (1, 1, seq_len, seq_len) or similar broadcastable shape
    
    Returns:
        output: (batch, n_heads, seq_len, d_k)
        weights: (batch, n_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


def create_causal_mask(seq_len):
    """Create causal mask with -inf above diagonal."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with learnable parameters.
    
    LEARNABLE PARAMETERS:
    - W_Q, W_K, W_V, W_O: nn.Linear(d_model, d_model)
    
    MATRIX DIMENSIONS THROUGH FORWARD PASS:
    ========================================
    Input:  (batch, seq_len, d_model)
    
    1. Linear projections:
       Q = W_Q(x): (batch, seq_len, d_model)
       K = W_K(x): (batch, seq_len, d_model)
       V = W_V(x): (batch, seq_len, d_model)
    
    2. Split heads:
       Q: (batch, n_heads, seq_len, d_k)
       K: (batch, n_heads, seq_len, d_k)
       V: (batch, n_heads, seq_len, d_k)
    
    3. Attention:
       scores: (batch, n_heads, seq_len, seq_len)
       weights: (batch, n_heads, seq_len, seq_len)
       output: (batch, n_heads, seq_len, d_k)
    
    4. Combine heads:
       (batch, seq_len, d_model)
    
    5. Output projection:
       W_O(output): (batch, seq_len, d_model)
    """
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Learnable projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.register_buffer('causal_mask', None)
    
    def forward(self, x, use_causal_mask=True):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, d_model)
            use_causal_mask: Whether to apply causal masking
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)  # (batch, seq_len, d_model)
        V = self.W_V(x)  # (batch, seq_len, d_model)
        
        # 2. Split into heads
        Q = split_heads(Q, self.n_heads, self.d_k)  # (batch, n_heads, seq_len, d_k)
        K = split_heads(K, self.n_heads, self.d_k)  # (batch, n_heads, seq_len, d_k)
        V = split_heads(V, self.n_heads, self.d_k)  # (batch, n_heads, seq_len, d_k)
        
        # 3. Create causal mask if needed
        if use_causal_mask:
            if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
                mask = create_causal_mask(seq_len)
                self.causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
        else:
            mask = None
        
        # 4. Compute attention for all heads
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        # output: (batch, n_heads, seq_len, d_k)
        # attention_weights: (batch, n_heads, seq_len, seq_len)
        
        # 5. Combine heads
        output = combine_heads(output, self.n_heads, self.d_k)  # (batch, seq_len, d_model)
        
        # 6. Final projection
        output = self.W_O(output)  # (batch, seq_len, d_model)
        
        return output, attention_weights


# =============================================================================
# STEP 4: Demonstration
# =============================================================================

print("\n" + "="*70)
print("DEMONSTRATION: Multi-Head Attention")
print("="*70)

# Configuration
d_model = 64
n_heads = 4
d_k = d_model // n_heads  # 16
seq_len = 5
batch_size = 1

print(f"\nConfiguration:")
print(f"  d_model = {d_model}")
print(f"  n_heads = {n_heads}")
print(f"  d_k = {d_k} (per head)")
print(f"  seq_len = {seq_len}")

# Create model
mha = MultiHeadAttention(d_model, n_heads)

# Count parameters
total_params = sum(p.numel() for p in mha.parameters())
print(f"\nTotal parameters: {total_params:,}")
print(f"  W_Q: {mha.W_Q.weight.shape}")
print(f"  W_K: {mha.W_K.weight.shape}")
print(f"  W_V: {mha.W_V.weight.shape}")
print(f"  W_O: {mha.W_O.weight.shape}")

# Create input
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, d_model)

print(f"\nInput shape: {x.shape}")

# Forward pass
output, attn_weights = mha(x, use_causal_mask=True)

print(f"\nOutput shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")

# Show attention patterns
words = ["The", "cat", "sat", "on", "mat"]
print(f"\nAttention patterns for: {' '.join(words)}")

for h in range(n_heads):
    print(f"\nHead {h}:")
    for i in range(seq_len):
        weights = attn_weights[0, h, i]
        max_idx = torch.argmax(weights)
        print(f"  {words[i]:6s} → attends most to → {words[max_idx]:6s} ({weights[max_idx]:.2%})")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Multi-Head Attention")
print("="*70)

print("""
WHAT WE LEARNED:
================
1. Multiple heads = multiple experts analyzing same text
2. Each head learns DIFFERENT attention patterns
3. d_k = d_model // n_heads (split dimensions)
4. Combine all heads with W_O projection

MATRIX DIMENSIONS:
==================
Input:        (batch, seq_len, d_model)
Q/K/V heads:  (batch, n_heads, seq_len, d_k)
Attention:    (batch, n_heads, seq_len, seq_len)
Combined:     (batch, seq_len, d_model)
Output:       (batch, seq_len, d_model)

LEARNABLE PARAMETERS:
=====================
W_Q, W_K, W_V, W_O: each (d_model, d_model)
Total: 4 × d_model²

NEXT: Transformer Block (Lesson 5)
Run: python 05_transformer_block.py
""")

# =============================================================================
# EXERCISES
# =============================================================================

print("\n" + "="*70)
print("EXERCISES")
print("="*70)

print("""
1. Change n_heads and observe how attention patterns differ
2. Try without causal mask (use_causal_mask=False)
3. Visualize attention weights as heatmaps for each head
4. Compare single-head vs multi-head on same input
""")

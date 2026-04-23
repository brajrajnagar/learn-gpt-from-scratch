"""
GPT from Scratch - Lesson 4: Multi-Head Attention
==================================================

Continuing our text predictor from Lessons 1-3:
- Lesson 1: We built a neural network to predict next word
- Lesson 2: We converted "The cat" to embeddings
- Lesson 3: We learned self-attention (one way to focus)
- Lesson 4: We'll learn MULTI-HEAD attention (multiple ways to focus)

EXAMPLE FLOW: "The cat sat on the ___" → Multiple experts analyze → predict "mat"

NOTE: This lesson uses PyTorch to show LEARNABLE parameters (nn.Linear)

MATRIX DIMENSIONS WE'LL COVER:
==============================
- Input embeddings:  (batch, seq_len, d_model)
- Q, K, V:           (batch, seq_len, d_model) each
- Split heads:       (batch, n_heads, seq_len, d_k) where d_k = d_model / n_heads
- Attention scores:  (batch, n_heads, seq_len, seq_len)
- Output per head:   (batch, n_heads, seq_len, d_k)
- Combined output:   (batch, seq_len, d_model)

LEARNABLE PARAMETERS:
=====================
- W_q: nn.Linear(d_model, d_model) - Query projection, weight (d_model, d_model)
- W_k: nn.Linear(d_model, d_model) - Key projection, weight (d_model, d_model)
- W_v: nn.Linear(d_model, d_model) - Value projection, weight (d_model, d_model)
- W_o: nn.Linear(d_model, d_model) - Output projection, weight (d_model, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# RECAP: Our Text Predictor So Far
# =============================================================================

print("\n" + "="*70)
print("RECAP: Our Text Predictor")
print("="*70)
print("""
FROM LESSON 3: We built self-attention that focuses on relevant words

Single attention flow for "The cat sat on the ___":
  → "sat" attends to "cat" (the subject)
  → "on" attends to "sat" (the action)
  → "the" attends to previous words

THE LIMITATION OF SINGLE ATTENTION:
===================================

With ONE attention head, each word has only ONE way to attend.
But language is complex - words have MULTIPLE relationships!

EXAMPLE: "The cat sat on the bank"

The word "bank" has multiple interpretations:
  - Financial bank (where you deposit money)
  - River bank (edge of a river)

Single attention CANNOT capture both meanings simultaneously!
It has to choose ONE way to attend.

THE SOLUTION: Multi-Head Attention
==================================

Instead of ONE attention, we run MULTIPLE in PARALLEL:
  - Head 1: Attends to financial context
  - Head 2: Attends to geographical context
  - Head 3: Attends to grammatical structure
  - Head 4: Attends to positional patterns

Each head learns DIFFERENT patterns!
Combined, they give RICH understanding!

WHAT WE'LL BUILD:
1. Multiple attention heads (parallel experts)
2. Split embeddings among heads
3. Compute attention for each head
4. Combine all head outputs
5. Final projection for prediction
=============================================================================""")


# =============================================================================
# STEP 1: Why Multi-Head? The Limitation of Single Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Why Multi-Head Attention?")
print("="*70)

print("""
THE PROBLEM WITH SINGLE ATTENTION:
==================================

Our predictor from Lesson 3 uses ONE attention mechanism.
Each word has ONE way to attend to other words.

EXAMPLE: Processing "The cat sat on the mat"

With single attention:
  "mat" → attends to → [some weighted combination of previous words]

This is ONE perspective on what matters!

BUT language is complex. Consider what "mat" might need to know:

1. GRAMMATICAL: "mat" is object of preposition "on"
   → Should attend to "on" strongly

2. SEMANTIC: "mat" is where cats sit
   → Should attend to "cat" and "sat"

3. REFERENTIAL: "the mat" - which mat?
   → Should attend to "the"

4. POSITIONAL: "mat" comes at end of clause
   → Should attend to sentence structure

SINGLE ATTENTION CAN'T CAPTURE ALL OF THESE!
It produces ONE attention pattern, not four.

MULTI-HEAD ATTENTION FIXES THIS:
================================

Run MULTIPLE attention mechanisms in PARALLEL:

Head 0: Grammar expert
  → Learns: "mat" attends to "on" (preposition-object)

Head 1: Meaning expert
  → Learns: "mat" attends to "cat", "sat" (semantic role)

Head 2: Reference expert
  → Learns: "mat" attends to "the" (article-noun)

Head 3: Position expert
  → Learns: "mat" attends to sentence boundaries

COMBINED: Rich, multi-dimensional understanding!

Let's implement this!
""")


# =============================================================================
# STEP 2: Multi-Head Architecture - Splitting Work Among Heads
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Multi-Head Architecture")
print("="*70)

print("""
HOW MULTI-HEAD WORKS:
=====================

Instead of ONE attention computation, we run NUM_HEADS computations.

BUT: We don't just duplicate the same computation!
Each head should learn DIFFERENT patterns.

KEY INSIGHT: Split the embedding dimension among heads!

EXAMPLE: embedding_dim = 64, num_heads = 4

Single attention:
  → One head processes all 64 dimensions
  → One attention pattern

Multi-head attention:
  → Head 0 processes dimensions 0-15 (16 dims)
  → Head 1 processes dimensions 16-31 (16 dims)
  → Head 2 processes dimensions 32-47 (16 dims)
  → Head 3 processes dimensions 48-63 (16 dims)
  → Each head learns DIFFERENT patterns!
  → Combine all 4 outputs = rich representation

WHY SPLIT DIMENSIONS?
=====================

1. EFFICIENCY: Each head processes smaller vectors
   → 64-dim attention = O(64²) operations
   → 4 × 16-dim attention = 4 × O(16²) = O(1024) operations
   → Much faster!

2. DIVERSITY: Each head can specialize
   → Head 0: Grammar patterns
   → Head 1: Semantic patterns
   → Head 2: Reference patterns
   → Head 3: Positional patterns

3. CAPACITY: More total parameters
   → Single: 64 × 64 = 4096 parameters
   → Multi: 4 × (16 × 16) × 4 heads = 4096 parameters
   → Same params, more expressiveness!

THE FORMULA:
============

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

Let's implement this step by step!
""")


def create_causal_mask(seq_len):
    """
    Create causal mask - prevents seeing future tokens.
    
    MATRIX DIMENSIONS:
    ==================
    Input:  seq_len (scalar)
    Output: (1, seq_len, seq_len) for broadcasting with attention scores
    
    Visual for seq_len=4:
    ```
    [[ 0,  -inf, -inf, -inf],   ← position 0 sees only itself
     [ 0,   0,   -inf, -inf],   ← position 1 sees 0,1
     [ 0,   0,    0,   -inf],   ← position 2 sees 0,1,2
     [ 0,   0,    0,    0  ]]   ← position 3 sees all
    ```
    
    After adding to scores and softmax:
    - Positions with -inf become 0 (no attention)
    - Positions with 0 get normal softmax values
    
    Returns:
        Mask tensor, shape (1, seq_len, seq_len)
    """
    # Create mask with shape (seq_len, seq_len)
    # 1s above diagonal, 0s on and below
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
    # Add batch dimension for broadcasting: (1, seq_len, seq_len)
    return mask.unsqueeze(0)


print("\n--- Multi-Head Setup: Our Text Predictor ---")
print("-"*50)

# Configuration for our mini predictor
embedding_dim = 64  # Size of token embeddings
num_heads = 4       # Number of attention heads
head_dim = embedding_dim // num_heads  # 16 dimensions per head

print(f"Configuration:")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Num heads: {num_heads}")
print(f"  Head dim: {head_dim} (each head processes {head_dim} features)")

print(f"\nThis is like having {num_heads} experts, each analyzing {head_dim} features!")


# =============================================================================
# STEP 3: Splitting Embeddings Among Heads
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Splitting Embeddings for Each Head")
print("="*70)

print("""
HOW WE SPLIT WORK AMONG HEADS:
==============================

Each token has embedding_dim features (e.g., 64).
We have num_heads experts (e.g., 4).
Each expert processes head_dim features (e.g., 16).

STEP 1: Project embeddings to Q, K, V for ALL heads
STEP 2: Reshape to split among heads
STEP 3: Transpose to organize by head

VISUAL EXAMPLE:
===============

Input embeddings: (batch, seq_len=5, embedding_dim=64)
  → 5 tokens ("The cat sat on mat"), each with 64 features

After projection: Q, K, V each have shape (batch, 5, 64)

After split_heads:
  → Q: (batch, num_heads=4, seq_len=5, head_dim=16)
  → K: (batch, num_heads=4, seq_len=5, head_dim=16)
  → V: (batch, num_heads=4, seq_len=5, head_dim=16)

Now each head has its own Q, K, V to process!
""")


def split_heads(x, num_heads, head_dim):
    """
    Split embeddings among heads.
    
    OUR EXAMPLE: Distributing work to 4 experts
    
    INPUT: Projected Q/K/V for "The cat sat on mat"
      Shape: (batch, 5 tokens, 64 features)
      
    PROCESS:
      1. Reshape: (batch, 5, 64) → (batch, 5, 4 heads, 16 features per head)
         Splits the last dimension (64) into (4 heads × 16 features)
      
      2. Transpose: (batch, 5, 4, 16) → (batch, 4, 5, 16)
         Moves head dimension to position 1 for efficient batch processing
      
    OUTPUT: Each head has its own data to process
      Shape: (batch, 4 heads, 5 tokens, 16 features)
      
    MATRIX OPERATION:
    =================
    x.view(batch, seq_len, num_heads, head_dim)
      → Reshapes last dimension: 64 → (4, 16)
    
    x.transpose(1, 2)
      → Swaps dimensions 1 and 2: (batch, 4, seq_len, 16)
      
    Args:
        x: Input tensor, shape (batch, seq_len, embedding_dim)
        num_heads: Number of heads
        head_dim: Features per head
    
    Returns:
        Split tensor, shape (batch, num_heads, seq_len, head_dim)
    """
    batch, seq_len, _ = x.shape
    
    # Reshape: (batch, seq_len, embedding_dim) → (batch, seq_len, num_heads, head_dim)
    # Example: (2, 10, 64) → (2, 10, 4, 16)
    x = x.view(batch, seq_len, num_heads, head_dim)
    
    # Transpose: (batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
    # Example: (2, 10, 4, 16) → (2, 4, 10, 16)
    # Now we can index by head: x[:, head_idx] gives that head's data
    return x.transpose(1, 2)


def combine_heads(x, num_heads, embedding_dim):
    """
    Combine head outputs back together.
    
    OUR EXAMPLE: Gathering reports from 4 experts
    
    INPUT: Each head's output
      Shape: (batch, 4 heads, 5 tokens, 16 features)
      
    PROCESS:
      1. Transpose: (batch, 4, 5, 16) → (batch, 5, 4, 16)
         Moves head dimension back to position 2
      
      2. Reshape: (batch, 5, 4, 16) → (batch, 5, 64)
         Concatenates all heads: 4 × 16 = 64
      
    OUTPUT: Combined representation
      Shape: (batch, 5 tokens, 64 features)
      
    MATRIX OPERATION:
    =================
    x.transpose(1, 2)
      → Swaps dimensions 1 and 2: (batch, seq_len, num_heads, head_dim)
    
    x.contiguous().view(batch, seq_len, embedding_dim)
      → Reshapes last two dims: (num_heads × head_dim) → embedding_dim
      → Example: (2, 10, 4, 16) → (2, 10, 64)
      
    Args:
        x: Input tensor, shape (batch, num_heads, seq_len, head_dim)
        num_heads: Number of heads
        embedding_dim: Total features after combining (num_heads × head_dim)
    
    Returns:
        Combined tensor, shape (batch, seq_len, embedding_dim)
    """
    batch, _, seq_len, _ = x.shape
    
    # Transpose: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
    # Example: (2, 4, 10, 16) → (2, 10, 4, 16)
    x = x.transpose(1, 2)
    
    # Reshape: (batch, seq_len, num_heads, head_dim) → (batch, seq_len, embedding_dim)
    # Example: (2, 10, 4, 16) → (2, 10, 64)
    # Concatenates: num_heads × head_dim = 4 × 16 = 64
    return x.contiguous().view(batch, seq_len, embedding_dim)


print("\n--- Demo: Splitting and Combining ---")
print("-"*50)

# Create sample embeddings for "The cat sat on mat" (5 tokens)
torch.manual_seed(42)
batch_size = 1
seq_len = 5
embeddings = torch.randn(batch_size, seq_len, embedding_dim)

print(f"Input embeddings shape: {embeddings.shape}")
print(f"  → batch={batch_size}, {seq_len} tokens, {embedding_dim} features each")

# Simulate Q projection (in real model, this is W_q · embeddings)
Q = torch.randn(batch_size, seq_len, embedding_dim)

print(f"\nProjected Q shape: {Q.shape}")

# Split among heads
Q_split = split_heads(Q, num_heads, head_dim)

print(f"\nAfter split_heads: {Q_split.shape}")
print(f"  → batch={batch_size}, {num_heads} heads, {seq_len} tokens, {head_dim} features")

print(f"\nEach head's Q shape:")
for i in range(num_heads):
    print(f"  Head {i}: {Q_split[:, i].shape}")

# Combine back
Q_combined = combine_heads(Q_split, num_heads, embedding_dim)

print(f"\nAfter combine_heads: {Q_combined.shape}")
print(f"  → Back to original shape!")

print(f"\n✓ Split and combine works correctly!")


# =============================================================================
# STEP 4: Single Head Attention (What Each Head Does)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: What Each Head Computes")
print("="*70)

print("""
EACH HEAD runs its OWN attention computation!

This is the SAME scaled dot-product attention from Lesson 3,
but each head has its OWN weights and learns its OWN patterns.

SINGLE HEAD ATTENTION:
======================

For each head:
  1. Compute attention scores: Q · K^T / √(head_dim)
  2. Apply causal mask (block future)
  3. Softmax to get attention weights
  4. Weighted sum of values: weights · V

KEY POINT: Each head has DIFFERENT weights!
  → Head 0 learns: W_q^0, W_k^0, W_v^0
  → Head 1 learns: W_q^1, W_k^1, W_v^1
  → etc.

This means each head computes DIFFERENT attention!
""")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention for a single head.
    
    OUR EXAMPLE: One expert analyzing "The cat sat on mat"
    
    This expert:
      1. Creates queries (what it's looking for)
      2. Creates keys (what each word offers)
      3. Creates values (information each word carries)
      4. Matches queries to keys (attention scores)
      5. Gathers information from attended words
    
    MATRIX DIMENSIONS:
    ==================
    Inputs:
      Q: (batch, seq_len, head_dim)  - Query vectors
      K: (batch, seq_len, head_dim)  - Key vectors
      V: (batch, seq_len, head_dim)  - Value vectors
    
    Intermediate:
      scores: (batch, seq_len, seq_len)  - Raw attention scores
      attention_weights: (batch, seq_len, seq_len)  - Normalized weights
    
    Output:
      output: (batch, seq_len, head_dim)  - Contextualized representations
      attention_weights: (batch, seq_len, seq_len)  - For visualization
    
    MATRIX OPERATIONS:
    ==================
    Step 1 - Scaled Dot Product:
      Q @ K^T / sqrt(head_dim)
      (batch, seq_len, head_dim) @ (batch, head_dim, seq_len) → (batch, seq_len, seq_len)
      
      For each query position, compute similarity with all key positions.
      Higher score = more similar = more attention.
      Dividing by sqrt(head_dim) prevents large values → better gradients.
    
    Step 2 - Softmax:
      softmax(scores, dim=-1)
      (batch, seq_len, seq_len) → (batch, seq_len, seq_len)
      
      Converts scores to probabilities. Each row sums to 1.0.
      attention_weights[i, j] = "how much position i attends to position j"
    
    Step 3 - Weighted Sum:
      attention_weights @ V
      (batch, seq_len, seq_len) @ (batch, seq_len, head_dim) → (batch, seq_len, head_dim)
      
      For each query position, compute weighted sum of all value vectors.
      Positions with higher attention weights contribute more.
    
    Args:
        Q: Query, shape (batch, seq_len, head_dim)
        K: Key, shape (batch, seq_len, head_dim)
        V: Value, shape (batch, seq_len, head_dim)
        mask: Optional causal mask, shape (1, seq_len, seq_len)
    
    Returns:
        attention_output: Shape (batch, seq_len, head_dim)
        attention_weights: Shape (batch, seq_len, seq_len)
    """
    d_k = K.shape[-1]  # head_dim
    
    # Step 1: Compute attention scores
    # Q: (batch, seq_len, head_dim)
    # K^T: (batch, head_dim, seq_len) [transposed]
    # scores: (batch, seq_len, seq_len)
    # For each (query_pos, key_pos) pair, compute dot product similarity
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Step 2: Apply mask (if provided)
    # mask: (1, seq_len, seq_len) or (batch, seq_len, seq_len)
    # After masking: masked positions have very negative values (-inf)
    # After softmax: these become ~0 (no attention)
    if mask is not None:
        scores = scores + mask
    
    # Step 3: Softmax to get attention weights
    # Each row (query position) now has a probability distribution over all key positions
    # attention_weights[i, j] = "how much position i attends to position j"
    weights = F.softmax(scores, dim=-1)
    # weights shape: (batch, seq_len, seq_len)
    # Property: weights.sum(dim=-1) = 1.0 for all query positions
    
    # Step 4: Weighted sum of values
    # weights: (batch, seq_len, seq_len)
    # V: (batch, seq_len, head_dim)
    # output: (batch, seq_len, head_dim)
    # output[i] = sum_j(weights[i,j] * V[j])
    output = torch.matmul(weights, V)
    
    return output, weights


print("\n--- Demo: Single Head Attention ---")
print("-"*50)

# Create Q, K, V for one head (16-dimensional)
torch.manual_seed(42)
Q_head = torch.randn(batch_size, seq_len, head_dim)
K_head = torch.randn(batch_size, seq_len, head_dim)
V_head = torch.randn(batch_size, seq_len, head_dim)

print(f"Head input shapes:")
print(f"  Q: {Q_head.shape}")
print(f"  K: {K_head.shape}")
print(f"  V: {V_head.shape}")

# Create causal mask
mask = create_causal_mask(seq_len)

# Compute attention
output, weights = scaled_dot_product_attention(Q_head, K_head, V_head, mask)

print(f"\nOutput shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print(f"\nAttention weights (who this head attends to):")
print(f"  Each row sums to 1.0: {weights.sum(dim=-1)}")


# =============================================================================
# STEP 5: Complete Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Complete Multi-Head Attention")
print("="*70)

print("""
PUTTING IT ALL TOGETHER:
========================

Multi-head attention combines everything:
  1. Project embeddings to Q, K, V using LEARNABLE nn.Linear
  2. Split among heads
  3. Each head computes attention
  4. Combine head outputs
  5. Final projection using LEARNABLE nn.Linear

Let's implement the complete layer!
""")


class MultiHeadAttention(nn.Module):
    """
    Complete multi-head attention layer with LEARNABLE parameters.
    
    OUR EXAMPLE: Team of experts analyzing "The cat sat on mat"
    
    Think of this as managing {num_heads} experts:
      - Each expert (head) has their own specialization
      - Each expert analyzes the text independently
      - We combine all expert reports
      - Final output goes to next layer
    
    This is EXACTLY how GPT's attention works!
    
    LEARNABLE PARAMETERS:
    - W_q, W_k, W_v: Project input to Q, K, V
    - W_o: Project combined output
    
    MATRIX DIMENSIONS THROUGH FORWARD PASS:
    =======================================
    
    INPUT:
      embeddings: (batch, seq_len, embedding_dim)
      Example: (2, 10, 64) - 2 sequences, 10 tokens, 64-dim embeddings
    
    STEP 1 - LINEAR PROJECTIONS:
      W_q.weight: (embedding_dim, embedding_dim) = (64, 64)
      W_k.weight: (embedding_dim, embedding_dim) = (64, 64)
      W_v.weight: (embedding_dim, embedding_dim) = (64, 64)
      
      Q = W_q(x): (batch, seq_len, embedding_dim)
      K = W_k(x): (batch, seq_len, embedding_dim)
      V = W_v(x): (batch, seq_len, embedding_dim)
    
    STEP 2 - SPLIT HEADS:
      Q_heads: (batch, n_heads, seq_len, head_dim)
      K_heads: (batch, n_heads, seq_len, head_dim)
      V_heads: (batch, n_heads, seq_len, head_dim)
      
      Example: (2, 4, 10, 16) - 2 batches, 4 heads, 10 positions, 16-dim per head
    
    STEP 3 - ATTENTION SCORES:
      scores = Q @ K^T / sqrt(head_dim)
      Q_heads: (batch, n_heads, seq_len, head_dim)
      K_heads^T: (batch, n_heads, head_dim, seq_len)
      scores: (batch, n_heads, seq_len, seq_len)
      
      Example: (2, 4, 10, 16) @ (2, 4, 16, 10) → (2, 4, 10, 10)
    
    STEP 4 - APPLY MASK:
      mask: (1, 1, seq_len, seq_len) or (1, seq_len, seq_len)
      scores + mask: (batch, n_heads, seq_len, seq_len)
    
    STEP 5 - SOFTMAX:
      attn = softmax(scores, dim=-1)
      attn: (batch, n_heads, seq_len, seq_len)
    
    STEP 6 - ATTENTION OUTPUT:
      output = attn @ V_heads
      attn: (batch, n_heads, seq_len, seq_len)
      V_heads: (batch, n_heads, seq_len, head_dim)
      output: (batch, n_heads, seq_len, head_dim)
      
      Example: (2, 4, 10, 10) @ (2, 4, 10, 16) → (2, 4, 10, 16)
    
    STEP 7 - COMBINE HEADS:
      combined: (batch, seq_len, embedding_dim)
      Reshapes (batch, n_heads, seq_len, head_dim) → (batch, seq_len, n_heads × head_dim)
      Example: (2, 10, 64)
    
    STEP 8 - OUTPUT PROJECTION:
      output = W_o(combined)
      combined: (batch, seq_len, embedding_dim)
      W_o.weight: (embedding_dim, embedding_dim)
      output: (batch, seq_len, embedding_dim)
    """
    
    def __init__(self, embedding_dim, num_heads):
        """
        Initialize multi-head attention with LEARNABLE parameters.
        
        Args:
            embedding_dim: Size of token embeddings
            num_heads: Number of attention heads
        
        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        ───────────────────────────────────────────────────────────
        self.W_q = nn.Linear(embedding_dim, embedding_dim)  # Query projection
           Weight shape: (embedding_dim, embedding_dim)
           Bias shape: (embedding_dim,)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)  # Key projection
           Weight shape: (embedding_dim, embedding_dim)
           Bias shape: (embedding_dim,)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)  # Value projection
           Weight shape: (embedding_dim, embedding_dim)
           Bias shape: (embedding_dim,)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)  # Output projection
           Weight shape: (embedding_dim, embedding_dim)
           Bias shape: (embedding_dim,)
        
        These use Xavier/Glorot initialization by default.
        They will be UPDATED via backpropagation during training!
        
        WHY head_dim = embedding_dim // num_heads?
        ──────────────────────────────────────────
        The MULTI-HEAD attention splits the embedding_dim dimensions ACROSS the heads.
        
        Think of it like splitting a deck of cards:
          Total cards = embedding_dim = 64
          Players = num_heads = 4
          Cards per player = head_dim = 64 / 4 = 16
        
        Each head operates independently in its own head_dim-dimensional space.
        Then we concatenate all heads back together: num_heads × head_dim = embedding_dim
        
        EXAMPLE with embedding_dim=64, num_heads=4:
          head_dim = 64 // 4 = 16
          Head 1: operates in dimensions [0:16]
          Head 2: operates in dimensions [16:32]
          Head 3: operates in dimensions [32:48]
          Head 4: operates in dimensions [48:64]
        
        After attention, we concatenate: [head1, head2, head3, head4] → 4×16 = 64
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Verify embedding_dim is divisible by num_heads
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        
        print(f"Multi-Head Attention initialized:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  → {num_heads} parallel experts, each processing {self.head_dim} features")
        
        # LEARNABLE weight matrices for Q, K, V, O
        # These are nn.Linear layers - PyTorch will initialize and update them!
        # Weight shape: (embedding_dim, embedding_dim), Bias shape: (embedding_dim,)
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
        print(f"\nLEARNABLE PARAMETERS:")
        print(f"  W_q: weight {self.W_q.weight.shape} + bias {self.W_q.bias.shape}")
        print(f"  W_k: weight {self.W_k.weight.shape} + bias {self.W_k.bias.shape}")
        print(f"  W_v: weight {self.W_v.weight.shape} + bias {self.W_v.bias.shape}")
        print(f"  W_o: weight {self.W_o.weight.shape} + bias {self.W_o.bias.shape}")
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTotal learnable parameters: {total_params:,}")
        
        # Register causal mask as buffer (not a parameter, but saved with model)
        self.register_buffer('causal_mask', None)
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of multi-head attention.
        
        Args:
            embeddings: Input embeddings, shape (batch, seq_len, embedding_dim)
                        Example: (2, 10, 64) - 2 sequences, 10 tokens, 64-dim embeddings
            use_causal_mask: Whether to apply causal mask
        
        Returns:
            output: Contextualized embeddings, shape (batch, seq_len, embedding_dim)
                    Each token now has a representation that incorporates attended context.
            attention_weights: Attention weights per head, shape (batch, num_heads, seq_len, seq_len)
                    Shows which positions each head attends to.
        
        DIMENSION FLOW THROUGH FORWARD PASS:
        ====================================
        
        1. INPUT:
           embeddings: (batch, seq_len, embedding_dim) = (2, 10, 64)
        
        2. LINEAR PROJECTIONS (LEARNABLE):
           Q = self.W_q(embeddings): (batch, seq_len, embedding_dim)
           K = self.W_k(embeddings): (batch, seq_len, embedding_dim)
           V = self.W_v(embeddings): (batch, seq_len, embedding_dim)
        
        3. SPLIT HEADS:
           Q_heads: (batch, n_heads, seq_len, head_dim) = (2, 4, 10, 16)
           K_heads: (batch, n_heads, seq_len, head_dim) = (2, 4, 10, 16)
           V_heads: (batch, n_heads, seq_len, head_dim) = (2, 4, 10, 16)
        
        4. ATTENTION SCORES:
           scores = Q_heads @ K_heads^T / sqrt(head_dim)
           (2, 4, 10, 16) @ (2, 4, 16, 10) → (2, 4, 10, 10)
        
        5. APPLY MASK:
           mask: (1, seq_len, seq_len) = (1, 10, 10)
           scores + mask: (2, 4, 10, 10)
        
        6. SOFTMAX:
           attn = softmax(scores, dim=-1): (2, 4, 10, 10)
        
        7. ATTENTION OUTPUT:
           output = attn @ V_heads
           (2, 4, 10, 10) @ (2, 4, 10, 16) → (2, 4, 10, 16)
        
        8. COMBINE HEADS:
           combined: (batch, seq_len, embedding_dim) = (2, 10, 64)
        
        9. OUTPUT PROJECTION (LEARNABLE):
           output = self.W_o(combined): (2, 10, 64)
        """
        batch, seq_len, _ = embeddings.shape
        
        print(f"\n" + "="*50)
        print(f"FORWARD PASS: Multi-head attention")
        print(f"="*50)
        print(f"\nInput: {embeddings.shape}")
        print(f"  → batch={batch}, {seq_len} tokens, {self.embedding_dim} features")
        
        # Step 1: Linear projections (LEARNABLE!)
        # embeddings: (batch, seq_len, embedding_dim)
        # W_q.weight: (embedding_dim, embedding_dim)
        # Q: (batch, seq_len, embedding_dim)
        print(f"\n[Step 1] Projecting to Q, K, V (using nn.Linear)...")
        Q = self.W_q(embeddings)  # (batch, seq_len, embedding_dim)
        K = self.W_k(embeddings)  # (batch, seq_len, embedding_dim)
        V = self.W_v(embeddings)  # (batch, seq_len, embedding_dim)
        print(f"  Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        
        # Step 2: Split among heads
        # Each: (batch, seq_len, embedding_dim) → (batch, n_heads, seq_len, head_dim)
        print(f"\n[Step 2] Splitting among {self.num_heads} heads...")
        Q_heads = split_heads(Q, self.num_heads, self.head_dim)
        K_heads = split_heads(K, self.num_heads, self.head_dim)
        V_heads = split_heads(V, self.num_heads, self.head_dim)
        print(f"  Q_heads: {Q_heads.shape}")
        print(f"  → Each head gets {self.head_dim} features")
        
        # Step 3: Create causal mask
        mask = None
        if use_causal_mask:
            if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
                self.causal_mask = create_causal_mask(seq_len)
            mask = self.causal_mask[:, :seq_len, :seq_len]
            print(f"\n[Step 3] Causal mask applied (no cheating!)")
        
        # Step 4: Compute attention for each head (in parallel!)
        # Q, K, V: (batch, n_heads, seq_len, head_dim)
        print(f"\n[Step 4] Computing attention for all heads in parallel...")
        head_outputs, attention_weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask
        )
        # head_outputs: (batch, n_heads, seq_len, head_dim)
        
        print(f"  Attention computed for all {self.num_heads} heads")
        
        # Step 5: Combine heads
        # head_outputs: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, embedding_dim)
        print(f"\n[Step 5] Combining head outputs...")
        combined = combine_heads(head_outputs, self.num_heads, self.embedding_dim)
        print(f"  Combined: {combined.shape}")
        
        # Step 6: Output projection (LEARNABLE!)
        # combined: (batch, seq_len, embedding_dim)
        # W_o.weight: (embedding_dim, embedding_dim)
        # output: (batch, seq_len, embedding_dim)
        print(f"\n[Step 6] Final projection (using nn.Linear)...")
        output = self.W_o(combined)  # (batch, seq_len, embedding_dim)
        print(f"  Output: {output.shape}")
        
        return output, attention_weights


print("\n" + "="*70)
print("DEMO: Multi-Head Attention on 'The cat sat on mat'")
print("="*70)

# Create multi-head attention layer
mha = MultiHeadAttention(embedding_dim, num_heads)

# Create sample embeddings for "The cat sat on mat"
torch.manual_seed(42)
embeddings = torch.randn(batch_size, seq_len, embedding_dim)

words = ["The", "cat", "sat", "on", "mat"]
print(f"\nInput: '{' '.join(words)}'")
print(f"Embeddings: {embeddings.shape}")

# Forward pass
output, attn_weights = mha(embeddings)

print(f"\n" + "="*50)
print("RESULTS:")
print("="*50)
print(f"Output shape: {output.shape}")
print(f"  → Same as input! Each token transformed")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"  → (batch, num_heads, seq_len, seq_len)")


# =============================================================================
# STEP 6: Visualizing Attention Patterns Across Heads
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Different Heads, Different Patterns")
print("="*70)

print("""
EACH HEAD learns DIFFERENT attention patterns!

Let's examine what each head pays attention to.
""")

print("\n" + "-"*50)
print("ATTENTION PATTERNS PER HEAD:")
print("-"*50)

for head_idx in range(num_heads):
    weights = attn_weights[0, head_idx]  # (seq_len, seq_len)
    print(f"\nHEAD {head_idx}:")
    print(f"  Shape: {weights.shape}")
    print(f"  Row sums (should be 1.0): {weights.sum(dim=-1)}")
    
    # Show what each token attends to
    print(f"  Attention distribution:")
    for i, word in enumerate(words):
        row = weights[i]
        max_idx = torch.argmax(row)
        print(f"    '{word}' → max attention to '{words[max_idx]}' ({row[max_idx]*100:.1f}%)")

print(f"""
KEY OBSERVATION:
================
Different heads show DIFFERENT attention patterns!

This is the POWER of multi-head attention:
  - Head 0 might focus on: Recent context
  - Head 1 might focus: Grammatical relationships
  - Head 2 might focus: Self-attention
  - Head 3 might focus: Beginning of sequence

Each head learns to attend differently!
Combined, they capture rich patterns!
""")


# =============================================================================
# SUMMARY: Multi-Head Attention in Our Text Predictor
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Multi-Head Attention")
print("="*70)

print("""
WHAT WE BUILT:
==============
1. Multiple attention heads (parallel experts)
2. Split embeddings among heads
3. Each head computes its own attention
4. Combine all head outputs
5. Final projection for next layer

LEARNABLE PARAMETERS:
=====================
- W_q = nn.Linear(embedding_dim, embedding_dim) - Query projection
  Weight shape: (embedding_dim, embedding_dim)
- W_k = nn.Linear(embedding_dim, embedding_dim) - Key projection
  Weight shape: (embedding_dim, embedding_dim)
- W_v = nn.Linear(embedding_dim, embedding_dim) - Value projection
  Weight shape: (embedding_dim, embedding_dim)
- W_o = nn.Linear(embedding_dim, embedding_dim) - Output projection
  Weight shape: (embedding_dim, embedding_dim)

These are initialized by PyTorch and LEARNED during training!

HOW THIS CONNECTS TO OUR PREDICTOR:
===================================

Complete flow for "The cat ___":

1. INPUT: "The cat"
   ↓
2. EMBEDDINGS (Lesson 2): Token + Position vectors
   ↓
3. MULTI-HEAD ATTENTION (this lesson): Multiple experts analyze
   - Head 0: Grammar patterns
   - Head 1: Semantic patterns
   - Head 2: Reference patterns
   - Head 3: Positional patterns
   ↓
4. COMBINE: All expert reports merged
   ↓
5. NEURAL NETWORK (Lesson 1): Process combined representation
   ↓
6. OUTPUT: Word probabilities

HOW THIS CONNECTS TO GPT:
=========================

GPT-2 Small:
  - embedding_dim = 768
  - num_heads = 12
  - head_dim = 64
  - 12 parallel experts!

GPT-3 Large:
  - embedding_dim = 12288
  - num_heads = 96
  - head_dim = 128
  - 96 parallel experts!

SAME ARCHITECTURE, different scale!

NEXT: Transformer Block
=======================
Now we have multi-head attention!
Next, we add:
  - Feed-forward network (more processing)
  - Layer normalization (stable training)
  - Residual connections (gradient flow)

Together = Complete Transformer Block!

Next: 05_transformer_block.py
=============================================================================""")


print("\n" + "="*70)
print("EXERCISE: Experiment with Multi-Head Attention")
print("="*70)

print("""
Try these experiments:

1. CHANGE NUMBER OF HEADS:
   num_heads = 8  # More experts
   head_dim = 64 // 8 = 8  # Smaller per head
   
   Question: How does this affect capacity?
   Answer: More heads = more diverse patterns, but less per head

2. SINGLE HEAD (back to Lesson 3):
   num_heads = 1
   head_dim = 64
   
   Question: How is this different?
   Answer: This is exactly single attention from Lesson 3!

3. ANALYZE ATTENTION PATTERNS:
   Print attention weights for each head
   Do different heads show different patterns?
   
4. WITHOUT CAUSAL MASK:
   output, weights = mha(embeddings, use_causal_mask=False)
   
   Question: How does attention change?
   Answer: All tokens can see all tokens!

KEY TAKEAWAY:
=============
Multi-head attention = multiple experts in parallel!
- Each head learns different patterns
- Combined = rich understanding
- This is why GPT is so powerful!
=============================================================================""")
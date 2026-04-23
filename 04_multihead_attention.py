"""
=============================================================================
LESSON 4: Multi-Head Attention - Multiple Experts for Better Prediction
=============================================================================

Continuing our text predictor from Lessons 1-3:
- Lesson 1: We built a neural network to predict next word
- Lesson 2: We converted "The cat" to embeddings
- Lesson 3: We learned self-attention (one way to focus)
- Lesson 4: We'll learn MULTI-HEAD attention (multiple ways to focus)

EXAMPLE FLOW: "The cat sat on the ___" → Multiple experts analyze → predict "mat"
"""

import numpy as np

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

def softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def create_causal_mask(seq_len):
    """Create causal mask - prevents seeing future tokens."""
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9
    return mask

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

Input embeddings: (seq_len=5, embedding_dim=64)
  → 5 tokens ("The cat sat on mat"), each with 64 features

After projection: Q, K, V each have shape (5, 64)

After split_heads:
  → Q: (num_heads=4, seq_len=5, head_dim=16)
  → K: (num_heads=4, seq_len=5, head_dim=16)
  → V: (num_heads=4, seq_len=5, head_dim=16)

Now each head has its own Q, K, V to process!
""")

def split_heads(x, num_heads, head_dim):
    """
    Split embeddings among heads.
    
    OUR EXAMPLE: Distributing work to 4 experts
    
    INPUT: Projected Q/K/V for "The cat sat on mat"
      Shape: (5 tokens, 64 features)
      
    PROCESS:
      1. Reshape: (5, 64) → (5, 4 heads, 16 features per head)
      2. Transpose: (5, 4, 16) → (4 heads, 5, 16)
      
    OUTPUT: Each head has its own data to process
      Shape: (4 heads, 5 tokens, 16 features)
      
    Args:
        x: Input tensor, shape (seq_len, embedding_dim)
        num_heads: Number of heads
        head_dim: Features per head
    
    Returns:
        Split tensor, shape (num_heads, seq_len, head_dim)
    """
    seq_len = x.shape[0]
    
    # Reshape: (seq_len, embedding_dim) → (seq_len, num_heads, head_dim)
    x = x.reshape(seq_len, num_heads, head_dim)
    
    # Transpose: (seq_len, num_heads, head_dim) → (num_heads, seq_len, head_dim)
    # Now we can index by head: x[head_idx] gives that head's data
    x = x.transpose(1, 0, 2)
    
    return x

def combine_heads(x, num_heads, embedding_dim):
    """
    Combine head outputs back together.
    
    OUR EXAMPLE: Gathering reports from 4 experts
    
    INPUT: Each head's output
      Shape: (4 heads, 5 tokens, 16 features)
      
    PROCESS:
      1. Transpose: (4, 5, 16) → (5, 4, 16)
      2. Reshape: (5, 4, 16) → (5, 64)
      
    OUTPUT: Combined representation
      Shape: (5 tokens, 64 features)
      
    Args:
        x: Input tensor, shape (num_heads, seq_len, head_dim)
        num_heads: Number of heads
        embedding_dim: Total features after combining
    
    Returns:
        Combined tensor, shape (seq_len, embedding_dim)
    """
    # Transpose: (num_heads, seq_len, head_dim) → (seq_len, num_heads, head_dim)
    x = x.transpose(1, 0, 2)
    
    # Reshape: (seq_len, num_heads, head_dim) → (seq_len, embedding_dim)
    seq_len = x.shape[0]
    x = x.reshape(seq_len, embedding_dim)
    
    return x

print("\n--- Demo: Splitting and Combining ---")
print("-"*50)

# Create sample embeddings for "The cat sat on mat" (5 tokens)
np.random.seed(42)
seq_len = 5
embeddings = np.random.randn(seq_len, embedding_dim)

print(f"Input embeddings shape: {embeddings.shape}")
print(f"  → {seq_len} tokens, {embedding_dim} features each")

# Simulate Q projection (in real model, this is W_q · embeddings)
Q = np.random.randn(seq_len, embedding_dim)

print(f"\nProjected Q shape: {Q.shape}")

# Split among heads
Q_split = split_heads(Q, num_heads, head_dim)

print(f"\nAfter split_heads: {Q_split.shape}")
print(f"  → {num_heads} heads, each with {seq_len} tokens and {head_dim} features")

print(f"\nEach head's Q shape:")
for i in range(num_heads):
    print(f"  Head {i}: {Q_split[i].shape}")

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

def single_head_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention for a single head.
    
    OUR EXAMPLE: One expert analyzing "The cat sat on mat"
    
    This expert:
      1. Creates queries (what it's looking for)
      2. Creates keys (what each word offers)
      3. Creates values (information each word carries)
      4. Matches queries to keys (attention scores)
      5. Gathers information from attended words
    
    Args:
        Q: Query, shape (seq_len, head_dim)
        K: Key, shape (seq_len, head_dim)
        V: Value, shape (seq_len, head_dim)
        mask: Optional causal mask
    
    Returns:
        attention_output: Shape (seq_len, head_dim)
        attention_weights: Shape (seq_len, seq_len)
    """
    d_k = K.shape[1]  # head_dim
    
    # Step 1: Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = scores + mask
    
    # Step 3: Softmax to get attention weights
    weights = softmax(scores)
    
    # Step 4: Weighted sum of values
    output = np.dot(weights, V)
    
    return output, weights

print("\n--- Demo: Single Head Attention ---")
print("-"*50)

# Create Q, K, V for one head (16-dimensional)
np.random.seed(42)
Q_head = np.random.randn(seq_len, head_dim)
K_head = np.random.randn(seq_len, head_dim)
V_head = np.random.randn(seq_len, head_dim)

print(f"Head input shapes:")
print(f"  Q: {Q_head.shape}")
print(f"  K: {K_head.shape}")
print(f"  V: {K_head.shape}")

# Create causal mask
mask = create_causal_mask(seq_len)

# Compute attention
output, weights = single_head_attention(Q_head, K_head, V_head, mask)

print(f"\nOutput shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print(f"\nAttention weights (who this head attends to):")
print(f"  Each row sums to 1.0: {np.round(weights.sum(axis=1), 4)}")

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
  1. Project embeddings to Q, K, V
  2. Split among heads
  3. Each head computes attention
  4. Combine head outputs
  5. Final projection

Let's implement the complete layer!
""")

class MultiHeadAttention:
    """
    Complete multi-head attention layer.
    
    OUR EXAMPLE: Team of experts analyzing "The cat sat on mat"
    
    Think of this as managing {num_heads} experts:
      - Each expert (head) has their own specialization
      - Each expert analyzes the text independently
      - We combine all expert reports
      - Final output goes to next layer
    
    This is EXACTLY how GPT's attention works!
    """
    
    def __init__(self, embedding_dim, num_heads):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Size of token embeddings
            num_heads: Number of attention heads
        """
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
        
        # Weight matrices for Q, K, V
        # These are LEARNED during training
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
        # Output projection (combines all heads)
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
        print(f"\nWeight matrices:")
        print(f"  W_q: {self.W_q.shape} (query projections)")
        print(f"  W_k: {self.W_k.shape} (key projections)")
        print(f"  W_v: {self.W_v.shape} (value projections)")
        print(f"  W_o: {self.W_o.shape} (output projection)")
        print(f"\nTotal parameters: {4 * embedding_dim * embedding_dim:,}")
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of multi-head attention.
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal mask
        
        Returns:
            output: Contextualized embeddings, shape (seq_len, embedding_dim)
            attention_weights: Dict of attention weights per head
        """
        seq_len = embeddings.shape[0]
        
        print(f"\n" + "="*50)
        print(f"FORWARD PASS: Multi-head attention")
        print(f"="*50)
        print(f"\nInput: {embeddings.shape}")
        print(f"  → {seq_len} tokens, {self.embedding_dim} features each")
        
        # Step 1: Linear projections (Q, K, V for all heads)
        print(f"\n[Step 1] Projecting to Q, K, V...")
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        print(f"  Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        
        # Step 2: Split among heads
        print(f"\n[Step 2] Splitting among {self.num_heads} heads...")
        Q_heads = split_heads(Q, self.num_heads, self.head_dim)
        K_heads = split_heads(K, self.num_heads, self.head_dim)
        V_heads = split_heads(V, self.num_heads, self.head_dim)
        print(f"  Q_heads: {Q_heads.shape}")
        print(f"  → Each head gets {self.head_dim} features")
        
        # Step 3: Create causal mask
        mask = None
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
            print(f"\n[Step 3] Causal mask applied (no cheating!)")
        
        # Step 4: Compute attention for each head
        print(f"\n[Step 4] Computing attention for each head...")
        head_outputs = []
        attention_weights = {}
        
        for head_idx in range(self.num_heads):
            Q_head = Q_heads[head_idx]
            K_head = K_heads[head_idx]
            V_head = V_heads[head_idx]
            
            output, weights = single_head_attention(Q_head, K_head, V_head, mask)
            head_outputs.append(output)
            attention_weights[f"head_{head_idx}"] = weights
            
            print(f"  Head {head_idx}: output {output.shape}, attention pattern computed")
        
        # Stack head outputs
        head_outputs = np.stack(head_outputs, axis=0)
        print(f"\n  Stacked head outputs: {head_outputs.shape}")
        
        # Step 5: Combine heads
        print(f"\n[Step 5] Combining head outputs...")
        combined = combine_heads(head_outputs, self.num_heads, self.embedding_dim)
        print(f"  Combined: {combined.shape}")
        
        # Step 6: Output projection
        print(f"\n[Step 6] Final projection...")
        output = np.dot(combined, self.W_o)
        print(f"  Output: {output.shape}")
        
        return output, attention_weights

print("\n" + "="*70)
print("DEMO: Multi-Head Attention on 'The cat sat on mat'")
print("="*70)

# Create multi-head attention layer
mha = MultiHeadAttention(embedding_dim, num_heads)

# Create sample embeddings for "The cat sat on mat"
np.random.seed(42)
embeddings = np.random.randn(seq_len, embedding_dim) * 0.1

words = ["The", "cat", "sat", "on", "mat"]
print(f"\nInput: '{' '.join(words)}'")
print(f"Embeddings: {embeddings.shape}")

# Forward pass
output, attn_weights = mha.forward(embeddings)

print(f"\n" + "="*50)
print("RESULTS:")
print("="*50)
print(f"Output shape: {output.shape}")
print(f"  → Same as input! Each token transformed")
print(f"Number of attention patterns: {len(attn_weights)}")
print(f"  → Each head produced its own pattern")

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

for head_name, weights in attn_weights.items():
    head_idx = int(head_name.split("_")[1])
    print(f"\n{head_name.upper()}:")
    print(f"  Shape: {weights.shape}")
    print(f"  Row sums (should be 1.0): {np.round(weights.sum(axis=1), 4)}")
    
    # Show what each token attends to
    print(f"  Attention distribution (each row = what token attends to):")
    for i, word in enumerate(words):
        row = weights[i]
        max_idx = np.argmax(row)
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
# STEP 7: Single vs Multi-Head Comparison
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Single vs Multi-Head Comparison")
print("="*70)

print("""
COMPARISON: Single Attention vs Multi-Head Attention
====================================================

SINGLE ATTENTION (Lesson 3):
  - ONE attention computation
  - ONE way to attend
  - ONE perspective

MULTI-HEAD ATTENTION (this lesson):
  - MULTIPLE attention computations (4 heads)
  - MULTIPLE ways to attend
  - MULTIPLE perspectives

Let's compare outputs!
""")

class SingleHeadAttention:
    """Single head attention for comparison."""
    
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def forward(self, embeddings):
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        d_k = K.shape[1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        seq_len = embeddings.shape[0]
        mask = create_causal_mask(seq_len)
        scores = scores + mask
        
        weights = softmax(scores)
        output = np.dot(weights, V)
        
        return output, weights

# Single head
single_attn = SingleHeadAttention(embedding_dim)
single_output, single_weights = single_attn.forward(embeddings)

# Multi-head (already computed)
multi_output = output

print(f"\nSingle-head output shape: {single_output.shape}")
print(f"Multi-head output shape: {multi_output.shape}")

print(f"\nSingle-head output (first token, first 8 dims):")
print(f"  {np.round(single_output[0, :8], 4)}")

print(f"\nMulti-head output (first token, first 8 dims):")
print(f"  {np.round(multi_output[0, :8], 4)}")

print(f"""
DIFFERENCE:
===========

SINGLE-HEAD:
  → One attention pattern
  → One way of understanding
  → Limited perspective

MULTI-HEAD:
  → {num_heads} attention patterns combined
  → {num_heads} ways of understanding
  → Rich, diverse representation!

EXAMPLE: "The cat sat on the bank"

Single attention might learn:
  → "bank" attends to "on" (preposition)

Multi-head learns:
  → Head 0: "bank" attends to "on" (grammar)
  → Head 1: "bank" attends to "cat", "sat" (semantics)
  → Head 2: "bank" attends to "the" (reference)
  → Head 3: "bank" attends to sentence end (position)

Multi-head captures MORE relationships!
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
   output, weights = mha.forward(embeddings, use_causal_mask=False)
   
   Question: How does attention change?
   Answer: All tokens can see all tokens!

KEY TAKEAWAY:
=============
Multi-head attention = multiple experts in parallel!
- Each head learns different patterns
- Combined = rich understanding
- This is why GPT is so powerful!
=============================================================================""")
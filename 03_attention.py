"""
=============================================================================
LESSON 3: Self-Attention - The Core Magic
=============================================================================

This is the heart of GPT! Self-attention allows each word to "look at" 
other words in the sentence to understand context.

REAL-WORLD ANALOGY: Reading Comprehension
=========================================

Imagine reading: "The cat sat on the mat and it was soft."

When you read "it", you need to know what "it" refers to.
- "it" looks back at previous words
- "it" pays most attention to "mat" (not "cat", "sat", "on", "the")
- "it" updates its meaning based on "mat"

Self-attention does exactly this! Each word decides which other words
are important for understanding its meaning.

MATRIX DIMENSIONS WE'LL COVER:
==============================
- Input embeddings:  (batch, seq_len, d_model)
- Query (Q):         (batch, seq_len, d_k)
- Key (K):           (batch, seq_len, d_k)
- Value (V):         (batch, seq_len, d_v)
- Attention scores:  (batch, seq_len, seq_len)
- Attention weights: (batch, seq_len, seq_len)
- Output:            (batch, seq_len, d_v)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# STEP 1: Understanding Self-Attention with a Simple Example
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Self-Attention - The Core Idea")
print("="*70)

print("""
SIMPLE EXAMPLE: "The cat sat"
============================

For each word, we want to know: "Which other words help me understand 
myself?"

"The" (position 0):
  - Looks at: ["The", "cat", "sat"]
  - Decides: "cat" is most important (subject follows article)
  - Attention weights: [0.3, 0.5, 0.2]

"cat" (position 1):
  - Looks at: ["The", "cat", "sat"]
  - Decides: "sat" is most important (verb follows subject)
  - Attention weights: [0.2, 0.2, 0.6]

"sat" (position 2):
  - Looks at: ["The", "cat", "sat"]
  - Decides: "cat" is most important (subject of verb)
  - Attention weights: [0.1, 0.7, 0.2]

ATTENTION WEIGHTS MATRIX (3x3):
┌──────┬──────┬──────┬──────┐
│      │ The  │ cat  │ sat  │
├──────┼──────┼──────┼──────┤
│ The  │ 0.3  │ 0.5  │ 0.2  │
│ cat  │ 0.2  │ 0.2  │ 0.6  │
│ sat  │ 0.1  │ 0.7  │ 0.2  │
└──────┴──────┴──────┴──────┘

Each ROW shows how much a word attends to ALL words.
Each COLUMN shows how much ALL words attend to one word.
""")

# =============================================================================
# STEP 2: Query, Key, Value - The Three Roles
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Query, Key, Value - Three Roles")
print("="*70)

print("""
REAL-WORLD ANALOGY: Library Search
==================================

Imagine searching for books in a library:

QUERY (Q): "What am I looking for?"
  - You: "I want books about cats"
  - Your query represents what you need

KEY (K): "What does each book contain?"
  - Book 1 key: "dogs, pets, animals"
  - Book 2 key: "cats, felines, pets"
  - Book 3 key: "history, ancient"
  - Each book has a key describing its contents

VALUE (V): "What information does each book provide?"
  - Book 1 value: "Dogs are loyal pets..."
  - Book 2 value: "Cats are independent..."
  - Book 3 value: "Ancient Rome was..."
  - The actual content you want to read

ATTENTION PROCESS:
==================
1. Match QUERY with KEYS: "cats" query matches Book 2 key strongly
2. Get attention weights: [low, HIGH, low]
3. Read VALUES weighted by attention: mostly Book 2's content

IN GPT:
=======
Each word plays ALL three roles:
  - As QUERY: "What am I looking for?"
  - As KEY: "What do I offer?"
  - As VALUE: "What information do I carry?"

For "The cat sat":
  "The" is Query → matches Keys of all words → gets Values
  "cat" is Query → matches Keys of all words → gets Values
  "sat" is Query → matches Keys of all words → gets Values
""")

# =============================================================================
# STEP 3: Scaled Dot-Product Attention (The Math)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Scaled Dot-Product Attention")
print("="*70)

print("""
THE FORMULA:
============

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

STEP-BY-STEP:
=============

1. COMPUTE SCORES: Q · K^T
   - Q shape: (seq_len, d_k)
   - K^T shape: (d_k, seq_len)
   - Result: (seq_len, seq_len) - attention scores
   
   Each entry [i,j] = how much position i should attend to position j

2. SCALE: Divide by √d_k
   - Prevents dot products from getting too large
   - Keeps values in good range for softmax
   - d_k = dimension of key vectors

3. SOFTMAX: Convert to probabilities
   - Each row sums to 1.0
   - Higher score = more attention

4. WEIGHTED SUM: Multiply by V
   - V shape: (seq_len, d_v)
   - Result: (seq_len, d_v) - contextualized representation
   
   Each position gets a weighted combination of all value vectors

MATRIX DIMENSIONS:
==================
Q: (seq_len, d_k)
K: (seq_len, d_k)
V: (seq_len, d_v)

scores = Q @ K^T: (seq_len, seq_len)
weights = softmax(scores): (seq_len, seq_len)
output = weights @ V: (seq_len, d_v)
""")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    MATRIX DIMENSIONS:
    ==================
    Input:
      Q: (batch, seq_len, d_k) - Query vectors
      K: (batch, seq_len, d_k) - Key vectors
      V: (batch, seq_len, d_v) - Value vectors
      mask: Optional (seq_len, seq_len) or (1, seq_len, seq_len)
    
    Intermediate:
      scores: (batch, seq_len, seq_len) - Raw attention scores
      weights: (batch, seq_len, seq_len) - Normalized attention weights
    
    Output:
      output: (batch, seq_len, d_v) - Contextualized representations
      weights: (batch, seq_len, seq_len) - For visualization
    
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        mask: Optional mask tensor (0 or -inf)
    
    Returns:
        output, attention_weights
    """
    # Get dimension of keys
    d_k = Q.size(-1)
    
    # Step 1: Compute raw attention scores
    # Q: (batch, seq_len, d_k)
    # K^T: (batch, d_k, seq_len)
    # scores: (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by sqrt(d_k)
    # Prevents dot products from getting too large
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask if provided
    # mask: (seq_len, seq_len) with 0 (keep) or -inf (mask)
    if mask is not None:
        scores = scores + mask
    
    # Step 4: Softmax to get attention weights
    # Each row sums to 1.0
    # weights[i, j] = how much position i attends to position j
    weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    # weights: (batch, seq_len, seq_len)
    # V: (batch, seq_len, d_v)
    # output: (batch, seq_len, d_v)
    output = torch.matmul(weights, V)
    
    return output, weights


# =============================================================================
# STEP 4: Create Query, Key, Value from Input
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Creating Q, K, V from Input Embeddings")
print("="*70)

print("""
HOW TO CREATE Q, K, V:
======================

Input: Embeddings of shape (seq_len, d_model)

We create Q, K, V by multiplying with LEARNABLE weight matrices:

Q = X · W_Q    where W_Q is (d_model, d_k)
K = X · W_K    where W_K is (d_model, d_k)
V = X · W_V    where W_V is (d_model, d_v)

These weight matrices are LEARNED during training!
They tell the model HOW to transform embeddings into Q, K, V.

EXAMPLE:
========
X (embeddings): (3, 64) - 3 words, 64-dim each
W_Q: (64, 64) - learned projection
Q: (3, 64) - 3 query vectors, 64-dim each

Same for K and V.
""")


def create_qkv_projections(d_model, d_k=None, d_v=None):
    """
    Create learnable projection matrices for Q, K, V.
    
    In real GPT, these are nn.Linear layers (learnable).
    For this demo, we'll create them as parameters.
    
    MATRIX DIMENSIONS:
    ==================
    W_Q: (d_model, d_k) - projects input to Query space
    W_K: (d_model, d_k) - projects input to Key space
    W_V: (d_model, d_v) - projects input to Value space
    
    Args:
        d_model: Dimension of input embeddings
        d_k: Dimension of keys (default: d_model)
        d_v: Dimension of values (default: d_model)
    
    Returns:
        W_Q, W_K, W_V: Learnable projection matrices
    """
    if d_k is None:
        d_k = d_model
    if d_v is None:
        d_v = d_model
    
    # Initialize with small random values
    # These would be LEARNED during training
    W_Q = torch.randn(d_model, d_k) * 0.01
    W_K = torch.randn(d_model, d_k) * 0.01
    W_V = torch.randn(d_model, d_v) * 0.01
    
    return W_Q, W_K, W_V


# =============================================================================
# STEP 5: Causal (Masked) Attention for GPT
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Causal Attention - No Peeking at the Future!")
print("="*70)

print("""
WHY CAUSAL ATTENTION?
=====================

GPT predicts the NEXT token. During training:
- Position i should ONLY see positions 0 to i-1
- Position i should NOT see positions i+1 onwards

REAL-WORLD ANALOGY: Taking a Test
==================================

Question 1: What is 2+2?
Question 2: What is 3×4?
Question 3: What is 5+7?

You must answer Q1 before seeing Q2!
You must answer Q2 before seeing Q3!

CAUSAL MASK:
============

For sequence of length 4:

        pos 0   pos 1   pos 2   pos 3
pos 0: [  0   , -inf  , -inf  , -inf  ]  ← sees only itself
pos 1: [  0   ,   0   , -inf  , -inf  ]  ← sees pos 0, 1
pos 2: [  0   ,   0   ,   0   , -inf  ]  ← sees pos 0, 1, 2
pos 3: [  0   ,   0   ,   0   ,   0   ]  ← sees all (last position)

After adding -inf and applying softmax, masked positions get weight 0.
""")


def create_causal_mask(seq_len):
    """
    Create causal mask for autoregressive attention.
    
    MATRIX DIMENSIONS:
    ==================
    Input:  seq_len (scalar)
    Output: (seq_len, seq_len)
    
    The mask has:
    - 0 on and below diagonal (visible)
    - -inf above diagonal (masked)
    
    Args:
        seq_len: Length of sequence
    
    Returns:
        Mask tensor of shape (seq_len, seq_len)
    """
    # Create upper triangular matrix with -inf above diagonal
    # torch.triu with diagonal=1 gives 1s above diagonal, 0s elsewhere
    # But we need to avoid 0 * -inf = NaN, so use torch.full
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


# =============================================================================
# STEP 6: Complete Self-Attention Demo
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete Self-Attention Demonstration")
print("="*70)

# Create sample data
seq_len = 4
d_model = 8
words = ["The", "cat", "sat", "mat"]

print(f"\nSentence: '{' '.join(words)}'")
print(f"Sequence length: {seq_len}")
print(f"Embedding dimension: {d_model}")

# Create random embeddings (in real model, these come from nn.Embedding)
torch.manual_seed(42)
X = torch.randn(seq_len, d_model)

print(f"\nInput embeddings shape: {X.shape}")
print(f"  (seq_len={seq_len}, d_model={d_model})")

# Create Q, K, V projection matrices
W_Q, W_K, W_V = create_qkv_projections(d_model)

print(f"\nProjection matrices:")
print(f"  W_Q: {W_Q.shape} (d_model={d_model} → d_k={d_model})")
print(f"  W_K: {W_K.shape} (d_model={d_model} → d_k={d_model})")
print(f"  W_V: {W_V.shape} (d_model={d_model} → d_v={d_model})")

# Project to Q, K, V
Q = torch.matmul(X, W_Q)  # (seq_len, d_k)
K = torch.matmul(X, W_K)  # (seq_len, d_k)
V = torch.matmul(X, W_V)  # (seq_len, d_v)

print(f"\nAfter projection:")
print(f"  Q: {Q.shape}")
print(f"  K: {K.shape}")
print(f"  V: {V.shape}")

# Create causal mask
mask = create_causal_mask(seq_len)
print(f"\nCausal mask shape: {mask.shape}")
print(f"Mask (0=visible, -inf=masked):")
print(mask)

# Compute attention
output, weights = scaled_dot_product_attention(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), mask.unsqueeze(0))
output = output.squeeze(0)  # Remove batch dimension
weights = weights.squeeze(0)

print(f"\nAttention weights (who attends to whom):")
print(f"Shape: {weights.shape}")
print("\n" + " "*8 + " ".join([f"{w:>6s}" for w in words]))
for i, word in enumerate(words):
    row = weights[i]
    print(f"{word:>6s}: " + " ".join([f"{v.item():6.3f}" for v in row]))

print(f"\nRow sums (should be 1.0): {weights.sum(dim=-1)}")

print(f"\nOutput shape: {output.shape}")
print(f"  Each word now has a contextualized representation!")

# =============================================================================
# STEP 7: Multi-Head Attention Preview
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Multi-Head Attention (Preview)")
print("="*70)

print("""
WHY MULTIPLE HEADS?
===================

One attention head can only learn ONE type of relationship.
Multiple heads can learn DIFFERENT types:

Head 1: Subject-verb relationships
Head 2: Pronoun references
Head 3: Adjective-noun pairs
Head 4: Temporal relationships
Head 5: Semantic similarity
Head 6: Positional patterns
...

Each head has its own W_Q, W_K, W_V matrices!

FORMULA:
========
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W_O

where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

MATRIX DIMENSIONS:
==================
If we have h heads, each with d_k dimensions:
- Total d_model = h × d_k
- Each head processes d_k dimensions
- Concatenated output: h × d_k = d_model
- Final projection W_O: (d_model, d_model)
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Self-Attention")
print("="*70)

print("""
WHAT WE LEARNED:
================
1. Self-attention lets each word attend to all other words
2. Query, Key, Value are three roles each word plays
3. Attention weights = softmax(Q · K^T / √d_k)
4. Output = weighted sum of Values
5. Causal mask prevents peeking at future tokens

MATRIX DIMENSIONS:
==================
Input:     (seq_len, d_model)
Q, K:      (seq_len, d_k)
V:         (seq_len, d_v)
Scores:    (seq_len, seq_len)
Weights:   (seq_len, seq_len)
Output:    (seq_len, d_v)

KEY INSIGHT:
============
Self-attention is a DIFFERENTIABLE way to compute:
"For each position, which other positions are most relevant?"

This is why GPT works so well - it learns to focus on what matters!

NEXT: Multi-Head Attention (Lesson 4)
Run: python 04_multihead_attention.py
""")

# =============================================================================
# EXERCISES
# =============================================================================

print("\n" + "="*70)
print("EXERCISES")
print("="*70)

print("""
1. Try changing the sentence and see how attention changes:
   words = ["I", "love", "machine", "learning"]

2. Modify the causal mask to allow full attention (no masking):
   mask = torch.zeros(seq_len, seq_len)

3. Add multiple attention heads and see how they differ:
   Create 4 heads and compare their attention patterns

4. Visualize attention weights as a heatmap
""")

"""
=============================================================================
LESSON 3: Self-Attention - The Heart of Transformers
=============================================================================

This is THE most important concept in GPT! Self-attention is what allows
the model to understand relationships between words, regardless of their
distance in the sequence.

KEY CONCEPTS:
1. Attention Mechanism - How it works intuitively
2. Query, Key, Value - The three components
3. Scaled Dot-Product Attention - The math
4. Causal Masking - Making it autoregressive (GPT-specific)

ATTENTION INTUITION:
When reading "The animal didn't cross the street because it was too tired",
you know "it" refers to "animal", not "street". Attention does this!

Let's build self-attention from scratch!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding Attention Intuitively
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Understanding Attention")
print("="*70)

print("""
ATTENTION MECHANISM EXPLAINED:

Imagine you're reading: "The cat sat on the mat because it was comfortable"

When processing "it", the model needs to know what "it" refers to.
Attention allows "it" to LOOK AT other words and assign importance:

  "it" → attends to → "cat" (80% attention)
  "it" → attends to → "comfortable" (15% attention)
  "it" → attends to → "mat" (5% attention)

This is how the model learns that "it" = "cat"!

TECHNICAL DETAILS:
- Each word can attend to ALL words (including itself)
- Attention weights are learned during training
- Computed dynamically based on the input

=============================================================================""")

# =============================================================================
# STEP 2: Query, Key, Value - The Core Components
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Query, Key, Value")
print("="*70)

print("""
ATTENTION uses three vectors for each token:

1. QUERY (Q): "What am I looking for?"
   - Represents what the current token wants to attend to
   
2. KEY (K): "What do I contain?"
   - Represents what information the token offers
   - Used by other tokens to find this token
   
3. VALUE (V): "What information do I carry?"
   - The actual content/information of the token
   - Gets weighted and summed based on attention

ANALOGY: Information Retrieval System
- QUERY = Your search query
- KEY = Document keywords/tags  
- VALUE = Document content

Attention(Q, K, V) = Weighted sum of VALUES
                     where weights come from Q·K similarity

=============================================================================""")

def compute_qkv(embedding, weights_q, weights_k, weights_v):
    """
    Compute Query, Key, Value vectors for a token.
    
    Args:
        embedding: Token embedding, shape (embedding_dim,)
        weights_q: Query weight matrix, shape (embedding_dim, d_k)
        weights_k: Key weight matrix, shape (embedding_dim, d_k)
        weights_v: Value weight matrix, shape (embedding_dim, d_v)
    
    Returns:
        query, key, value vectors
    """
    query = np.dot(embedding, weights_q)
    key = np.dot(embedding, weights_k)
    value = np.dot(embedding, weights_v)
    
    return query, key, value

# Example: Compute QKV for a single token
print("\n--- QKV Example ---")

np.random.seed(42)
embedding_dim = 8  # Input embedding size
d_k = d_v = 4  # QKV dimension (often smaller than embedding_dim)

# Token embedding
token_embedding = np.random.randn(embedding_dim)

# Learnable weight matrices
W_q = np.random.randn(embedding_dim, d_k)
W_k = np.random.randn(embedding_dim, d_k)
W_v = np.random.randn(embedding_dim, d_v)

query, key, value = compute_qkv(token_embedding, W_q, W_k, W_v)

print(f"Token embedding shape: {token_embedding.shape}")
print(f"Query shape: {query.shape}")
print(f"Key shape: {key.shape}")
print(f"Value shape: {value.shape}")

# =============================================================================
# STEP 3: Scaled Dot-Product Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Scaled Dot-Product Attention")
print("="*70)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention scores and weighted values.
    
    Args:
        Q: Query matrix, shape (seq_len, d_k)
        K: Key matrix, shape (seq_len, d_k)
        V: Value matrix, shape (seq_len, d_v)
        mask: Optional mask for causal attention
    
    Returns:
        attention_output: Weighted sum of values, shape (seq_len, d_v)
        attention_weights: Attention scores, shape (seq_len, seq_len)
    """
    d_k = K.shape[1]
    
    # Step 1: Compute attention scores (Q · K^T)
    # This measures similarity between each query and key
    scores = np.dot(Q, K.T)
    
    # Step 2: Scale by sqrt(d_k) - prevents softmax saturation
    # When d_k is large, dot products can be very large
    # Scaling keeps gradients stable
    scores = scores / np.sqrt(d_k)
    
    print(f"  Raw scores shape: {scores.shape}")
    print(f"  Raw scores (first row): {scores[0]}")
    
    # Step 3: Apply mask (for causal attention in GPT)
    if mask is not None:
        scores = scores + mask  # Add -inf to masked positions
    
    # Step 4: Softmax to get attention weights
    # Converts scores to probabilities (sum to 1)
    attention_weights = softmax(scores)
    
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Attention weights (first row): {attention_weights[0]}")
    
    # Step 5: Weighted sum of values
    # Each output is a combination of all values, weighted by attention
    attention_output = np.dot(attention_weights, V)
    
    print(f"  Output shape: {attention_output.shape}")
    
    return attention_output, attention_weights

def softmax(x):
    """
    Numerically stable softmax.
    
    Args:
        x: Input array (can be 1D or 2D)
    
    Returns:
        Softmax output (probabilities that sum to 1)
    """
    # Subtract max for numerical stability
    # Prevents overflow when computing exp(large_number)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("\n--- Computing Attention for a Sequence ---")

# Create a sequence of 5 tokens, each with 8-dimensional embedding
np.random.seed(42)
seq_len = 5
embedding_dim = 8
d_k = d_v = 4

# Simulate embeddings for 5 tokens
embeddings = np.random.randn(seq_len, embedding_dim)
print(f"Input embeddings shape: {embeddings.shape}")

# Compute Q, K, V for all tokens at once
W_q = np.random.randn(embedding_dim, d_k)
W_k = np.random.randn(embedding_dim, d_k)
W_v = np.random.randn(embedding_dim, d_v)

Q = np.dot(embeddings, W_q)  # Shape: (seq_len, d_k)
K = np.dot(embeddings, W_k)  # Shape: (seq_len, d_k)
V = np.dot(embeddings, W_v)  # Shape: (seq_len, d_v)

print(f"\nQuery matrix shape: {Q.shape}")
print(f"Key matrix shape: {K.shape}")
print(f"Value matrix shape: {V.shape}")

# Compute attention
print("\nComputing scaled dot-product attention:")
attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\nAttention weights matrix:")
print(attention_weights)

print("\n" + "-"*70)
print("INTERPRETING ATTENTION WEIGHTS:")
print("-"*70)
print("""
Each ROW shows what a token attends to:
- Row 0: What token 0 attends to
- Row 1: What token 1 attends to
- etc.

Each COLUMN shows what attends to a token:
- Col 0: How much other tokens attend to token 0
- Col 1: How much other tokens attend to token 1
- etc.

Higher weight = more attention = more influence
=============================================================================""")

# =============================================================================
# STEP 4: Causal Masking (GPT-Specific)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Causal Masking for Autoregressive Generation")
print("="*70)

print("""
WHY CAUSAL MASKING?

GPT is trained to predict the NEXT token. During training:
- Token 0 can only see token 0
- Token 1 can only see tokens 0, 1
- Token 2 can only see tokens 0, 1, 2
- etc.

This prevents "cheating" - seeing future tokens!

CAUSAL MASK:
  [[ 0, -inf, -inf, -inf, -inf],
   [ 0,   0, -inf, -inf, -inf],
   [ 0,   0,   0, -inf, -inf],
   [ 0,   0,   0,   0, -inf],
   [ 0,   0,   0,   0,   0  ]]

After softmax, -inf becomes 0 (no attention)
""")

def create_causal_mask(seq_len):
    """
    Create a causal (triangular) mask.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Mask matrix where future positions have -inf
    """
    mask = np.zeros((seq_len, seq_len))
    
    # Upper triangle (future positions) gets -inf
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9  # Large negative number (becomes ~0 after softmax)
    
    return mask

print("\n--- Causal Mask Example ---")
seq_len = 5
causal_mask = create_causal_mask(seq_len)

print(f"Causal mask for sequence length {seq_len}:")
print(causal_mask)

print("\nNow computing attention WITH causal mask:")
attention_output_masked, attention_weights_masked = scaled_dot_product_attention(
    Q, K, V, mask=causal_mask
)

print(f"\nMasked attention weights:")
print(attention_weights_masked)

print("\n" + "-"*70)
print("NOTICE:")
print("-"*70)
print("""
With causal mask:
- Token 0 only attends to itself (100%)
- Token 1 attends to tokens 0 and 1
- Token 2 attends to tokens 0, 1, 2
- etc.

Each token can ONLY see itself and PREVIOUS tokens!
This is essential for autoregressive generation.
=============================================================================""")

# =============================================================================
# STEP 5: Complete Self-Attention Layer
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Complete Self-Attention Layer")
print("="*70)

class SelfAttention:
    """
    Self-attention layer with causal masking.
    This is the core of GPT!
    """
    
    def __init__(self, embedding_dim, d_k, d_v):
        """
        Initialize self-attention.
        
        Args:
            embedding_dim: Input embedding dimension
            d_k: Dimension for query and key
            d_v: Dimension for value
        """
        self.embedding_dim = embedding_dim
        self.d_k = d_k
        self.d_v = d_v
        
        # Learnable weight matrices
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_k = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_v = np.random.randn(embedding_dim, d_v) * 0.1
        
        print(f"Self-Attention initialized")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Q/K dim: {d_k}")
        print(f"  V dim: {d_v}")
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of self-attention.
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal mask (True for GPT)
        
        Returns:
            attention_output: Shape (seq_len, d_v)
            attention_weights: Shape (seq_len, seq_len)
        """
        seq_len = embeddings.shape[0]
        
        # Compute Q, K, V
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
        
        # Compute attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask
        )
        
        return attention_output, attention_weights

print("\n--- Self-Attention Layer Example ---")

# Create self-attention layer
self_attn = SelfAttention(embedding_dim=8, d_k=4, d_v=4)

# Create sample embeddings (5 tokens, 8 dimensions each)
embeddings = np.random.randn(5, 8)

# Forward pass
output, weights = self_attn.forward(embeddings)

print(f"\nInput shape: {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print("\nAttention weights (who attends to whom):")
print(weights)

# =============================================================================
# STEP 6: Understanding the Output
# =============================================================================

print("\n" + "="*70)
print("STEP 6: What Does Self-Attention Output Mean?")
print("="*70)

print("""
SELF-ATTENTION OUTPUT:

Each output vector is a CONTEXTUALIZED representation of a token.

Key insight: The output for each token now CONTAINS INFORMATION 
from the tokens it attended to!

Example: "The cat sat on the mat"

After self-attention:
- "cat" embedding now contains info about "The" (it attended to it)
- "sat" embedding contains info about "The", "cat"
- "the" (second) contains info about "The", "cat", "sat", "on"

This is how GPT builds understanding!

In GPT specifically:
1. Each token predicts the next token
2. Causal mask ensures it can only use previous tokens
3. Self-attention lets it focus on relevant previous tokens
4. The output is used to predict the next word

=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Self-Attention")
print("="*70)

print("""
SELF-ATTENTION FORMULA:

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

WHERE:
- Q = Query = What am I looking for?
- K = Key = What do I offer?
- V = Value = What information do I carry?
- d_k = Key dimension (for scaling)

CAUSAL MASK:
- Essential for autoregressive language modeling
- Prevents attending to future tokens
- Lower triangular mask with -inf

WHY IT WORKS:
- Learns relationships between tokens
- Captures long-range dependencies
- Parallelizable (unlike RNNs)

LIMITATION OF SINGLE ATTENTION:
- Each token only has ONE way to attend
- Can't capture different types of relationships

SOLUTION: Multi-Head Attention (next lesson!)
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Attention")
print("="*70)

print("""
Try these:

1. Change sequence length:
   seq_len = 10  # Longer sequence

2. Change dimensions:
   d_k = 8, d_v = 8  # Larger QKV dimensions

3. Without causal mask:
   self_attn.forward(embeddings, use_causal_mask=False)
   How does attention change?

4. Analyze attention patterns:
   - Which tokens get the most attention?
   - How does the mask affect the weights?

Key Takeaway:
- Self-attention lets tokens attend to each other
- QKV mechanism computes attention weights
- Causal mask makes it autoregressive (GPT-specific)
- Output is contextualized token representations

Next: 04_multihead_attention.py - Multiple attention heads!
=============================================================================""")
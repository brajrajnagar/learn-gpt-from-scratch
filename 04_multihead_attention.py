"""
=============================================================================
LESSON 4: Multi-Head Attention - Multiple Perspectives
=============================================================================

Single attention is powerful, but GPT uses MULTI-HEAD attention.
This is like having multiple attention mechanisms working in parallel!

KEY CONCEPTS:
1. Why Multi-Head? - Multiple representation subspaces
2. Multiple Heads - Parallel attention computations
3. Concatenation + Projection - Combining head outputs
4. Implementation - Building the complete layer

WHY MULTI-HEAD?

Think of it like reading a sentence with different focus:
- Head 1: Attends to syntactic relationships (grammar)
- Head 2: Attends to semantic relationships (meaning)
- Head 3: Attends to coreferences (pronoun references)
- Head 4: Attends to positional patterns

Each head learns DIFFERENT patterns!

Let's build multi-head attention!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Why Multi-Head Attention?")
print("="*70)

print("""
SINGLE ATTENTION LIMITATION:

With one attention head, each token has only ONE way to attend 
to other tokens. But language is complex!

Example: "The cat sat on the mat because it was comfortable"

Different relationships to capture:
- "it" → "cat" (coreference - what does "it" refer to?)
- "comfortable" → "cat" (semantic - what is comfortable?)
- "on" → "mat" (syntactic - preposition-object)
- "sat" → "cat" (subject-verb relationship)

MULTI-HEAD SOLUTION:

Instead of one attention computation, we do MULTIPLE in parallel!
Each head has its own Q, K, V weight matrices.

GPT-2 Small: 12 heads
GPT-3 Large: 96 heads

=============================================================================""")

# =============================================================================
# STEP 2: Multi-Head Architecture
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Multi-Head Architecture")
print("="*70)

print("""
MULTI-HEAD ATTENTION FLOW:

1. INPUT: embeddings, shape (seq_len, embedding_dim)
              e.g., (5, 512) for 5 tokens, 512-dim embeddings

2. LINEAR PROJECTIONS (for each head):
   - Project embedding to smaller dimension for each head
   - If embedding_dim=512 and num_heads=8, each head gets 512/8=64 dims
   
3. PARALLEL ATTENTION (for each head):
   - Compute Q, K, V for this head
   - Compute attention: softmax(QK^T/√d_k)·V
   - Output shape: (seq_len, head_dim)
   
4. CONCATENATE:
   - Concatenate all head outputs
   - Shape: (seq_len, num_heads * head_dim) = (seq_len, embedding_dim)
   
5. OUTPUT PROJECTION:
   - Final linear layer to mix head information
   - Output shape: (seq_len, embedding_dim)

FORMULA:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

=============================================================================""")

# =============================================================================
# STEP 3: Implementing Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Implementing Multi-Head Attention")
print("="*70)

def softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def create_causal_mask(seq_len):
    """Create causal (triangular) mask."""
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9
    return mask

class MultiHeadAttention:
    """
    Multi-head attention layer.
    This is the exact implementation used in GPT!
    """
    
    def __init__(self, embedding_dim, num_heads):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_heads: Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Each head works with a smaller dimension
        # Total dimension is split across heads
        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads"
        
        self.head_dim = embedding_dim // num_heads
        
        print(f"Multi-Head Attention initialized")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        
        # Weight matrices for Q, K, V (project to all heads at once)
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
        # Output projection matrix (concatenate heads → output dim)
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Split tensor, shape (num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[0]
        # Reshape: (seq_len, embedding_dim) → (seq_len, num_heads, head_dim)
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        # Transpose: (seq_len, num_heads, head_dim) → (num_heads, seq_len, head_dim)
        x = x.transpose(1, 0, 2)
        return x
    
    def _combine_heads(self, x):
        """
        Combine heads back together.
        
        Args:
            x: Input, shape (num_heads, seq_len, head_dim)
        
        Returns:
            Combined tensor, shape (seq_len, embedding_dim)
        """
        # Transpose: (num_heads, seq_len, head_dim) → (seq_len, num_heads, head_dim)
        x = x.transpose(1, 0, 2)
        # Reshape: (seq_len, num_heads, head_dim) → (seq_len, embedding_dim)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.embedding_dim)
        return x
    
    def _single_head_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention for a single head.
        
        Args:
            Q: Query, shape (seq_len, head_dim)
            K: Key, shape (seq_len, head_dim)
            V: Value, shape (seq_len, head_dim)
            mask: Optional causal mask
        
        Returns:
            attention_output: Shape (seq_len, head_dim)
            attention_weights: Shape (seq_len, seq_len)
        """
        d_k = K.shape[1]
        
        # Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax to get weights
        weights = softmax(scores)
        
        # Weighted sum of values
        output = np.dot(weights, V)
        
        return output, weights
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of multi-head attention.
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal mask
        
        Returns:
            output: Shape (seq_len, embedding_dim)
            attention_weights: Dict of attention weights per head
        """
        seq_len = embeddings.shape[0]
        
        # Step 1: Linear projections
        Q = np.dot(embeddings, self.W_q)  # (seq_len, embedding_dim)
        K = np.dot(embeddings, self.W_k)  # (seq_len, embedding_dim)
        V = np.dot(embeddings, self.W_v)  # (seq_len, embedding_dim)
        
        # Step 2: Split into heads
        Q_heads = self._split_heads(Q)  # (num_heads, seq_len, head_dim)
        K_heads = self._split_heads(K)  # (num_heads, seq_len, head_dim)
        V_heads = self._split_heads(V)  # (num_heads, seq_len, head_dim)
        
        # Step 3: Create causal mask (same for all heads)
        mask = None
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
        
        # Step 4: Compute attention for each head (in parallel conceptually)
        head_outputs = []
        attention_weights = {}
        
        for head_idx in range(self.num_heads):
            Q_head = Q_heads[head_idx]  # (seq_len, head_dim)
            K_head = K_heads[head_idx]  # (seq_len, head_dim)
            V_head = V_heads[head_idx]  # (seq_len, head_dim)
            
            # Compute attention
            output, weights = self._single_head_attention(Q_head, K_head, V_head, mask)
            head_outputs.append(output)
            attention_weights[f"head_{head_idx}"] = weights
        
        # Stack head outputs: (num_heads, seq_len, head_dim)
        head_outputs = np.stack(head_outputs, axis=0)
        
        # Step 5: Combine heads
        combined = self._combine_heads(head_outputs)  # (seq_len, embedding_dim)
        
        # Step 6: Output projection
        output = np.dot(combined, self.W_o)  # (seq_len, embedding_dim)
        
        return output, attention_weights

# =============================================================================
# STEP 4: Example Usage
# =============================================================================

print("\n--- Multi-Head Attention Example ---")

# Parameters (similar to GPT-2 small)
embedding_dim = 64  # Using smaller for demo (GPT-2 uses 768)
num_heads = 4       # GPT-2 small uses 12
seq_len = 6

print(f"\nConfiguration:")
print(f"  Sequence length: {seq_len}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Number of heads: {num_heads}")
print(f"  Head dimension: {embedding_dim // num_heads}")

# Create multi-head attention layer
mha = MultiHeadAttention(embedding_dim, num_heads)

# Create sample embeddings
np.random.seed(42)
embeddings = np.random.randn(seq_len, embedding_dim)
print(f"\nInput embeddings shape: {embeddings.shape}")

# Forward pass
output, attn_weights = mha.forward(embeddings)

print(f"\nOutput shape: {output.shape}")
print(f"Number of attention weight matrices: {len(attn_weights)}")

# =============================================================================
# STEP 5: Visualizing Attention Patterns
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Attention Patterns Across Heads")
print("="*70)

print("\nEach head learns different attention patterns!")
print("Let's look at the attention weights for each head:\n")

for head_name, weights in attn_weights.items():
    print(f"{head_name} attention weights (shape {weights.shape}):")
    print(f"  Row sums (should be 1.0): {weights.sum(axis=1)}")
    print(f"  Mean attention per position: {weights.mean(axis=0)}")
    print()

print("-"*70)
print("NOTICE:")
print("-"*70)
print("""
- Each head produces DIFFERENT attention patterns
- Some heads might focus on nearby tokens
- Some heads might have more uniform attention
- Some heads might learn specific patterns (like attending to verbs)

This diversity is the power of multi-head attention!
=============================================================================""")

# =============================================================================
# STEP 6: Comparing Single vs Multi-Head
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Single vs Multi-Head Comparison")
print("="*70)

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

print("\nComparing single-head vs multi-head:")

# Single head
single_attn = SingleHeadAttention(embedding_dim)
single_output, single_weights = single_attn.forward(embeddings)

# Multi-head (already computed above)
multi_output = output

print(f"\nSingle-head output shape: {single_output.shape}")
print(f"Multi-head output shape: {multi_output.shape}")

print(f"\nSingle-head output (first token, first 5 dims):")
print(f"  {single_output[0, :5]}")

print(f"\nMulti-head output (first token, first 5 dims):")
print(f"  {multi_output[0, :5]}")

print("\n" + "-"*70)
print("KEY DIFFERENCE:")
print("-"*70)
print("""
Single-head: One way to attend, one set of patterns
Multi-head: Multiple perspectives, diverse patterns

Multi-head allows the model to:
1. Capture different types of relationships simultaneously
2. Learn specialized attention patterns per head
3. Have more representational capacity

This is why all modern LLMs use multi-head attention!
=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Multi-Head Attention")
print("="*70)

print("""
MULTI-HEAD ATTENTION STEPS:

1. PROJECT: embeddings → Q, K, V using weight matrices
2. SPLIT: Divide Q, K, V into num_heads parts
3. ATTEND: Compute attention for each head independently
4. COMBINE: Concatenate all head outputs
5. PROJECT: Final linear layer to mix information

FORMULA:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

PARAMETERS:
- W_q, W_k, W_v: embedding_dim × embedding_dim each
- W_o: embedding_dim × embedding_dim
- Total: 4 × embedding_dim² parameters

EXAMPLE (GPT-2 Small):
- embedding_dim = 768
- num_heads = 12
- head_dim = 768/12 = 64
- Parameters: 4 × 768² ≈ 2.4 million

NEXT: We'll combine multi-head attention with feed-forward networks
      to build the complete Transformer Block!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Multi-Head Attention")
print("="*70)

print("""
Try these:

1. Change number of heads:
   mha = MultiHeadAttention(embedding_dim=64, num_heads=8)
   How does head_dim change?

2. Compare attention patterns:
   - Look at attention_weights for different heads
   - Do they show different patterns?

3. Without causal mask:
   mha.forward(embeddings, use_causal_mask=False)
   How does attention change?

4. Scale up:
   embedding_dim = 128, num_heads = 8
   What's the head dimension now?

Key Takeaway:
- Multi-head = multiple attention computations in parallel
- Each head learns different attention patterns
- Outputs are concatenated and projected
- This gives the model more representational power!

Next: 05_transformer_block.py - Complete transformer block!
=============================================================================""")
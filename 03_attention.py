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

NOTE: This lesson uses PyTorch to show LEARNABLE parameters (nn.Linear)

MATRIX DIMENSIONS WE'LL COVER:
==============================
- Query (Q):     (batch, seq_len, d_k)
- Key (K):       (batch, seq_len, d_k)
- Value (V):     (batch, seq_len, d_v)
- Attention:     (batch, n_heads, seq_len, seq_len)
- Output:        (batch, seq_len, d_model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# STEP 1: Scaled Dot-Product Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Scaled Dot-Product Attention")
print("="*70)

print("""
THE CORE ATTENTION MECHANISM
============================

COMPUTE:
─────────
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

WHERE:
──────
Q (Query):    "What am I looking for?"   shape: (batch, seq_q, d_k)
K (Key):      "What do I contain?"       shape: (batch, seq_k, d_k)
V (Value):    "What info do I carry?"    shape: (batch, seq_k, d_v)

STEPS:
──────
1. Compute similarity: scores = Q @ K^T / sqrt(d_k)
   shape: (batch, seq_q, seq_k)
   
2. Apply softmax: attention_weights = softmax(scores, axis=-1)
   Each row sums to 1.0 → these are "attention weights"
   
3. Weighted sum: output = attention_weights @ V
   shape: (batch, seq_q, d_v)

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
""")


class ScaledDotProductAttention(nn.Module):
    """
    The fundamental attention operation.
    
    MATRIX DIMENSIONS:
    ==================
    Inputs:
      Q: (batch, seq_q, d_k)  - Query vectors
      K: (batch, seq_k, d_k)  - Key vectors
      V: (batch, seq_k, d_v)  - Value vectors
    
    Intermediate:
      scores: (batch, seq_q, seq_k)  - Attention scores before softmax
      attention_weights: (batch, seq_q, seq_k)  - Normalized weights
    
    Output:
      output: (batch, seq_q, d_v)  - Weighted sum of values
      attention_weights: (batch, seq_q, seq_k)  - For visualization
    
    MATRIX OPERATIONS:
    ==================
    Step 1 - Scaled Dot Product:
      Q @ K^T / sqrt(d_k)
      (batch, seq_q, d_k) @ (batch, d_k, seq_k) → (batch, seq_q, seq_k)
      
      For each query position, compute similarity with all key positions.
      Higher score = more similar = more attention.
    
    Step 2 - Softmax:
      softmax(scores, dim=-1)
      (batch, seq_q, seq_k) → (batch, seq_q, seq_k)
      
      Converts scores to probabilities. Each row sums to 1.0.
    
    Step 3 - Weighted Sum:
      attention_weights @ V
      (batch, seq_q, seq_k) @ (batch, seq_k, d_v) → (batch, seq_q, d_v)
      
      For each query position, compute weighted sum of all value vectors.
      Positions with higher attention weights contribute more.
    """
    
    def __init__(self):
        super().__init__()
        # No learnable parameters in basic attention
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            Q: Query tensor, shape (batch, seq_q, d_k)
               Example: (2, 10, 16) - 2 sequences, 10 query positions, 16-dim keys
            K: Key tensor, shape (batch, seq_k, d_k)
               Example: (2, 10, 16) - 2 sequences, 10 key positions, 16-dim keys
            V: Value tensor, shape (batch, seq_k, d_v)
               Example: (2, 10, 16) - 2 sequences, 10 value positions, 16-dim values
            mask: Optional mask tensor, shape (seq_q, seq_k) or (batch, 1, seq_q, seq_k)
                  Where mask[i,j] = -inf means "don't attend to position j from position i"
        
        Returns:
            output: Attention output, shape (batch, seq_q, d_v)
                    Each query position now has a d_v-dimensional vector
                    that incorporates information from attended key positions.
            attention_weights: Softmax weights, shape (batch, seq_q, seq_k)
                    Shows how much each position attends to every other position.
        
        DIMENSION FLOW:
        ===============
        Input Q: (batch, seq_q, d_k)
             ↓
        scores = Q @ K^T / sqrt(d_k)
             ↓
        scores: (batch, seq_q, seq_k)
             ↓
        attention_weights = softmax(scores)
             ↓
        attention_weights: (batch, seq_q, seq_k)
             ↓
        output = attention_weights @ V
             ↓
        Output: (batch, seq_q, d_v)
        """
        batch_size = Q.shape[0]
        d_k = Q.shape[-1]
        
        # Step 1: Compute raw attention scores
        # Q: (batch, seq_q, d_k)
        # K^T: (batch, d_k, seq_k)  [transposed]
        # scores: (batch, seq_q, seq_k)
        # For each (query_pos, key_pos) pair, compute dot product similarity
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: (batch, seq_q, seq_k)
        
        # Apply mask if provided (set masked positions to -inf)
        # This blocks attention to certain positions (e.g., future tokens in GPT)
        if mask is not None:
            # mask: (seq_q, seq_k) or (batch, 1, seq_q, seq_k)
            # After masking: masked positions have very negative values
            # After softmax: these become ~0 (no attention)
            scores = scores + mask
        
        # Step 2: Softmax → attention weights
        # Each row (query position) now has a probability distribution over all key positions
        # attention_weights[i, j] = "how much position i attends to position j"
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch, seq_q, seq_k)
        # Property: attention_weights.sum(dim=-1) = 1.0 for all query positions
        
        # Step 3: Weighted sum of values
        # For each query position, compute weighted sum of all value vectors
        # attention_weights: (batch, seq_q, seq_k)
        # V: (batch, seq_k, d_v)
        # output: (batch, seq_q, d_v)
        # output[i] = sum_j(attention_weights[i,j] * V[j])
        output = torch.matmul(attention_weights, V)
        # output shape: (batch, seq_q, d_v)
        
        return output, attention_weights


# =============================================================================
# STEP 2: Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Multi-Head Attention")
print("="*70)

print("""
MULTI-HEAD ATTENTION: Run attention multiple times in parallel.

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
   using LEARNABLE linear layers (nn.Linear)
2. Split into multiple heads (each head operates in d_k dimensions)
3. Apply scaled dot-product attention to EACH head
4. Concatenate all heads
5. Project back to d_model using LEARNABLE linear layer

DIMENSIONS:
───────────
Input:  (batch, seq_len, d_model)
Where:  d_model = n_heads × d_k

LEARNABLE PARAMETERS:
─────────────────────
W_q: d_model → d_model  (query projection)
W_k: d_model → d_model  (key projection)
W_v: d_model → d_model  (value projection)
W_o: d_model → d_model  (output projection)

These are initialized by PyTorch and LEARNED during training!

GPT NOTE:
─────────
GPT uses MASKED multi-head attention in the decoder.
The mask prevents positions from attending to future positions.
""")


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Run attention multiple times in parallel.
    
    MATRIX DIMENSIONS THROUGH FORWARD PASS:
    =======================================
    
    INPUT:
      x: (batch, seq_len, d_model)
      Example: (2, 10, 64) - 2 sequences, 10 tokens, 64-dim embeddings
    
    STEP 1 - LINEAR PROJECTIONS:
      W_q.weight: (d_model, d_model) = (64, 64)
      W_k.weight: (d_model, d_model) = (64, 64)
      W_v.weight: (d_model, d_model) = (64, 64)
      
      Q = W_q(x): (batch, seq_len, d_model)
      K = W_k(x): (batch, seq_len, d_model)
      V = W_v(x): (batch, seq_len, d_model)
    
    STEP 2 - SPLIT HEADS:
      Q_heads: (batch, n_heads, seq_len, d_k)
      K_heads: (batch, n_heads, seq_len, d_k)
      V_heads: (batch, n_heads, seq_len, d_k)
      
      Example: (2, 4, 10, 16) - 2 batches, 4 heads, 10 positions, 16-dim per head
    
    STEP 3 - ATTENTION SCORES:
      scores = Q @ K^T / sqrt(d_k)
      Q_heads: (batch, n_heads, seq_len, d_k)
      K_heads^T: (batch, n_heads, d_k, seq_len)
      scores: (batch, n_heads, seq_len, seq_len)
      
      Example: (2, 4, 10, 16) @ (2, 4, 16, 10) → (2, 4, 10, 10)
      Each of 10 positions gets a score over all 10 positions, for each head.
    
    STEP 4 - APPLY MASK:
      mask: (1, 1, seq_len, seq_len)
      scores + mask: (batch, n_heads, seq_len, seq_len)
      Upper triangle becomes -inf (blocked after softmax)
    
    STEP 5 - SOFTMAX:
      attn = softmax(scores, dim=-1)
      attn: (batch, n_heads, seq_len, seq_len)
      Each row (for each head, each query position) sums to 1.0
    
    STEP 6 - ATTENTION OUTPUT:
      output = attn @ V_heads
      attn: (batch, n_heads, seq_len, seq_len)
      V_heads: (batch, n_heads, seq_len, d_k)
      output: (batch, n_heads, seq_len, d_k)
      
      Example: (2, 4, 10, 10) @ (2, 4, 10, 16) → (2, 4, 10, 16)
    
    STEP 7 - COMBINE HEADS:
      combined: (batch, seq_len, d_model)
      Reshapes (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads × d_k)
      Example: (2, 10, 64)
    
    STEP 8 - OUTPUT PROJECTION:
      output = W_o(combined)
      combined: (batch, seq_len, d_model)
      W_o.weight: (d_model, d_model)
      output: (batch, seq_len, d_model)
    """
    
    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention with LEARNABLE parameters.
        
        Args:
            d_model: Dimension of input/output embeddings
            n_heads: Number of attention heads
        
        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        ───────────────────────────────────────────────────────────
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
           Weight shape: (d_model, d_model)
           Bias shape: (d_model,)
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
           Weight shape: (d_model, d_model)
           Bias shape: (d_model,)
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
           Weight shape: (d_model, d_model)
           Bias shape: (d_model,)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
           Weight shape: (d_model, d_model)
           Bias shape: (d_model,)
        
        These use Xavier/Glorot initialization by default.
        They will be UPDATED via backpropagation during training!
        
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
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Same as d_k in standard attention
        
        # LEARNABLE linear projections for Q, K, V and output
        # These are initialized by PyTorch and updated during training!
        # Weight shape: (d_model, d_model), Bias shape: (d_model,)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention module (no learnable params)
        self.attention = ScaledDotProductAttention()
        
        # Causal mask buffer (registered, not learnable)
        self.register_buffer('causal_mask', None)
    
    def _split_heads(self, x):
        """
        Split the last dimension into (n_heads, d_k) and transpose.
        
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
               Example: (2, 10, 64)
        
        Returns:
            Split tensor, shape (batch, n_heads, seq_len, d_k)
            Example: (2, 4, 10, 16)
        
        MATRIX OPERATION:
        =================
        Step 1 - Reshape:
          (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
          Example: (2, 10, 64) → (2, 10, 4, 16)
          
          Splits d_model=64 into n_heads=4 groups of d_k=16 dimensions.
        
        Step 2 - Transpose:
          (batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
          Example: (2, 10, 4, 16) → (2, 4, 10, 16)
          
          Moves n_heads dimension to position 1 for efficient batch processing.
        
        This allows each head to operate independently on its d_k-dimensional subspace.
        """
        batch, seq_len, _ = x.shape
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        # Transpose: (batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def _combine_heads(self, x):
        """
        Merge the heads back together.
        
        Args:
            x: Input tensor, shape (batch, n_heads, seq_len, d_k)
               Example: (2, 4, 10, 16)
        
        Returns:
            Combined tensor, shape (batch, seq_len, d_model)
            Example: (2, 10, 64)
        
        MATRIX OPERATION:
        =================
        Step 1 - Transpose:
          (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
          Example: (2, 4, 10, 16) → (2, 10, 4, 16)
        
        Step 2 - Reshape (concatenate):
          (batch, seq_len, n_heads, d_k) → (batch, seq_len, n_heads × d_k)
          Example: (2, 10, 4, 16) → (2, 10, 64)
          
          Concatenates all head outputs: 4 heads × 16 dims = 64 dims total.
        """
        batch, _, seq_len, _ = x.shape
        # Transpose: (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape: (batch, seq_len, n_heads, d_k) → (batch, seq_len, n_heads × d_k)
        return x.contiguous().view(batch, seq_len, self.d_model)
    
    def forward(self, x, use_causal_mask=True):
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
               Example: (2, 10, 64) - 2 sequences, 10 tokens, 64-dim embeddings
            use_causal_mask: Whether to apply causal mask (for GPT)
        
        Returns:
            output: shape (batch, seq_len, d_model)
                    Contextualized representations after multi-head attention.
            attention_weights: shape (batch, n_heads, seq_len, seq_len)
                    Attention weights for each head, useful for visualization.
        
        DIMENSION FLOW THROUGH FORWARD PASS:
        ====================================
        
        1. INPUT:
           x: (batch, seq_len, d_model) = (2, 10, 64)
        
        2. LINEAR PROJECTIONS (LEARNABLE):
           Q = self.W_q(x): (batch, seq_len, d_model)
           K = self.W_k(x): (batch, seq_len, d_model)
           V = self.W_v(x): (batch, seq_len, d_model)
        
        3. SPLIT HEADS:
           Q_heads: (batch, n_heads, seq_len, d_k) = (2, 4, 10, 16)
           K_heads: (batch, n_heads, seq_len, d_k) = (2, 4, 10, 16)
           V_heads: (batch, n_heads, seq_len, d_k) = (2, 4, 10, 16)
        
        4. ATTENTION SCORES:
           scores = Q_heads @ K_heads^T / sqrt(d_k)
           (2, 4, 10, 16) @ (2, 4, 16, 10) → (2, 4, 10, 10)
        
        5. APPLY MASK:
           mask: (1, 1, seq_len, seq_len) = (1, 1, 10, 10)
           scores + mask: (2, 4, 10, 10)
        
        6. SOFTMAX:
           attn = softmax(scores, dim=-1): (2, 4, 10, 10)
        
        7. ATTENTION OUTPUT:
           output = attn @ V_heads
           (2, 4, 10, 10) @ (2, 4, 10, 16) → (2, 4, 10, 16)
        
        8. COMBINE HEADS:
           combined: (batch, seq_len, d_model) = (2, 10, 64)
        
        9. OUTPUT PROJECTION (LEARNABLE):
           output = self.W_o(combined): (2, 10, 64)
        """
        batch, seq_len, _ = x.shape
        
        # Step 1: Linear projections (LEARNABLE!)
        # x: (batch, seq_len, d_model)
        # W_q.weight: (d_model, d_model)
        # Q: (batch, seq_len, d_model)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # Step 2: Split into multiple heads
        # Each: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        Q_heads = self._split_heads(Q)  # (batch, n_heads, seq_len, d_k)
        K_heads = self._split_heads(K)  # (batch, n_heads, seq_len, d_k)
        V_heads = self._split_heads(V)  # (batch, n_heads, seq_len, d_k)
        
        # Create causal mask if needed
        if use_causal_mask:
            # Create or get cached causal mask
            if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
                self.causal_mask = self._create_causal_mask(seq_len)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
        else:
            mask = None
        
        # Step 3 & 4: Apply attention to all heads in parallel
        # Q, K, V: (batch, n_heads, seq_len, d_k)
        output, attention_weights = self.attention.forward(Q_heads, K_heads, V_heads, mask)
        # output: (batch, n_heads, seq_len, d_k)
        # attention_weights: (batch, n_heads, seq_len, seq_len)
        
        # Step 5: Combine heads
        # output: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        combined = self._combine_heads(output)  # (batch, seq_len, d_model)
        
        # Step 6: Output projection (LEARNABLE!)
        # combined: (batch, seq_len, d_model)
        # W_o.weight: (d_model, d_model)
        # output: (batch, seq_len, d_model)
        output = self.W_o(combined)  # (batch, seq_len, d_model)
        
        return output, attention_weights
    
    def _create_causal_mask(self, seq_len):
        """
        Create causal (triangular) mask for autoregressive generation.
        
        GPT uses this mask to prevent positions from attending to FUTURE positions.
        
        MASK SHAPE:
        ===========
        Returns: (1, 1, seq_len, seq_len) for broadcasting with 
                 (batch, n_heads, seq_len, seq_len)
        
        Visual representation for seq_len=4:
        ```
        Position 0: [0,  -inf, -inf, -inf]  ← can only see itself
        Position 1: [0,  0,    -inf, -inf]  ← can see 0 and 1
        Position 2: [0,  0,    0,    -inf]  ← can see 0, 1, 2
        Position 3: [0,  0,    0,    0   ]  ← can see all
        ```
        
        After adding to scores and applying softmax:
        - Positions with -inf become 0 (no attention)
        - Positions with 0 get normal softmax values
        
        Returns:
            Mask tensor, shape (1, 1, seq_len, seq_len) for broadcasting
        """
        # Create upper triangular mask (1s above diagonal, 0s on and below)
        # Then convert to additive mask: 1 becomes -inf, 0 stays 0
        # mask: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask * float('-inf')
        # Add batch and head dimensions for broadcasting
        # (seq_len, seq_len) → (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Demonstration Functions
# =============================================================================

def demonstrate_scaled_dot_product_attention():
    """Show how scaled dot-product attention works step by step."""
    print("=" * 70)
    print("SCALED DOT-PRODUCT ATTENTION DEMO")
    print("=" * 70)
    print()
    
    batch_size = 1
    seq_len = 4
    d_k = 8
    
    # Create simple Q, K, V
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    print(f"Input shapes:")
    print(f"  Q (queries):  {Q.shape}")
    print(f"  K (keys):     {K.shape}")
    print(f"  V (values):   {V.shape}")
    print()
    
    attention = ScaledDotProductAttention()
    output, attention_weights = attention(Q, K, V)
    
    print(f"Output shapes:")
    print(f"  output:            {output.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    print("-" * 70)
    words = ["[BOS]", "the", "cat", "sat"]
    print(f"  Words: {words}")
    print()
    for i, word in enumerate(words):
        print(f"  '{word}' attends to:")
        weights_i = attention_weights[0, i]
        for j, (w, wt) in enumerate(zip(words, weights_i)):
            bar = "█" * int(wt.item() * 40)
            print(f"    → {w:8s}: {wt.item():.3f} {bar}")
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
    
    # Show learnable parameters
    print(f"LEARNABLE PARAMETERS:")
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"  Total: {total_params:,} parameters")
    print(f"  W_q: {mha.W_q.weight.shape} + {mha.W_q.bias.shape}")
    print(f"  W_k: {mha.W_k.weight.shape} + {mha.W_k.bias.shape}")
    print(f"  W_v: {mha.W_v.weight.shape} + {mha.W_v.bias.shape}")
    print(f"  W_o: {mha.W_o.weight.shape} + {mha.W_o.bias.shape}")
    print()
    
    seq_len = 5
    batch_size = 1
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Configuration:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_k = d_model / n_heads = {d_k}")
    print()
    
    print(f"Input shape:")
    print(f"  x: {x.shape}")
    print()
    
    output, attention_weights = mha(x, use_causal_mask=True)
    
    print(f"Output shapes:")
    print(f"  output:            {output.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    print(f"    (batch, n_heads, seq_len, seq_len)")
    print()
    
    print(f"Attention weights for head 0 (causal - lower triangular):")
    print(f"  {attention_weights[0, 0]}")
    print()
    
    print(f"Each head learns DIFFERENT attention patterns:")
    for h in range(n_heads):
        weights = attention_weights[0, h]
        # For each position, show which position gets most attention
        max_positions = torch.argmax(weights, dim=-1)
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
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
    print(f"Causal mask (lower triangular):")
    print(f"  0 = visible, -inf = masked (future)")
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
    
    batch_size = 1
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attn = mha(x, use_causal_mask=True)
    
    print("With causal mask, attention weights are:")
    for i in range(seq_len):
        weights = attn[0, 0, i]
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
    batch_size = 1
    
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, attn = mha(x, use_causal_mask=False)  # No causal mask for visualization
    
    words = ["[BOS]", "the", "cat", "sat", "on", "mat"]
    
    print(f"Self-attention patterns for: {' '.join(words)}")
    print(f"(Without causal mask - full attention)")
    print()
    
    for h in range(n_heads):
        print(f"Head {h}:")
        for i in range(seq_len):
            weights = attn[0, h, i]
            # Show which positions get most attention
            max_idx = torch.argmax(weights)
            marker = "←" if i == max_idx else "  "
            bar = "█" * int(weights[max_idx].item() * 30)
            print(f"  {marker} {words[i]:8s} → {words[max_idx]:8s} ({weights[max_idx].item():.2%}) {bar}")
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
    print("✓ LEARNABLE parameters: W_q, W_k, W_v, W_o (nn.Linear layers)")
    print("✓ Causal mask: Prevents GPT from looking ahead (autoregressive)")
    print()
    print("LEARNABLE PARAMETERS:")
    print("  W_q, W_k, W_v, W_o: nn.Linear(d_model, d_model)")
    print("  Weight shape: (d_model, d_model), Bias shape: (d_model,)")
    print("  These are initialized by PyTorch and updated via backprop!")
    print()
    print("KEY FORMULAS:")
    print("  scores = Q @ K^T / sqrt(d_k)  →  (batch, seq_q, seq_k)")
    print("  attention = softmax(scores) @ V  →  (batch, seq_q, d_v)")
    print("  multi_head = concat(head_1, ..., head_n) @ W_O  →  (batch, seq_len, d_model)")
    print()
    print("DIMENSION SUMMARY:")
    print("  Input:        (batch, seq_len, d_model)")
    print("  Q, K, V:      (batch, seq_len, d_model) each")
    print("  Split heads:  (batch, n_heads, seq_len, d_k)")
    print("  Attention:    (batch, n_heads, seq_len, seq_len)")
    print("  Output:       (batch, seq_len, d_model)")
    print()
    print("NEXT: Transformer Block (attention + feed-forward + residual + layernorm)")
    print("Run: python 05_transformer_block.py")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
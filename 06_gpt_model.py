"""
GPT from Scratch - Lesson 6: Complete GPT Model
================================================

Now we assemble all components into the complete GPT model!

ARCHITECTURE OVERVIEW (from the original GPT paper):
====================================================

    ┌─────────────────────────────────────────────────────────┐
    │                    GPT MODEL                            │
    │                                                         │
    │  Input Text ────► Token IDs                             │
    │                      │                                  │
    │                      ▼                                  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ Token Embedding + Position Embedding             │  │
    │  │   (d_model dimensional vectors)                  │  │
    │  └──────────────────────────────────────────────────┘  │
    │                      │                                  │
    │                      ▼                                  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │           Transformer Block 1                    │  │
    │  │   ┌─────────────────────────────────────────┐   │  │
    │  │   │ Multi-Head Self-Attention               │   │  │
    │  │   │ + Residual + LayerNorm                  │   │  │
    │  │   │ Feed-Forward Network                    │   │  │
    │  │   │ + Residual + LayerNorm                  │   │  │
    │  │   └─────────────────────────────────────────┘   │  │
    │  └──────────────────────────────────────────────────┘  │
    │                      │                                  │
    │                      ▼                                  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │           Transformer Block 2                    │  │
    │  │   (Same structure as Block 1)                    │  │
    │  └──────────────────────────────────────────────────┘  │
    │                      │                                  │
    │                      ▼                                  │
    │                   ... (more blocks) ...                │
    │                      │                                  │
    │                      ▼                                  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ Layer Normalization                              │  │
    │  └──────────────────────────────────────────────────┘  │
    │                      │                                  │
    │                      ▼                                  │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │ Linear Projection → Softmax → Next Token         │  │
    │  └──────────────────────────────────────────────────┘  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

KEY HYPERPARAMETERS (matching Transformer repo naming):
========================================================
- d_model: Dimension of embeddings (the "width" of the model)
- n_heads: Number of attention heads  
- n_blocks: Number of transformer blocks (the "depth")
- d_ff: Hidden dimension of feed-forward network
- d_k, d_v: Dimension per head for keys/values

GPT is a "decoder-only" transformer because:
- It only uses causal (masked) self-attention
- It predicts the next token (autoregressive)
- No encoder (unlike original Transformer for translation)

NOTE: This lesson uses PyTorch to show LEARNABLE parameters (nn.Linear, nn.Embedding)

Let's build the complete GPT model!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# STEP 1: GPT Configuration Class (matching TransformerConfig pattern)
# =============================================================================

print("\n" + "="*70)
print("STEP 1: GPT Configuration Class")
print("="*70)

print("""
WHY A CONFIG CLASS?
===================
Just like the Transformer repo uses TransformerConfig, we use GPTConfig to:
1. Centralize all hyperparameters in one place
2. Make it easy to create different model variants
3. Enable saving/loading configurations
4. Provide clear documentation of all parameters

REAL-WORLD EXAMPLE: Restaurant Blueprint
========================================
Before building a restaurant, you need a blueprint:
- How many tables? (n_blocks = depth)
- How many chefs per station? (n_heads = width)
- How large is the menu? (vocab_size)
- How detailed are descriptions? (d_model)

The config is this blueprint!
""")


class GPTConfig:
    """
    Configuration class for GPT model, matching the pattern from Transformer repo.
    
    This centralizes all hyperparameters in one place, making it easy to:
    - Create different model variants
    - Save/load configurations
    - Compare model sizes
    
    Attributes:
        vocab_size: Size of vocabulary (e.g., 50,257 for GPT-2)
        d_model: Dimension of embedding (the "width" of the model)
        n_heads: Number of attention heads
        n_blocks: Number of transformer blocks (the "depth" of the model)
        d_ff: Hidden dimension of feed-forward network
        dropout: Dropout rate (for future use with training)
        max_sequence_length: Maximum sequence length the model handles
    
    Derived Attributes (computed from above):
        d_k: Dimension per head for Query and Key (d_model // n_heads)
        d_v: Dimension per head for Value (d_model // n_heads)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,    # Size of vocabulary
        d_model: int = 64,          # Dimension of embedding (the "width")
        n_heads: int = 4,           # Number of attention heads
        n_blocks: int = 2,          # Number of transformer blocks (the "depth")
        d_ff: int = 256,            # Hidden dimension of feed-forward network
        dropout: float = 0.1,       # Dropout rate (for future use)
        max_sequence_length: int = 100,  # Max sequence length
    ):
        """
        Initialize GPT configuration.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embedding (the "width" of the model)
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks (the "depth" of the model)
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout rate
            max_sequence_length: Maximum sequence length the model handles
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        
        # Sanity checks (matching Transformer repo)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads  # Dimension per head (for Q, K)
        self.d_v = d_model // n_heads  # Dimension per head (for V)
    
    def __repr__(self):
        return (
            f"GPTConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_blocks={self.n_blocks},\n"
            f"  d_ff={self.d_ff},\n"
            f"  d_k={self.d_k},\n"
            f"  d_v={self.d_v},\n"
            f"  dropout={self.dropout},\n"
            f")"
        )


# Show example config
print("\n" + "-"*70)
print("Example GPT Configuration (matching Transformer repo style):")
print("-"*70)

config = GPTConfig(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_blocks=2,
    d_ff=256,
    dropout=0.1,
    max_sequence_length=100
)
print(config)

print("\n" + "-"*70)
print("Configuration Explained:")
print("-"*70)
print(f"""
vocab_size={config.vocab_size}
  → Number of unique tokens in vocabulary
  → GPT-2 uses 50,257 (BPE tokens)
  → We use 1,000 for demonstration

d_model={config.d_model}
  → Dimension of embeddings (the "width" of the model)
  → GPT-2 uses 768 (small) to 1600 (XL) [CORRECTED]
  → We use 64 for demonstration

n_heads={config.n_heads}
  → Number of attention heads (specialists per layer)
  → GPT-2 uses 12 (small) to 25 (XL) [CORRECTED]
  → We use 4 for demonstration

n_blocks={config.n_blocks}
  → Number of transformer blocks (the "depth")
  → GPT-2 uses 12 (small) to 96 (XL)
  → We use 2 for demonstration

d_ff={config.d_ff}
  → Hidden dimension of feed-forward network
  → Typically 4x d_model (so 64*4=256) ✓
  → GPT-2 uses 3072 (4*768)

d_k=d_v={config.d_k}
  → Dimension per attention head
  → Computed as d_model // n_heads
  → Each head focuses on {config.d_k} dimensions

dropout={config.dropout}
  → Regularization rate (10% dropout)
  → Helps prevent overfitting

max_sequence_length={config.max_sequence_length}
  → Maximum tokens the model can handle
  → GPT-2 supports 1024 tokens
  → We use 100 for demonstration
""")


# =============================================================================
# STEP 2: Helper Functions
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Helper Functions")
print("="*70)

print("""
CAUSAL MASK = "No Peeking" Rule
-------------------------------
Prevents tokens from seeing future tokens (like taking a test where
you can only see questions you've already answered)
""")


def create_causal_mask(seq_len):
    """
    Create causal (triangular) mask.
    
    Prevents positions from attending to future positions.
    
    WHY CAUSAL MASK?
    ================
    GPT is trained to predict the NEXT token. During training, each token
    should ONLY see tokens that came BEFORE it, not after. This is called
    "causal" because cause must precede effect!
    
    REAL-WORLD ANALOGY: Taking a Test
    ==================================
    Imagine taking a multiple-choice test:
    
    GOOD (causal): You answer questions 1, 2, 3 in order.
                   Each answer only uses knowledge from earlier questions.
    
    BAD (not causal): You could peek at future questions before answering.
                      This would give unfair advantage!
    
    In GPT training, we must prevent "peeking" at future tokens!
    
    HOW THE MASK WORKS:
    ====================
    
    Example: Sequence "The cat sat" (3 tokens)
    
    WITHOUT MASK (bad - allows peeking):
    ┌─────────┬─────┬─────┬─────┐
    │ attends │ The │ cat │ sat │
    ├─────────┼─────┼─────┼─────┤
    │ The     │  ✓  │  ✗  │  ✗  │  ← "The" can only see itself
    │ cat     │  ✓  │  ✓  │  ✗  │  ← "cat" can see "The" and itself
    │ sat     │  ✓  │  ✓  │  ✓  │  ← "sat" can see ALL (BAD!)
    └─────────┴─────┴─────┴─────┘
    
    Wait! The third row shows "sat" can see future tokens relative to
    earlier positions. We need to BLOCK this!
    
    WITH CAUSAL MASK:
    ┌─────────┬─────┬─────┬─────┐
    │ attends │ The │ cat │ sat │
    ├─────────┼─────┼─────┼─────┤
    │ The     │  ✓  │  ✗  │  ✗  │  ← position 0 sees only itself
    │ cat     │  ✓  │  ✓  │  ✗  │  ← position 1 sees 0 and itself
    │ sat     │  ✓  │  ✓  │  ✓  │  ← position 2 sees all past
    └─────────┴─────┴─────┴─────┘
    
    THE MASK MATRIX (3x3):
    ┌──────┬──────┬──────┬──────┐
    │      │  0   │  1   │  2   │
    ├──────┼──────┼──────┼──────┤
    │  0   │  0   │ -inf  │ -inf │  ← "The" blocked from cat, sat
    │  1   │  0   │  0    │ -inf │  ← "cat" blocked from sat
    │  2   │  0   │  0    │  0   │  ← "sat" sees all (it's last!)
    └──────┴──────┴──────┴──────┘
    
    -inf (negative infinity) blocks attention!
    After softmax: exp(-inf) = 0, so attention weight = 0
    
    VISUAL: Lower Triangular Matrix
    ┌──────────────────┐
    │ █ ░ ░ ░ ░ ░ ░ ░ ░│  ← Row 0: only position 0 visible
    │ █ █ ░ ░ ░ ░ ░ ░ ░│  ← Row 1: positions 0,1 visible
    │ █ █ █ ░ ░ ░ ░ ░ ░│  ← Row 2: positions 0,1,2 visible
    │ █ █ █ █ ░ ░ ░ ░ ░│
    │ █ █ █ █ █ ░ ░ ░ ░│
    └──────────────────┘
    █ = can attend (0)
    ░ = blocked (-inf)
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Mask tensor where future positions have -inf (effectively zero after softmax)
    """
    # Create mask matrix with -inf in upper triangle
    # This blocks attention to future tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
    return mask


# =============================================================================
# STEP 3: Embedding Layers with nn.Embedding (LEARNABLE)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Layers (with nn.Embedding)")
print("="*70)

print("""
TOKEN EMBEDDING: Convert token IDs to dense vectors (LEARNABLE)
POSITION EMBEDDING: Add position information (LEARNABLE)

Combined: final_embedding = token_embedding + position_embedding

Both use nn.Embedding - PyTorch's learnable embedding layer!
""")


class TokenEmbedding(nn.Module):
    """
    Token embedding layer using nn.Embedding (LEARNABLE).
    
    This is a learnable look-up table:
      - Input: tensor of token IDs, shape (batch, seq_len)
      - Output: tensor of embeddings, shape (batch, seq_len, d_model)
    
    LEARNABLE PARAMETERS:
    - self.embedding.weight: shape (vocab_size, d_model)
      This is the actual embedding table, learned during training!
    
    Example:
      vocab_size = 1000
      d_model = 64
      
      token_ids = [10, 25, 67]  (3 tokens)
      embeddings = lookup[token_ids]  → shape (3, 64)
    """
    
    def __init__(self, vocab_size, d_model):
        """
        Initialize token embedding with LEARNABLE parameters.
        
        Args:
            vocab_size: Number of unique tokens
            d_model: Dimension of embedding vectors
        
        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        - self.embedding = nn.Embedding(vocab_size, d_model)
          Weight shape: (vocab_size, d_model)
          This is the embedding table that gets learned!
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        print(f"TokenEmbedding created:")
        print(f"  vocab_size = {vocab_size}, d_model = {d_model}")
        print(f"  LEARNABLE PARAMETERS:")
        print(f"    embedding.weight: {self.embedding.weight.shape}")
        print(f"    Total: {vocab_size * d_model:,} parameters")
    
    def forward(self, token_ids):
        """
        Get embeddings for token IDs.
        
        Args:
            token_ids: Tensor of token IDs, shape (batch, seq_len)
        
        Returns:
            Token embeddings, shape (batch, seq_len, d_model)
        """
        return self.embedding(token_ids)


class PositionEmbedding(nn.Module):
    """
    Position embedding layer using nn.Embedding (LEARNABLE).
    
    Each position in the sequence gets a unique learnable embedding.
    
    LEARNABLE PARAMETERS:
    - self.embedding.weight: shape (max_seq_len, d_model)
      This is the position embedding table, learned during training!
    
    Example:
      max_seq_len = 100
      d_model = 64
      
      position 0 → [0.01, -0.02, ...] (64-dim, LEARNED)
      position 1 → [0.02, -0.01, ...] (64-dim, LEARNED)
      position 2 → [0.03, 0.01, ...] (64-dim, LEARNED)
    """
    
    def __init__(self, max_seq_len, d_model):
        """
        Initialize position embedding with LEARNABLE parameters.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of embedding vectors
        
        LEARNABLE PARAMETERS (automatically initialized by PyTorch):
        - self.embedding = nn.Embedding(max_seq_len, d_model)
          Weight shape: (max_seq_len, d_model)
          These position vectors are learned during training!
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
        print(f"PositionEmbedding created:")
        print(f"  max_seq_len = {max_seq_len}, d_model = {d_model}")
        print(f"  LEARNABLE PARAMETERS:")
        print(f"    embedding.weight: {self.embedding.weight.shape}")
        print(f"    Total: {max_seq_len * d_model:,} parameters")
    
    def forward(self, seq_len):
        """
        Get position embeddings for positions 0 to seq_len-1.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Position embeddings, shape (seq_len, d_model)
        """
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long)
        return self.embedding(positions)


# =============================================================================
# STEP 4: Complete GPT Model with nn.Module
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Complete GPT Model (with nn.Module)")
print("="*70)

print("""
Now we build the complete GPT model using PyTorch's nn.Module!

All parameters are LEARNABLE via:
- nn.Embedding for token and position embeddings
- nn.Linear for attention projections and output
- nn.LayerNorm for normalization
""")


class GPT(nn.Module):
    """
    Complete GPT Model using PyTorch nn.Module.
    
    This is a decoder-only transformer (like GPT-2/3) that predicts next tokens.
    
    Architecture:
        1. Token Embedding (nn.Embedding) + Position Embedding (nn.Embedding)
        2. Stacked Transformer Blocks (with nn.Linear for attention/FFN)
        3. Layer Normalization (nn.LayerNorm)
        4. Output Projection (nn.Linear) → Vocabulary logits
    
    ALL PARAMETERS ARE LEARNABLE!
    
    LEARNABLE PARAMETERS:
    - Token Embedding: vocab_size × d_model
    - Position Embedding: max_seq_len × d_model
    - Per Transformer Block:
      - Attention: 4 × d_model² (Q, K, V, output projections)
      - FFN: 2 × d_model × d_ff (two linear layers)
      - LayerNorm: 2 × d_model (weight, bias) × 2 norms
    - Output: d_model × vocab_size
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize GPT model with LEARNABLE parameters.
        
        Args:
            config: Optional GPTConfig object
            **kwargs: Individual hyperparameters (if config not provided)
        
        LEARNABLE PARAMETERS (all initialized by PyTorch):
        - Token Embedding: nn.Embedding(vocab_size, d_model)
        - Position Embedding: nn.Embedding(max_seq_len, d_model)
        - Per Transformer Block:
          - Attention: nn.Linear layers for Q, K, V, output
          - FFN: nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
          - LayerNorm: nn.LayerNorm (learnable weight/bias)
        - Final LayerNorm: nn.LayerNorm(d_model)
        - Output: nn.Linear(d_model, vocab_size)
        """
        super().__init__()
        
        # Use config if provided, otherwise create from kwargs
        if config is not None:
            self.config = config
        else:
            self.config = GPTConfig(**kwargs)
        
        # Extract config values
        vocab_size = self.config.vocab_size
        max_seq_len = self.config.max_sequence_length
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        n_blocks = self.config.n_blocks
        d_ff = self.config.d_ff
        
        print(f"\n{'='*50}")
        print(f"GPT Model Configuration")
        print(f"{'='*50}")
        print(f"  vocab_size = {vocab_size}")
        print(f"  d_model = {d_model}")
        print(f"  n_heads = {n_heads}")
        print(f"  n_blocks = {n_blocks}")
        print(f"  d_ff = {d_ff}")
        print(f"  d_k = d_v = {self.config.d_k}")
        print(f"  max_seq_len = {max_seq_len}")
        print(f"{'='*50}")
        
        # LEARNABLE Embedding layers (nn.Embedding)
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(max_seq_len, d_model)
        
        # LEARNABLE Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_blocks)
        ])
        
        for i, block in enumerate(self.blocks):
            print(f"  Block {i+1}/{n_blocks} created")
        
        # LEARNABLE Final layer norm (nn.LayerNorm)
        self.ln_final = nn.LayerNorm(d_model)
        
        # LEARNABLE Output projection (nn.Linear)
        self.output = nn.Linear(d_model, vocab_size)
        
        print(f"\n  Final LayerNorm: nn.LayerNorm({d_model})")
        print(f"  Output projection: nn.Linear({d_model}, {vocab_size})")
        
        # Print total parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n  Total LEARNABLE parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"{'='*50}")
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        Args:
            token_ids: Input token IDs, shape (batch, seq_len)
        
        Returns:
            logits: Output logits, shape (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape
        
        # Get token embeddings (LEARNABLE)
        token_embs = self.token_embedding(token_ids)  # (batch, seq_len, d_model)
        
        # Get position embeddings (LEARNABLE)
        pos_embs = self.position_embedding(seq_len)  # (seq_len, d_model)
        
        # Combine: broadcast position embeddings to batch
        x = token_embs + pos_embs  # (batch, seq_len, d_model)
        
        # Pass through transformer blocks (LEARNABLE)
        for i, block in enumerate(self.blocks):
            x = block(x)
        
        # Final layer norm (LEARNABLE)
        x = self.ln_final(x)
        
        # Output projection (LEARNABLE)
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            token_ids: Input token IDs, shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token IDs, shape (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop sequence if too long
            idx_cond = token_ids[:, -self.config.max_sequence_length:]
            
            # Forward pass
            logits = self(idx_cond)
            
            # Get last token's logits
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from softmax distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            token_ids = torch.cat((token_ids, idx_next), dim=1)
        
        return token_ids


# =============================================================================
# STEP 5: Transformer Block (Minimal Version for Standalone Execution)
# =============================================================================

# Minimal TransformerBlock for standalone execution
# (In the full project, this imports from lesson 5)
class TransformerBlock(nn.Module):
        """Minimal Transformer block for standalone execution."""
        
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.attention = MultiHeadAttention(d_model, n_heads)
            self.ffn = FeedForwardNetwork(d_model, d_ff)
        
        def forward(self, x):
            x = x + self.attention(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x


# Minimal MultiHeadAttention for standalone execution
class MultiHeadAttention(nn.Module):
    """Minimal MultiHeadAttention for standalone execution."""
    
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
    
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)
        
        # Create causal mask
        if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
            self.causal_mask = mask.unsqueeze(0).unsqueeze(0)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        d_k = Q_heads.shape[-1]
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, V_heads)
        output = self._combine_heads(output)
        output = self.W_o(output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Minimal FeedForwardNetwork for standalone execution."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# =============================================================================
# STEP 6: Example Usage and Demonstration
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Example Usage")
print("="*70)

# Create GPT model with config
config = GPTConfig(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_blocks=2,
    d_ff=256,
    max_sequence_length=128
)

gpt = GPT(config=config)

print("\n" + "-"*70)
print("Processing sample input...")
print("-"*70)

# Create sample input
torch.manual_seed(42)
batch_size = 1
seq_len = 5
input_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
print(f"Input tokens: {input_tokens}")
print(f"Input shape: {input_tokens.shape}")

# Forward pass
logits = gpt(input_tokens)
print(f"\nOutput logits shape: {logits.shape}")
print(f"  → (batch={batch_size}, seq_len={seq_len}, vocab_size={config.vocab_size})")

# Get predictions for last position
last_logits = logits[:, -1, :]  # (batch, vocab_size)
probs = F.softmax(last_logits, dim=-1)

# Top predictions
top_values, top_indices = torch.topk(probs[0], k=5)
print(f"\nTop 5 predictions for next token:")
for idx, (val, token_id) in enumerate(zip(top_values, top_indices)):
    print(f"  {idx+1}. Token {token_id.item()}: {val.item()*100:.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY OF LESSON 6")
print("="*70)
print("""
WHAT WE BUILT:
==============
1. GPTConfig - Centralized configuration (matching Transformer repo)
2. TokenEmbedding - nn.Embedding (LEARNABLE token vectors)
3. PositionEmbedding - nn.Embedding (LEARNABLE position vectors)
4. TransformerBlock - From Lesson 5 (with nn.Linear layers)
5. GPT - Complete model using nn.Module

LEARNABLE PARAMETERS (all via PyTorch):
=======================================
- Token Embedding: nn.Embedding(vocab_size, d_model)
- Position Embedding: nn.Embedding(max_seq_len, d_model)
- Attention: nn.Linear layers (W_q, W_k, W_v, W_o)
- FFN: nn.Linear layers (fc1, fc2)
- LayerNorm: nn.LayerNorm (learnable weight/bias)
- Output: nn.Linear(d_model, vocab_size)

KEY NAMING CONVENTIONS (aligned with Transformer repo):
=======================================================
- d_model (not embed_dim) - embedding dimension
- n_heads (not num_heads) - attention heads
- n_blocks (not num_blocks) - transformer blocks
- d_ff (not ff_dim) - feed-forward hidden dim
- d_k, d_v - dimension per head

NEXT: Training the model (Lesson 7)
Run: python 07_training.py
=============================================================================""")
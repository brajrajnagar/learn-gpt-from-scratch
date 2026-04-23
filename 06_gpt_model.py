"""
=============================================================================
LESSON 6: Complete GPT Model - Assembling the Full Architecture
=============================================================================

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

Let's build the complete GPT model!
"""

import numpy as np

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
SOFTMAX = Probability Converter
-------------------------------
Converts raw scores (logits) to probabilities that sum to 1.0

CAUSAL MASK = "No Peeking" Rule
-------------------------------
Prevents tokens from seeing future tokens (like taking a test where
you can only see questions you've already answered)
""")

def softmax(x):
    """
    Numerically stable softmax.
    
    Converts logits to probabilities.
    
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
    Create causal (triangular) mask.
    
    Prevents positions from attending to future positions.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Mask matrix where future positions have -1e9 (effectively zero after softmax)
    """
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9
    return mask

# =============================================================================
# STEP 3: Embedding Layers
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Layers")
print("="*70)

print("""
TOKEN EMBEDDING: Convert token IDs to dense vectors
POSITION EMBEDDING: Add position information

Combined: final_embedding = token_embedding + position_embedding
""")


class TokenEmbedding:
    """
    Token embedding layer - converts token IDs to dense vectors.
    
    This is a look-up table:
      - Input: tensor of token IDs, shape (seq_len,)
      - Output: tensor of embeddings, shape (seq_len, d_model)
    
    Example:
      vocab_size = 1000
      d_model = 64
      
      token_ids = [10, 25, 67]  (3 tokens)
      embeddings = lookup[token_ids]  → shape (3, 64)
    """
    
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Number of unique tokens
            d_model: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        np.random.seed(42)
        self.weights = np.random.randn(vocab_size, d_model) * 0.02
        
        print(f"TokenEmbedding: vocab={vocab_size}, d_model={d_model}")
    
    def forward(self, token_ids):
        """
        Get embeddings for token IDs.
        
        Args:
            token_ids: Array of token IDs, shape (seq_len,)
        
        Returns:
            Token embeddings, shape (seq_len, d_model)
        """
        return self.weights[token_ids]


class PositionEmbedding:
    """
    Position embedding layer - adds position information.
    
    Each position in the sequence gets a unique embedding.
    
    Example:
      max_seq_len = 100
      d_model = 64
      
      position 0 → [0.01, -0.02, ...] (64-dim)
      position 1 → [0.02, -0.01, ...] (64-dim)
      position 2 → [0.03, 0.01, ...] (64-dim)
    """
    
    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of embedding vectors
        """
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        np.random.seed(42)
        self.weights = np.random.randn(max_seq_len, d_model) * 0.02
        
        print(f"PositionEmbedding: max_len={max_seq_len}, d_model={d_model}")
    
    def forward(self, seq_len):
        """
        Get position embeddings.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Position embeddings, shape (seq_len, d_model)
        """
        return self.weights[:seq_len]

# =============================================================================
# STEP 4: Core Components (from previous lessons)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Core Components")
print("="*70)


class LayerNorm:
    """Layer Normalization - stabilizes training."""
    
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class FeedForward:
    """Feed-Forward Network - transforms representations."""
    
    def __init__(self, d_model, d_ff):
        np.random.seed(42)
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        hidden = np.dot(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        return np.dot(hidden, self.W2) + self.b2


class MultiHeadAttention:
    """Multi-Head Self-Attention - the core of transformer."""
    
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # d_k = d_v
        
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def _split_heads(self, x):
        """Split into heads for parallel attention."""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 0, 2)
    
    def _combine_heads(self, x):
        """Combine heads back to d_model."""
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass for multi-head attention.
        
        Args:
            embeddings: Input embeddings, shape (seq_len, d_model)
            use_causal_mask: Whether to apply causal mask
        
        Returns:
            Output embeddings, shape (seq_len, d_model)
        """
        seq_len = embeddings.shape[0]
        
        # Project to Q, K, V
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        # Split into heads
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)
        
        # Create mask if needed
        mask = create_causal_mask(seq_len) if use_causal_mask else None
        
        # Process each head
        head_outputs = []
        for head_idx in range(self.n_heads):
            Q_head = Q_heads[head_idx]
            K_head = K_heads[head_idx]
            V_head = V_heads[head_idx]
            
            # Attention scores
            scores = np.dot(Q_head, K_head.T) / np.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            weights = softmax(scores)
            output = np.dot(weights, V_head)
            head_outputs.append(output)
        
        # Combine heads
        combined = np.stack(head_outputs, axis=0)
        combined = self._combine_heads(combined)
        return np.dot(combined, self.W_o)


class TransformerBlock:
    """
    Complete Transformer Block.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → x + attn_out
          → LayerNorm → FeedForward → x + ffn_out
    """
    
    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        # Attention sub-layer (Pre-LayerNorm)
        ln1_out = self.ln1.forward(x)
        attn_out = self.attention.forward(ln1_out)
        x = x + attn_out  # Residual
        
        # FFN sub-layer (Pre-LayerNorm)
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out  # Residual
        
        return x

# =============================================================================
# STEP 5: Complete GPT Model
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Complete GPT Model")
print("="*70)


class GPT:
    """
    Complete GPT Model - Autoregressive Language Model.
    
    Architecture:
        1. Token Embedding + Position Embedding → Input representation
        2. Stacked Transformer Blocks → Contextual understanding
        3. Layer Normalization → Final normalization
        4. Output Projection → Probability distribution over vocabulary
    
    This is a decoder-only transformer (like GPT-2/3).
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize GPT model.
        
        Args:
            config: Optional GPTConfig object
            **kwargs: Individual hyperparameters (if config not provided)
        """
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
        print(f"  vocab_size={vocab_size}")
        print(f"  d_model={d_model}")
        print(f"  n_heads={n_heads}")
        print(f"  n_blocks={n_blocks}")
        print(f"  d_ff={d_ff}")
        print(f"  d_k=d_v={self.config.d_k}")
        print(f"  max_seq_len={max_seq_len}")
        print(f"{'='*50}")
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = []
        for i in range(n_blocks):
            block = TransformerBlock(d_model, n_heads, d_ff)
            self.blocks.append(block)
            print(f"  Block {i+1}/{n_blocks} created")
        
        # Final layer norm
        self.ln_final = LayerNorm(d_model)
        
        # Output projection
        np.random.seed(42)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1
        
        # Print parameter count
        self._print_parameter_count(n_blocks)
    
    def _print_parameter_count(self, n_blocks):
        """Print approximate parameter count."""
        emb_params = self.config.vocab_size * self.config.d_model
        pos_params = self.config.max_sequence_length * self.config.d_model
        
        # Per block params
        block_params = (4 * self.config.d_model**2 +  # Attention
                       8 * self.config.d_model**2 +   # FFN
                       4 * self.config.d_model)       # LayerNorms
        total_block_params = n_blocks * block_params
        output_params = self.config.d_model * self.config.vocab_size
        
        total = emb_params + pos_params + total_block_params + output_params
        print(f"\n  Total parameters: {total:,} ({total/1e6:.2f}M)")
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
        
        Returns:
            logits: Output logits, shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        
        # Token embeddings
        token_embs = self.token_embedding.forward(token_ids)
        
        # Position embeddings
        pos_embs = self.position_embedding.forward(seq_len)
        
        # Combine
        x = token_embs + pos_embs
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
        
        # Final layer norm
        x = self.ln_final.forward(x)
        
        # Output projection
        logits = np.dot(x, self.W_out)
        
        return logits
    
    def predict_next_token(self, token_ids, temperature=1.0):
        """
        Predict next token probabilities.
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
            temperature: Sampling temperature
        
        Returns:
            probabilities: Next token probabilities, shape (vocab_size,)
        """
        logits = self.forward(token_ids)
        last_logits = logits[-1]
        
        # Temperature scaling
        if temperature != 1.0:
            last_logits = last_logits / temperature
        
        # Convert to probabilities
        probs = softmax(last_logits)
        
        return probs

# =============================================================================
# STEP 6: Example Usage
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

np.random.seed(42)
input_tokens = np.array([10, 25, 67, 89, 123])
print(f"Input tokens: {input_tokens}")

# Forward pass
logits = gpt.forward(input_tokens)
print(f"\nOutput logits shape: {logits.shape}")

# Get predictions
probs = gpt.predict_next_token(input_tokens)
print(f"Prediction probabilities shape: {probs.shape}")

# Top predictions
top_indices = np.argsort(probs)[-5:][::-1]
print(f"\nTop 5 predictions:")
for idx in top_indices:
    print(f"  Token {idx}: {probs[idx]*100:.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
WHAT WE BUILT:
1. GPTConfig - Centralized configuration (matching Transformer repo)
2. TokenEmbedding - Token IDs → dense vectors
3. PositionEmbedding - Position info
4. TransformerBlock - Attention + FFN + Residuals
5. GPT - Complete model with forward pass

KEY NAMING CONVENTIONS (aligned with Transformer repo):
- d_model (not embed_dim) - embedding dimension
- n_heads (not num_heads) - attention heads
- n_blocks (not num_blocks) - transformer blocks
- d_ff (not ff_dim) - feed-forward hidden dim
- d_k, d_v - dimension per head

NEXT: Training the model (Lesson 7)
""")
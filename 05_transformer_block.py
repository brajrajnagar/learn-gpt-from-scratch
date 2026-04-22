"""
=============================================================================
LESSON 5: The Complete Transformer Block
=============================================================================

Now we combine everything to build the fundamental unit of GPT:
the Transformer Block (also called Transformer Decoder Layer).

KEY COMPONENTS:
1. Multi-Head Self-Attention - Relationships between tokens
2. Feed-Forward Network - Processing and transformation
3. Layer Normalization - Stabilizing training
4. Residual Connections - Gradient flow and learning

GPT ARCHITECTURE:
- GPT-2 Small: 12 transformer blocks stacked
- GPT-2 Medium: 24 transformer blocks
- GPT-3: Up to 96 transformer blocks

Each block has the SAME structure, just different learned weights!

Let's build the complete transformer block!
"""

import numpy as np

# =============================================================================
# STEP 1: Layer Normalization
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Layer Normalization")
print("="*70)

print("""
WHY NORMALIZATION?

Deep networks can have unstable training:
- Gradients can explode (become too large)
- Gradients can vanish (become too small)
- Different layers can have very different scales

SOLUTION: Layer Normalization

Normalizes the activations to have:
- Mean = 0
- Standard deviation = 1

Then learns scale (γ) and shift (β) parameters to maintain
representational power.

BENEFITS:
- Faster convergence
- More stable training
- Less sensitive to initialization
=============================================================================""")

class LayerNorm:
    """
    Layer Normalization.
    
    Unlike BatchNorm (which normalizes across batch),
    LayerNorm normalizes across the feature dimension for each sample.
    """
    
    def __init__(self, embedding_dim, eps=1e-5):
        """
        Args:
            embedding_dim: Dimension of input
            eps: Small constant for numerical stability
        """
        self.embedding_dim = embedding_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(embedding_dim)  # Scale
        self.beta = np.zeros(embedding_dim)  # Shift
        
        print(f"LayerNorm initialized for dim={embedding_dim}")
    
    def forward(self, x):
        """
        Normalize the input.
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Normalized output, same shape as input
        """
        # Step 1: Compute mean across embedding dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # Step 2: Compute variance
        var = np.var(x, axis=-1, keepdims=True)
        
        # Step 3: Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Step 4: Scale and shift (learnable)
        output = self.gamma * x_norm + self.beta
        
        return output
    
    def __call__(self, x):
        return self.forward(x)

print("\n--- LayerNorm Example ---")

# Create LayerNorm
layer_norm = LayerNorm(embedding_dim=8)

# Sample input (sequence of 5 tokens, 8-dim each)
np.random.seed(42)
x = np.random.randn(5, 8) * 10 + 5  # Large values with offset

print(f"\nInput statistics:")
print(f"  Mean: {x.mean():.4f}")
print(f"  Std: {x.std():.4f}")
print(f"  Min: {x.min():.4f}")
print(f"  Max: {x.max():.4f}")

# Normalize
x_norm = layer_norm.forward(x)

print(f"\nNormalized statistics:")
print(f"  Mean: {x_norm.mean():.4f}")
print(f"  Std: {x_norm.std():.4f}")
print(f"  Min: {x_norm.min():.4f}")
print(f"  Max: {x_norm.max():.4f}")

print("\nLayerNorm normalizes each token's embedding to ~N(0,1)!")

# =============================================================================
# STEP 2: Feed-Forward Network
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Feed-Forward Network (FFN)")
print("="*70)

print("""
FEED-FORWARD NETWORK:

After attention, each token's representation is processed by
a small neural network (same for all tokens).

STRUCTURE:
  Input → Linear → ReLU → Linear → Output
  
  (seq_len, d_model) → (seq_len, 4*d_model) → (seq_len, 4*d_model) → (seq_len, d_model)

KEY POINTS:
- Applied independently to each token position
- Expands to 4x dimension, then compresses back
- Adds non-linearity and processing capacity
- Same weights used for all positions

WHY 4X EXPANSION?
- Gives the network capacity to learn complex transformations
- Common design choice in transformers
- Some models use different expansion ratios
=============================================================================""")

class FeedForward:
    """
    Feed-Forward Network for transformer.
    """
    
    def __init__(self, embedding_dim, ff_dim):
        """
        Args:
            embedding_dim: Input/output dimension (d_model)
            ff_dim: Hidden dimension (typically 4 * embedding_dim)
        """
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim
        
        # Two linear layers with ReLU in between
        np.random.seed(42)
        self.W1 = np.random.randn(embedding_dim, ff_dim) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embedding_dim) * np.sqrt(2.0 / ff_dim)
        self.b2 = np.zeros(embedding_dim)
        
        print(f"FeedForward initialized")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Hidden dim: {ff_dim}")
        print(f"  Output dim: {embedding_dim}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Output, shape (seq_len, embedding_dim)
        """
        # First linear layer + ReLU
        hidden = np.dot(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
    
    def __call__(self, x):
        return self.forward(x)

print("\n--- FeedForward Example ---")

# Create FFN (GPT-2 style: 768 → 3072 → 768)
ffn = FeedForward(embedding_dim=64, ff_dim=256)

# Sample input
x = np.random.randn(5, 64)
print(f"\nInput shape: {x.shape}")

# Forward pass
hidden = np.dot(x, ffn.W1) + ffn.b1
hidden_relu = np.maximum(0, hidden)
output = np.dot(hidden_relu, ffn.W2) + ffn.b2

print(f"After first linear: {hidden.shape}")
print(f"After ReLU (non-zero): {(hidden_relu > 0).sum() / hidden_relu.size * 100:.1f}%")
print(f"Output shape: {output.shape}")

# =============================================================================
# STEP 3: Residual Connections
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Residual Connections")
print("="*70)

print("""
RESIDUAL CONNECTIONS (Skip Connections):

Instead of:  output = F(input)
We do:       output = F(input) + input

WHY?

1. GRADIENT FLOW: Gradients can flow directly through skip connections
   - Helps train deep networks (GPT has many layers!)
   - Prevents vanishing gradients

2. IDENTITY MAPPING: Easy to learn "do nothing" if needed
   - Just zero out F(input)
   - Important for model flexibility

3. STABILITY: Makes training more stable
   - Prevents dramatic changes between layers
   - Works well with LayerNorm

IN TRANSFORMER:
- Attention: output = LayerNorm(x + Attention(x))
- FFN: output = LayerNorm(x + FFN(x))

This is called "Pre-LayerNorm" architecture.
=============================================================================""")

def residual_connection(x, sublayer_output):
    """
    Add residual connection.
    
    Args:
        x: Original input
        sublayer_output: Output from attention or FFN
    
    Returns:
        x + sublayer_output
    """
    return x + sublayer_output

print("\n--- Residual Connection Example ---")

# Sample input
x = np.random.randn(3, 8)
print(f"Input shape: {x.shape}")

# Simulate sublayer output (some transformation)
sublayer_out = np.random.randn(3, 8) * 0.1  # Small values

# Add residual
output = residual_connection(x, sublayer_out)

print(f"Sublayer output (small): mean={sublayer_out.mean():.4f}, std={sublayer_out.std():.4f}")
print(f"Residual output: mean={output.mean():.4f}, std={output.std():.4f}")
print(f"Output is dominated by input (residual)!")

# =============================================================================
# STEP 4: Complete Transformer Block
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Complete Transformer Block")
print("="*70)

# First, we need the MultiHeadAttention class (import from previous lesson)
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
    """Multi-head attention layer."""
    
    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def _split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 0, 2)
    
    def _combine_heads(self, x):
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.embedding_dim)
    
    def forward(self, embeddings, use_causal_mask=True):
        seq_len = embeddings.shape[0]
        
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)
        
        mask = create_causal_mask(seq_len) if use_causal_mask else None
        
        head_outputs = []
        for head_idx in range(self.num_heads):
            Q_head = Q_heads[head_idx]
            K_head = K_heads[head_idx]
            V_head = V_heads[head_idx]
            
            scores = np.dot(Q_head, K_head.T) / np.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            weights = softmax(scores)
            output = np.dot(weights, V_head)
            head_outputs.append(output)
        
        combined = np.stack(head_outputs, axis=0)
        combined = self._combine_heads(combined)
        output = np.dot(combined, self.W_o)
        
        return output

class TransformerBlock:
    """
    Complete Transformer Block (Decoder Layer).
    
    This is the fundamental building block of GPT!
    
    Architecture (Pre-LayerNorm):
    
    input
      │
      ├─────────────────────────────┐
      │                             │
      ↓                             │
    LayerNorm                       │
      │                             │
      ↓                             │
    Multi-Head Attention (causal)   │
      │                             │
      ↓                             │
      + ←───────────────────────────┘  (Residual)
      │
      ├─────────────────────────────┐
      │                             │
      ↓                             │
    LayerNorm                       │
      │                             │
      ↓                             │
    Feed-Forward Network            │
      │                             │
      ↓                             │
      + ←───────────────────────────┘  (Residual)
      │
      ↓
    output
    """
    
    def __init__(self, embedding_dim, num_heads, ff_dim):
        """
        Initialize transformer block.
        
        Args:
            embedding_dim: Dimension of embeddings (d_model)
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension (typically 4 * embedding_dim)
        """
        self.embedding_dim = embedding_dim
        
        # Components
        self.ln1 = LayerNorm(embedding_dim)
        self.ln2 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim, ff_dim)
        
        print(f"TransformerBlock initialized")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  FF hidden dim: {ff_dim}")
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Output, shape (seq_len, embedding_dim)
        """
        # ATTENTION SUB-LAYER
        # Pre-LayerNorm → Attention → Residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attention.forward(ln1_out)
        x = x + attn_out  # Residual connection
        
        # FEED-FORWARD SUB-LAYER
        # Pre-LayerNorm → FFN → Residual
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out  # Residual connection
        
        return x

print("\n--- Transformer Block Example ---")

# Create transformer block (GPT-2 small style, scaled down)
block = TransformerBlock(embedding_dim=64, num_heads=4, ff_dim=256)

# Sample input (sequence of 10 tokens)
np.random.seed(42)
x = np.random.randn(10, 64)
print(f"\nInput shape: {x.shape}")

# Forward pass
output = block.forward(x)
print(f"Output shape: {output.shape}")

print("\nTransformer block complete!")
print("Input and output have the same shape - ready to stack!")

# =============================================================================
# STEP 5: Stacking Multiple Blocks
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Stacking Transformer Blocks")
print("="*70)

print("""
GPT is built by STACKING multiple transformer blocks!

GPT-2 Small: 12 blocks
GPT-2 Medium: 24 blocks
GPT-2 Large: 36 blocks
GPT-3: Up to 96 blocks

Each block:
1. Takes input of shape (seq_len, embedding_dim)
2. Applies attention + FFN with residuals
3. Outputs same shape

Blocks are connected sequentially:
  input → Block 1 → Block 2 → ... → Block N → output
""")

class StackedTransformerBlocks:
    """
    Stack of multiple transformer blocks.
    """
    
    def __init__(self, num_blocks, embedding_dim, num_heads, ff_dim):
        """
        Args:
            num_blocks: Number of transformer blocks to stack
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
        """
        self.num_blocks = num_blocks
        self.blocks = []
        
        for i in range(num_blocks):
            print(f"Creating block {i+1}/{num_blocks}...")
            block = TransformerBlock(embedding_dim, num_heads, ff_dim)
            self.blocks.append(block)
        
        print(f"\nStackedTransformerBlocks: {num_blocks} blocks created!")
    
    def forward(self, x):
        """
        Forward pass through all blocks.
        
        Args:
            x: Input embeddings
        
        Returns:
            Final output after all blocks
        """
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            print(f"After Block {i+1}: mean={x.mean():.4f}, std={x.std():.4f}")
        
        return x

print("\n--- Stacking 3 Transformer Blocks ---")

stack = StackedTransformerBlocks(num_blocks=3, embedding_dim=64, num_heads=4, ff_dim=256)

# Forward pass
x = np.random.randn(8, 64)
print(f"\nInput shape: {x.shape}")
final_output = stack.forward(x)
print(f"\nFinal output shape: {final_output.shape}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Transformer Block")
print("="*70)

print("""
TRANSFORMER BLOCK ARCHITECTURE:

1. INPUT: (seq_len, embedding_dim)

2. ATTENTION SUB-LAYER:
   - LayerNorm(x)
   - Multi-Head Self-Attention (causal)
   - Residual: x + attention_output

3. FEED-FORWARD SUB-LAYER:
   - LayerNorm(x)
   - Feed-Forward Network
   - Residual: x + ffn_output

4. OUTPUT: (seq_len, embedding_dim)

KEY DESIGN CHOICES:

- Pre-LayerNorm: LayerNorm BEFORE sub-layer (more stable)
- Residual Connections: Enable gradient flow
- Same dimension throughout: Easy to stack

PARAMETER COUNT (per block, embedding_dim=768, num_heads=12):
- Attention: 4 × 768² ≈ 2.4M
- FFN: 768 × 3072 + 3072 × 768 ≈ 4.7M
- Total per block: ~7M parameters
- GPT-2 Small (12 blocks): ~84M parameters

NEXT: We'll add the final output layer to complete GPT!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Transformer Blocks")
print("="*70)

print("""
Try these:

1. Change block configuration:
   block = TransformerBlock(embedding_dim=128, num_heads=8, ff_dim=512)

2. Stack more blocks:
   stack = StackedTransformerBlocks(num_blocks=6, ...)

3. Analyze output statistics:
   - How do mean/std change through blocks?
   - Does LayerNorm keep values normalized?

4. Compare different ff_dim ratios:
   ff_dim = 2 * embedding_dim  # Smaller
   ff_dim = 8 * embedding_dim  # Larger

Key Takeaway:
- Transformer block = Attention + FFN + LayerNorm + Residuals
- Same input/output shape enables stacking
- This is the core building block of GPT!

Next: 06_gpt_model.py - Complete GPT architecture!
=============================================================================""")
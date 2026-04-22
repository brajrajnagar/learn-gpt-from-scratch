"""
=============================================================================
LESSON 6: Complete GPT Model Architecture
=============================================================================

Now we assemble all components into the complete GPT model!

GPT ARCHITECTURE OVERVIEW:

1. INPUT
   ↓
2. TOKEN EMBEDDINGS + POSITION EMBEDDINGS
   ↓
3. TRANSFORMER BLOCKS (stacked N times)
   ↓
4. LAYER NORM
   ↓
5. OUTPUT PROJECTION (to vocabulary size)
   ↓
6. SOFTMAX → NEXT TOKEN PROBABILITIES

This is called a "decoder-only" transformer because:
- It only uses causal (masked) self-attention
- It predicts the next token (autoregressive)
- No encoder (unlike original Transformer)

Let's build the complete GPT model!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding GPT Architecture
# =============================================================================

print("\n" + "="*70)
print("STEP 1: GPT Architecture Overview")
print("="*70)

print("""
GPT MODEL COMPONENTS:

┌─────────────────────────────────────────────┐
│           GPT MODEL                         │
│                                             │
│  Input: token_ids (seq_len,)                │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ TOKEN EMBEDDING                   │     │
│  │ (vocab_size, embedding_dim)       │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ POSITION EMBEDDING                │     │
│  │ (max_seq_len, embedding_dim)      │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│         ADD                                 │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ TRANSFORMER BLOCK 1               │     │
│  │ - Multi-Head Attention (causal)   │     │
│  │ - LayerNorm                       │     │
│  │ - Feed-Forward Network            │     │
│  │ - Residual connections            │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ TRANSFORMER BLOCK 2               │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│         ... (N blocks total)                │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ FINAL LAYER NORM                  │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│  ┌───────────────────────────────────┐     │
│  │ OUTPUT PROJECTION                 │     │
│  │ (embedding_dim → vocab_size)      │     │
│  └───────────────────────────────────┘     │
│         ↓                                   │
│  Output: logits (seq_len, vocab_size)       │
│         ↓                                   │
│  Softmax → probabilities                    │
└─────────────────────────────────────────────┘

KEY PARAMETERS (GPT-2 Small):
- vocab_size: 50,257 (BPE vocabulary)
- max_seq_len: 1024 tokens
- embedding_dim: 768
- num_heads: 12
- num_blocks: 12
- ff_dim: 3072 (4 × embedding_dim)
- Total parameters: ~124 million
""")

# =============================================================================
# STEP 2: Helper Functions
# =============================================================================

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

# =============================================================================
# STEP 3: Embedding Layers
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Layers")
print("="*70)

class TokenEmbedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        np.random.seed(42)
        # Embedding matrix: each row is a token's embedding
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02
        
        print(f"TokenEmbedding: vocab={vocab_size}, dim={embedding_dim}")
    
    def forward(self, token_ids):
        """
        Get embeddings for token IDs.
        
        Args:
            token_ids: Array of token IDs, shape (seq_len,)
        
        Returns:
            Token embeddings, shape (seq_len, embedding_dim)
        """
        return self.weights[token_ids]

class PositionEmbedding:
    """Position embedding layer."""
    
    def __init__(self, max_seq_len, embedding_dim):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        np.random.seed(42)
        self.weights = np.random.randn(max_seq_len, embedding_dim) * 0.02
        
        print(f"PositionEmbedding: max_len={max_seq_len}, dim={embedding_dim}")
    
    def forward(self, seq_len):
        """
        Get position embeddings.
        
        Returns:
            Position embeddings, shape (seq_len, embedding_dim)
        """
        return self.weights[:seq_len]

# =============================================================================
# STEP 4: Core Components (from previous lessons)
# =============================================================================

class LayerNorm:
    """Layer Normalization."""
    
    def __init__(self, embedding_dim, eps=1e-5):
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class FeedForward:
    """Feed-Forward Network."""
    
    def __init__(self, embedding_dim, ff_dim):
        np.random.seed(42)
        self.W1 = np.random.randn(embedding_dim, ff_dim) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embedding_dim) * np.sqrt(2.0 / ff_dim)
        self.b2 = np.zeros(embedding_dim)
    
    def forward(self, x):
        hidden = np.dot(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        return np.dot(hidden, self.W2) + self.b2

class MultiHeadAttention:
    """Multi-Head Self-Attention."""
    
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
        return np.dot(combined, self.W_o)

# =============================================================================
# STEP 5: Transformer Block
# =============================================================================

class TransformerBlock:
    """Complete Transformer Block."""
    
    def __init__(self, embedding_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embedding_dim)
        self.ln2 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim, ff_dim)
    
    def forward(self, x):
        # Attention sub-layer
        ln1_out = self.ln1.forward(x)
        attn_out = self.attention.forward(ln1_out)
        x = x + attn_out  # Residual
        
        # FFN sub-layer
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out  # Residual
        
        return x

# =============================================================================
# STEP 6: Complete GPT Model
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete GPT Model")
print("="*70)

class GPT:
    """
    Complete GPT Model.
    
    This is the full autoregressive language model!
    """
    
    def __init__(self, vocab_size, max_seq_len, embedding_dim, 
                 num_heads, num_blocks, ff_dim):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary (e.g., 50257 for GPT-2)
            max_seq_len: Maximum sequence length (e.g., 1024)
            embedding_dim: Dimension of embeddings (e.g., 768)
            num_heads: Number of attention heads (e.g., 12)
            num_blocks: Number of transformer blocks (e.g., 12)
            ff_dim: Feed-forward hidden dimension (e.g., 3072)
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        print("\n" + "="*50)
        print("Initializing GPT Model")
        print("="*50)
        print(f"Configuration:")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Max sequence length: {max_seq_len}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Number of transformer blocks: {num_blocks}")
        print(f"  FFN hidden dimension: {ff_dim}")
        print("="*50)
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(max_seq_len, embedding_dim)
        
        # Transformer blocks
        self.blocks = []
        for i in range(num_blocks):
            print(f"Creating transformer block {i+1}/{num_blocks}...")
            block = TransformerBlock(embedding_dim, num_heads, ff_dim)
            self.blocks.append(block)
        
        # Final layer norm
        self.ln_final = LayerNorm(embedding_dim)
        
        # Output projection (to vocabulary size)
        print(f"Creating output projection...")
        np.random.seed(42)
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.1
        
        print("="*50)
        print("GPT Model initialized!")
        print("="*50)
        
        # Calculate approximate parameter count
        self._print_parameter_count(num_blocks)
    
    def _print_parameter_count(self, num_blocks):
        """Print approximate parameter count."""
        emb_params = self.vocab_size * self.embedding_dim
        pos_params = self.max_seq_len * self.embedding_dim
        
        # Per block: attention (4 * d^2) + FFN (2 * d * 4d) + layer norms
        block_params = (4 * self.embedding_dim**2 +  # Attention
                       8 * self.embedding_dim**2 +   # FFN (d*4d + 4d*d)
                       4 * self.embedding_dim)       # LayerNorm
        total_block_params = num_blocks * block_params
        
        output_params = self.embedding_dim * self.vocab_size
        
        total = emb_params + pos_params + total_block_params + output_params
        print(f"\nApproximate parameter count: {total:,} ({total/1e6:.1f}M)")
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
        
        Returns:
            logits: Output logits for next token prediction, 
                    shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        
        # Step 1: Token embeddings
        token_embs = self.token_embedding.forward(token_ids)
        
        # Step 2: Position embeddings
        pos_embs = self.position_embedding.forward(seq_len)
        
        # Step 3: Combine embeddings
        x = token_embs + pos_embs
        
        # Step 4: Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
        
        # Step 5: Final layer norm
        x = self.ln_final.forward(x)
        
        # Step 6: Output projection to vocabulary
        logits = np.dot(x, self.W_out)
        
        return logits
    
    def predict_next_token(self, token_ids, temperature=1.0):
        """
        Predict next token probabilities.
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
            temperature: Sampling temperature (1.0 = normal)
        
        Returns:
            probabilities: Next token probabilities, shape (vocab_size,)
        """
        # Get logits for the last position
        logits = self.forward(token_ids)
        last_logits = logits[-1]  # Shape: (vocab_size,)
        
        # Apply temperature scaling
        if temperature != 1.0:
            last_logits = last_logits / temperature
        
        # Convert to probabilities
        probs = softmax(last_logits)
        
        return probs

# =============================================================================
# STEP 7: Example Usage
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Example - Creating and Using GPT")
print("="*70)

# Create a small GPT model for demonstration
# (Real GPT-2 Small would use: vocab=50257, max_len=1024, emb=768, heads=12, blocks=12)

gpt = GPT(
    vocab_size=1000,      # Small vocab for demo
    max_seq_len=128,      # Short sequences
    embedding_dim=64,     # Small embedding
    num_heads=4,          # Fewer heads
    num_blocks=2,         # Just 2 blocks for demo
    ff_dim=256            # Smaller FFN
)

print("\n" + "-"*70)
print("Running forward pass...")

# Simulate input token IDs
np.random.seed(42)
input_tokens = np.array([10, 25, 67, 89, 123, 45, 78, 234])
print(f"\nInput tokens: {input_tokens}")
print(f"Input length: {len(input_tokens)}")

# Forward pass
logits = gpt.forward(input_tokens)
print(f"\nOutput logits shape: {logits.shape}")
print(f"  (seq_len={len(input_tokens)}, vocab_size=1000)")

# Get next token probabilities
probs = gpt.predict_next_token(input_tokens)
print(f"\nNext token probabilities shape: {probs.shape}")
print(f"  (vocab_size=1000)")

# Find most likely next tokens
top_indices = np.argsort(probs)[-10:][::-1]
print(f"\nTop 10 most likely next tokens:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Token {idx}: probability {probs[idx]:.6f}")

# =============================================================================
# STEP 8: Understanding the Output
# =============================================================================

print("\n" + "="*70)
print("STEP 8: Understanding GPT Output")
print("="*70)

print("""
GPT OUTPUT EXPLAINED:

1. LOGITS (raw output):
   - Shape: (seq_len, vocab_size)
   - Each row is the score for each vocabulary item
   - Higher logit = more likely
   
2. PROBABILITIES (after softmax):
   - Shape: (vocab_size,) for next token prediction
   - Sum to 1.0
   - Used for sampling the next token

3. NEXT TOKEN PREDICTION:
   - GPT is trained to predict the next token
   - Given tokens [t0, t1, t2, t3], it predicts t4
   - Uses only past tokens (causal mask)

4. TEMPERATURE:
   - Temperature = 1.0: Normal probabilities
   - Temperature < 1.0: More confident (peaky distribution)
   - Temperature > 1.0: More random (flatter distribution)
   - Temperature → 0: Greedy (always pick highest)

HOW GPT GENERATES TEXT:
1. Start with input tokens (prompt)
2. Predict next token probabilities
3. Sample or select next token
4. Append to sequence
5. Repeat from step 2

This is called "autoregressive generation"!
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Complete GPT Model")
print("="*70)

print("""
GPT MODEL COMPONENTS:

1. TOKEN EMBEDDINGS: Convert token IDs → vectors
2. POSITION EMBEDDINGS: Add position information
3. TRANSFORMER BLOCKS: Process with attention + FFN
4. LAYER NORM: Normalize activations
5. OUTPUT PROJECTION: Convert to vocabulary logits

FORWARD PASS:
  token_ids → embeddings → blocks → norm → logits → probs

KEY PARAMETERS (GPT-2 Small):
- vocab_size: 50,257
- max_seq_len: 1,024
- embedding_dim: 768
- num_heads: 12
- num_blocks: 12
- ff_dim: 3,072
- Total: ~124M parameters

WHAT MAKES GPT WORK:
1. Self-attention captures token relationships
2. Multi-head allows multiple perspectives
3. Stacked blocks build hierarchical representations
4. Causal mask enables autoregressive prediction
5. Large scale (parameters + data) gives capability

NEXT: Training the model with loss function and optimization!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with GPT Model")
print("="*70)

print("""
Try these:

1. Change model size:
   gpt = GPT(vocab_size=2000, max_seq_len=64, 
             embedding_dim=128, num_heads=8, 
             num_blocks=4, ff_dim=512)

2. Analyze output:
   - What's the shape of logits?
   - How do probabilities change with temperature?

3. Temperature effect:
   probs_cold = gpt.predict_next_token(input_tokens, temperature=0.1)
   probs_hot = gpt.predict_next_token(input_tokens, temperature=2.0)
   Compare the distributions!

4. Longer input:
   input_tokens = np.arange(20)  # 20 tokens
   How does output change?

Key Takeaway:
- GPT combines embeddings + transformer blocks + output projection
- Output is logits that become next-token probabilities
- Model is autoregressive (predicts one token at a time)

Next: 07_training.py - Training GPT with loss and optimization!
=============================================================================""")
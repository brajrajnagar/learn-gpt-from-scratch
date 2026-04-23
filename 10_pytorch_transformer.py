"""
=============================================================================
BONUS LESSON 10: PyTorch Transformer - From NumPy to Production
=============================================================================

Throughout this course, we built GPT from scratch using NumPy to understand
the fundamentals. Now let's see how PyTorch provides built-in components!

KEY PYTORCH COMPONENTS:
- nn.MultiheadAttention - Self-attention mechanism
- nn.TransformerEncoderLayer - Complete transformer block
- nn.TransformerEncoder - Stack of transformer blocks
- nn.Embedding - Token embeddings
- Automatic differentiation - Real backpropagation!

WHAT YOU'LL LEARN:
1. How our NumPy implementation maps to PyTorch APIs
2. Building GPT using PyTorch primitives
3. Real training with gradient descent
4. Production-ready transformer code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("\n" + "="*70)
print("BONUS LESSON: PyTorch Transformer")
print("="*70)

# =============================================================================
# STEP 1: PyTorch MultiheadAttention vs Our NumPy Implementation
# =============================================================================

print("\n" + "="*70)
print("STEP 1: MultiheadAttention in PyTorch")
print("="*70)

print("""
OUR NUMPY IMPLEMENTATION (Lesson 4):
====================================
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.1

PYTORCH EQUIVALENT:
===================
nn.MultiheadAttention(embed_dim, num_heads)

PyTorch handles:
- Q, K, V projections internally
- Splitting into heads
- Attention computation
- Output projection
- Optional masking
- Key padding masks
""")

# Create PyTorch multihead attention
embed_dim = 64
num_heads = 4

print(f"\n--- Creating MultiheadAttention ---")
print(f"  embed_dim: {embed_dim}")
print(f"  num_heads: {num_heads}")

pytorch_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

print(f"\nPyTorch MultiheadAttention created!")
print(f"  in_proj_weight shape: {pytorch_attention.in_proj_weight.shape}")
print(f"    → Combines W_q, W_k, W_v into one weight matrix")
print(f"  out_proj.weight shape: {pytorch_attention.out_proj.weight.shape}")
print(f"    → Output projection (our W_o)")

# Test the attention
print("\n" + "-"*50)
print("Testing MultiheadAttention:")
print("-"*50)

# Create sample input: batch_size=2, seq_len=5, embed_dim=64
batch_size = 2
seq_len = 5
x = torch.randn(batch_size, seq_len, embed_dim)

print(f"\nInput shape: {x.shape}")
print(f"  → (batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim})")

# Apply attention
attn_output, attn_weights = pytorch_attention(x, x, x)

print(f"\nOutput shape: {attn_output.shape}")
print(f"  → Same shape as input (residual connection ready!)")

print(f"\nAttention weights shape: {attn_weights.shape}")
print(f"  → (seq_len={seq_len}, seq_len={seq_len})")
print(f"  → PyTorch returns averaged attention weights across heads (not per-head)")
print(f"  → For per-head weights, use need_weights=True and access differently")

print("\n" + "-"*50)
print("COMPARISON: NumPy vs PyTorch")
print("-"*50)
print("""
NUMPY (Manual):
- Forward pass only (no gradients)
- Manual Q, K, V computation
- Manual attention score calculation
- Manual softmax implementation

PYTORCH (Automatic):
- Full autograd support (real backprop!)
- Optimized C++ backend
- GPU acceleration ready
- Masking support built-in
""")

# =============================================================================
# STEP 2: TransformerEncoderLayer - Complete Block
# =============================================================================

print("\n" + "="*70)
print("STEP 2: TransformerEncoderLayer")
print("="*70)

print("""
OUR TRANSFORMER BLOCK (Lesson 5):
==================================
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(...)
        self.layernorm1 = LayerNorm(embed_dim)
        self.layernorm2 = LayerNorm(embed_dim)

PYTORCH EQUIVALENT:
===================
nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=ff_dim,
    dropout=0.1,
    batch_first=True
)

PyTorch includes:
- Multihead attention
- Feed-forward network (2 layers + GELU)
- Layer normalization
- Dropout
- Residual connections
""")

# Create transformer encoder layer
ff_dim = 256

print(f"\n--- Creating TransformerEncoderLayer ---")
print(f"  d_model: {embed_dim}")
print(f"  nhead: {num_heads}")
print(f"  dim_feedforward: {ff_dim}")
print(f"  dropout: 0.1")

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=ff_dim,
    dropout=0.1,
    batch_first=True
)

print(f"\nTransformerEncoderLayer created!")
print(f"  Parameters: {sum(p.numel() for p in encoder_layer.parameters()):,}")

# Test the layer
x = torch.randn(batch_size, seq_len, embed_dim)
output = encoder_layer(x)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"  → Shape preserved (ready for stacking)")

# =============================================================================
# STEP 3: Building a Mini GPT with PyTorch
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Mini GPT Model in PyTorch")
print("="*70)

print("""
Now let's build a complete GPT-like model using PyTorch primitives!

ARCHITECTURE:
=============
1. Token Embedding (nn.Embedding)
2. Position Embedding (learned)
3. Stack of TransformerEncoderLayers
4. Layer Normalization
5. Output Projection (nn.Linear)
""")


class MiniGPT(nn.Module):
    """
    Mini GPT model using PyTorch primitives.
    
    This is a REAL, trainable GPT-like model that can be trained
    with actual backpropagation!
    
    Architecture based on GPT-2:
    - Token embeddings
    - Learned position embeddings
    - Stacked transformer blocks
    - Layer normalization
    - Output projection to vocabulary
    """
    
    def __init__(self, vocab_size, max_seq_len, embed_dim=128, 
                 num_heads=4, num_layers=4, ff_dim=512, dropout=0.1):
        """
        Initialize Mini GPT.
        
        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks - NOTE: We need to handle causal masking ourselves
        # For GPT, we use causal mask in forward() to enable causal attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Layer norm and dropout
        self.ln_f = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with small values (like our NumPy version)."""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.position_embedding.weight.data.uniform_(-initrange, initrange)
        self.lm_head.weight.data.uniform_(-initrange, initrange)
    
    def generate_causal_mask(self, seq_len, device):
        """
        Generate causal mask for GPT (prevents attending to future).
        
        For PyTorch TransformerEncoder, the mask shape should be (T, T) for 2D
        or (batch_size, T, T) for 3D.
        
        Returns:
            causal_mask: Boolean mask where True means "attend to this position"
                        Shape: (seq_len, seq_len)
        """
        # Create a triangular mask - lower triangle is 0 (attend), upper is -inf (block)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask  # (seq_len, seq_len)
    
    def forward(self, idx, targets=None):
        """
        Forward pass with causal masking for GPT.
        
        Args:
            idx: Input token IDs, shape (batch_size, seq_len)
            targets: Optional target token IDs for loss computation
        
        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            loss: Optional loss value if targets provided
        """
        B, T = idx.shape  # Batch size, sequence length
        device = idx.device
        
        # Get token embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, embed_dim)
        
        # Get position embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(positions)  # (T, embed_dim)
        
        # Combine embeddings
        x = tok_emb + pos_emb  # (B, T, embed_dim)
        x = self.dropout(x)
        
        # Create causal mask (prevents attending to future tokens)
        # This is CRUCIAL for GPT - autoregressive language modeling!
        causal_mask = self.generate_causal_mask(T, device)
        
        # Apply transformer blocks with causal mask
        # is_causal=True tells PyTorch to use causal attention
        x = self.transformer(x, mask=causal_mask)  # (B, T, embed_dim)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token IDs, shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (None = no filtering)
        
        Returns:
            Generated token IDs, shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop sequence if too long
            idx_cond = idx[:, -100:]  # Support up to 100 tokens context
            
            # Forward pass
            logits, _ = self.forward(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


print("\n--- Creating Mini GPT Model ---")

# Model configuration
vocab_size = 1000  # Small vocabulary for demo
max_seq_len = 64
embed_dim = 128
num_heads = 4
num_layers = 4
ff_dim = 512

# Create model
model = MiniGPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=ff_dim
)

print(f"\nMini GPT Configuration:")
print(f"  Vocabulary: {vocab_size}")
print(f"  Max sequence: {max_seq_len}")
print(f"  Embedding dim: {embed_dim}")
print(f"  Attention heads: {num_heads}")
print(f"  Transformer layers: {num_layers}")
print(f"  FFN dim: {ff_dim}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# =============================================================================
# STEP 4: Real Training with Backpropagation
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Real Training with Autograd")
print("="*70)

print("""
NOW THE MAGIC: Real backpropagation!

Our NumPy version only did forward pass. PyTorch computes gradients
automatically using autograd!

TRAINING LOOP:
==============
1. Forward pass: logits, loss = model(input, targets)
2. Backward pass: loss.backward()  ← Automatic gradients!
3. Update weights: optimizer.step()
4. Zero gradients: optimizer.zero_grad()
""")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"\nOptimizer: AdamW")
print(f"  Learning rate: 3e-4")

# Create sample training data
batch_size = 4
seq_len = 16

# Random token sequences (simulating tokenized text)
inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"\nTraining data:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Input shape: {inputs.shape}")
print(f"  Target shape: {targets.shape}")

# Training step
print("\n" + "-"*50)
print("Training Step:")
print("-"*50)

model.train()

# Forward pass
logits, loss = model(inputs, targets)

print(f"\nForward pass:")
print(f"  Logits shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")

# Backward pass
loss.backward()
print(f"\nBackward pass: Gradients computed!")

# Count gradients
grad_params = sum(1 for p in model.parameters() if p.grad is not None)
print(f"  Parameters with gradients: {grad_params}")

# Update weights
optimizer.step()
optimizer.zero_grad()

print(f"\nWeights updated! (optimizer.step)")
print(f"Gradients zeroed (optimizer.zero_grad)")

# =============================================================================
# STEP 5: Training Loop Demo
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Mini Training Demo")
print("="*70)

print("\nTraining for a few steps to show loss decreasing...")

# Create more training samples
num_samples = 32
train_inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
train_targets = torch.randint(0, vocab_size, (num_samples, seq_len))

model.train()
print(f"\n{'Step':<6} {'Loss':<10} {'Perplexity':<10}")
print("-" * 30)

for step in range(10):
    # Get a random batch
    batch_idx = torch.randperm(num_samples)[:batch_size]
    batch_inputs = train_inputs[batch_idx]
    batch_targets = train_targets[batch_idx]
    
    # Forward pass
    logits, loss = model(batch_inputs, batch_targets)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # Compute perplexity
    perplexity = math.exp(loss.item())
    
    print(f"{step:<6} {loss.item():<10.4f} {perplexity:<10.2f}")

print("\n" + "-"*50)
print("Training complete!")
print("-"*50)

# =============================================================================
# STEP 6: Text Generation
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Text Generation")
print("="*70)

print("""
GENERATION PROCESS:
===================
1. Start with prompt tokens
2. Forward pass to get logits
3. Sample next token
4. Append to sequence
5. Repeat!

This is autoregressive generation - same as GPT!
""")

model.eval()

# Generate from random start
print("\n--- Generating Sequences ---")

# Start with random tokens
start_tokens = torch.randint(0, vocab_size // 2, (1, 8))
print(f"Start tokens: {start_tokens.tolist()}")

# Generate 16 new tokens
generated = model.generate(start_tokens, max_new_tokens=16, temperature=1.0, top_k=50)
print(f"Generated tokens: {generated.tolist()}")
print(f"Total sequence length: {generated.shape[1]}")

# Different temperatures
print("\n--- Temperature Comparison ---")

for temp in [0.5, 1.0, 2.0]:
    generated = model.generate(start_tokens, max_new_tokens=8, temperature=temp)
    print(f"  Temp {temp}: {generated[0, -8:].tolist()}")

# =============================================================================
# STEP 7: Comparison Summary
# =============================================================================

print("\n" + "="*70)
print("STEP 7: NumPy vs PyTorch Summary")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    NUMPY (Lessons 1-9)                          │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Educational - Understanding fundamentals                      │
│ ✓ Manual implementation of every component                      │
│ ✓ Forward pass only (no real backprop)                          │
│ ✓ CPU only                                                      │
│ ✓ Great for learning concepts                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PYTORCH (This Lesson)                        │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Production-ready code                                         │
│ ✓ Built-in optimized components                                 │
│ ✓ Automatic differentiation (real backprop!)                    │
│ ✓ GPU acceleration                                              │
│ ✓ Ready for large-scale training                                │
└─────────────────────────────────────────────────────────────────┘

KEY MAPPINGS:
=============
NumPy Component              → PyTorch Equivalent
─────────────────────────────────────────────────
MultiHeadAttention           → nn.MultiheadAttention
TransformerBlock             → nn.TransformerEncoderLayer
LayerNorm                    → nn.LayerNorm
Dense/Linear                 → nn.Linear
Embedding                    → nn.Embedding
Manual gradients             → torch.autograd (automatic!)
NumPy arrays                 → torch.Tensor
CPU only                     → .cuda() for GPU

NEXT STEPS:
===========
1. Train on real text data (use a tokenizer like BPE)
2. Scale up model size (more layers, larger embed_dim)
3. Add learning rate scheduling
4. Use mixed precision training (amp)
5. Deploy to GPU for faster training
6. Experiment with GPT variants (GPT-J, LLaMA architecture)

RESOURCES:
==========
- PyTorch Documentation: https://pytorch.org/docs/
- "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- nanoGPT (Andrej Karpathy): https://github.com/karpathy/nanoGPT
- HuggingFace Transformers: https://huggingface.co/docs/transformers
=============================================================================""")

print("\n" + "="*70)
print("CONGRATULATIONS!")
print("="*70)
print("""
You've completed the entire GPT learning course!

From NumPy basics → PyTorch production code
You now understand:
- Neural network fundamentals
- Embeddings (token + position)
- Self-attention mechanism
- Multi-head attention
- Transformer blocks
- Complete GPT architecture
- Training with backpropagation
- Text generation strategies
- PyTorch implementation

You're ready to:
- Read the GPT papers with understanding
- Explore HuggingFace transformers
- Build your own language models
- Contribute to open-source LLM projects

Keep learning and building! 🚀
=============================================================================""")
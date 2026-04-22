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
REAL-WORLD EXAMPLE: Calibrating Measurement Instruments
========================================================

Imagine a science lab with multiple measurement devices:
- Thermometer (temperature)
- Scale (weight)
- Ruler (length)
- Voltmeter (voltage)

PROBLEM: Each device uses different scales!
- Temperature: -10 to 40 degrees Celsius
- Weight: 0 to 1000 grams
- Voltage: 0 to 220 volts

If you feed these directly into a machine learning model:
- Large values dominate (voltage 220 >> temperature 25)
- Training becomes unstable
- Model struggles to learn from all features equally

SOLUTION: Normalize all measurements to the same scale!

LAYER NORMALIZATION WORKS LIKE THIS:

For each token's embedding vector:
1. Compute mean and standard deviation
2. Subtract mean → centers at 0
3. Divide by std → scales to unit variance
4. Apply learned scale (γ) and shift (β)

Result: All features are on comparable scales!

BENEFITS:
- Faster convergence (no scale fighting)
- More stable training (no exploding/vanishing)
- Less sensitive to initialization
- Works with any batch size

WHY "LAYER" NORMALIZATION?
- Normalizes across features within ONE sample
- Different from BatchNorm (normalizes across samples)
- Perfect for sequence models like GPT
=============================================================================""")

class LayerNorm:
    """
    Layer Normalization.
    
    REAL-WORLD EXAMPLE: Audio Volume Normalizer
    ===========================================
    
    Think of LayerNorm like an audio engineer normalizing song volumes:
    
    PROBLEM: Songs have different volumes
    - Rock song: Very loud (peaks at -3 dB)
    - Classical: Quiet (peaks at -18 dB)
    - Podcast: Medium (peaks at -12 dB)
    
    SOLUTION: Normalize each song
    1. Measure average volume (mean)
    2. Measure dynamic range (variance)
    3. Adjust to standard level (normalize)
    4. Apply custom EQ (learnable γ and β)
    
    RESULT: All songs play at similar perceived volume!
    
    In neural networks:
    - Each token embedding = one "song"
    - Each feature dimension = one "frequency"
    - LayerNorm = audio engineer for activations
    """
    
    def __init__(self, embedding_dim, eps=1e-5):
        """
        Args:
            embedding_dim: Dimension of input
            eps: Small constant for numerical stability (prevents division by zero)
        """
        self.embedding_dim = embedding_dim
        self.eps = eps
        
        # Learnable parameters
        # These let the network undo normalization if needed!
        self.gamma = np.ones(embedding_dim)  # Scale (like volume knob)
        self.beta = np.zeros(embedding_dim)  # Shift (like bass/treble)
        
        print(f"LayerNorm initialized for dim={embedding_dim}")
        print(f"  → Like having {embedding_dim} independent volume knobs")
    
    def forward(self, x):
        """
        Normalize the input.
        
        REAL-WORLD EXAMPLE: Grade Normalization
        ----------------------------------------
        Imagine normalizing test scores from different classes:
        
        Class A: Mean=70, Std=15 (hard test)
        Class B: Mean=85, Std=5 (easy test)
        Class C: Mean=60, Std=20 (varied results)
        
        To compare fairly:
        1. Subtract class mean: score - mean
        2. Divide by std: (score - mean) / std
        3. Now all scores are "standardized" (z-scores!)
        
        LayerNorm does the same for neural activations!
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Normalized output, same shape as input
        """
        # Step 1: Compute mean across embedding dimension
        # Like computing class average
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # Step 2: Compute variance
        # Like computing how spread out grades are
        var = np.var(x, axis=-1, keepdims=True)
        
        # Step 3: Normalize (z-score normalization)
        # Now all features have mean=0, variance=1
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Step 4: Scale and shift (learnable)
        # This restores representational power!
        # If network needs original scale, it can learn γ=std, β=mean
        output = self.gamma * x_norm + self.beta
        
        return output
    
    def __call__(self, x):
        return self.forward(x)

print("\n--- LayerNorm Example: Audio Volume Analogy ---")
print("="*50)
print("""
SCENARIO: Normalizing audio tracks for a playlist

Track 1: Rock song (loud, peaks at 0.9)
Track 2: Classical (quiet, peaks at 0.2)
Track 3: Podcast (medium, peaks at 0.5)

LayerNorm is like an audio engineer making all tracks
play at similar perceived volume!
""")

# Create LayerNorm
layer_norm = LayerNorm(embedding_dim=8)

# Sample input (sequence of 5 tokens, 8-dim each)
# Simulating "loud" activations (large values with offset)
np.random.seed(42)
x = np.random.randn(5, 8) * 10 + 5  # Mean ~5, Std ~10

print(f"\nInput (before normalization):")
print(f"  Mean: {x.mean():.4f} ← Like a 'loud' audio track")
print(f"  Std: {x.std():.4f} ← High dynamic range")
print(f"  Min: {x.min():.4f}")
print(f"  Max: {x.max():.4f}")

# Normalize
x_norm = layer_norm.forward(x)

print(f"\nOutput (after LayerNorm):")
print(f"  Mean: {x_norm.mean():.4f} ← Centered at 0")
print(f"  Std: {x_norm.std():.4f} ← Normalized to ~1")
print(f"  Min: {x_norm.min():.4f}")
print(f"  Max: {x_norm.max():.4f}")

print("\n✓ LayerNorm normalizes each token's embedding!")
print("  This keeps training stable across all layers!")

# =============================================================================
# STEP 2: Feed-Forward Network
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Feed-Forward Network (FFN)")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Information Processing Factory
===================================================

After attention gathers information, the Feed-Forward Network
(PROCESSING DEPARTMENT) analyzes and transforms it.

FACTORY ANALOGY:

INPUT DEPARTMENT (Attention Output):
  Raw materials arrive from suppliers
  Shape: (seq_len, embedding_dim) = (5 tokens, 768 features)

EXPANSION DEPARTMENT (Linear 1):
  Materials are unpackED and spread out
  768 features → 3072 features (4x expansion!)
  Like unfolding a packed suitcase

PROCESSING FLOOR (ReLU Activation):
  Workers process each feature independently
  ReLU: Keep positive signals, zero out negative
  Like quality control - only useful signals pass

COMPRESSION DEPARTMENT (Linear 2):
  Processed materials are repacked
  3072 features → 768 features (back to original size)
  Like repacking with new organization

OUTPUT DEPARTMENT:
  Finished products ready for next stage
  Shape: (seq_len, embedding_dim) - same as input!

WHY 4X EXPANSION?
- Gives room for complex processing
- Like having a large workbench
- More space = more sophisticated transformations
- Common design choice in transformers

KEY INSIGHT: FFN processes EACH TOKEN INDEPENDENTLY
- Token 0: Processed by same FFN as token 1
- But token 0 doesn't mix with token 1
- Like identical factories at each position
=============================================================================""")

class FeedForward:
    """
    Feed-Forward Network for transformer.
    
    REAL-WORLD EXAMPLE: Photo Filter App
    =====================================
    
    Think of FFN like applying filters to photos:
    
    INPUT: One photo (token embedding)
           Shape: (768,) features
    
    FILTER 1 (Linear + ReLU):
    - Apply 3072 different filters
    - Each filter detects something specific
    - ReLU: Only keep positive detections
    
    FILTER 2 (Linear):
    - Combine filter results
    - Produce final enhanced photo
    - Output: Same size as input (768,)
    
    The "filters" are learned during training!
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
        # He initialization helps ReLU networks train better
        np.random.seed(42)
        self.W1 = np.random.randn(embedding_dim, ff_dim) * np.sqrt(2.0 / embedding_dim)
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embedding_dim) * np.sqrt(2.0 / ff_dim)
        self.b2 = np.zeros(embedding_dim)
        
        print(f"FeedForward initialized")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Hidden dim: {ff_dim} ({ff_dim/embedding_dim:.0f}x expansion)")
        print(f"  Output dim: {embedding_dim}")
        print(f"  Parameters: {embedding_dim * ff_dim * 2}")
    
    def forward(self, x):
        """
        Forward pass.
        
        REAL-WORLD EXAMPLE: Document Translation Pipeline
        --------------------------------------------------
        Imagine translating documents through an intermediate form:
        
        Step 1: Expand to intermediate representation
                English → Universal semantic form
                Shape: (seq_len, 768) → (seq_len, 3072)
        
        Step 2: Apply non-linear activation (ReLU)
                Keep only meaningful semantic features
                Like keeping only relevant concepts
        
        Step 3: Compress to output
                Universal form → Translated English
                Shape: (seq_len, 3072) → (seq_len, 768)
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Output, shape (seq_len, embedding_dim)
        """
        # First linear layer + ReLU
        # Expand and activate
        hidden = np.dot(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU: Keep positive, zero negative
        
        # Second linear layer
        # Compress back to original dimension
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
    
    def __call__(self, x):
        return self.forward(x)

print("\n--- FeedForward Example: Document Processing ---")
print("="*50)
print("""
SCENARIO: Processing documents through expansion/compression

Think of this like summarizing a long document:
1. Read full document (expand to understand)
2. Extract key points (ReLU - keep important info)
3. Write summary (compress back)

The document length stays same, but content is refined!
""")

# Create FFN (GPT-2 style ratio: 64 → 256 → 64)
ffn = FeedForward(embedding_dim=64, ff_dim=256)

# Sample input (5 tokens, 64-dim each)
x = np.random.randn(5, 64)
print(f"\nInput shape: {x.shape}")
print(f"  → 5 tokens, each with 64 features")

# Forward pass - trace through the network
hidden = np.dot(x, ffn.W1) + ffn.b1
print(f"\nAfter expansion (Linear 1):")
print(f"  Shape: {hidden.shape}")
print(f"  → Expanded to 256 features (4x!)")

hidden_relu = np.maximum(0, hidden)
active_pct = (hidden_relu > 0).sum() / hidden_relu.size * 100
print(f"\nAfter ReLU activation:")
print(f"  Active neurons: {active_pct:.1f}%")
print(f"  → ReLU zeroed out negative values")

output = np.dot(hidden_relu, ffn.W2) + ffn.b2
print(f"\nAfter compression (Linear 2):")
print(f"  Output shape: {output.shape}")
print(f"  → Back to original 64 features!")
print(f"  → But now transformed with new information!")

# =============================================================================
# STEP 3: Residual Connections
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Residual Connections")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Mountain Hiking Trail with Shortcuts
=========================================================

Imagine hiking up a 100-layer mountain (like a 100-layer neural network):

WITHOUT RESidual (Normal Trail):
  Layer 1 → Layer 2 → Layer 3 → ... → Layer 100
  
  Problem: If any section is blocked (bad gradient),
  you can't continue! You're stuck!

WITH RESIDUAL (Shortcut Paths):
  Layer 1 → Layer 2 → Layer 3 → ... → Layer 100
     ↓         ↓         ↓              ↓
  Shortcut paths let you skip sections!
  
  Benefit: Even if one section is hard, you can skip ahead!
  Gradients flow freely through shortcuts!

RESIDUAL CONNECTION FORMULA:
  output = F(input) + input
  
  Where:
  - F(input) is the sublayer (attention or FFN)
  - input is the shortcut (identity/skip connection)
  
  The network learns F(input), but the signal
  always has the original input as a "safety net"!

WHY RESIDUALS WORK:

1. GRADIENT HIGHWAY:
   Gradients can flow directly through skip connections
   → No vanishing gradient problem
   → Can train very deep networks (100+ layers!)

2. EASY IDENTITY:
   Network can learn "do nothing" easily
   → Just set F(input) = 0
   → Output = 0 + input = input (identity)

3. STABLE TRAINING:
   Each layer makes small modifications
   → Prevents dramatic changes
   → Training is more predictable

IN TRANSFORMER/GPT:
  - After attention: output = LayerNorm(x + Attention(x))
  - After FFN: output = LayerNorm(x + FFN(x))
  
  This is called "Pre-LayerNorm" architecture!
=============================================================================""")

def residual_connection(x, sublayer_output):
    """
    Add residual connection.
    
    REAL-WORLD EXAMPLE: Original Document + Annotations
    ----------------------------------------------------
    Think of residual connection like annotating a document:
    
    Original document (x):
    "The cat sat on the mat."
    
    Annotations (sublayer_output):
    [Note: "cat" is subject, "sat" is verb, "mat" is object]
    
    Combined (residual):
    "The cat sat on the mat." + [annotations]
    
    The original text is preserved!
    Annotations add information, not replace!
    
    Args:
        x: Original input
        sublayer_output: Output from attention or FFN
    
    Returns:
        x + sublayer_output
    """
    return x + sublayer_output

print("\n--- Residual Connection Example: Document Annotation ---")
print("="*50)
print("""
SCENARIO: Adding annotations to documents while preserving original

Original document (x): Contains full content
Annotations (sublayer_output): Small additions/modifications
Combined: Original + annotations = Enhanced document
""")

# Sample input (original document)
np.random.seed(42)
x = np.random.randn(3, 8)
print(f"Original input shape: {x.shape}")
print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")

# Simulate sublayer output (annotations - small modifications)
sublayer_out = np.random.randn(3, 8) * 0.1  # Small values (subtle annotations)
print(f"\nSublayer output (annotations):")
print(f"  Mean: {sublayer_out.mean():.4f}, Std: {sublayer_out.std():.4f}")
print(f"  → Small modifications, not replacing content")

# Add residual (combine original + annotations)
output = residual_connection(x, sublayer_out)

print(f"\nResidual output (enhanced document):")
print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
print(f"  → Original content preserved!")
print(f"  → Annotations added subtly")

print("\n✓ Residual connections let original signal flow through!")
print("  This prevents information loss in deep networks!")

# =============================================================================
# STEP 4: Complete Transformer Block
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Complete Transformer Block")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Complete Manufacturing Assembly Line
=========================================================

The Transformer Block is a complete processing unit that combines:
- Attention (gather information)
- FFN (process information)
- LayerNorm (stabilize)
- Residual (preserve original)

FACTORY LAYOUT:

                    ┌─────────────────────────────────┐
                    │     TRANSFORMER BLOCK           │
                    │     (Processing Unit)           │
                    └─────────────────────────────────┘

INPUT RAW MATERIALS (x)
    │
    ├────────────────────────────────────┐
    │                                    │
    ↓                                    │
┌─────────────────┐                      │
│ LAYER NORM 1    │ ← Normalize          │
│ (Calibration)   │   activations        │
└─────────────────┘                      │
    │                                    │
    ↓                                    │
┌─────────────────┐                      │
│ MULTI-HEAD      │ ← Gather info        │
│ ATTENTION       │   from context       │
└─────────────────┘                      │
    │                                    │
    ↓                                    │
    + ←──────────────────────────────────┘  (RESIDUAL: Add back original)
    │
    ├────────────────────────────────────┐
    │                                    │
    ↓                                    │
┌─────────────────┐                      │
│ LAYER NORM 2    │ ← Normalize          │
│ (Calibration)   │   again              │
└─────────────────┘                      │
    │                                    │
    ↓                                    │
┌─────────────────┐                      │
│ FEED-FORWARD    │ ← Process            │
│ NETWORK         │   independently      │
└─────────────────┘                      │
    │                                    │
    ↓                                    │
    + ←──────────────────────────────────┘  (RESIDUAL: Add back again)
    │
    ↓
OUTPUT PROCESSED MATERIALS (x)

KEY INSIGHT: Input and output have SAME SHAPE!
This allows stacking multiple blocks!
=============================================================================""")

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
    
    REAL-WORLD EXAMPLE: Complete Document Processing Station
    =========================================================
    
    This is the fundamental building block of GPT!
    
    Think of it as a document processing station:
    
    1. CALIBRATION (LayerNorm 1):
       Normalize the document for consistent processing
    
    2. RESEARCH (Multi-Head Attention):
       Look up related documents, find connections
       "What does 'it' refer to?"
       "Which words are related?"
    
    3. ANNOTATION (Residual):
       Add research notes to original document
    
    4. CALIBRATION (LayerNorm 2):
       Normalize again for next stage
    
    5. ANALYSIS (Feed-Forward Network):
       Process each document independently
       Extract insights, add understanding
    
    6. ANNOTATION (Residual):
       Add analysis notes to document
    
    OUTPUT: Enhanced document with context and analysis!
    
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
        
        # Components - like departments in a factory
        self.ln1 = LayerNorm(embedding_dim)  # Calibration station 1
        self.ln2 = LayerNorm(embedding_dim)  # Calibration station 2
        self.attention = MultiHeadAttention(embedding_dim, num_heads)  # Research dept
        self.ffn = FeedForward(embedding_dim, ff_dim)  # Analysis dept
        
        print(f"TransformerBlock initialized")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  FF hidden dim: {ff_dim}")
        print(f"  → Complete processing unit ready!")
    
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        REAL-WORLD EXAMPLE: Processing Pipeline
        ----------------------------------------
        
        INPUT: Document embeddings
               Shape: (seq_len, embedding_dim)
        
        STAGE 1 - ATTENTION SUB-LAYER:
        1. Calibrate (LayerNorm)
        2. Research connections (Attention)
        3. Add notes to original (Residual)
        
        STAGE 2 - FEED-FORWARD SUB-LAYER:
        4. Calibrate again (LayerNorm)
        5. Analyze content (FFN)
        6. Add analysis to document (Residual)
        
        OUTPUT: Enhanced embeddings
                Shape: (seq_len, embedding_dim) - same!
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Output, shape (seq_len, embedding_dim)
        """
        # ATTENTION SUB-LAYER
        # Pre-LayerNorm → Attention → Residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attention.forward(ln1_out)
        x = x + attn_out  # Residual connection (add back original)
        
        # FEED-FORWARD SUB-LAYER
        # Pre-LayerNorm → FFN → Residual
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out  # Residual connection (add back original)
        
        return x

print("\n--- Transformer Block Example: Complete Processing ---")
print("="*50)
print("""
SCENARIO: Processing a 10-token sentence through one transformer block

Input: 10 word embeddings (each 64-dimensional)
Block: Complete transformer processing
Output: 10 enhanced word embeddings (same shape!)
""")

# Create transformer block (GPT-2 small style, scaled down)
print("Initializing transformer block...")
block = TransformerBlock(embedding_dim=64, num_heads=4, ff_dim=256)

# Sample input (sequence of 10 tokens representing a sentence)
np.random.seed(42)
x = np.random.randn(10, 64)
print(f"\nInput shape: {x.shape}")
print(f"  → 10 tokens (words), each with 64 features")

# Forward pass through the complete block
print(f"\nRunning forward pass...")
output = block.forward(x)

print(f"\nOutput shape: {output.shape}")
print(f"  → Same shape as input!")
print(f"  → But now each token has CONTEXT + ANALYSIS!")

print("\n" + "="*50)
print("TRANSFORMER BLOCK COMPLETE!")
print("="*50)
print("""
What happened inside:

1. LayerNorm 1 calibrated the input
2. Multi-Head Attention gathered context
   - Each token attended to previous tokens
   - Built understanding of relationships
3. Residual added attention output to original
4. LayerNorm 2 calibrated again
5. Feed-Forward Network processed independently
   - Each token transformed with learned patterns
6. Residual added FFN output to previous

RESULT: Each token embedding now contains:
- Original word meaning
- Context from attended words
- Processed understanding from FFN

This is how GPT builds contextual understanding!
=============================================================================""")

# =============================================================================
# STEP 5: Stacking Multiple Blocks
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Stacking Transformer Blocks")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Multi-Stage Rocket
======================================

GPT is built by STACKING multiple transformer blocks!

ROCKET STAGE ANALOGY:

Stage 1 (Block 1): Initial processing
  - Basic pattern recognition
  - Simple word relationships
  - Like rocket leaving atmosphere

Stage 2 (Block 2): Deeper processing
  - More abstract patterns
  - Complex relationships
  - Like rocket gaining altitude

...

Stage 12 (Block 12): High-level understanding
  - Abstract semantic understanding
  - Long-range dependencies
  - Like rocket reaching orbit!

GPT CONFIGURATIONS:
- GPT-2 Small: 12 blocks (12-stage rocket)
- GPT-2 Medium: 24 blocks
- GPT-2 Large: 36 blocks
- GPT-3: Up to 96 blocks!

Each block:
1. Takes input of shape (seq_len, embedding_dim)
2. Applies attention + FFN with residuals
3. Outputs same shape - ready for next block!

Blocks are connected sequentially:
  input → Block 1 → Block 2 → ... → Block N → output

DEEPER = MORE CAPABLE (but harder to train)
- Residual connections make deep training possible
- LayerNorm keeps everything stable
- Each block builds on previous understanding
=============================================================================""")

class StackedTransformerBlocks:
    """
    Stack of multiple transformer blocks.
    
    REAL-WORLD EXAMPLE: Assembly Line with Multiple Stations
    ========================================================
    
    Think of this as a factory assembly line:
    
    Station 1: Basic processing
    Station 2: Build on Station 1's work
    Station 3: Build on Station 2's work
    ...
    Station N: Final refinement
    
    Each station:
    - Receives partially processed material
    - Adds its specialized processing
    - Passes to next station
    
    Final product has been through ALL stations!
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
        
        print(f"Building stacked transformer with {num_blocks} blocks...")
        print("="*50)
        
        for i in range(num_blocks):
            print(f"Creating block {i+1}/{num_blocks}...")
            block = TransformerBlock(embedding_dim, num_heads, ff_dim)
            self.blocks.append(block)
        
        print(f"\nStackedTransformerBlocks: {num_blocks} blocks created!")
        print(f"  → Ready to process deep understanding!")
    
    def forward(self, x):
        """
        Forward pass through all blocks.
        
        REAL-WORLD EXAMPLE: Document Through Review Stages
        ---------------------------------------------------
        Imagine a document going through multiple review stages:
        
        Reviewer 1: Basic grammar and structure
        Reviewer 2: Content accuracy
        Reviewer 3: Logical flow
        ...
        Reviewer N: Final polish
        
        Each reviewer:
        - Reads the document (with all previous notes)
        - Adds their expertise
        - Passes to next reviewer
        
        Final document has been refined by ALL reviewers!
        
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
print("="*50)
print("""
SCENARIO: Building a mini-GPT with 3 transformer blocks

This is like a 3-stage rocket:
- Stage 1: Basic understanding
- Stage 2: Deeper processing
- Stage 3: Abstract reasoning

Let's build and run it!
""")

stack = StackedTransformerBlocks(num_blocks=3, embedding_dim=64, num_heads=4, ff_dim=256)

# Forward pass
x = np.random.randn(8, 64)
print(f"\nInput shape: {x.shape}")
print(f"  → 8 tokens, 64 features each")

print(f"\nRunning through all blocks...")
print("="*50)
final_output = stack.forward(x)

print(f"\n" + "="*50)
print("FINAL OUTPUT:")
print("="*50)
print(f"Shape: {final_output.shape}")
print(f"  → Same shape as input!")
print(f"  → But now with DEEP contextual understanding!")

print("\n" + "="*70)
print("WHAT HAPPENED:")
print("="*70)
print("""
The input passed through 3 transformer blocks:

Block 1: Basic processing
  - Learned simple patterns
  - Attended to nearby tokens
  - Basic feature extraction

Block 2: Intermediate processing
  - Built on Block 1's understanding
  - More complex patterns
  - Medium-range dependencies

Block 3: Advanced processing
  - Built on Block 2's understanding
  - Abstract patterns
  - Long-range dependencies

FINAL RESULT:
Each token embedding now contains rich contextual information
gathered from all previous tokens through multiple layers
of attention and processing!

This is the essence of how GPT understands language!
=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Transformer Block")
print("="*70)

print("""
TRANSFORMER BLOCK ARCHITECTURE:
===============================

1. INPUT: (seq_len, embedding_dim)

2. ATTENTION SUB-LAYER:
   - LayerNorm(x)           ← Calibrate
   - Multi-Head Attention   ← Gather context
   - x + attention_output   ← Residual (preserve original)

3. FEED-FORWARD SUB-LAYER:
   - LayerNorm(x)           ← Calibrate again
   - Feed-Forward Network   ← Process independently
   - x + ffn_output         ← Residual (preserve original)

4. OUTPUT: (seq_len, embedding_dim)

KEY DESIGN CHOICES:
===================

- Pre-LayerNorm: LayerNorm BEFORE sub-layer (more stable)
- Residual Connections: Enable gradient flow through deep networks
- Same dimension throughout: Easy to stack blocks
- 4x FFN expansion: Capacity for complex processing

PARAMETER COUNT (per block, embedding_dim=768, num_heads=12):
============================================================

- Attention: 4 × 768² ≈ 2.4 million parameters
- FFN: 768 × 3072 + 3072 × 768 ≈ 4.7 million parameters
- LayerNorm: 768 × 2 × 2 ≈ 3,000 parameters (negligible)
- Total per block: ~7 million parameters

GPT-2 Small (12 blocks): ~84 million parameters
GPT-2 Medium (24 blocks): ~350 million parameters
GPT-2 Large (36 blocks): ~760 million parameters
GPT-2 XL (48 blocks): ~1.5 billion parameters

REAL-WORLD ANALOGIES RECAP:
===========================

1. LAYER NORM: Audio volume normalizer
   - Calibrates activations to stable range

2. FEED-FORWARD: Document processing factory
   - Expands, processes, compresses
   - Each token processed independently

3. RESIDUAL: Mountain trail with shortcuts
   - Original signal always preserved
   - Enables training deep networks

4. TRANSFORMER BLOCK: Complete processing station
   - Attention gathers context
   - FFN processes independently
   - Residuals preserve original
   - LayerNorm stabilizes everything

5. STACKED BLOCKS: Multi-stage rocket
   - Each block builds on previous
   - Deeper = more abstract understanding
   - Final block = highest-level reasoning

NEXT: We'll add the final output layer to complete GPT!
      (Token prediction head for language generation)
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Transformer Blocks")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. CHANGE BLOCK CONFIGURATION:
   block = TransformerBlock(embedding_dim=128, num_heads=8, ff_dim=512)
   
   Question: How does capacity change?
   Answer: Larger dimensions = more parameters = more learning capacity

2. STACK MORE BLOCKS:
   stack = StackedTransformerBlocks(num_blocks=6, embedding_dim=64, ...)
   
   Question: How does depth affect understanding?
   Expectation: Deeper = more abstract reasoning

3. ANALYZE OUTPUT STATISTICS:
   After each block, print mean and std:
   print(f"After Block {i}: mean={x.mean():.4f}, std={x.std():.4f}")
   
   Question: Does LayerNorm keep values stable?
   Expectation: Yes! Std should stay around 1

4. COMPARE FFN RATIOS:
   ff_dim = 2 * embedding_dim  # Smaller (less processing)
   ff_dim = 8 * embedding_dim  # Larger (more processing)
   
   Question: How does FFN size affect capacity?
   Answer: Larger FFN = more transformation capacity

5. VISUALIZE (MENTALLY):
   Imagine each block as a filter:
   - Block 1: Removes noise, finds basic patterns
   - Block 2: Finds relationships between patterns
   - Block 3: Abstract concepts emerge
   - ...
   - Block N: High-level understanding

KEY TAKEAWAY:
=============
- Transformer block = Attention + FFN + LayerNorm + Residuals
- Same input/output shape enables stacking
- Each block builds deeper understanding
- This is the core building block of GPT!
- Stack many blocks = deep language understanding!

Next: 06_gpt_model.py - Complete GPT architecture!
=============================================================================""")
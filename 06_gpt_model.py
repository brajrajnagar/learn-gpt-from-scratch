"""
=============================================================================
LESSON 6: Complete GPT Model Architecture
=============================================================================

Now we assemble all components into the complete GPT model!

REAL-WORLD ANALOGY: Building a Complete Restaurant
==================================================

Think of GPT as a restaurant that serves words:

1. ENTRANCE (Input)
   - Customers arrive with orders (token IDs)
   
2. MENU TRANSLATOR (Token Embeddings)
   - Converts order names to kitchen codes
   - "Margherita Pizza" → Code #42
   
3. SEATING CHART (Position Embeddings)
   - Tracks order sequence (first appetizer, then main, then dessert)
   
4. KITCHEN STATIONS (Transformer Blocks)
   - Station 1: Prep (basic patterns)
   - Station 2: Cooking (complex patterns)
   - Station 3: Plating (final refinement)
   - Each station builds on previous work
   
5. QUALITY CHECK (Final Layer Norm)
   - Ensure consistent presentation
   
6. SERVING (Output Projection)
   - Present final dish to customer
   - Customer chooses from menu (softmax over vocabulary)

GPT is a "decoder-only" transformer because:
- It only uses causal (masked) self-attention
- It predicts the next token (autoregressive)
- No encoder (unlike original Transformer for translation)

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
REAL-WORLD EXAMPLE: Amazon Fulfillment Center
=============================================

GPT processes language like Amazon processes orders:

┌─────────────────────────────────────────────────────────┐
│                    GPT MODEL                             │
│                                                          │
│  CUSTOMER ORDER ARRIVES (Input token IDs)               │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ ORDER TRANSLATOR (Token Embedding)             │     │
│  │ "Wireless Mouse" → SKU #12345                  │     │
│  │ Converts IDs to dense vectors                  │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ WAREHOUSE LOCATION (Position Embedding)        │     │
│  │ Aisle 1, Shelf 2, Position 3                   │     │
│  │ Adds sequence position info                    │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ PROCESSING STATION 1 (Transformer Block 1)     │     │
│  │ - Scan item (Attention)                        │     │
│  │ - Package appropriately (FFN)                  │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ PROCESSING STATION 2 (Transformer Block 2)     │     │
│  │ - Add shipping labels                          │     │
│  │ - Quality check                                │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│         ... (N stations total)                           │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ FINAL INSPECTION (Final Layer Norm)            │     │
│  │ Standardize all packages                       │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│  ┌────────────────────────────────────────────────┐     │
│  │ SHIPPING LABEL PRINTER (Output Projection)     │     │
│  │ Generates delivery options                     │     │
│  │ (embedding_dim → vocab_size)                   │     │
│  └────────────────────────────────────────────────┘     │
│         ↓                                                │
│  DELIVERY OPTIONS (Softmax → Probabilities)             │
│  - Option A: 45% chance                                 │
│  - Option B: 30% chance                                 │
│  - Option C: 25% chance                                 │
└─────────────────────────────────────────────────────────┘

KEY PARAMETERS (GPT-2 Small):
- vocab_size: 50,257 (BPE vocabulary)
- max_seq_len: 1024 tokens
- embedding_dim: 768
- num_heads: 12
- num_blocks: 12
- ff_dim: 3072 (4 × embedding_dim)
- Total parameters: ~124 million

GPT-2 Small is like a medium-sized fulfillment center:
- 50K+ products in catalog (vocabulary)
- Can handle orders up to 1024 items (sequence length)
- 12 processing stations (transformer blocks)
- 12 specialized scanning teams per station (attention heads)
=============================================================================""")

# =============================================================================
# STEP 2: Helper Functions
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Helper Functions")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Restaurant Kitchen Tools
=============================================

Just like a kitchen needs basic tools (knives, pans, timers),
GPT needs helper functions:

SOFTMAX = Food Portioning
- Takes raw ingredients (logits)
- Divides into proper portions (probabilities)
- All portions add up to 100% (sum to 1.0)

CAUSAL MASK = "No Peeking" Rule
- Like students taking a test
- Can only see your own paper (past tokens)
- Can't see future papers (future tokens)
- Ensures fair testing (autoregressive generation)
""")

def softmax(x):
    """
    Numerically stable softmax.
    
    REAL-WORLD EXAMPLE: Dividing Pizza Fairly
    ==========================================
    
    Imagine you have 5 friends and 1 pizza:
    
    Friend ratings (logits): [2, 5, 3, 8, 1]
    - Friend 4 rated highest (8) - loves pizza most!
    - Friend 5 rated lowest (1) - not very hungry
    
    Softmax divides the pizza proportionally:
    - Friend 4 gets largest slice (highest probability)
    - Friend 5 gets smallest slice
    - All slices add up to 1 whole pizza (sum = 1.0)
    
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
    
    REAL-WORLD EXAMPLE: Movie Spoiler Protection
    ============================================
    
    Imagine watching a movie with spoiler protection:
    
    At minute 1: You can only see minute 1
    At minute 5: You can see minutes 1-5
    At minute 10: You can see minutes 1-10
    
    You CANNOT see future minutes (spoilers!)
    
    Causal mask works the same way:
    - Token 0: Can only see token 0
    - Token 5: Can see tokens 0-5
    - Token 10: Can see tokens 0-10
    
    This prevents "spoilers" (cheating by seeing future)!
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Mask matrix where future positions have -1e9 (hidden)
    """
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9  # Large negative (becomes ~0 after softmax)
    return mask

# =============================================================================
# STEP 3: Embedding Layers
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Layers")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Library Card Catalog System
================================================

TOKEN EMBEDDING = Book Lookup
-----------------------------

Imagine a library with 50,000 books (vocabulary):

PATRON REQUEST: "Harry Potter and the Sorcerer's Stone"
  ↓
LIBRARIAN LOOKS UP in catalog (embedding table)
  ↓
RETURNS: Book location code [0.23, -0.45, 0.89, ...] (768-dim vector)
  ↓
This vector uniquely identifies the book!

Each book has a unique embedding (location code):
- "Harry Potter" → [0.23, -0.45, 0.89, ...]
- "Lord of the Rings" → [-0.12, 0.67, -0.34, ...]
- "1984" → [0.45, 0.23, -0.78, ...]

POSITION EMBEDDING = Reading Order
----------------------------------

Books on a shelf need ORDER:
- Book 1: First in series
- Book 2: Second in series
- Book 3: Third in series

Position embeddings add this order information:
- Position 0 → [0.00, 1.00, 0.00, ...]
- Position 1 → [0.50, 0.87, 0.12, ...]
- Position 2 → [0.87, 0.50, 0.34, ...]

Combined = Book + Position
- "Harry Potter" at position 0
- "Chamber of Secrets" at position 1
- "Prisoner of Azkaban" at position 2

This tells GPT both WHAT and WHERE!
=============================================================================""")

class TokenEmbedding:
    """
    Token embedding layer.
    
    REAL-WORLD EXAMPLE: Restaurant Menu Translator
    ===============================================
    
    Think of TokenEmbedding as a menu translator:
    
    CUSTOMER ORDER (token ID):
    "I'll have item #42"
    
    TRANSLATOR LOOKS UP (embedding lookup):
    Item #42 = "Margherita Pizza" = [0.23, -0.45, 0.89, ...]
    
    KITCHEN RECEIVES (dense vector):
    Full description of the dish in chef's language
    
    The embedding table is like the menu:
    - Each item has a unique ID
    - Each ID maps to a detailed description
    - Chef knows exactly what to make
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Number of unique tokens (e.g., 50,257 for GPT-2)
            embedding_dim: Size of embedding vectors (e.g., 768)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        np.random.seed(42)
        # Embedding matrix: each row is a token's embedding
        # Like a dictionary: token_id → embedding_vector
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02
        
        print(f"TokenEmbedding created")
        print(f"  Vocabulary: {vocab_size:,} unique tokens")
        print(f"  Embedding dim: {embedding_dim} features per token")
        print(f"  → Like a dictionary with {vocab_size:,} entries!")
    
    def forward(self, token_ids):
        """
        Get embeddings for token IDs.
        
        REAL-WORLD EXAMPLE: Looking Up Words in Dictionary
        ---------------------------------------------------
        Input: [10, 25, 67] (token IDs = word numbers)
        
        Process:
        - Look up word #10 → "The" → [0.1, -0.2, ...]
        - Look up word #25 → "cat" → [-0.3, 0.4, ...]
        - Look up word #67 → "sat" → [0.5, -0.1, ...]
        
        Output: Stack of embeddings for all tokens
        
        Args:
            token_ids: Array of token IDs, shape (seq_len,)
        
        Returns:
            Token embeddings, shape (seq_len, embedding_dim)
        """
        return self.weights[token_ids]

class PositionEmbedding:
    """
    Position embedding layer.
    
    REAL-WORLD EXAMPLE: Train Seat Assignment
    =========================================
    
    Think of PositionEmbedding as seat assignments:
    
    PASSENGER: "I have a ticket" (token)
    CONDUCTOR: "Your seat is Car 3, Seat 15" (position)
    
    The position tells you WHERE in the sequence:
    - Token "The" at position 0 (beginning of sentence)
    - Token "cat" at position 1 (subject of sentence)
    - Token "sat" at position 2 (verb/action)
    
    Same word, different position = different meaning!
    """
    
    def __init__(self, max_seq_len, embedding_dim):
        """
        Args:
            max_seq_len: Maximum sequence length (e.g., 1024)
            embedding_dim: Size of embedding vectors
        """
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        np.random.seed(42)
        # Position embedding table
        # Each position gets a unique embedding
        self.weights = np.random.randn(max_seq_len, embedding_dim) * 0.02
        
        print(f"PositionEmbedding created")
        print(f"  Max length: {max_seq_len} positions")
        print(f"  Embedding dim: {embedding_dim} features per position")
        print(f"  → Can handle sequences up to {max_seq_len} tokens!")
    
    def forward(self, seq_len):
        """
        Get position embeddings.
        
        Args:
            seq_len: Current sequence length
        
        Returns:
            Position embeddings, shape (seq_len, embedding_dim)
        """
        return self.weights[:seq_len]

# =============================================================================
# STEP 4: Core Components (from previous lessons)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Core Components")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Restaurant Equipment
=========================================

These components were built in previous lessons:

LayerNorm = Food Scale/Measuring Cup
- Ensures consistent portions
- Keeps everything standardized
- Prevents extreme values

FeedForward = Food Processor
- Takes ingredients in
- Chops, mixes, transforms
- Outputs processed food

MultiHeadAttention = Team of Food Critics
- Critic 1: Tastes seasoning
- Critic 2: Evaluates texture
- Critic 3: Analyzes presentation
- Combined: Complete evaluation

All these tools work together in the kitchen!
""")

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

print("\n" + "="*70)
print("STEP 5: Transformer Block")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Sushi Assembly Line Station
================================================

Each transformer block is like a sushi chef station:

INPUT: Rice and fish arrive (embeddings from previous layer)
       ↓
┌─────────────────────────────────────────────┐
│  STATION 1: PREPARATION (LayerNorm)         │
│  - Measure rice precisely (normalize)       │
│  - Ensure consistent portions               │
│         ↓                                    │
│  STATION 2: ASSEMBLY (Attention)            │
│  - Chef examines ingredients                │
│  - Understands relationships                │
│  - "Fish goes ON rice, not under"           │
│         ↓                                    │
│  RESIDUAL: Add original ingredients back    │
│  - Don't lose the original flavor!          │
│         ↓                                    │
│  STATION 3: PREPARATION (LayerNorm 2)       │
│  - Final measurement check                  │
│         ↓                                    │
│  STATION 4: SHAPING (FeedForward)           │
│  - Form into proper sushi shape             │
│  - Apply transformation                     │
│         ↓                                    │
│  RESIDUAL: Add previous stage back          │
│  - Preserve accumulated flavor              │
└─────────────────────────────────────────────┘
       ↓
OUTPUT: Finished sushi (enhanced embeddings)
        Same format as input, but transformed!

Multiple blocks = Multiple chef stations in sequence!
Each station refines the sushi further!
=============================================================================""")

class TransformerBlock:
    """
    Complete Transformer Block.
    
    REAL-WORLD EXAMPLE: Document Review Station
    ===========================================
    
    Think of TransformerBlock as a document reviewer:
    
    INPUT DRAFT (x): Original document
    
    REVIEW CYCLE 1 (Attention):
    - Read entire document (self-attention)
    - Find connections between sections
    - "Section 3 references Section 1"
    - Add review notes (residual)
    
    REVIEW CYCLE 2 (FFN):
    - Process each paragraph independently
    - Improve wording and clarity
    - Add final polish
    - Add edits to document (residual)
    
    OUTPUT: Enhanced document (same format, better content!)
    """
    
    def __init__(self, embedding_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embedding_dim)
        self.ln2 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim, ff_dim)
    
    def forward(self, x):
        # Attention sub-layer (Pre-LayerNorm architecture)
        ln1_out = self.ln1.forward(x)
        attn_out = self.attention.forward(ln1_out)
        x = x + attn_out  # Residual connection
        
        # FFN sub-layer (Pre-LayerNorm architecture)
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out  # Residual connection
        
        return x

# =============================================================================
# STEP 6: Complete GPT Model
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete GPT Model")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Complete Ice Cream Shop
============================================

GPT is like an ice cream shop that predicts your next flavor:

┌──────────────────────────────────────────────────────────┐
│                    GPT ICE CREAM SHOP                     │
│                                                           │
│  CUSTOMER ORDER (Input token IDs)                        │
│  "I want: Vanilla → Chocolate → ?"                       │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ FLAVOR ENCODER (Token Embedding)                 │   │
│  │ Vanilla → [0.2, -0.5, 0.8, ...]                  │   │
│  │ Chocolate → [-0.3, 0.7, -0.2, ...]               │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ORDER SEQUENCE (Position Embedding)              │   │
│  │ First flavor + position 0                        │   │
│  │ Second flavor + position 1                       │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ PROCESSING STATION 1 (Transformer Block 1)       │   │
│  │ "Customer started with vanilla..."               │   │
│  │ Basic pattern recognition                        │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ PROCESSING STATION 2 (Transformer Block 2)       │   │
│  │ "...then chocolate, likely wants sweet!"         │   │
│  │ Deeper pattern understanding                     │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│         ... (more stations for deeper understanding)     │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ FINAL QUALITY CHECK (Final LayerNorm)            │   │
│  │ Ensure consistent recommendations                │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ RECOMMENDATION GENERATOR (Output Projection)     │   │
│  │ Maps understanding to flavor options             │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  FLAVOR PROBABILITIES (Softmax)                          │
│  - Strawberry: 35% ← Most likely!                       │
│  - Cookies & Cream: 25%                                 │
│  - Mint Chip: 20%                                       │
│  - Vanilla: 10%                                         │
│  - ... (50,000+ flavors total)                          │
│         ↓                                                 │
│  SHOPKEEPER: "I recommend Strawberry!"                  │
└──────────────────────────────────────────────────────────┘

KEY INSIGHT: GPT doesn't "know" the next word.
It calculates PROBABILITIES based on patterns!
=============================================================================""")

class GPT:
    """
    Complete GPT Model.
    
    REAL-WORLD EXAMPLE: Full-Service Restaurant
    ===========================================
    
    This is the complete autoregressive language model!
    
    Think of it as running a restaurant:
    
    1. HOST (Input Processing)
       - Greets customers (receives token IDs)
       - Seats them properly (embeddings)
    
    2. WAITER (Information Flow)
       - Takes order to kitchen (forward pass)
       - Brings food back (output logits)
    
    3. KITCHEN (Transformer Blocks)
       - Prep cook (Block 1): Basic preparation
       - Line cook (Block 2): Main cooking
       - Sous chef (Block 3): Refinement
       - Executive chef (Block N): Final touches
    
    4. EXPEDITOR (Output Projection)
       - Plates the food (projects to vocab)
       - Presents to customer (softmax)
    
    5. CUSTOMER (Next Token Selection)
       - Chooses from options (samples)
       - "I'll have the strawberry!"
    
    The cycle repeats for each new token!
    """
    
    def __init__(self, vocab_size, max_seq_len, embedding_dim, 
                 num_heads, num_blocks, ff_dim):
        """
        Initialize GPT model.
        
        REAL-WORLD EXAMPLE: Restaurant Setup
        -------------------------------------
        
        Setting up a new restaurant requires:
        
        vocab_size = Menu size
          - GPT-2: 50,257 items (like a massive food court!)
          - Our demo: 1,000 items (small cafe)
        
        max_seq_len = Maximum order complexity
          - How many courses can be ordered
          - GPT-2: 1024 (banquet-sized!)
        
        embedding_dim = Chef's vocabulary richness
          - How detailed flavor descriptions are
          - GPT-2: 768 dimensions (sommelier-level!)
        
        num_heads = Number of specialist chefs
          - Each handles different aspects
          - GPT-2: 12 specialists
        
        num_blocks = Number of kitchen stations
          - More stations = more refined dishes
          - GPT-2: 12 stations (assembly line!)
        
        ff_dim = Processing capacity
          - How much transformation per station
          - GPT-2: 3072 (4x embedding = plenty of room!)
        
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
        print("🍦 GPT Language Restaurant Opening!")
        print("="*50)
        print(f"Restaurant Configuration:")
        print(f"  Menu size (vocab): {vocab_size:,} items")
        print(f"  Max courses (seq_len): {max_seq_len}")
        print(f"  Chef vocabulary (emb_dim): {embedding_dim}")
        print(f"  Specialist chefs (heads): {num_heads}")
        print(f"  Kitchen stations (blocks): {num_blocks}")
        print(f"  Processing capacity (ff_dim): {ff_dim}")
        print("="*50)
        
        # Embedding layers (Host stand)
        print("\nSetting up host stand...")
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(max_seq_len, embedding_dim)
        
        # Transformer blocks (Kitchen stations)
        print("\nSetting up kitchen stations...")
        self.blocks = []
        for i in range(num_blocks):
            print(f"  Station {i+1}/{num_blocks}: Opening...")
            block = TransformerBlock(embedding_dim, num_heads, ff_dim)
            self.blocks.append(block)
        
        # Final layer norm (Quality check)
        print("\nSetting up quality control...")
        self.ln_final = LayerNorm(embedding_dim)
        
        # Output projection (Menu printer)
        print("Setting up menu printer...")
        np.random.seed(42)
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.1
        
        print("="*50)
        print("🎉 Restaurant is now open for business!")
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
        print(f"\n💰 Estimated investment (parameters): {total:,} ({total/1e6:.1f}M)")
        print("   (Each parameter is like a recipe adjustment!)")
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        REAL-WORLD EXAMPLE: Processing a Customer Order
        -----------------------------------------------
        
        INPUT: Customer's order history
               token_ids = [10, 25, 67] (appetizer, salad, ?)
        
        STEP 1: Look up each item (Token Embedding)
                10 → "Spring Rolls" → [0.2, -0.5, ...]
                25 → "Caesar Salad" → [-0.3, 0.7, ...]
                67 → "Soup" → [0.1, 0.4, ...]
        
        STEP 2: Add course numbers (Position Embedding)
                Course 1: Spring Rolls + pos_0
                Course 2: Caesar Salad + pos_1
                Course 3: Soup + pos_2
        
        STEP 3: Send through kitchen (Transformer Blocks)
                Station 1: Basic prep
                Station 2: Main cooking
                ...
                Station N: Final touches
        
        STEP 4: Quality check (Final LayerNorm)
                Ensure consistent presentation
        
        STEP 5: Print recommendations (Output Projection)
                Generate scores for all menu items
        
        OUTPUT: Recommendation scores (logits)
                "Based on your order, we recommend..."
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
        
        Returns:
            logits: Output logits for next token prediction, 
                    shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        
        # Step 1: Token embeddings (Look up items in catalog)
        token_embs = self.token_embedding.forward(token_ids)
        
        # Step 2: Position embeddings (Add course numbers)
        pos_embs = self.position_embedding.forward(seq_len)
        
        # Step 3: Combine (Item + Position)
        x = token_embs + pos_embs
        
        # Step 4: Pass through transformer blocks (Kitchen stations)
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
        
        # Step 5: Final layer norm (Quality control)
        x = self.ln_final.forward(x)
        
        # Step 6: Output projection to vocabulary (Print recommendations)
        logits = np.dot(x, self.W_out)
        
        return logits
    
    def predict_next_token(self, token_ids, temperature=1.0):
        """
        Predict next token probabilities.
        
        REAL-WORLD EXAMPLE: Weather Forecast
        -------------------------------------
        
        Given recent weather pattern [Sunny, Cloudy, ?]
        Predict tomorrow's weather:
        
        TEMPERATURE SCALING:
        - Cold forecast (temp=0.1): Very confident
          "95% chance of Sunny!"
        
        - Normal forecast (temp=1.0): Standard
          "45% Sunny, 35% Cloudy, 20% Rain"
        
        - Wild forecast (temp=2.0): Uncertain/random
          "30% Sunny, 25% Cloudy, 25% Rain, 20% Snow!"
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
            temperature: Sampling temperature
                - Low (<1): Confident/conservative
                - Normal (=1): Standard
                - High (>1): Creative/random
        
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

print("""
REAL-WORLD SCENARIO: Opening a Mini Language Restaurant
========================================================

We're opening a small cafe (not a massive food court like GPT-2):
- Menu: 1,000 items (manageable size)
- Max order: 128 courses (reasonable meal)
- Staff: 4 specialist chefs (small but skilled team)
- Stations: 2 kitchen stops (efficient workflow)

Let's see our restaurant in action!
""")

# Create a small GPT model for demonstration
gpt = GPT(
    vocab_size=1000,      # Small vocab for demo (like a cafe menu)
    max_seq_len=128,      # Short sequences (reasonable meal)
    embedding_dim=64,     # Small embedding (compact descriptions)
    num_heads=4,          # Fewer heads (smaller team)
    num_blocks=2,         # Just 2 blocks (efficient workflow)
    ff_dim=256            # Smaller FFN (appropriate capacity)
)

print("\n" + "-"*70)
print("📝 Processing a customer order...")
print("-"*70)

# Simulate input token IDs (customer's order history)
np.random.seed(42)
input_tokens = np.array([10, 25, 67, 89, 123, 45, 78, 234])
print(f"\nCustomer order history: {input_tokens}")
print(f"  → {len(input_tokens)} courses ordered so far")

# Forward pass - generate recommendations
print(f"\n🔄 Sending order through kitchen...")
logits = gpt.forward(input_tokens)
print(f"\n📊 Kitchen recommendations:")
print(f"  Output shape: {logits.shape}")
print(f"  → Scores for all {logits.shape[1]} menu items at each position!")

# Get next token probabilities
print(f"\n🎯 Predicting next course...")
probs = gpt.predict_next_token(input_tokens)
print(f"  Probability distribution shape: {probs.shape}")
print(f"  → Probability for each of {probs.shape[0]} menu items!")

# Find most likely next tokens
top_indices = np.argsort(probs)[-10:][::-1]
print(f"\n🏆 Top 10 recommended next courses:")
for i, idx in enumerate(top_indices):
    confidence = "⭐⭐⭐" if probs[idx] > 0.05 else "⭐⭐" if probs[idx] > 0.02 else "⭐"
    print(f"  {i+1}. Item #{idx}: {probs[idx]*100:.2f}% chance {confidence}")

# =============================================================================
# STEP 8: Understanding the Output
# =============================================================================

print("\n" + "="*70)
print("STEP 8: Understanding GPT Output")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Horse Race Betting
======================================

GPT OUTPUT EXPLAINED:

1. LOGITS (Raw Scores) = Odds Before Conversion
   - Shape: (seq_len, vocab_size)
   - Each item has a raw score
   - Higher = more favored
   
   Example: [2.5, -1.3, 0.8, ...]
   → Item 0 is favored, Item 1 is underdog

2. PROBABILITIES (After Softmax) = Betting Odds
   - Shape: (vocab_size,) for next token
   - All odds sum to 100%
   - Used to determine payouts
   
   Example: [0.45, 0.15, 0.25, ...]
   → Item 0: 45% chance (favorite)
   → Item 1: 15% chance (longshot)
   → Item 2: 25% chance (contender)

3. NEXT TOKEN PREDICTION = Picking the Winner
   - GPT is trained to predict the next token
   - Like picking which horse wins
   - Uses only past information (causal mask)

4. TEMPERATURE = Confidence Level
   
   COLD (temp=0.1): Ultra-confident
   - "I'm 99% sure it's Horse #5!"
   - Very peaky distribution
   - Good for factual completion
   
   NORMAL (temp=1.0): Standard confidence
   - "Horse #5 is favored at 45%"
   - Natural probability distribution
   - Good for general use
   
   HOT (temp=2.0): Feeling adventurous
   - "Any horse could win today!"
   - Flatter distribution
   - Good for creative generation

HOW GPT GENERATES TEXT = Continuous Betting
============================================

1. Start with prompt: "The cat sat on the"
2. Calculate odds for next word
3. Sample from odds (roll the dice)
4. Append chosen word: "The cat sat on the mat"
5. Repeat from step 2: "The cat sat on the mat because"
6. Continue until done!

This is "autoregressive generation"!
Like a snowball rolling downhill, growing with each step!
=============================================================================""")

# =============================================================================
# STEP 9: Temperature Demonstration
# =============================================================================

print("\n" + "="*70)
print("STEP 9: Temperature Effect Demonstration")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Choosing a Restaurant
=========================================

Imagine choosing where to eat:

CONSERVATIVE MODE (temperature=0.1):
  "Always go to your favorite!"
  - Italian: 95% ← Overwhelming favorite
  - Chinese: 3%
  - Mexican: 2%
  → Safe, predictable choice

NORMAL MODE (temperature=1.0):
  "Go with your usual preferences"
  - Italian: 40% ← Favorite but not guaranteed
  - Chinese: 25%
  - Mexican: 20%
  - Thai: 10%
  - Others: 5%
  → Balanced choice

ADVENTUROUS MODE (temperature=2.0):
  "Try something new!"
  - Italian: 25% ← Still likely, but less dominant
  - Chinese: 22%
  - Mexican: 20%
  - Thai: 18%
  - Ethiopian: 10%
  - Indian: 5%
  → Open to surprises!

Let's see temperature in action!
""")

print("\n--- Temperature Comparison ---")
print("="*50)

# Cold sampling (conservative)
probs_cold = gpt.predict_next_token(input_tokens, temperature=0.1)
print(f"\nCOLD (temp=0.1):")
print(f"  Top 3: {[f'#{idx}={p*100:.1f}%' for idx, p in zip(np.argsort(probs_cold)[-3:][::-1], np.sort(probs_cold)[-3:][::-1])]}")
print(f"  → Confident, peaky distribution")

# Normal sampling
probs_normal = gpt.predict_next_token(input_tokens, temperature=1.0)
print(f"\nNORMAL (temp=1.0):")
print(f"  Top 3: {[f'#{idx}={p*100:.1f}%' for idx, p in zip(np.argsort(probs_normal)[-3:][::-1], np.sort(probs_normal)[-3:][::-1])]}")
print(f"  → Natural distribution")

# Hot sampling (adventurous)
probs_hot = gpt.predict_next_token(input_tokens, temperature=2.0)
print(f"\nHOT (temp=2.0):")
print(f"  Top 3: {[f'#{idx}={p*100:.1f}%' for idx, p in zip(np.argsort(probs_hot)[-3:][::-1], np.sort(probs_hot)[-3:][::-1])]}")
print(f"  → Diverse, spread-out distribution")

print("\n" + "-"*70)
print("KEY INSIGHT:")
print("-"*70)
print("""
Temperature controls the "personality" of GPT:

🥶 COLD (0.1-0.5):
   - Focused, deterministic
   - Good for: Facts, code, math
   - "The capital of France is ___" → "Paris" (always)

😐 NORMAL (0.7-1.0):
   - Balanced, natural
   - Good for: General conversation
   - "Once upon a time" → varied but coherent

🔥 HOT (1.2-2.0):
   - Creative, surprising
   - Good for: Stories, poetry, brainstorming
   - "Write a poem about" → unique each time
=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Complete GPT Model")
print("="*70)

print("""
REAL-WORLD ANALOGIES RECAP:
===========================

1. AMAZON FULFILLMENT CENTER:
   - Order arrives → Translator → Warehouse → Stations → Quality → Shipping
   - Maps to: Input → Embeddings → Blocks → Norm → Output

2. LIBRARY CARD CATALOG:
   - Token embedding = Book lookup system
   - Position embedding = Reading order tracker

3. ICE CREAM SHOP:
   - Predicts next flavor based on order history
   - Calculates probabilities, doesn't "know" the answer

4. RESTAURANT KITCHEN:
   - Multiple stations (blocks) refine the dish
   - Each station has specialists (heads)
   - Quality control (LayerNorm) ensures consistency

GPT MODEL COMPONENTS:
=====================

1. TOKEN EMBEDDINGS: Convert token IDs → vectors
   "Like looking up words in a dictionary"

2. POSITION EMBEDDINGS: Add position information
   "Like numbering pages in a book"

3. TRANSFORMER BLOCKS: Process with attention + FFN
   "Like kitchen stations refining a dish"

4. LAYER NORM: Normalize activations
   "Like a food scale ensuring consistent portions"

5. OUTPUT PROJECTION: Convert to vocabulary logits
   "Like printing a menu of options"

FORWARD PASS:
=============
  token_ids → embeddings → blocks → norm → logits → probs
  
  "Customer order → Kitchen → Stations → Quality → Menu → Odds"

KEY PARAMETERS (GPT-2 Small):
=============================
- vocab_size: 50,257 (massive menu!)
- max_seq_len: 1,024 (long meals!)
- embedding_dim: 768 (detailed descriptions)
- num_heads: 12 (specialist chefs)
- num_blocks: 12 (kitchen stations)
- ff_dim: 3,072 (processing capacity)
- Total: ~124M parameters (recipe adjustments!)

WHAT MAKES GPT WORK:
====================
1. Self-attention captures token relationships
2. Multi-head allows multiple perspectives
3. Stacked blocks build hierarchical representations
4. Causal mask enables autoregressive prediction
5. Temperature controls creativity vs. focus
6. Large scale (parameters + data) gives capability

NEXT: Training the model - teaching GPT to predict!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with GPT Model")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. CHANGE MODEL SIZE:
   gpt = GPT(vocab_size=2000, max_seq_len=64, 
             embedding_dim=128, num_heads=8, 
             num_blocks=4, ff_dim=512)
   
   Question: How does output quality change?
   Expectation: Larger model = more nuanced predictions

2. ANALYZE LOGITS:
   logits = gpt.forward(input_tokens)
   print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
   
   Question: What do positive/negative logits mean?
   Answer: Positive = favored, Negative = unlikely

3. TEMPERATURE EFFECT:
   probs_cold = gpt.predict_next_token(input_tokens, temperature=0.1)
   probs_hot = gpt.predict_next_token(input_tokens, temperature=2.0)
   
   Question: How does entropy change?
   Cold: Low entropy (confident)
   Hot: High entropy (uncertain)

4. LONGER INPUT:
   input_tokens = np.arange(20)  # 20 tokens
   logits = gpt.forward(input_tokens)
   print(f"Output shape: {logits.shape}")
   
   Question: Does GPT handle longer sequences?
   Answer: Yes! Up to max_seq_len

5. PREDICTION CONFIDENCE:
   top_prob = probs.max()
   print(f"Top prediction confidence: {top_prob*100:.1f}%")
   
   Question: When is GPT most confident?
   Answer: Depends on learned patterns!

6. VISUALIZE (MENTALLY):
   Imagine probability distribution as a bar chart:
   - Cold: One tall bar, many tiny bars
   - Normal: Few medium bars
   - Hot: Many similar-height bars

KEY TAKEAWAY:
=============
- GPT combines all components to predict next token
- Output is a probability distribution over vocabulary
- Temperature controls sampling behavior
- Model is autoregressive (predicts one token at a time)
- Like a restaurant that learns your preferences!

Next: 07_training.py - Teaching GPT with loss and optimization!
=============================================================================""")
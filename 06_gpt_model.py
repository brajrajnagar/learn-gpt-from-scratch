"""
=============================================================================
LESSON 6: Complete GPT Model - Assembling the Full Architecture
=============================================================================

Now we assemble all components into the complete GPT model!

REAL-WORLD ANALOGY: Building a Complete Restaurant
==================================================

Think of GPT as a restaurant that serves words:

1. ENTRANCE (Input)
   - Customers arrive with orders (token IDs)
   
2. MENU TRANSLATOR (Token Embeddings)
   - Converts order names to kitchen codes
   - "Margherita Pizza" -> Code #42
   
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

+------------------------------------------------------------------+
|                    GPT MODEL                                      |
|                                                                   |
|  CUSTOMER ORDER ARRIVES (Input token IDs)                        |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | ORDER TRANSLATOR (Token Embedding)                       |    |
|  | "Wireless Mouse" -> SKU #12345                           |    |
|  | Converts IDs to dense vectors                            |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | WAREHOUSE LOCATION (Position Embedding)                  |    |
|  | Aisle 1, Shelf 2, Position 3                             |    |
|  | Adds sequence position info                              |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | PROCESSING STATION 1 (Transformer Block 1)               |    |
|  | - Scan item (Attention)                                  |    |
|  | - Package appropriately (FFN)                            |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | PROCESSING STATION 2 (Transformer Block 2)               |    |
|  | - Add shipping labels                                    |    |
|  | - Quality check                                          |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|         ... (N stations total)                                    |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | FINAL INSPECTION (Final Layer Norm)                      |    |
|  | Standardize all packages                                 |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|  +----------------------------------------------------------+    |
|  | SHIPPING LABEL PRINTER (Output Projection)               |    |
|  | Generates delivery options                               |    |
|  | (embedding_dim -> vocab_size)                            |    |
|  +----------------------------------------------------------+    |
|         |                                                         |
|         v                                                         |
|  DELIVERY OPTIONS (Softmax -> Probabilities)                     |
|  - Option A: 45% chance                                          |
|  - Option B: 30% chance                                          |
|  - Option C: 25% chance                                          |
+-------------------------------------------------------------------+

KEY PARAMETERS (GPT-2 Small):
- vocab_size: 50,257 (BPE vocabulary)
- max_seq_len: 1024 tokens
- embedding_dim: 768
- num_heads: 12
- num_blocks: 12
- ff_dim: 3072 (4 x embedding_dim)
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
            mask[i, j] = -1e9
    return mask

# =============================================================================
# STEP 3: Embedding Layers
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Layers - Converting IDs to Vectors")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Library Card Catalog System
===============================================

TOKEN EMBEDDING = Book Lookup
-----------------------------

Imagine a library with 50,000 books (vocabulary):

PATRON REQUEST: "Harry Potter and the Sorcerer's Stone"
  |
LIBRARIAN LOOKS UP in catalog (embedding table)
  |
RETURNS: Book location code [0.23, -0.45, 0.89, ...] (768-dim vector)
  |
This vector uniquely identifies the book!

Each book has a unique embedding (location code):
- "Harry Potter" -> [0.23, -0.45, 0.89, ...]
- "Lord of the Rings" -> [-0.12, 0.67, -0.34, ...]
- "1984" -> [0.45, 0.23, -0.78, ...]

POSITION EMBEDDING = Reading Order
----------------------------------

Books on a shelf need ORDER:
- Book 1: First in series
- Book 2: Second in series
- Book 3: Third in series

Position embeddings add this order information:
- Position 0 -> [0.00, 1.00, 0.00, ...]
- Position 1 -> [0.50, 0.87, 0.12, ...]
- Position 2 -> [0.87, 0.50, 0.34, ...]

Combined = Book + Position
- "Harry Potter" at position 0
- "Chamber of Secrets" at position 1
- "Prisoner of Azkaban" at position 2

This tells GPT both WHAT and WHERE!
=============================================================================""")

class TokenEmbedding:
    """
    Token embedding layer.
    
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
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02
        
        print(f"TokenEmbedding created")
        print(f"  Vocabulary: {vocab_size:,} unique tokens")
        print(f"  Embedding dim: {embedding_dim} features per token")
    
    def forward(self, token_ids):
        """
        Get embeddings for token IDs.
        
        Input: [10, 25, 67] (token IDs = word numbers)
        
        Process:
        - Look up word #10 -> "The" -> [0.1, -0.2, ...]
        - Look up word #25 -> "cat" -> [-0.3, 0.4, ...]
        - Look up word #67 -> "sat" -> [0.5, -0.1, ...]
        
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
        self.weights = np.random.randn(max_seq_len, embedding_dim) * 0.02
        
        print(f"PositionEmbedding created")
        print(f"  Max length: {max_seq_len} positions")
        print(f"  Embedding dim: {embedding_dim} features per position")
    
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
print("STEP 4: Core Components - Reusing Building Blocks")
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
print("STEP 5: Transformer Block - The Processing Unit")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Sushi Assembly Line Station
===============================================

Each transformer block is like a sushi chef station:

INPUT: Rice and fish arrive (embeddings from previous layer)
       |
+---------------------------------------------+
|  STATION 1: PREPARATION (LayerNorm)         |
|  - Measure rice precisely (normalize)       |
|  - Ensure consistent portions               |
|         |                                   |
|  STATION 2: ASSEMBLY (Attention)            |
|  - Chef examines ingredients                |
|  - Understands relationships                |
|  - "Fish goes ON rice, not under"           |
|         |                                   |
|  RESIDUAL: Add original ingredients back    |
|  - Don't lose the original flavor!          |
|         |                                   |
|  STATION 3: PREPARATION (LayerNorm 2)       |
|  - Final measurement check                  |
|         |                                   |
|  STATION 4: SHAPING (FeedForward)           |
|  - Form into proper sushi shape             |
|  - Apply transformation                     |
|         |                                   |
|  RESIDUAL: Add previous stage back          |
|  - Preserve accumulated flavor              |
+---------------------------------------------+
       |
OUTPUT: Finished sushi (enhanced embeddings)
        Same format as input, but transformed!

Multiple blocks = Multiple chef stations in sequence!
Each station refines the sushi further!
=============================================================================""")

class TransformerBlock:
    """
    Complete Transformer Block.
    
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
print("STEP 6: Complete GPT Model - The Full Architecture")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Complete Ice Cream Shop
===========================================

GPT is like an ice cream shop that predicts your next flavor:

+------------------------------------------------------------+
|                    GPT ICE CREAM SHOP                       |
|                                                             |
|  CUSTOMER ORDER (Input token IDs)                          |
|  "I want: Vanilla -> Chocolate -> ?"                       |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | FLAVOR ENCODER (Token Embedding)                       | |
|  | Vanilla -> [0.2, -0.5, 0.8, ...]                       | |
|  | Chocolate -> [-0.3, 0.7, -0.2, ...]                    | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | ORDER SEQUENCE (Position Embedding)                    | |
|  | First flavor + position 0                              | |
|  | Second flavor + position 1                             | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | PROCESSING STATION 1 (Transformer Block 1)             | |
|  | "Customer started with vanilla..."                     | |
|  | Basic pattern recognition                              | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | PROCESSING STATION 2 (Transformer Block 2)             | |
|  | "...then chocolate, likely wants sweet!"               | |
|  | Deeper pattern understanding                           | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|         ... (more stations for deeper understanding)       |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | FINAL QUALITY CHECK (Final LayerNorm)                  | |
|  | Ensure consistent recommendations                      | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|  +--------------------------------------------------------+ |
|  | RECOMMENDATION GENERATOR (Output Projection)           | |
|  | Maps understanding to flavor options                   | |
|  +--------------------------------------------------------+ |
|         |                                                   |
|  FLAVOR PROBABILITIES (Softmax)                            |
|  - Strawberry: 35% <- Most likely!                        |
|  - Cookies & Cream: 25%                                   |
|  - Mint Chip: 20%                                         |
|  - Vanilla: 10%                                           |
|  - ... (50,000+ flavors total)                            |
|         |                                                   |
|  SHOPKEEPER: "I recommend Strawberry!"                    |
+------------------------------------------------------------+

KEY INSIGHT: GPT doesn't "know" the next word.
It calculates PROBABILITIES based on patterns!
=============================================================================""")

class GPT:
    """
    Complete GPT Model - Autoregressive Language Model.
    
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
        
        Setting up a restaurant requires:
        
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
        print("GPT Language Restaurant Opening!")
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
        print("Restaurant is now open for business!")
        print("="*50)
        
        # Calculate approximate parameter count
        self._print_parameter_count(num_blocks)
    
    def _print_parameter_count(self, num_blocks):
        """Print approximate parameter count."""
        emb_params = self.vocab_size * self.embedding_dim
        pos_params = self.max_seq_len * self.embedding_dim
        
        # Per block: attention (4 * d^2) + FFN (2 * d * 4d) + layer norms
        block_params = (4 * self.embedding_dim**2 +
                       8 * self.embedding_dim**2 +
                       4 * self.embedding_dim)
        total_block_params = num_blocks * block_params
        
        output_params = self.embedding_dim * self.vocab_size
        
        total = emb_params + pos_params + total_block_params + output_params
        print(f"\nEstimated investment (parameters): {total:,} ({total/1e6:.1f}M)")
    
    def forward(self, token_ids):
        """
        Forward pass of GPT model.
        
        INPUT: Customer's order history
               token_ids = [10, 25, 67] (appetizer, salad, ?)
        
        STEP 1: Look up each item (Token Embedding)
                10 -> "Spring Rolls" -> [0.2, -0.5, ...]
                25 -> "Caesar Salad" -> [-0.3, 0.7, ...]
                67 -> "Soup" -> [0.1, 0.4, ...]
        
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
        
        # Step 1: Token embeddings
        token_embs = self.token_embedding.forward(token_ids)
        
        # Step 2: Position embeddings
        pos_embs = self.position_embedding.forward(seq_len)
        
        # Step 3: Combine (token + position)
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
        last_logits = logits[-1]
        
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
    vocab_size=1000,      # Small vocab for demo
    max_seq_len=128,      # Short sequences
    embedding_dim=64,     # Small embedding
    num_heads=4,          # Fewer heads
    num_blocks=2,         # Just 2 blocks
    ff_dim=256            # Smaller FFN
)

print("\n" + "-"*70)
print("Processing a customer order...")
print("-"*70)

# Simulate input token IDs
np.random.seed(42)
input_tokens = np.array([10, 25, 67, 89, 123, 45, 78, 234])
print(f"\nCustomer order history: {input_tokens}")
print(f"  -> {len(input_tokens)} courses ordered so far")

# Forward pass
print(f"\nSending order through kitchen...")
logits = gpt.forward(input_tokens)
print(f"\nKitchen recommendations:")
print(f"  Output shape: {logits.shape}")
print(f"  -> Scores for all {logits.shape[1]} menu items at each position!")

# Get next token probabilities
print(f"\nPredicting next course...")
probs = gpt.predict_next_token(input_tokens)
print(f"  Probability distribution shape: {probs.shape}")

# Find most likely next tokens
top_indices = np.argsort(probs)[-10:][::-1]
print(f"\nTop 10 recommended next courses:")
for i, idx in enumerate(top_indices):
    confidence = "***" if probs[idx] > 0.05 else "**" if probs[idx] > 0.02 else "*"
    print(f"  {i+1}. Item #{idx}: {probs[idx]*100:.2f}% chance {confidence}")

# =============================================================================
# SUMMARY: Complete GPT Model
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Complete GPT Architecture")
print("="*70)

print("""
WHAT WE BUILT:
==============
1. Token Embeddings - Convert IDs to vectors
2. Position Embeddings - Add sequence order
3. Transformer Blocks - Process and understand
4. Final LayerNorm - Stabilize output
5. Output Projection - Map to vocabulary
6. Softmax - Convert to probabilities

COMPLETE FLOW:
==============

Input: "The cat sat on the" (token IDs)
  |
  v
Token Embeddings: Look up vectors
  |
  v
Position Embeddings: Add position info
  |
  v
Transformer Block 1: Basic patterns
  |
  v
Transformer Block 2: Deeper patterns
  |
  v
... (more blocks)
  |
  v
Final LayerNorm: Normalize
  |
  v
Output Projection: vocab_size logits
  |
  v
Softmax: probabilities for each word
  |
  v
Sample: "mat" (most likely next word)

HOW THIS CONNECTS TO GPT:
=========================

GPT-2 Small:
  - vocab_size = 50,257
  - embedding_dim = 768
  - num_heads = 12
  - num_blocks = 12
  - ff_dim = 3072
  - Total params: ~124M

GPT-3 Large:
  - vocab_size = 50,257
  - embedding_dim = 12288
  - num_heads = 96
  - num_blocks = 96
  - ff_dim = 49152
  - Total params: ~175B

SAME ARCHITECTURE, different scale!

NEXT: Training the Model
========================
Now we have the complete GPT architecture!
Next, we learn how to TRAIN it:
- Loss functions (cross-entropy)
- Backpropagation (gradient descent)
- Training loop (iterate over data)
- Evaluation (perplexity)

Next: 07_training.py
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Experiment with GPT Architecture")
print("="*70)

print("""
Try these experiments:

1. CHANGE VOCABULARY SIZE:
   gpt = GPT(vocab_size=5000, ...)  # Larger vocab
   
   Question: How does this affect parameters?
   Answer: More vocabulary = more embedding params

2. CHANGE NUMBER OF BLOCKS:
   gpt = GPT(num_blocks=4, ...)  # Deeper model
   
   Question: How does depth affect output?
   Answer: More blocks = deeper understanding

3. CHANGE EMBEDDING DIMENSION:
   gpt = GPT(embedding_dim=128, ...)  # Richer embeddings
   
   Question: How does this affect capacity?
   Answer: Larger dim = more expressive power

4. TEMPERATURE SCALING:
   probs = gpt.predict_next_token(tokens, temperature=0.5)
   probs = gpt.predict_next_token(tokens, temperature=2.0)
   
   Question: How does temperature affect predictions?
   Answer: Low = confident, High = diverse/random

KEY TAKEAWAY:
=============
GPT = Embeddings + Transformer Blocks + Output Projection!
- Token embeddings convert IDs to vectors
- Position embeddings add order
- Transformer blocks process and understand
- Output projection maps to vocabulary
- Softmax gives probabilities

This is the COMPLETE autoregressive language model!
=============================================================================""")
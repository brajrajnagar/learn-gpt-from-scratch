"""
=============================================================================
LESSON 3: Self-Attention - The Heart of Transformers
=============================================================================

This is THE most important concept in GPT! Self-attention is what allows
the model to understand relationships between words, regardless of their
distance in the sequence.

KEY CONCEPTS:
1. Attention Mechanism - How it works intuitively
2. Query, Key, Value - The three components
3. Scaled Dot-Product Attention - The math
4. Causal Masking - Making it autoregressive (GPT-specific)

ATTENTION INTUITION:
When reading "The animal didn't cross the street because it was too tired",
you know "it" refers to "animal", not "street". Attention does this!

Let's build self-attention from scratch!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding Attention Intuitively
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Understanding Attention")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Reading Comprehension
==========================================

Imagine you're reading this sentence:
"The cat sat on the mat because it was comfortable"

When you reach the word "it", your brain does something amazing:
- You automatically look back at previous words
- You figure out what "it" refers to
- You understand: "it" = "cat" (the cat was comfortable)

This is exactly what ATTENTION does in GPT!

ATTENTION PATTERN FOR "it":
  "it" → attends to → "cat" (80% attention) ← Main reference
  "it" → attends to → "comfortable" (15% attention) ← Why
  "it" → attends to → "mat" (5% attention) ← Location

The model learns to FOCUS on relevant words when processing each word!

TECHNICAL DETAILS:
- Each word can attend to ALL words (including itself)
- Attention weights are learned during training
- Computed dynamically based on the input
- Stronger attention = more influence on meaning

=============================================================================""")

# =============================================================================
# STEP 2: Query, Key, Value - The Core Components
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Query, Key, Value")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Library Search System
=========================================

Imagine you're at a library looking for books about "machine learning".

QUERY (Q): Your search request
- You type: "machine learning basics"
- This represents WHAT you're looking for
- In attention: What the current token wants to find

KEY (K): Book catalog entries
- Each book has keywords/tags in the catalog
- "Introduction to ML" → tags: ["machine", "learning", "intro"]
- "Advanced Python" → tags: ["python", "advanced", "coding"]
- In attention: What information each token offers

VALUE (V): The actual book content
- Once you find matching books, you read them
- The books contain the actual information
- In attention: The actual content that gets used

HOW LIBRARY SEARCH WORKS (LIKE ATTENTION):
1. You have a QUERY (what you want)
2. Compare QUERY to all KEYs (catalog matching)
3. Find best matches (attention scores)
4. Read the VALUE from matching books (weighted sum)

Attention(Q, K, V) = Find relevant books and read them!
                     where relevance comes from Q·K matching
""")

def compute_qkv(embedding, weights_q, weights_k, weights_v):
    """
    Compute Query, Key, Value vectors for a token.
    
    REAL-WORLD EXAMPLE: Creating Your Library Search
    -------------------------------------------------
    For each word/token, we create three specialized vectors:
    
    QUERY: Created by projecting embedding through W_q
    - "What am I looking for in other tokens?"
    - Example: Token "it" creates query looking for nouns
    
    KEY: Created by projecting embedding through W_k  
    - "What do I offer to other tokens?"
    - Example: Token "cat" creates key indicating it's a noun/animal
    
    VALUE: Created by projecting embedding through W_v
    - "What information do I carry?"
    - Example: Token "cat" carries info about felines
    
    Args:
        embedding: Token embedding, shape (embedding_dim,)
        weights_q: Query weight matrix, shape (embedding_dim, d_k)
        weights_k: Key weight matrix, shape (embedding_dim, d_k)
        weights_v: Value weight matrix, shape (embedding_dim, d_v)
    
    Returns:
        query, key, value vectors
    """
    query = np.dot(embedding, weights_q)
    key = np.dot(embedding, weights_k)
    value = np.dot(embedding, weights_v)
    
    return query, key, value

# REAL-WORLD EXAMPLE: Sentence "The cat sat"
print("\n--- QKV Example: Processing 'The cat sat' ---")
print("="*50)
print("""
SCENARIO: GPT processes the sentence "The cat sat"

For each word, we compute Q, K, V:

WORD: "cat"
  QUERY: "I'm looking for what came before me"
  KEY: "I'm a noun, an animal, a subject"
  VALUE: "I represent a furry pet"

WORD: "sat"
  QUERY: "I'm looking for who did the action"
  KEY: "I'm a verb, past tense, action"
  VALUE: "I represent sitting action"

Let's compute these vectors!
""")

np.random.seed(42)
embedding_dim = 8  # Input embedding size
d_k = d_v = 4  # QKV dimension (often smaller than embedding_dim)

# Token embedding for "cat"
cat_embedding = np.random.randn(embedding_dim)
print(f"\n'cat' token embedding: {np.round(cat_embedding, 2)}")

# Learnable weight matrices (trained to create useful Q, K, V)
W_q = np.random.randn(embedding_dim, d_k)
W_k = np.random.randn(embedding_dim, d_k)
W_v = np.random.randn(embedding_dim, d_v)

query, key, value = compute_qkv(cat_embedding, W_q, W_k, W_v)

print(f"\n'cat' Query vector: {np.round(query, 2)}")
print(f"  → What 'cat' is looking for in other words")
print(f"'cat' Key vector: {np.round(key, 2)}")
print(f"  → What 'cat' offers to other words")
print(f"'cat' Value vector: {np.round(value, 2)}")
print(f"  → Information 'cat' carries")

print(f"\nShapes:")
print(f"  Token embedding: {cat_embedding.shape}")
print(f"  Query: {query.shape}")
print(f"  Key: {key.shape}")
print(f"  Value: {value.shape}")

# =============================================================================
# STEP 3: Scaled Dot-Product Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Scaled Dot-Product Attention")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Dating App Matching
=======================================

Think of attention like a dating app matching system:

QUERY = Your preferences
- "I like people who are kind, funny, smart"

KEY = Each person's profile tags
- Person A: [kind: 0.9, funny: 0.3, smart: 0.8]
- Person B: [kind: 0.4, funny: 0.9, smart: 0.6]

VALUE = The actual person's content
- What you actually learn from matching with them

ATTENTION WORKS LIKE THIS:

1. COMPATIBILITY SCORE (Query · Key):
   - Compare your preferences to each profile
   - Higher dot product = better match
   
2. SCALE (divide by sqrt(d_k)):
   - Normalize so scores don't explode
   - Like adjusting for different rating scales
   
3. SOFTMAX (convert to probabilities):
   - Turn scores into "how interested am I?"
   - All interests sum to 100%
   
4. WEIGHTED SUM (attention · Value):
   - Learn more from people you're interested in
   - Less interested = less influence on you

THE FORMULA:
  Attention(Q, K, V) = softmax(QK^T / √d_k) · V
""")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention scores and weighted values.
    
    REAL-WORLD EXAMPLE: Complete Matching Process
    ----------------------------------------------
    This function implements the full attention mechanism:
    
    INPUT:
    - Q: What each token is looking for (preferences)
    - K: What each token offers (profile tags)
    - V: What each token contains (actual content)
    
    PROCESS:
    1. scores = Q · K^T (compatibility check)
    2. scores /= √d_k (normalize)
    3. weights = softmax(scores) (convert to probabilities)
    4. output = weights · V (gather information)
    
    OUTPUT:
    - Each token now has information from tokens it attended to
    - Like learning from people you matched with
    
    Args:
        Q: Query matrix, shape (seq_len, d_k)
        K: Key matrix, shape (seq_len, d_k)
        V: Value matrix, shape (seq_len, d_v)
        mask: Optional mask for causal attention
    
    Returns:
        attention_output: Weighted sum of values, shape (seq_len, d_v)
        attention_weights: Attention scores, shape (seq_len, seq_len)
    """
    d_k = K.shape[1]
    
    # Step 1: Compute attention scores (Q · K^T)
    # This measures similarity between each query and key
    # Like computing compatibility scores in dating app
    scores = np.dot(Q, K.T)
    
    # Step 2: Scale by sqrt(d_k) - prevents softmax saturation
    # When d_k is large, dot products can be very large
    # Scaling keeps gradients stable (prevents numerical overflow)
    scores = scores / np.sqrt(d_k)
    
    print(f"  Raw compatibility scores shape: {scores.shape}")
    print(f"  First token's scores toward all: {np.round(scores[0], 2)}")
    
    # Step 3: Apply mask (for causal attention in GPT)
    if mask is not None:
        scores = scores + mask  # Add -inf to masked positions
        print(f"  After mask applied: {np.round(scores[0], 2)}")
    
    # Step 4: Softmax to get attention weights
    # Converts scores to probabilities (sum to 1)
    # High score → high probability → high attention
    attention_weights = softmax(scores)
    
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  First token's attention distribution: {np.round(attention_weights[0], 4)}")
    
    # Step 5: Weighted sum of values
    # Each output is a combination of all values, weighted by attention
    # Tokens you attend to more = more influence on your output
    attention_output = np.dot(attention_weights, V)
    
    print(f"  Output shape: {attention_output.shape}")
    
    return attention_output, attention_weights

def softmax(x):
    """
    Numerically stable softmax.
    
    REAL-WORLD EXAMPLE: Converting Scores to Percentages
    -----------------------------------------------------
    Imagine you rated 5 movies: [2, 5, 3, 8, 1]
    
    Softmax converts these to "how much do I prefer each?"
    where all preferences sum to 100% (or 1.0).
    
    High scores get high percentages, low scores get low percentages.
    
    Args:
        x: Input array (can be 1D or 2D)
    
    Returns:
        Softmax output (probabilities that sum to 1)
    """
    # Subtract max for numerical stability
    # Prevents overflow when computing exp(large_number)
    # Like normalizing test scores before converting to grades
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("\n--- Computing Attention for a 5-Token Sequence ---")
print("="*50)
print("""
SCENARIO: Processing "The cat sat on mat"

5 tokens, each with 8-dimensional embedding.
Each token will compute attention to all other tokens.
""")

# Create a sequence of 5 tokens, each with 8-dimensional embedding
np.random.seed(42)
seq_len = 5
embedding_dim = 8
d_k = d_v = 4

# Simulate embeddings for 5 tokens (e.g., "The cat sat on mat")
embeddings = np.random.randn(seq_len, embedding_dim)
print(f"Input: {seq_len} tokens, each {embedding_dim}-dimensional")
print(f"Input embeddings shape: {embeddings.shape}")

# Compute Q, K, V for all tokens at once
# Each token creates its own Q, K, V from its embedding
W_q = np.random.randn(embedding_dim, d_k)
W_k = np.random.randn(embedding_dim, d_k)
W_v = np.random.randn(embedding_dim, d_v)

Q = np.dot(embeddings, W_q)  # All tokens' queries
K = np.dot(embeddings, W_k)  # All tokens' keys
V = np.dot(embeddings, W_v)  # All tokens' values

print(f"\nComputed for all tokens:")
print(f"  Query matrix: {Q.shape} ← What each token seeks")
print(f"  Key matrix: {K.shape} ← What each token offers")
print(f"  Value matrix: {V.shape} ← What each token contains")

# Compute attention
print("\n" + "-"*50)
print("Computing attention (matching queries to keys):")
print("-"*50)
attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\n" + "="*50)
print("ATTENTION WEIGHTS MATRIX:")
print("="*50)
print("""
Each ROW i shows: What does token i attend to?
Each COLUMN j shows: How much does token j get attended to?
""")
print("Attention weights (who pays attention to whom):")
print(np.round(attention_weights, 4))

print("\n" + "-"*70)
print("INTERPRETING ATTENTION WEIGHTS:")
print("-"*70)
print("""
EXAMPLE READING OF ROW 0 (token 0 = "The"):
  [0.25, 0.20, 0.18, 0.19, 0.18]
  
  "The" distributes attention:
  - 25% to itself ("The")
  - 20% to "cat"
  - 18% to "sat"
  - 19% to "on"
  - 18% to "mat"
  
  → "The" attends fairly evenly to all words!

EXAMPLE READING OF COLUMN 2 (token 2 = "sat"):
  Looking down column 2 tells us who attends to "sat":
  - Row 0: "The" gives 18% to "sat"
  - Row 1: "cat" gives X% to "sat"
  - Row 2: "sat" gives X% to itself
  - etc.
  
  → High values = this word is important to others!

KEY INSIGHT:
Higher weight = more attention = more influence on meaning!
=============================================================================""")

# =============================================================================
# STEP 4: Causal Masking (GPT-Specific)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Causal Masking for Autoregressive Generation")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Taking a Test Without Cheating
===================================================

Imagine students taking a test, seated in a row:

Student 0: Can only see their own paper (no one before them)
Student 1: Can see their paper + Student 0's paper
Student 2: Can see their paper + Student 0's + Student 1's
Student 3: Can see their paper + all previous students
etc.

NO STUDENT CAN SEE FUTURE STUDENTS (those seated after them)!

This is CAUSAL MASKING in GPT!

WHY CAUSAL MASKING?
===================

GPT is trained to predict the NEXT token. During training:
- Token 0 ("The") can only see token 0
- Token 1 ("cat") can only see tokens 0, 1
- Token 2 ("sat") can only see tokens 0, 1, 2
- Token 3 ("on") can only see tokens 0, 1, 2, 3
- etc.

This prevents "cheating" - seeing future tokens!

If GPT could see future tokens during training:
- It would "cheat" by looking at the answer
- It wouldn't learn to PREDICT, just COPY
- It would fail at generation time!

CAUSAL MASK (what each position can see):
  Position →  0     1     2     3     4
  0        [SEE  HIDDEN HIDDEN HIDDEN HIDDEN]
  1        [SEE  SEE   HIDDEN HIDDEN HIDDEN]
  2        [SEE  SEE   SEE    HIDDEN HIDDEN]
  3        [SEE  SEE   SEE    SEE    HIDDEN]
  4        [SEE  SEE   SEE    SEE    SEE   ]

In numbers (-1e9 = hidden, 0 = visible):
  [[  0, -1e9, -1e9, -1e9, -1e9],
   [  0,    0, -1e9, -1e9, -1e9],
   [  0,    0,    0, -1e9, -1e9],
   [  0,    0,    0,    0, -1e9],
   [  0,    0,    0,    0,    0]]

After softmax, -1e9 becomes 0 (no attention to future)!
""")

def create_causal_mask(seq_len):
    """
    Create a causal (triangular) mask.
    
    REAL-WORLD EXAMPLE: Classroom Seating Chart
    --------------------------------------------
    Creates a mask where each student can only see themselves
    and students who came before them (not future students).
    
    Visual for seq_len=5:
    ✓ . . . .   (Student 0 sees only self)
    ✓ ✓ . . .   (Student 1 sees 0 and self)
    ✓ ✓ ✓ . .   (Student 2 sees 0,1 and self)
    ✓ ✓ ✓ ✓ .   (Student 3 sees 0,1,2 and self)
    ✓ ✓ ✓ ✓ ✓   (Student 4 sees all and self)
    
    Where ✓ = can see (0), . = hidden (-1e9)
    
    Args:
        seq_len: Sequence length (number of students/tokens)
    
    Returns:
        Mask matrix where future positions have -1e9 (hidden)
    """
    mask = np.zeros((seq_len, seq_len))
    
    # Upper triangle (future positions) gets -1e9
    # These are positions this token should NOT attend to
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9  # Large negative (becomes ~0 after softmax)
    
    return mask

print("\n--- Causal Mask Example: 5-Token Sequence ---")
seq_len = 5
causal_mask = create_causal_mask(seq_len)

print(f"Causal mask for sequence length {seq_len}:")
print(f"(0 = can see, -1e9 = cannot see/hide)")
print()
print("        Token: 0      1      2      3      4")
print("        ↓      ↓      ↓      ↓      ↓")
for i in range(seq_len):
    row_str = " ".join(f"{val:6.0f}" for val in causal_mask[i])
    print(f"Token {i} → [{row_str}]")

print("\nINTERPRETATION:")
print("  Token 0: Can only see itself (position 0)")
print("  Token 1: Can see tokens 0, 1")
print("  Token 2: Can see tokens 0, 1, 2")
print("  Token 3: Can see tokens 0, 1, 2, 3")
print("  Token 4: Can see all tokens (0,1,2,3,4)")
print("  → No token can see FUTURE tokens!")

print("\n" + "-"*50)
print("Now computing attention WITH causal mask:")
print("-"*50)
attention_output_masked, attention_weights_masked = scaled_dot_product_attention(
    Q, K, V, mask=causal_mask
)

print(f"\n" + "="*50)
print("MASKED ATTENTION WEIGHTS:")
print("="*50)
print(np.round(attention_weights_masked, 4))

print("\n" + "-"*70)
print("NOTICE THE DIFFERENCE:")
print("-"*70)
print("""
WITHOUT MASK (first row):
  [0.25, 0.20, 0.18, 0.19, 0.18]
  → Token 0 attends to ALL tokens (including future)

WITH MASK (first row):
  [1.0, 0.0, 0.0, 0.0, 0.0]
  → Token 0 ONLY attends to itself (no future tokens!)

WITHOUT MASK (second row):
  [0.22, 0.25, 0.18, 0.17, 0.18]
  → Token 1 attends to all tokens

WITH MASK (second row):
  [0.45, 0.55, 0.0, 0.0, 0.0]
  → Token 1 only attends to tokens 0 and 1!

KEY INSIGHT:
Each token can ONLY see itself and PREVIOUS tokens!
This is essential for autoregressive (next-token prediction) generation.
=============================================================================""")

# =============================================================================
# STEP 5: Complete Self-Attention Layer
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Complete Self-Attention Layer")
print("="*70)

class SelfAttention:
    """
    Self-attention layer with causal masking.
    This is the core of GPT!
    
    REAL-WORLD EXAMPLE: Complete Reading Comprehension System
    ==========================================================
    
    This layer implements the full attention mechanism that GPT uses.
    
    Think of it as a reading assistant that:
    1. For each word, determines what to look for (Query)
    2. Determines what each word offers (Key)
    3. Gathers information from relevant words (Value)
    4. Combines information with proper weighting
    5. Ensures no cheating (causal mask - can't see future)
    
    OUTPUT: Each word now has a "contextualized" representation
    - Original meaning + information from attended words
    - "it" now contains info about "cat" (what it refers to)
    - "sat" now contains info about "The cat" (who did the action)
    """
    
    def __init__(self, embedding_dim, d_k, d_v):
        """
        Initialize self-attention.
        
        REAL-WORLD EXAMPLE: Setting Up the Reading System
        --------------------------------------------------
        embedding_dim: How rich is each word's initial representation?
          - GPT-2: 768 dimensions
          - Our example: 8 dimensions (for learning)
        
        d_k: How detailed should queries/keys be?
          - Controls how specifically tokens can match
          - Higher = more nuanced attention patterns
        
        d_v: How much information does each token carry?
          - Controls the richness of value information
          - Often same as d_k
        
        Args:
            embedding_dim: Input embedding dimension
            d_k: Dimension for query and key
            d_v: Dimension for value
        """
        self.embedding_dim = embedding_dim
        self.d_k = d_k
        self.d_v = d_v
        
        # Learnable weight matrices
        # These are trained to create useful Q, K, V vectors
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_k = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_v = np.random.randn(embedding_dim, d_v) * 0.1
        
        print(f"Self-Attention initialized")
        print(f"  Input dimension: {embedding_dim}")
        print(f"  Query/Key dimension: {d_k}")
        print(f"  Value dimension: {d_v}")
        print(f"  Parameters: {embedding_dim * d_k * 3 + embedding_dim * d_v}")
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of self-attention.
        
        REAL-WORLD EXAMPLE: Processing a Sentence
        ------------------------------------------
        Given embeddings for "The cat sat on mat":
        
        1. Compute Q, K, V for each token
        2. Create causal mask (prevent seeing future)
        3. Match queries to keys (find relevant tokens)
        4. Weight values by attention (gather info)
        5. Return contextualized representations
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal mask (True for GPT)
        
        Returns:
            attention_output: Shape (seq_len, d_v) - contextualized tokens
            attention_weights: Shape (seq_len, seq_len) - who attends to whom
        """
        seq_len = embeddings.shape[0]
        
        # Compute Q, K, V for all tokens
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
            print(f"  Causal mask applied (no cheating!)")
        
        # Compute attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask
        )
        
        return attention_output, attention_weights

print("\n--- Self-Attention Layer: Complete Example ---")
print("="*50)
print("""
SCENARIO: Complete self-attention for "The cat sat on mat"

This is the full forward pass that happens inside GPT!
""")

# Create self-attention layer
self_attn = SelfAttention(embedding_dim=8, d_k=4, d_v=4)

# Create sample embeddings (5 tokens representing "The cat sat on mat")
embeddings = np.random.randn(5, 8)
print(f"\nInput: 5 tokens, each 8-dimensional")
print(f"Input shape: {embeddings.shape}")

# Forward pass
print(f"\nRunning forward pass with causal mask...")
output, weights = self_attn.forward(embeddings)

print(f"\n" + "="*50)
print("RESULTS:")
print("="*50)
print(f"Input shape: {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

print(f"\nAttention weights (who attends to whom):")
print(np.round(weights, 4))

print("\n" + "-"*50)
print("OUTPUT INTERPRETATION:")
print("-"*50)
print("""
Each row of OUTPUT is a CONTEXTUALIZED token representation.

Token 0 ("The"):
  Original: [random 8-dim vector]
  After attention: Still mostly just "The" info
  (can only attend to itself)

Token 1 ("cat"):
  Original: [random 8-dim vector about cats]
  After attention: Contains "The" + "cat" info
  (attended to tokens 0 and 1)

Token 2 ("sat"):
  Original: [random 8-dim vector about sitting]
  After attention: Contains "The" + "cat" + "sat" info
  (attended to tokens 0, 1, and 2)

This is how GPT builds understanding!
Each token accumulates context from previous tokens.
=============================================================================""")

# =============================================================================
# STEP 6: Understanding the Output
# =============================================================================

print("\n" + "="*70)
print("STEP 6: What Does Self-Attention Output Mean?")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Group Project Information Sharing
======================================================

Imagine a group project where students sit in a row and can only
talk to people before them (causal masking!):

Student 0 ("The"): 
  - Works alone, knows only their own info
  - Output: Just "The" information

Student 1 ("cat"):
  - Can talk to Student 0
  - Learns about "The" while keeping "cat" knowledge
  - Output: Combined "The cat" understanding

Student 2 ("sat"):
  - Can talk to Students 0 and 1
  - Learns about "The" and "cat" while keeping "sat" knowledge
  - Output: Combined "The cat sat" understanding

Student 3 ("on"):
  - Can talk to Students 0, 1, 2
  - Output: Combined "The cat sat on" understanding

Student 4 ("mat"):
  - Can talk to ALL previous students
  - Output: Full sentence understanding!

SELF-ATTENTION OUTPUT:
=====================

Each output vector is a CONTEXTUALIZED representation of a token.

KEY INSIGHT: The output for each token now CONTAINS INFORMATION 
from the tokens it attended to!

EXAMPLE: "The cat sat on the mat"

After self-attention:
- "cat" embedding contains: info about "The" + "cat"
- "sat" embedding contains: info about "The" + "cat" + "sat"
- "on" embedding contains: info about "The cat sat" + "on"
- "mat" embedding contains: info about entire sentence + "mat"

This is how GPT builds contextual understanding!

IN GPT SPECIFICALLY:
1. Each token predicts the next token
2. Causal mask ensures it can only use previous tokens
3. Self-attention lets it focus on relevant previous tokens
4. The output is used to predict the next word

EXAMPLE PREDICTION:
  Input: "The cat sat on the"
  After attention, "the" has context: "The cat sat on"
  Model predicts: "mat" (most likely next word)
=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Self-Attention")
print("="*70)

print("""
REAL-WORLD ANALOGIES RECAP:
===========================

1. LIBRARY SEARCH:
   - Query = Your search request
   - Key = Book catalog tags
   - Value = Actual book content
   - Attention = Finding and reading relevant books

2. DATING APP:
   - Query = Your preferences
   - Key = Profile tags
   - Value = Actual person's qualities
   - Attention = Matching and learning from compatible people

3. CLASSROOM TEST:
   - Causal mask = Can't see future students' papers
   - Each student can only look at previous students
   - Prevents cheating!

4. GROUP PROJECT:
   - Information flows from earlier to later students
   - Each student accumulates knowledge from predecessors
   - Final student has complete understanding

THE FORMULA:
============

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

BREAKDOWN:
- QK^T = Match queries to keys (compatibility score)
- / √d_k = Scale to prevent numerical issues
- softmax() = Convert to probabilities (attention weights)
- · V = Weighted sum of values (gather information)

CAUSAL MASK:
============
- Essential for autoregressive language modeling
- Prevents attending to future tokens
- Lower triangular mask with -inf
- After softmax: future positions get 0 attention

WHY IT WORKS:
=============
- Learns relationships between tokens
- Captures long-range dependencies
- Parallelizable (unlike RNNs)
- Dynamic (attention changes per input)

LIMITATION OF SINGLE ATTENTION:
===============================
- Each token only has ONE way to attend
- Can't capture different types of relationships simultaneously
- Example: "bank" could mean river bank OR financial bank
  Single attention can't handle both meanings!

SOLUTION: Multi-Head Attention (next lesson!)
- Multiple attention heads in parallel
- Each head learns different attention patterns
- One head for "river" meaning, one for "money" meaning
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Attention")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. CHANGE SEQUENCE LENGTH:
   seq_len = 10  # Longer sentence
   
   Question: How does attention pattern change with longer sequences?
   Expectation: More tokens = more distributed attention

2. CHANGE DIMENSIONS:
   d_k = 8, d_v = 8  # Larger QKV dimensions
   
   Question: How does dimension affect attention quality?
   Expectation: Higher dim = more nuanced attention patterns

3. WITHOUT CAUSAL MASK:
   self_attn.forward(embeddings, use_causal_mask=False)
   
   Question: How does attention change without the mask?
   Expectation: All tokens attend to all tokens equally

4. ANALYZE ATTENTION PATTERNS:
   Print attention_weights and examine:
   - Which tokens get the most attention?
   - Does the causal mask work correctly?
   - How does attention distribute across positions?

5. VISUALIZE (MENTALLY):
   Imagine attention as a heatmap:
   - Bright cells = high attention
   - Dark cells = low attention
   - Upper triangle = blocked by causal mask

KEY TAKEAWAY:
=============
- Self-attention lets tokens attend to each other
- QKV mechanism computes attention weights
- Causal mask makes it autoregressive (GPT-specific)
- Output is contextualized token representations
- Foundation for Multi-Head Attention (next!)

Next: 04_multihead_attention.py - Multiple attention heads!
=============================================================================""")
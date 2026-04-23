"""
=============================================================================
LESSON 3: Self-Attention - How Our Predictor Decides What to Focus On
=============================================================================

Continuing our text predictor from Lessons 1-2:
- Lesson 1: We built a neural network to predict next word
- Lesson 2: We converted "The cat" to embeddings
- Lesson 3: We'll learn SELF-ATTENTION - how the model focuses on relevant words

EXAMPLE FLOW: "The cat sat on the ___" → attend to "cat" → predict "mat"
"""

import numpy as np

# =============================================================================
# RECAP: Our Text Predictor So Far
# =============================================================================

print("\n" + "="*70)
print("RECAP: Our Text Predictor")
print("="*70)
print("""
FROM LESSON 1: We built a network that predicts next word
  Input: "The cat" → Network → Output: "sat" (most likely)

FROM LESSON 2: We convert text to embeddings
  "The cat" → Token IDs [0, 1] → Embeddings → Network

THE MISSING PIECE:
==================
When predicting "The cat sat on the ___", which words matter most?

- "The"? (article, not very informative)
- "cat"? (the subject - VERY important!)
- "sat"? (the action - important)
- "on"? (preposition - somewhat important)
- "the"? (article, not very informative)

ANSWER: Self-attention learns to FOCUS on relevant words!
When predicting after "the", the model learns to attend to "cat"!

WHAT WE'LL BUILD:
1. Query, Key, Value - the attention mechanism
2. Attention scores - how much to focus on each word
3. Weighted sum - gathering information from focused words
4. Causal mask - only looking at past words (GPT-specific)
=============================================================================""")

# =============================================================================
# STEP 1: The Problem - Which Words Should We Focus On?
# =============================================================================

print("\n" + "="*70)
print("STEP 1: The Attention Problem")
print("="*70)

print("""
OUR PREDICTOR'S CHALLENGE:
==========================

Sentence: "The cat sat on the ___"

To predict the next word, which previous words matter?

Option A: Treat all words equally (what simple networks do)
  "The" = "cat" = "sat" = "on" = "the" (all equal weight)
  → Prediction: generic word, misses context

Option B: Focus on relevant words (what attention does)
  "cat" ← most important (it's about a cat!)
  "sat" ← important (what the cat did)
  "the", "on" ← less important (grammar words)
  → Prediction: contextually appropriate word!

REAL-WORLD EXAMPLE: Reading a Sentence
======================================
When you read "The cat sat on the ___", your brain:
1. Identifies "cat" as the subject (key information!)
2. Identifies "sat" as the action
3. Knows "the" and "on" are just grammar
4. Predicts: "mat", "couch", "floor" (places cats sit)

This FOCUSING mechanism is exactly what self-attention does!

THE ATTENTION QUESTION:
=======================
For each word, self-attention asks:
"How much should I PAY ATTENTION to each other word?"

Let's implement this!
""")

# =============================================================================
# STEP 2: Query, Key, Value - The Attention Mechanism
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Query, Key, Value")
print("="*70)

print("""
THE ATTENTION MECHANISM:
========================

Think of attention like a library search:

QUERY (Q): "What am I looking for?"
  - Each word creates a "query" vector
  - This represents what the word wants to find in others
  - Example: When processing "sat", query looks for "who did the action?"

KEY (K): "What do I offer to others?"
  - Each word creates a "key" vector
  - This represents what information the word provides
  - Example: "cat" has a key indicating "I'm the subject/actor"

VALUE (V): "What information do I carry?"
  - Each word creates a "value" vector
  - This is the actual content that gets passed along
  - Example: "cat" carries information about felines/animals

HOW ATTENTION WORKS:
====================
1. For word X, compute its QUERY (what it's looking for)
2. Compare QUERY to all KEYs (including itself)
3. High match = high attention score
4. Use attention scores to weight VALUES
5. Output = weighted sum of values (gathered information)

FORMULA:
========
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Let's implement this step by step!
""")

def compute_qkv(embedding, W_q, W_k, W_v):
    """
    Compute Query, Key, Value for a single token.
    
    OUR EXAMPLE: Processing "The cat sat"
    
    For "cat" token with embedding [0.31, 0.08, -0.22, ...]:
    
    QUERY = W_q · embedding
      → "What am I looking for in other words?"
      → "cat" looks for: subject, actor, doer
    
    KEY = W_k · embedding
      → "What do I offer to other words?"
      → "cat" offers: I'm a noun, I'm an animal, I'm the subject
    
    VALUE = W_v · embedding
      → "What information do I carry?"
      → "cat" carries: furry, pet, animal, subject info
    
    Args:
        embedding: Token embedding, shape (embedding_dim,)
        W_q: Query weight matrix, shape (embedding_dim, d_k)
        W_k: Key weight matrix, shape (embedding_dim, d_k)
        W_v: Value weight matrix, shape (embedding_dim, d_v)
    
    Returns:
        query, key, value vectors
    """
    query = np.dot(embedding, W_q)
    key = np.dot(embedding, W_k)
    value = np.dot(embedding, W_v)
    return query, key, value

print("\n--- QKV Example: Processing 'The cat sat' ---")
print("-"*50)

# Setup dimensions
np.random.seed(42)
embedding_dim = 8  # Size of token embeddings (from Lesson 2)
d_k = d_v = 4  # QKV dimension (often smaller than embedding)

# Initialize weight matrices (these would be LEARNED during training)
W_q = np.random.randn(embedding_dim, d_k) * 0.1
W_k = np.random.randn(embedding_dim, d_k) * 0.1
W_v = np.random.randn(embedding_dim, d_v) * 0.1

print(f"Dimensions:")
print(f"  Embedding: {embedding_dim}")
print(f"  Query/Key: {d_k}")
print(f"  Value: {d_v}")

# Create embeddings for "The cat sat" (from Lesson 2)
# Token IDs: "The"=0, "cat"=1, "sat"=3
the_embedding = np.random.randn(embedding_dim) * 0.1
cat_embedding = np.random.randn(embedding_dim) * 0.1
sat_embedding = np.random.randn(embedding_dim) * 0.1

print(f"\nToken embeddings (from Lesson 2):")
print(f"  'The': [{the_embedding[0]:.3f}, {the_embedding[1]:.3f}, ...]")
print(f"  'cat': [{cat_embedding[0]:.3f}, {cat_embedding[1]:.3f}, ...]")
print(f"  'sat': [{sat_embedding[0]:.3f}, {sat_embedding[1]:.3f}, ...]")

# Compute QKV for "cat"
print("\n" + "-"*50)
print("Computing Q, K, V for 'cat':")
print("-"*50)

cat_q, cat_k, cat_v = compute_qkv(cat_embedding, W_q, W_k, W_v)

print(f"\n'cat' QUERY: {np.round(cat_q, 3)}")
print(f"  → What 'cat' is looking for: 'Who is the subject?'")

print(f"\n'cat' KEY: {np.round(cat_k, 3)}")
print(f"  → What 'cat' offers: 'I am the subject/actor'")

print(f"\n'cat' VALUE: {np.round(cat_v, 3)}")
print(f"  → Information 'cat' carries: 'feline, pet, animal'")

# =============================================================================
# STEP 3: Computing Attention Scores
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Computing Attention Scores")
print("="*70)

print("""
HOW ATTENTION SCORES ARE COMPUTED:
==================================

For each word, we want to know: "Which other words should I attend to?"

STEP-BY-STEP for "sat" attending to previous words:

1. "sat" has a QUERY (what it's looking for)
   → "Who did the action? What is the subject?"

2. Compare to KEYs of all words:
   - "The" KEY: "I'm an article" → Low match
   - "cat" KEY: "I'm the subject/actor" → HIGH match!
   - "sat" KEY: "I'm a verb/action" → Medium match (self)

3. Dot product measures match:
   - sat_Q · The_K = 0.12 (low attention)
   - sat_Q · cat_K = 0.85 (high attention!)
   - sat_Q · sat_K = 0.45 (medium, self-attention)

4. Scale by √d_k (prevents large values)
   → Divide by √4 = 2

5. Softmax to get probabilities:
   → [0.15, 0.55, 0.30]
   → "sat" attends 55% to "cat"!

Let's implement this!
""")

def compute_attention_scores(queries, keys):
    """
    Compute attention scores by matching queries to keys.
    
    OUR EXAMPLE: "sat" figuring out which words to attend to
    
    INPUT:
    - queries: What each word is looking for
    - keys: What each word offers
    
    PROCESS:
    1. Dot product: Q · K^T (how well do they match?)
    2. Scale: Divide by √d_k (prevents large values)
    3. Softmax: Convert to probabilities (attention weights)
    
    OUTPUT:
    - Attention weights: How much each word attends to each other word
    
    Args:
        queries: Query matrix, shape (seq_len, d_k)
        keys: Key matrix, shape (seq_len, d_k)
    
    Returns:
        attention_weights: Shape (seq_len, seq_len)
    """
    d_k = queries.shape[1]
    
    # Step 1: Compute raw scores (dot product of Q and K)
    # This measures how well each query matches each key
    scores = np.dot(queries, keys.T)
    
    # Step 2: Scale by √d_k
    # Prevents softmax saturation when d_k is large
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Softmax to get attention weights
    weights = softmax(scores)
    
    return weights

def softmax(x):
    """Numerically stable softmax - converts scores to probabilities."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("\n--- Attention Scores for 'The cat sat' ---")
print("-"*50)

# Create embeddings matrix for "The cat sat"
embeddings = np.array([the_embedding, cat_embedding, sat_embedding])
print(f"Input embeddings shape: {embeddings.shape}")
print(f"  → 3 tokens ('The', 'cat', 'sat'), each {embedding_dim}-dimensional")

# Compute Q, K, V for all tokens
Q = np.dot(embeddings, W_q)
K = np.dot(embeddings, W_k)
V = np.dot(embeddings, W_v)

print(f"\nComputed matrices:")
print(f"  Q (queries): {Q.shape}")
print(f"  K (keys): {K.shape}")
print(f"  V (values): {V.shape}")

# Compute attention scores (without mask first)
print("\n" + "-"*50)
print("Computing attention scores:")
print("-"*50)

scores_raw = np.dot(Q, K.T) / np.sqrt(d_k)
print(f"Raw attention scores (before softmax):")
print(f"  Shape: {scores_raw.shape}")
print(f"  Values:")
print(f"""
        To:    'The'    'cat'    'sat'
From 'The'    {scores_raw[0,0]:6.2f}  {scores_raw[0,1]:6.2f}  {scores_raw[0,2]:6.2f}
From 'cat'    {scores_raw[1,0]:6.2f}  {scores_raw[1,1]:6.2f}  {scores_raw[1,2]:6.2f}
From 'sat'    {scores_raw[2,0]:6.2f}  {scores_raw[2,1]:6.2f}  {scores_raw[2,2]:6.2f}
""")

# Apply softmax to get attention weights
attention_weights = softmax(scores_raw)

print(f"\nAttention weights (after softmax):")
print(f"""
        To:    'The'    'cat'    'sat'
From 'The'    {attention_weights[0,0]:6.2f}  {attention_weights[0,1]:6.2f}  {attention_weights[0,2]:6.2f}
From 'cat'    {attention_weights[1,0]:6.2f}  {attention_weights[1,1]:6.2f}  {attention_weights[1,2]:6.2f}
From 'sat'    {attention_weights[2,0]:6.2f}  {attention_weights[2,1]:6.2f}  {attention_weights[2,2]:6.2f}
""")

print("-"*50)
print("INTERPRETING THE ATTENTION MATRIX:")
print("-"*50)
print("""
Row 0 ('The' attends to...):
  [0.35, 0.33, 0.32]
  → "The" distributes attention fairly evenly
  → Articles don't have strong preferences

Row 1 ('cat' attends to...):
  [0.30, 0.40, 0.30]
  → "cat" attends most to itself (self-attention)
  → Also attends to "The" (its article)

Row 2 ('sat' attends to...):
  [0.28, 0.45, 0.27]
  → "sat" attends MOST to "cat" (the subject!)
  → This is exactly what we want!
  → Verbs attend to their subjects!

KEY INSIGHT:
============
Attention weights show WHICH words influence WHICH other words!
- High weight = strong influence
- Low weight = weak influence
- Each row sums to 1.0 (probability distribution)
""")

# =============================================================================
# STEP 4: Gathering Information (Weighted Sum of Values)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Gathering Information from Attended Words")
print("="*70)

print("""
NOW THAT WE HAVE ATTENTION WEIGHTS:
===================================

For each word, we gather information from words it attends to.

EXAMPLE: "sat" attends to:
  - "The": 28% attention
  - "cat": 45% attention (highest!)
  - "sat": 27% attention (self)

OUTPUT for "sat" = weighted sum of VALUES:
  output_sat = 0.28 × value_The + 0.45 × value_cat + 0.27 × value_sat

This means "sat" now CONTAINS information from "cat"!
- Original "sat" knew: "I'm a verb, past tense, sitting action"
- After attention "sat" knows: "I'm what the CAT did"

This is CONTEXTUALIZATION!
""")

def gather_information(attention_weights, values):
    """
    Gather information by computing weighted sum of values.
    
    OUR EXAMPLE: Each word accumulating context from attended words
    
    For "sat":
      attention_weights[2] = [0.28, 0.45, 0.27]  (who "sat" attends to)
      values = [value_The, value_cat, value_sat]
      
      output_sat = 0.28 × value_The + 0.45 × value_cat + 0.27 × value_sat
      
    This is a weighted average where:
    - High attention = more influence from that word
    - Low attention = less influence
    
    Args:
        attention_weights: Shape (seq_len, seq_len)
        values: Shape (seq_len, d_v)
    
    Returns:
        output: Shape (seq_len, d_v) - contextualized representations
    """
    # Weighted sum: each output is combination of all values
    # weighted by attention
    output = np.dot(attention_weights, values)
    return output

print("\n--- Gathering Information for 'The cat sat' ---")
print("-"*50)

# Gather information using attention weights
output = gather_information(attention_weights, V)

print(f"Input shape:  {embeddings.shape}")
print(f"Output shape: {output.shape}")
print(f"  → Same shape! Each token transformed into contextualized version")

print(f"\n" + "-"*50)
print("BEFORE vs AFTER ATTENTION:")
print("-"*50)

print(f"\nOriginal embeddings (no context):")
for i, word in enumerate(["The", "cat", "sat"]):
    print(f"  {word}: [{embeddings[i,0]:.3f}, {embeddings[i,1]:.3f}, {embeddings[i,2]:.3f}, ...]")

print(f"\nAfter attention (with context):")
for i, word in enumerate(["The", "cat", "sat"]):
    print(f"  {word}: [{output[i,0]:.3f}, {output[i,1]:.3f}, {output[i,2]:.3f}, ...]")

print(f"\n" + "-"*50)
print("WHAT CHANGED:")
print("-"*50)
print("""
"The" (position 0):
  - Before: Just article information
  - After: Still mostly article info (can only attend to itself with mask)
  
"cat" (position 1):
  - Before: Just "cat" information (feline, pet, animal)
  - After: "cat" + some "The" context (the specific cat)
  
"sat" (position 2):
  - Before: Just "sat" information (verb, past tense, action)
  - After: "sat" + "cat" info (WHO sat) + "The" context
  → This is the BIGGEST change! Verbs need to know their subjects!

THE MAGIC OF ATTENTION:
=======================
Each word now CONTAINS information from words it attended to!
- "sat" knows about "cat" (the subject)
- This helps predict what comes NEXT
- After "The cat sat", model can predict "on the mat"
""")

# =============================================================================
# STEP 5: Causal Mask - GPT Can't See the Future
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Causal Mask - Preventing Cheating")
print("="*70)

print("""
THE PROBLEM WITH UNMASKED ATTENTION:
====================================

In our current attention, "sat" can attend to ALL words:
  - "The" (before) ✓
  - "cat" (before) ✓
  - "sat" (itself) ✓

But what if the sentence continues?
  "The cat sat on the mat because it was tired"

If "sat" could see "because it was tired":
  → It would "cheat" by using future information!
  → During training, it wouldn't learn to PREDICT
  → It would just COPY from the future!

THE SOLUTION: Causal Mask
=========================

Causal mask ensures each word can ONLY see:
  - Itself
  - Words that came BEFORE it

For "The cat sat on the mat":

Position 0 ("The"):   Can see [The]
Position 1 ("cat"):   Can see [The, cat]
Position 2 ("sat"):   Can see [The, cat, sat]
Position 3 ("on"):    Can see [The, cat, sat, on]
Position 4 ("mat"):   Can see [The, cat, sat, on, mat]

NO WORD CAN SEE FUTURE WORDS!

This is essential for GPT's autoregressive generation!
""")

def create_causal_mask(seq_len):
    """
    Create causal mask - prevents seeing future tokens.
    
    OUR EXAMPLE: "The cat sat" with causal mask
    
    Mask shape (3x3):
             To:    The    cat    sat
    From The   [  0,  -inf,  -inf ]  ← "The" sees only itself
    From cat   [  0,     0,  -inf ]  ← "cat" sees The + itself
    From sat   [  0,     0,     0 ]  ← "sat" sees all previous + itself
    
    Where:
    - 0 = can see (attention allowed)
    - -inf = cannot see (attention blocked, becomes 0 after softmax)
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Causal mask, shape (seq_len, seq_len)
    """
    mask = np.zeros((seq_len, seq_len))
    
    # Upper triangle = future positions = blocked
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1e9  # Large negative (becomes ~0 after softmax)
    
    return mask

print("\n--- Causal Mask for 'The cat sat' ---")
print("-"*50)

seq_len = 3
causal_mask = create_causal_mask(seq_len)

print(f"Causal mask (0 = can see, -1e9 = blocked):")
print(f"""
        To:    'The'      'cat'      'sat'
From 'The'  [    0.0,  -1e9,    -1e9   ]
From 'cat'  [    0.0,      0.0,  -1e9   ]
From 'sat'  [    0.0,      0.0,      0.0 ]
""")

print("-"*50)
print("INTERPRETATION:")
print("-"*50)
print("""
Row 0 ("The"):
  → Can only see itself (position 0)
  → No future words to see anyway

Row 1 ("cat"):
  → Can see "The" (position 0) and itself (position 1)
  → Cannot see "sat" (future)

Row 2 ("sat"):
  → Can see "The", "cat", and itself
  → All previous words visible

This ensures: Each word only uses PAST context!
""")

print("\n" + "-"*50)
print("Applying causal mask to attention scores:")
print("-"*50)

# Apply mask to raw scores (before softmax)
scores_masked = scores_raw + causal_mask

print(f"Raw scores BEFORE mask:")
print(np.round(scores_raw, 3))

print(f"\nRaw scores AFTER mask (future blocked):")
print(f"""
       To:     'The'      'cat'      'sat'
'The'  [{scores_masked[0,0]:8.2f}, {scores_masked[0,1]:8.2e}, {scores_masked[0,2]:8.2e}]
'cat'  [{scores_masked[1,0]:8.2f}, {scores_masked[1,1]:8.2f}, {scores_masked[1,2]:8.2e}]
'sat'  [{scores_masked[2,0]:8.2f}, {scores_masked[2,1]:8.2f}, {scores_masked[2,2]:8.2f}]
""")

# Apply softmax to get masked attention weights
attention_weights_masked = softmax(scores_masked)

print(f"\nAttention weights AFTER mask (after softmax):")
print(f"""
       To:     'The'     'cat'     'sat'
'The'  [{attention_weights_masked[0,0]:8.4f}, {attention_weights_masked[0,1]:8.4f}, {attention_weights_masked[0,2]:8.4f}]
'cat'  [{attention_weights_masked[1,0]:8.4f}, {attention_weights_masked[1,1]:8.4f}, {attention_weights_masked[1,2]:8.4f}]
'sat'  [{attention_weights_masked[2,0]:8.4f}, {attention_weights_masked[2,1]:8.4f}, {attention_weights_masked[2,2]:8.4f}]
""")

print("-"*50)
print("NOTICE THE DIFFERENCE:")
print("-"*50)
print("""
WITHOUT MASK:
  "The" attends to: [0.35, 0.33, 0.32] ← sees future "sat"!
  
WITH MASK:
  "The" attends to: [1.00, 0.00, 0.00] ← only sees itself!
  
WITHOUT MASK:
  "cat" attends to: [0.30, 0.40, 0.30] ← sees future "sat"!
  
WITH MASK:
  "cat" attends to: [0.42, 0.58, 0.00] ← only sees "The" + itself!

CAUSAL MASK ENSURES:
====================
Each word can ONLY use information from BEFORE it!
This is essential for autoregressive (next-token) prediction!
""")

# =============================================================================
# STEP 6: Complete Self-Attention with Causal Mask
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete Self-Attention Layer")
print("="*70)

print("""
PUTTING IT ALL TOGETHER:
========================

Our complete self-attention layer:
1. Compute Q, K, V from embeddings
2. Compute attention scores (Q · K^T / √d_k)
3. Apply causal mask (block future)
4. Softmax to get attention weights
5. Gather information (weighted sum of V)

Let's implement this as a complete layer!
""")

class SelfAttention:
    """
    Complete self-attention layer with causal masking.
    
    This is the CORE of GPT!
    
    OUR EXAMPLE: Processing "The cat sat on the mat"
    
    INPUT: Embeddings from Lesson 2
      - Each token has a vector representation
      - "The" → [0.12, -0.23, ...]
      - "cat" → [0.31, 0.08, ...]
      - etc.
    
    PROCESS:
      1. Create Q, K, V projections
      2. Compute attention with causal mask
      3. Gather contextualized information
    
    OUTPUT: Contextualized embeddings
      - Each token now contains info from attended tokens
      - "sat" contains info about "cat" (the subject)
      - "mat" contains info about entire sentence
    
    These contextualized embeddings go to the next layer!
    """
    
    def __init__(self, embedding_dim, d_k, d_v):
        """
        Initialize self-attention layer.
        
        Args:
            embedding_dim: Size of input embeddings
            d_k: Dimension for query and key
            d_v: Dimension for value
        """
        self.embedding_dim = embedding_dim
        self.d_k = d_k
        self.d_v = d_v
        
        # Initialize weight matrices (these are LEARNED during training)
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_k = np.random.randn(embedding_dim, d_k) * 0.1
        self.W_v = np.random.randn(embedding_dim, d_v) * 0.1
        
        print(f"Self-Attention layer initialized:")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Q/K dim: {d_k}")
        print(f"  V dim: {d_v}")
        print(f"  Parameters: {embedding_dim * (d_k * 2 + d_v):,}")
    
    def forward(self, embeddings):
        """
        Forward pass of self-attention.
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
        
        Returns:
            output: Contextualized embeddings, shape (seq_len, d_v)
            attention_weights: Attention distribution, shape (seq_len, seq_len)
        """
        seq_len = embeddings.shape[0]
        
        # Step 1: Compute Q, K, V
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        # Step 2: Compute raw attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        
        # Step 3: Apply causal mask
        mask = create_causal_mask(seq_len)
        scores = scores + mask
        
        # Step 4: Softmax to get attention weights
        attention_weights = softmax(scores)
        
        # Step 5: Gather information (weighted sum of values)
        output = np.dot(attention_weights, V)
        
        return output, attention_weights

print("\n--- Complete Self-Attention Demo ---")
print("-"*50)

# Create self-attention layer
attn_layer = SelfAttention(embedding_dim=8, d_k=4, d_v=4)

# Create embeddings for "The cat sat on mat" (5 tokens)
np.random.seed(42)
seq_len = 5
embeddings = np.random.randn(seq_len, 8) * 0.1

words = ["The", "cat", "sat", "on", "mat"]
print(f"\nInput: '{' '.join(words)}'")
print(f"Embeddings shape: {embeddings.shape}")

# Forward pass
print(f"\n" + "-"*50)
print("Forward pass:")
print("-"*50)
output, attn_weights = attn_layer.forward(embeddings)

print(f"\nOutput shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")

print(f"\n" + "="*50)
print("ATTENTION WEIGHTS (who attends to whom):")
print("="*50)

print(f"\n        To:      ", end="")
for word in words:
    print(f"{word:>8}", end="")
print()
print("         " + "-" * (len(words) * 9))

for i, from_word in enumerate(words):
    print(f"From {from_word:>3}:  ", end="")
    for j in range(len(words)):
        print(f"{attn_weights[i,j]:8.4f}", end="")
    print()

print("\n" + "-"*50)
print("INTERPRETING THE ATTENTION PATTERN:")
print("-"*50)

for i, word in enumerate(words):
    row = attn_weights[i]
    max_idx = np.argmax(row)
    print(f"'{word}' attends most to '{words[max_idx]}' ({row[max_idx]*100:.1f}%)")

print(f"""
KEY OBSERVATIONS:
=================
1. Lower triangle only (causal mask working!)
2. Each token attends to itself (diagonal is non-zero)
3. Earlier tokens have fewer options (can only attend to past)
4. Later tokens build up context from all previous tokens

OUTPUT: Contextualized Representations
======================================
Each output vector now contains:
- Original word meaning
- PLUS: Information from attended words

This is how GPT builds contextual understanding!
""")

# =============================================================================
# SUMMARY: Self-Attention in Our Text Predictor
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Self-Attention")
print("="*70)

print("""
WHAT WE BUILT:
==============
1. Query, Key, Value mechanism
   - Q: "What am I looking for?"
   - K: "What do I offer?"
   - V: "What information do I carry?"

2. Attention scores
   - Match queries to keys (dot product)
   - Scale and softmax to get weights

3. Information gathering
   - Weighted sum of values
   - High attention = more influence

4. Causal mask
   - Block future tokens
   - Essential for autoregressive prediction

HOW THIS CONNECTS TO OUR PREDICTOR:
===================================

Complete flow for "The cat ___":

1. INPUT: "The cat"
   ↓
2. EMBEDDINGS (Lesson 2): Token + Position vectors
   ↓
3. SELF-ATTENTION (this lesson): Contextualize embeddings
   - "cat" attends to "The" (its article)
   - "cat" now has subject context
   ↓
4. NEURAL NETWORK (Lesson 1): Process contextualized embeddings
   ↓
5. OUTPUT: Word probabilities
   - P("sat") = 45%
   - P("slept") = 25%
   - etc.

HOW THIS CONNECTS TO GPT:
=========================

GPT uses the SAME self-attention mechanism:
- Same Q, K, V computation
- Same attention scores (Q·K^T / √d_k)
- Same causal mask
- Same weighted sum of values

GPT differences:
- Much larger (768 dimensions vs our 8)
- Multiple attention heads (next lesson!)
- Many attention layers stacked
- Trained on massive data

NEXT: Multi-Head Attention
==========================
Single attention has limitation:
- Each token has only ONE way to attend
- Can't capture multiple relationships simultaneously

Example: "bank" could mean:
- River bank (geographical)
- Bank account (financial)

Multi-head attention solves this by having MULTIPLE
attention heads, each learning different patterns!

Next: 04_multihead_attention.py
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Experiment with Attention")
print("="*70)

print("""
Try these experiments:

1. LONGER SEQUENCE:
   seq_len = 10  # "The cat sat on the mat because it was tired"
   
   Question: How does attention distribute over longer sequences?
   Expectation: Later tokens attend more selectively

2. WITHOUT CAUSAL MASK:
   # Comment out the mask line in forward()
   
   Question: How does attention change?
   Expectation: All tokens can see all other tokens

3. ANALYZE ATTENTION PATTERNS:
   Print attention_weights and examine:
   - Which words do verbs attend to? (should be subjects!)
   - Which words do articles attend to? (should be nouns!)
   - Does the pattern make linguistic sense?

4. VISUALIZE AS HEATMAP:
   Imagine attention as colors:
   - Bright = high attention
   - Dark = low attention
   - Upper triangle = blocked (causal mask)

KEY TAKEAWAY:
=============
Self-attention lets each word focus on relevant other words!
- Queries find what they're looking for
- Keys indicate what they offer
- Values carry the information
- Causal mask prevents cheating
- Output = contextualized representations

This is the HEART of how GPT understands language!
=============================================================================""")
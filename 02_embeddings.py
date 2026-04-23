"""
=============================================================================
LESSON 2: Embeddings - Converting "The cat" to Numbers for Our Predictor
=============================================================================

Continuing our text predictor from Lesson 1:
- Lesson 1: We predicted next word using manual features [cat=1, dog=0, the=1]
- Lesson 2: We'll learn how to AUTOMATICALLY convert "The cat" to numbers

EXAMPLE FLOW: "The cat" → tokens → embeddings → prediction
"""

import numpy as np

# =============================================================================
# RECAP: Our Text Predictor from Lesson 1
# =============================================================================

print("\n" + "="*70)
print("RECAP: Our Text Predictor")
print("="*70)
print("""
In Lesson 1, we manually created features:
  "The cat" → [cat=1, dog=0, the=1]

PROBLEM: This doesn't scale!
- What about "The dog"? → [cat=0, dog=1, the=1]
- What about "The bird"? → We didn't even have a "bird" feature!
- What about "The quick brown fox"? → Need features for every word!

SOLUTION: Embeddings - automatic conversion of ANY word to numbers!

WHAT WE'LL BUILD:
1. One-hot encoding → basic approach (sparse)
2. Token embeddings → dense vectors (meaningful)
3. Position embeddings → word order information
4. Combined → final input for our predictor
=============================================================================""")

# =============================================================================
# STEP 1: The Problem - How Do We Convert "The cat" to Numbers?
# =============================================================================

print("\n" + "="*70)
print("STEP 1: The Encoding Problem")
print("="*70)

print("""
OUR PREDICTOR NEEDS NUMBERS:
============================

We want to predict: "The cat ___" → "sat"

But our network can't process text directly!
We need to convert "The cat" to numbers.

OPTION 1: Manual Features (what we did in Lesson 1)
  "The cat" → [cat=1, animal=1, the=1]
  
  PROBLEM: We have to manually design features!
  - Who decides which words matter?
  - What about new words we haven't seen?
  - Doesn't scale to real vocabulary (50,000+ words)

OPTION 2: One-Hot Encoding
  Vocabulary: ["The", "cat", "dog", "sat", "slept", ...]
  
  "The" → [1, 0, 0, 0, 0, ...]  (1 at position 0)
  "cat" → [0, 1, 0, 0, 0, ...]  (1 at position 1)
  
  PROBLEM: Mostly zeros, no meaning captured!
  - "cat" and "dog" vectors are orthogonal (dot product = 0)
  - But they're similar concepts!

OPTION 3: Token Embeddings (what GPT uses)
  "The" → [0.23, -0.15, 0.47, ...]  (dense, learned)
  "cat" → [0.31, 0.08, -0.22, ...]  (dense, learned)
  
  ADVANTAGE: Similar words have similar vectors!
  - Learned from data
  - Captures meaning
  - Dense (no wasted zeros)

Let's implement all three to understand the evolution!
""")

# =============================================================================
# STEP 2: One-Hot Encoding (The Basic Approach)
# =============================================================================

print("\n" + "="*70)
print("STEP 2: One-Hot Encoding")
print("="*70)

def one_hot_encode(word_index, vocab_size):
    """
    Create a one-hot encoded vector.
    
    EXAMPLE: Our mini vocabulary for text prediction
    
    Index | Word
    ------|------
    0     | "The"
    1     | "cat"
    2     | "dog"
    3     | "sat"
    4     | "slept"
    5     | "ate"
    6     | "ran"
    7     | "on"
    8     | "the"
    9     | "mat"
    
    To encode "cat" (index 1):
    → [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
      ↑
      Only this position is 1
    
    Args:
        word_index: Position of word in vocabulary (0 to vocab_size-1)
        vocab_size: Total number of words in vocabulary
    
    Returns:
        One-hot vector with 1 at word_index, 0 elsewhere
    """
    vector = np.zeros(vocab_size)
    vector[word_index] = 1
    return vector

print("\n--- Our Mini Vocabulary ---")
print("-"*50)

# Define a mini vocabulary for our text predictor
vocabulary = ["The", "cat", "dog", "sat", "slept", "ate", "ran", "on", "the", "mat"]
vocab_size = len(vocabulary)

print(f"Vocabulary ({vocab_size} words):")
for i, word in enumerate(vocabulary):
    print(f"  {i}: {word}")

print("\n" + "-"*50)
print("One-Hot Encoding Examples:")
print("-"*50)

# Encode "cat" (index 1)
cat_index = 1
cat_one_hot = one_hot_encode(cat_index, vocab_size)
print(f"\n'cat' (index {cat_index}):")
print(f"  One-hot: {cat_one_hot}")
print(f"  → 1 at position {cat_index}, zeros everywhere else")

# Encode "dog" (index 2)
dog_index = 2
dog_one_hot = one_hot_encode(dog_index, vocab_size)
print(f"\n'dog' (index {dog_index}):")
print(f"  One-hot: {dog_one_hot}")

print("\n" + "-"*50)
print("PROBLEM: No Similarity Captured!")
print("-"*50)

# Dot product of one-hot vectors
dot_product = np.dot(cat_one_hot, dog_one_hot)
print(f"'cat' · 'dog' (dot product) = {dot_product}")
print("  → Zero! Mathematically unrelated.")
print("  → But cats and dogs ARE similar (both pets, animals)!")
print("  → One-hot can't capture this.")

# =============================================================================
# STEP 3: Token Embeddings (Dense, Learned Vectors)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Token Embeddings")
print("="*70)

class TokenEmbedding:
    """
    Token embedding layer - converts token IDs to dense vectors.
    
    OUR EXAMPLE: Learning vectors for our vocabulary
    
    Think of this as a lookup table:
    
    Token ID | Word   | Embedding (first 5 of 8 dims)
    ---------|--------|--------------------------------
    0        | "The"  | [0.12, -0.23, 0.45, -0.11, 0.33, ...]
    1        | "cat"  | [0.31, 0.08, -0.22, 0.15, -0.41, ...]
    2        | "dog"  | [0.28, 0.12, -0.18, 0.22, -0.35, ...]
    3        | "sat"  | [-0.15, 0.42, 0.18, -0.33, 0.27, ...]
    ...
    
    Each row is a LEARNED vector for that token.
    After training, similar words have similar vectors!
    """
    
    def __init__(self, vocab_size, embedding_dim, vocab_list=None):
        """
        Create embedding table for vocabulary.
        
        NOTE: In this educational example, we initialize with structured
        values to demonstrate the CONCEPT. In real training, these would
        be LEARNED from data through backpropagation!
        
        Args:
            vocab_size: Number of tokens (e.g., 10 for our mini vocab)
            embedding_dim: Size of each vector (e.g., 8 for demo, 768 for GPT)
            vocab_list: Optional list of words for structured initialization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab_list = vocab_list or []
        
        # Initialize embeddings with STRUCTURED values for demonstration
        # In real training, these would start random and be LEARNED!
        np.random.seed(42)
        
        if vocab_list:
            # Create SEMANTICALLY-MEANINGFUL initial embeddings
            # This simulates what embeddings might look like AFTER training
            # where similar words have similar vectors
            self.weights = self._create_structured_embeddings(embedding_dim)
        else:
            # Standard random initialization (how real models start)
            self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        print(f"\nCreated embedding table:")
        print(f"  Vocabulary: {vocab_size} tokens")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Table shape: {self.weights.shape}")
        print(f"  Total parameters: {vocab_size * embedding_dim:,}")
        if vocab_list:
            print(f"  Note: Using structured init (simulating trained embeddings)")
            print(f"        Real models LEARN these values from data!")
    
    def _create_structured_embeddings(self, dim):
        """
        Create embeddings with semantic structure for demonstration.
        
        This simulates what embeddings might look like AFTER training:
        - Similar words (cat/dog) have similar vectors
        - Action words (sat/ate/ran) cluster together
        - Function words (the) have distinct patterns
        
        REAL TRAINING: These patterns emerge from learning on data!
        """
        weights = np.zeros((self.vocab_size, dim))
        
        # Define semantic categories and their "prototype" vectors
        # Each category gets a base vector + small random noise
        categories = {
            'animals': ['cat', 'dog'],           # Living creatures
            'actions': ['sat', 'slept', 'ate', 'ran'],  # Verbs/actions
            'articles': ['The', 'the'],          # Function words
            'objects': ['mat'],                  # Inanimate objects
        }
        
        # Create prototype vectors for each category
        np.random.seed(42)
        prototypes = {}
        for cat_name in categories:
            # Each category gets a random prototype
            prototypes[cat_name] = np.random.randn(dim) * 0.5
        
        # Assign embeddings based on category
        for word_idx, word in enumerate(self.vocab_list):
            assigned = False
            for cat_name, cat_words in categories.items():
                if word in cat_words:
                    # Add small noise to prototype (so words aren't identical)
                    noise = np.random.randn(dim) * 0.05
                    weights[word_idx] = prototypes[cat_name] + noise
                    assigned = True
                    break
            
            if not assigned:
                # Unassigned words get random embeddings
                weights[word_idx] = np.random.randn(dim) * 0.1
        
        return weights
    
    def forward(self, token_ids):
        """
        Look up embeddings for token IDs.
        
        EXAMPLE: Convert token sequence to embeddings
        
        Input token IDs: [0, 1, 3]  →  "The cat sat"
        
        This looks up:
        - Row 0 → "The" embedding
        - Row 1 → "cat" embedding
        - Row 3 → "sat" embedding
        
        Output: 3x8 matrix (3 tokens, each with 8-dim vector)
        
        Args:
            token_ids: Array of token IDs, e.g., [0, 1, 3]
        
        Returns:
            Embeddings matrix, shape (num_tokens, embedding_dim)
        """
        return self.weights[token_ids]

print("\n--- Creating Token Embeddings ---")
print("-"*50)

# Create embedding layer for our vocabulary
embedding_dim = 8  # Small for demo (GPT uses 768)
token_embedding = TokenEmbedding(vocab_size, embedding_dim)

print("\n" + "-"*50)
print("Embedding Lookup Table (showing first 4 dims):")
print("-"*50)

print(f"{'ID':<4} {'Word':<8} {'Embedding (first 4 of ' + str(embedding_dim) + ' dims)':<40}")
print("-" * 50)
for i, word in enumerate(vocabulary[:7]):  # Show first 7 words
    emb = token_embedding.weights[i]
    print(f"{i:<4} {word:<8} [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, ...]")

print("\n" + "-"*50)
print("Using Embeddings in Our Predictor:")
print("-"*50)

# Convert "The cat" to token IDs
# "The" = index 0, "cat" = index 1
input_text = "The cat"
token_ids = np.array([0, 1])  # Token IDs for "The cat"

print(f"Input text: '{input_text}'")
print(f"Token IDs: {token_ids}")

# Get embeddings
embeddings = token_embedding.forward(token_ids)
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"  → {len(token_ids)} tokens, each with {embedding_dim} features")

print(f"\nEmbedding vectors:")
for i, (tid, emb) in enumerate(zip(token_ids, embeddings)):
    print(f"  Token {i}: '{vocabulary[tid]}' → [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, ...]")

# =============================================================================
# STEP 4: Position Embeddings (Adding Word Order)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Position Embeddings")
print("="*70)

print("""
WHY WE NEED POSITION EMBEDDINGS:
================================

Consider these sentences:
  A) "The cat sat"     ← Normal English
  B) "Sat the cat"     ← Yoda speak
  C) "Cat sat the"     ← Nonsense

Same words, DIFFERENT meanings!

Problem: Our token embeddings don't capture word order.
- "The" has same embedding in all three sentences
- "cat" has same embedding in all three sentences

Solution: Add position embeddings to tell GPT WHERE each word is!

POSITION EMBEDDINGS:
  Position 0 (first word):  [0.10, -0.05, 0.15, ...]
  Position 1 (second word): [0.15, -0.08, 0.20, ...]
  Position 2 (third word):  [0.20, -0.12, 0.25, ...]

Each position gets a unique learned vector!
""")

class PositionEmbedding:
    """
    Position embedding layer - encodes word order.
    
    OUR EXAMPLE: Tracking word positions in "The cat sat"
    
    Position | Token  | Position Embedding (first 4 dims)
    ---------|--------|----------------------------------
    0        | "The"  | [0.10, -0.05, 0.15, -0.08, ...]
    1        | "cat"  | [0.15, -0.08, 0.20, -0.10, ...]
    2        | "sat"  | [0.20, -0.12, 0.25, -0.15, ...]
    
    Position 0 embedding = "I am the FIRST word"
    Position 1 embedding = "I am the SECOND word"
    etc.
    """
    
    def __init__(self, max_positions, embedding_dim):
        """
        Create position embedding table.
        
        Args:
            max_positions: Maximum sequence length (e.g., 10 for demo, 1024 for GPT)
            embedding_dim: Size of position vectors (must match token embeddings)
        """
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        
        # Initialize position embeddings
        np.random.seed(42)
        self.weights = np.random.randn(max_positions, embedding_dim) * 0.1
        
        print(f"\nCreated position embeddings:")
        print(f"  Max positions: {max_positions}")
        print(f"  Embedding dim: {embedding_dim} (must match token embeddings)")
    
    def forward(self, sequence_length):
        """
        Get position embeddings for positions 0 to sequence_length-1.
        
        EXAMPLE: For "The cat sat" (3 words)
        Returns position embeddings for positions 0, 1, 2
        
        Args:
            sequence_length: Number of tokens in sequence
        
        Returns:
            Position embeddings, shape (sequence_length, embedding_dim)
        """
        return self.weights[:sequence_length]

print("\n--- Creating Position Embeddings ---")
print("-"*50)

max_positions = 10  # Support sequences up to 10 tokens
position_embedding = PositionEmbedding(max_positions, embedding_dim)

print("\n" + "-"*50)
print("Position Embedding Table (first 4 dims):")
print("-"*50)

for pos in range(5):
    pos_emb = position_embedding.weights[pos]
    print(f"Position {pos}: [{pos_emb[0]:6.3f}, {pos_emb[1]:6.3f}, {pos_emb[2]:6.3f}, {pos_emb[3]:6.3f}, ...]")

print("\n" + "-"*50)
print("Getting Position Embeddings for 'The cat sat':")
print("-"*50)

# Get position embeddings for 3-token sequence
seq_length = 3
pos_embs = position_embedding.forward(seq_length)

print(f"Sequence length: {seq_length}")
print(f"Position embeddings shape: {pos_embs.shape}")
print(f"\nPosition vectors:")
for i in range(seq_length):
    print(f"  Position {i}: [{pos_embs[i, 0]:6.3f}, {pos_embs[i, 1]:6.3f}, {pos_embs[i, 2]:6.3f}, {pos_embs[i, 3]:6.3f}, ...]")

# =============================================================================
# STEP 5: Combining Token + Position Embeddings
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Final Input = Token + Position")
print("="*70)

print("""
THE FINAL FORMULA:
==================

final_input[i] = token_embedding[i] + position_embedding[i]

This gives GPT BOTH:
- WHAT the word is (token embedding)
- WHERE the word is (position embedding)

EXAMPLE: "The cat sat"
""")

def combine_embeddings(token_embs, pos_embs):
    """
    Combine token and position embeddings.
    
    EXAMPLE: Preparing input for our predictor
    
    Token embeddings (WHAT):
      "The" → [0.12, -0.23, 0.45, ...]
      "cat" → [0.31, 0.08, -0.22, ...]
      "sat" → [-0.15, 0.42, 0.18, ...]
    
    Position embeddings (WHERE):
      Pos 0 → [0.10, -0.05, 0.15, ...]
      Pos 1 → [0.15, -0.08, 0.20, ...]
      Pos 2 → [0.20, -0.12, 0.25, ...]
    
    Combined (element-wise addition):
      Row 0: [0.22, -0.28, 0.60, ...] ← "The" as first word
      Row 1: [0.46, 0.00, -0.02, ...] ← "cat" as second word
      Row 2: [0.05, 0.30, 0.43, ...] ← "sat" as third word
    
    Args:
        token_embs: Token embeddings, shape (seq_len, dim)
        pos_embs: Position embeddings, shape (seq_len, dim)
    
    Returns:
        Combined embeddings, shape (seq_len, dim)
    """
    return token_embs + pos_embs

print("\n--- Combining for 'The cat sat' ---")
print("-"*50)

# Token IDs for "The cat sat"
token_ids = np.array([0, 1, 3])  # "The"=0, "cat"=1, "sat"=3

# Get token embeddings
token_embs = token_embedding.forward(token_ids)

# Get position embeddings
pos_embs = position_embedding.forward(len(token_ids))

# Combine
final_input = combine_embeddings(token_embs, pos_embs)

print(f"\nToken IDs: {token_ids}")
print(f"Words: {[vocabulary[tid] for tid in token_ids]}")

print(f"\n📊 TOKEN EMBEDDINGS (WHAT):")
for i, (tid, emb) in enumerate(zip(token_ids, token_embs)):
    print(f"  {vocabulary[tid]:<6} [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, ...]")

print(f"\n📍 POSITION EMBEDDINGS (WHERE):")
for i, emb in enumerate(pos_embs):
    print(f"  Pos {i:<3} [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, ...]")

print(f"\n📈 COMBINED (WHAT + WHERE):")
for i, (tid, emb) in enumerate(zip(token_ids, final_input)):
    print(f"  {vocabulary[tid]:<6}@{i} [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, ...]")

print("\n" + "-"*50)
print("WHY THIS WORKS:")
print("-"*50)
print("""
Each row now contains BOTH types of information:

Row 0: "The" + "first position" = "The as the first word"
Row 1: "cat" + "second position" = "cat as the second word"
Row 2: "sat" + "third position" = "sat as the third word"

This combined representation goes into our predictor!

Without position: "The cat sat" and "Sat the cat" would be identical!
With position: GPT can distinguish word order!
""")

# =============================================================================
# STEP 6: Using Embeddings in Our Text Predictor
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete Embedding Pipeline")
print("="*70)

print("""
COMPLETE FLOW FOR "The cat ___":
================================

1. TOKENIZE: Convert text to token IDs
   "The cat" → [0, 1]

2. TOKEN EMBEDDINGS: Look up word vectors
   [0, 1] → 2x8 matrix of token embeddings

3. POSITION EMBEDDINGS: Get position vectors
   Sequence length 2 → 2x8 matrix of position embeddings

4. COMBINE: Add token + position
   2x8 + 2x8 = 2x8 final input

5. PREDICT: Pass through neural network
   2x8 input → network → probabilities for next word

Let's see the complete pipeline!
""")

class EmbeddingPipeline:
    """
    Complete embedding pipeline for text prediction.
    
    This is EXACTLY how GPT prepares its input!
    """
    
    def __init__(self, vocab_size, embedding_dim, max_positions):
        """Initialize token and position embeddings."""
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(max_positions, embedding_dim)
    
    def encode(self, token_ids):
        """
        Convert token IDs to final input embeddings.
        
        Args:
            token_ids: Token IDs for input sequence
        
        Returns:
            Final input embeddings ready for the model
        """
        # Get token embeddings
        token_embs = self.token_embedding.forward(token_ids)
        
        # Get position embeddings
        pos_embs = self.position_embedding.forward(len(token_ids))
        
        # Combine
        return combine_embeddings(token_embs, pos_embs)

print("\n--- Complete Pipeline Demo ---")
print("-"*50)

# Create pipeline
pipeline = EmbeddingPipeline(vocab_size, embedding_dim, max_positions)

# Encode "The cat"
input_tokens = np.array([0, 1])
print(f"\nInput: 'The cat' → Token IDs: {input_tokens}")

final = pipeline.encode(input_tokens)
print(f"\nFinal output shape: {final.shape}")
print(f"  → Ready for neural network!")

print(f"\nFinal embeddings (what goes into our predictor):")
for i, (tid, emb) in enumerate(zip(input_tokens, final)):
    print(f"  Position {i}, Word '{vocabulary[tid]}': [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, ...]")

# =============================================================================
# SUMMARY: How GPT Uses Embeddings
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Embeddings in Our Text Predictor")
print("="*70)

print("""
WHAT WE BUILT:
==============
1. Token embeddings → Convert word IDs to dense vectors
2. Position embeddings → Encode word order
3. Combined → Final input with WHAT + WHERE information

HOW THIS CONNECTS TO GPT:
=========================

GPT-2 Small:
  - Vocabulary: 50,257 tokens
  - Embedding dim: 768
  - Max positions: 1,024
  - Token embedding params: 50,257 × 768 = 38.6M
  - Position embedding params: 1,024 × 768 = 786K

Our Mini Predictor:
  - Vocabulary: 10 tokens
  - Embedding dim: 8
  - Max positions: 10
  - Token embedding params: 10 × 8 = 80
  - Position embedding params: 10 × 8 = 80

SAME ARCHITECTURE, different scale!

THE COMPLETE FLOW:
==================

Input text: "The cat ___"
    ↓ (tokenize)
Token IDs: [0, 1]
    ↓ (embed)
Token embeddings: [[0.12, -0.23, ...], [0.31, 0.08, ...]]
    ↓ (add positions)
Position embeddings: [[0.10, -0.05, ...], [0.15, -0.08, ...]]
    ↓ (combine)
Final input: [[0.22, -0.28, ...], [0.46, 0.00, ...]]
    ↓ (neural network)
Prediction: P("sat")=45%, P("slept")=25%, ...

Next: 03_attention.py - How does the model use these embeddings?
      We'll learn SELF-ATTENTION - the core of GPT!
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Experiment with Embeddings")
print("="*70)

print("""
Try these experiments:

1. DIFFERENT EMBEDDING SIZES:
   embedding_dim = 16  # More expressive
   embedding_dim = 4   # Less expressive
   
   Question: How does size affect representation?

2. LARGER VOCABULARY:
   Add more words: ["bird", "quick", "brown", "fox", "mat"]
   
   Question: How many parameters for vocab_size=100, dim=64?
   Answer: 100 × 64 = 6,400 parameters

3. CHECK SIMILARITY:
   After training, similar words should have similar vectors!
   cos_similarity("cat", "dog") should be positive
   cos_similarity("cat", "mat") should be lower

4. POSITION MATTERS:
   Compare embeddings for:
   - "The cat" (The=position 0, cat=position 1)
   - "Cat the" (Cat=position 0, the=position 1)
   
   Same words, different positions → different final embeddings!

KEY TAKEAWAY:
=============
- Embeddings convert discrete tokens to continuous vectors
- Token embedding = "what" the word is
- Position embedding = "where" the word appears
- Combined = Complete representation for the model
- This is how GPT "reads" text!
=============================================================================""")
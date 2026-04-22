"""
=============================================================================
LESSON 2: Word Embeddings - How Words Become Numbers
=============================================================================

GPT (and all language models) can't process raw text - they need numbers.
Embeddings are how we convert words/tokens into meaningful vectors.

KEY CONCEPTS:
1. One-Hot Encoding - The basic approach
2. Word Embeddings - Dense vector representations
3. Token Embeddings + Position Embeddings - How GPT represents input
4. Embedding Lookup - Efficient retrieval of embeddings

In GPT:
- Each token has a learned embedding vector (e.g., 768 dimensions)
- Position information is added via positional embeddings
- Final input = token_embedding + position_embedding

Let's build embeddings from scratch!
"""

import numpy as np

# =============================================================================
# STEP 1: One-Hot Encoding (The Basic Approach)
# =============================================================================

print("\n" + "="*70)
print("STEP 1: One-Hot Encoding")
print("="*70)

def create_one_hot_encoding(word, vocabulary):
    """
    Create a one-hot encoded vector for a word.
    
    REAL-WORLD EXAMPLE: Library Book System
    ----------------------------------------
    Imagine a tiny library with 5 books. Each book gets a unique ID:
    - Book 0: "Cat in the Hat"
    - Book 1: "Dog Tales"
    - Book 2: "Fish Dreams"
    - Book 3: "Bird Songs"
    - Book 4: "Mouse Adventures"
    
    One-hot encoding is like saying "I want book #1" by marking only
    position 1 with a 1, and all others with 0.
    
    Args:
        word: The word to encode
        vocabulary: List of all words in vocabulary
    
    Returns:
        One-hot vector with 1 at word's index, 0 elsewhere
    """
    vector = np.zeros(len(vocabulary))
    if word in vocabulary:
        vector[vocabulary.index(word)] = 1
    return vector

# REAL-WORLD EXAMPLE: Pet Store Inventory
print("\n--- One-Hot Encoding: Pet Store Inventory ---")
print("="*50)
print("""
SCENARIO: You run a pet store with 5 types of animals.
You need to encode which animal a customer wants.

VOCABULARY: ["cat", "dog", "fish", "bird", "mouse"]
Each animal gets a unique position (index).
""")

vocabulary = ["cat", "dog", "fish", "bird", "mouse"]

print(f"Pet Store Vocabulary: {vocabulary}")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"\nIndex mapping:")
for i, word in enumerate(vocabulary):
    print(f"  {i}: {word}")

# Encode some words
print("\n" + "-"*50)
print("Customer Requests (One-Hot Encoded):")
print("-"*50)

for word in ["cat", "dog", "fish"]:
    one_hot = create_one_hot_encoding(word, vocabulary)
    print(f"\nCustomer wants '{word}':")
    print(f"  One-hot vector: {one_hot}")
    print(f"  Index: {vocabulary.index(word)} (position in vocabulary)")
    print(f"  → Only position {vocabulary.index(word)} is 'hot' (1), rest are 0")

print("\n" + "="*70)
print("PROBLEMS WITH ONE-HOT ENCODING:")
print("="*70)
print("""
REAL-WORLD LIMITATIONS:

1. SPARSE & INEFFICIENT:
   Pet store example: [0, 1, 0, 0, 0] for "dog"
   - 80% zeros! Wastes memory.
   - Real vocabulary: 50,000 words → 49,999 zeros per word!

2. NO SEMANTIC MEANING:
   "cat" = [1, 0, 0, 0, 0]
   "dog" = [0, 1, 0, 0, 0]
   
   Dot product = 0 → Mathematically orthogonal (unrelated)
   But cats and dogs ARE similar (both pets, mammals, furry)!
   
   One-hot encoding can't capture this similarity.

3. HIGH DIMENSIONAL:
   - Vector size = vocabulary size
   - GPT vocabulary: ~50,000 tokens
   - Each word = 50,000-dimensional vector (mostly zeros!)

4. NO GENERALIZATION:
   - Each word is completely independent
   - Can't infer that "kitten" relates to "cat"
   - Can't handle unknown words

SOLUTION: Word Embeddings (dense, meaningful vectors)!
=============================================================================""")

# =============================================================================
# STEP 2: Word Embeddings (Dense Vectors)
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Word Embeddings")
print("="*70)

class WordEmbedding:
    """
    Learn a dense vector representation for each word.
    
    REAL-WORLD EXAMPLE: Animal Characteristics
    -------------------------------------------
    Instead of [0, 1, 0, 0, 0], we learn vectors like:
    
    cat → [furry: 0.8, size: 0.3, domestic: 0.9, predator: 0.7, legs: 0.8]
    dog → [furry: 0.8, size: 0.5, domestic: 0.95, predator: 0.6, legs: 0.8]
    fish → [furry: 0.0, size: 0.2, domestic: 0.7, predator: 0.4, legs: 0.0]
    
    Each dimension represents a "feature" or "characteristic".
    Similar animals have similar vectors!
    
    In real embeddings, dimensions aren't this interpretable,
    but the principle is the same - similar things have similar vectors.
    """
    
    def __init__(self, vocabulary, embedding_dim):
        """
        Initialize embeddings for all words.
        
        Args:
            vocabulary: List of words
            embedding_dim: Size of embedding vector (e.g., 5, 100, 768)
        """
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        
        # Random initialization (will be learned during training)
        np.random.seed(42)
        # Use smaller values for better initialization
        self.embeddings = np.random.randn(len(vocabulary), embedding_dim) * 0.1
        
        print(f"Created embeddings for {len(vocabulary)} words")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Embedding matrix shape: {self.embeddings.shape}")
    
    def get_embedding(self, word):
        """Get the embedding vector for a word."""
        if word in self.word_to_idx:
            return self.embeddings[self.word_to_idx[word]]
        else:
            # Return zero vector for unknown words
            return np.zeros(self.embedding_dim)
    
    def similarity(self, word1, word2):
        """
        Compute cosine similarity between two word embeddings.
        
        REAL-WORLD EXAMPLE: Animal Similarity
        --------------------------------------
        Cosine similarity measures how "aligned" two vectors are.
        
        Imagine two animals as arrows in 5D space:
        - If arrows point same direction → similarity = 1 (identical)
        - If arrows perpendicular → similarity = 0 (unrelated)
        - If arrows opposite → similarity = -1 (opposite)
        
        Cosine similarity = (A · B) / (||A|| * ||B||)
        
        After training, "cat" and "dog" would have high similarity
        because they appear in similar contexts (both are pets).
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        similarity = dot_product / (norm1 * norm2)
        return similarity

# REAL-WORLD EXAMPLE: Pet Store with Embeddings
print("\n--- Word Embedding: Pet Store Animals ---")
print("="*50)
print("""
SCENARIO: Same pet store, but now using embeddings.

Instead of one-hot [0, 1, 0, 0, 0], each animal gets a 
dense vector capturing its characteristics.

VOCABULARY: ["cat", "dog", "fish", "bird", "mouse", "animal", "pet", "house"]
EMBEDDING DIM: 5 (each animal described by 5 features)
""")

vocabulary = ["cat", "dog", "fish", "bird", "mouse", "animal", "pet", "house"]
embedding_dim = 5

embeddings = WordEmbedding(vocabulary, embedding_dim)

# Get embeddings for some words
print("\n" + "-"*50)
print("Animal Embedding Vectors (5 features each):")
print("-"*50)
for word in ["cat", "dog", "fish", "bird"]:
    emb = embeddings.get_embedding(word)
    print(f"  {word:6} → [{emb[0]:6.3f}, {emb[1]:6.3f}, {emb[2]:6.3f}, {emb[3]:6.3f}, {emb[4]:6.3f}]")

print("\n" + "-"*50)
print("INTERPRETATION (if these were trained):")
print("-"*50)
print("""
Each dimension might represent a learned feature:
  Dim 1: Size (small ←→ large)
  Dim 2: Furriness (no fur ←→ very furry)
  Dim 3: Domesticity (wild ←→ domestic)
  Dim 4: Activity level (calm ←→ active)
  Dim 5: Common as pet (rare ←→ common)

After training on pet store data:
  - "cat" and "dog" would have similar vectors
  - "fish" would be different (no fur, different size)
  - "bird" would be unique (flies, feathers)
""")

# =============================================================================
# STEP 3: Understanding Embedding Similarity
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Similarity")
print("="*70)

print("\n" + "-"*50)
print("Cosine Similarity Between Animals:")
print("-"*50)

# In real trained embeddings, similar words have similar vectors
# Here we'll demonstrate the concept

word_pairs = [
    ("cat", "dog"),      # Both pets, mammals
    ("cat", "fish"),     # Both pets, but very different
    ("cat", "house"),    # Unrelated
    ("animal", "pet"),   # Related concepts
]

for word1, word2 in word_pairs:
    sim = embeddings.similarity(word1, word2)
    print(f"  {word1:8} ↔ {word2:8}: {sim:.4f}", end="")
    if sim > 0.5:
        print(" ← Similar!")
    elif sim > 0:
        print(" ← Somewhat related")
    else:
        print(" ← Different")

print("\n" + "="*70)
print("NOTE: These are RANDOM embeddings, so similarities are random.")
print("After training on real text data:")
print("  - 'cat' and 'dog' → high similarity (both pets)")
print("  - 'cat' and 'fish' → lower similarity")
print("  - 'cat' and 'house' → very low similarity")
print("="*70)

# =============================================================================
# STEP 4: Token Embeddings in GPT
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Token Embeddings in GPT")
print("="*70)

print("""
REAL-WORLD EXAMPLE: How GPT Tokenizes Text
===========================================

In GPT, we don't embed whole words - we embed TOKENS.

WHAT IS A TOKEN?
- Subword units (e.g., "playing" → "play" + "ing")
- Can be as short as a character or as long as a word
- GPT-2 uses Byte Pair Encoding (BPE) with ~50,000 tokens

EXAMPLE TOKENIZATION:
  Text: "I'm playing GPT"
  Tokens: ["I", "'m", " play", "ing", " G", "PT"]
  
  Notice:
  - "playing" split into " play" + "ing"
  - "GPT" split into " G" + "PT"
  - Spaces are part of tokens!

GPT EMBEDDING PROCESS:

1. Tokenize input text:
   "The cat sat" → ["The", " cat", " sat"]
   
2. Convert tokens to IDs (using vocabulary):
   ["The", " cat", " sat"] → [464, 257, 1234]
   
3. Lookup embeddings:
   Each token ID → embedding vector (e.g., 768 dimensions)
   - Token 464 → embedding[464]
   - Token 257 → embedding[257]
   - Token 1234 → embedding[1234]
   
4. Stack embeddings:
   Create matrix of shape (sequence_length, embedding_dim)
   Example: (3 tokens, 768 dims) → shape (3, 768)

Let's implement this!
=============================================================================""")

class TokenEmbedding:
    """
    Token embedding layer like in GPT.
    
    REAL-WORLD EXAMPLE: Embedding Lookup Table
    -------------------------------------------
    Think of this as a giant lookup table (like a dictionary):
    
    Token ID | Token    | Embedding (first 5 of 768 values)
    ---------|----------|-----------------------------------
    0        | <pad>    | [0.01, -0.02, 0.03, ...]
    1        | <unk>    | [0.00, 0.00, 0.00, ...]
    2        | "a"      | [0.15, -0.08, 0.12, ...]
    3        | "b"      | [-0.05, 0.22, -0.11, ...]
    ...      | ...      | ...
    464      | "The"    | [0.31, -0.15, 0.42, ...]
    ...      | ...      | ...
    
    To get embedding for token 464, just look up row 464!
    """
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            embedding_dim: Size of each embedding vector
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding matrix: each row is a token's embedding
        # Real GPT-2: (50257, 768) = 38.6 million parameters just for embeddings!
        np.random.seed(42)
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02
        
    def forward(self, token_ids):
        """
        Get embeddings for a sequence of token IDs.
        
        REAL-WORLD EXAMPLE: Processing a Sentence
        ------------------------------------------
        Input: token_ids = [464, 257, 1234]  ("The cat sat")
        
        This function looks up each token's embedding:
        - embeddings[464] = "The" embedding
        - embeddings[257] = "cat" embedding  
        - embeddings[1234] = "sat" embedding
        
        Returns: Matrix where row i = embedding of token_ids[i]
        
        Args:
            token_ids: Array of token IDs, shape (sequence_length,)
        
        Returns:
            Embeddings, shape (sequence_length, embedding_dim)
        """
        return self.weights[token_ids]

# REAL-WORLD EXAMPLE: Embedding a Sentence
print("\n--- Token Embedding: Processing 'The cat sat' ---")
print("="*50)
print("""
SCENARIO: GPT receives the text "The cat sat on the mat"

STEP 1: Tokenization
  Text: "The cat sat on the mat"
  Tokens: ["The", " cat", " sat", " on", " the", " mat"]
  
STEP 2: Convert to IDs (using vocabulary lookup)
  Token IDs: [464, 257, 1234, 89, 464, 567]
  
STEP 3: Look up embeddings (this is what we implement)
""")

# Example: Mini vocabulary
vocab_size = 100  # Mini vocabulary for demo
embedding_dim = 8  # Small for demonstration (GPT uses 768)

token_embedding = TokenEmbedding(vocab_size, embedding_dim)

# Simulate tokenized input: "The cat sat" → token IDs
token_ids = np.array([45, 12, 67, 23, 89])  # 5 tokens

print(f"\nToken IDs (from tokenizer): {token_ids}")
print(f"Token IDs shape: {token_ids.shape}")
print(f"  → {len(token_ids)} tokens in sequence")

# Get embeddings
embeddings_output = token_embedding.forward(token_ids)
print(f"\nEmbeddings shape: {embeddings_output.shape}")
print(f"  → {embeddings_output.shape[0]} tokens, each with {embeddings_output.shape[1]}-dim vector")

print(f"\nEmbeddings (first 3 tokens, showing first 4 of {embedding_dim} dims):")
for i in range(min(3, len(token_ids))):
    print(f"  Token {token_ids[i]:2} → [{embeddings_output[i, 0]:6.3f}, {embeddings_output[i, 1]:6.3f}, {embeddings_output[i, 2]:6.3f}, {embeddings_output[i, 3]:6.3f}, ...]")

# =============================================================================
# STEP 5: Position Embeddings
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Position Embeddings")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Why Position Matters
=========================================

PROBLEM: Self-attention is position-agnostic!

Consider these sentences:
  A) "The cat sat on the mat"
  B) "Sat the cat on the mat"  (Yoda speak)
  C) "The mat sat on the cat"  (Nonsense!)

Without position information, GPT would treat these similarly
because they contain the same words!

EXAMPLE: Pronoun Resolution
  "The cat ate the fish because it was hungry."
  "The cat ate the fish because it was fresh."
  
  What does "it" refer to?
  - Sentence 1: "it" = cat (hungry)
  - Sentence 2: "it" = fish (fresh)
  
  Position tells us which noun "it" connects to!

SOLUTION: Add position embeddings to token embeddings!

POSITION EMBEDDINGS:
- Learned vector for each position in sequence
- Position 0: [0.1, -0.2, 0.3, ...] ← embedding for "first word"
- Position 1: [0.2, -0.1, 0.4, ...] ← embedding for "second word"
- Position 2: [0.3, -0.3, 0.5, ...] ← embedding for "third word"
- etc.

FINAL INPUT = TOKEN EMBEDDING + POSITION EMBEDDING
""")

class PositionEmbedding:
    """
    Position embedding layer.
    
    REAL-WORLD EXAMPLE: Sentence Position Tracker
    -----------------------------------------------
    Think of this as keeping track of word order:
    
    Sentence: "The cat sat on the mat"
    
    Position | Token  | Position Embedding (first 5 dims)
    ---------|--------|----------------------------------
    0        | "The"  | [0.10, -0.05, 0.15, -0.08, 0.12, ...]
    1        | "cat"  | [0.15, -0.08, 0.20, -0.10, 0.18, ...]
    2        | "sat"  | [0.20, -0.12, 0.25, -0.15, 0.22, ...]
    3        | "on"   | [0.25, -0.15, 0.30, -0.18, 0.28, ...]
    4        | "the"  | [0.30, -0.18, 0.35, -0.22, 0.32, ...]
    5        | "mat"  | [0.35, -0.22, 0.40, -0.25, 0.38, ...]
    
    Each position gets a unique embedding vector.
    GPT-2 supports up to 1024 positions!
    """
    
    def __init__(self, max_position, embedding_dim):
        """
        Args:
            max_position: Maximum sequence length supported
            embedding_dim: Size of embedding vector
        """
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        
        # Learnable position embeddings
        # Real GPT-2: (1024, 768) = 786,432 parameters for positions
        np.random.seed(42)
        self.weights = np.random.randn(max_position, embedding_dim) * 0.02
        
    def forward(self, sequence_length):
        """
        Get position embeddings for positions 0 to sequence_length-1.
        
        REAL-WORLD EXAMPLE: Getting Positions for a Sentence
        -----------------------------------------------------
        For sentence "The cat sat" (3 words):
        
        Returns position embeddings for positions 0, 1, 2:
        - Position 0 embedding: for "The" (first word)
        - Position 1 embedding: for "cat" (second word)
        - Position 2 embedding: for "sat" (third word)
        
        These get added to token embeddings to give GPT
        information about word order.
        
        Returns:
            Position embeddings, shape (sequence_length, embedding_dim)
        """
        return self.weights[:sequence_length]

print("\n--- Position Embedding Example ---")
max_position = 1024  # GPT supports sequences up to 1024 tokens
embedding_dim = 8

position_embedding = PositionEmbedding(max_position, embedding_dim)

# Get position embeddings for a 5-token sequence
print(f"\nMax positions supported: {max_position}")
print(f"Embedding dimension: {embedding_dim}")

position_embeddings = position_embedding.forward(5)
print(f"\nPosition embeddings for 5-token sequence:")
print(f"Shape: {position_embeddings.shape}")
print(f"\nPosition vectors (first 4 dims shown):")
for i in range(5):
    print(f"  Position {i}: [{position_embeddings[i, 0]:6.3f}, {position_embeddings[i, 1]:6.3f}, {position_embeddings[i, 2]:6.3f}, {position_embeddings[i, 3]:6.3f}, ...]")

print("\n" + "-"*50)
print("INTERPRETATION:")
print("-"*50)
print("""
Each position gets a unique learned vector.
When added to token embeddings:
  - Position 0 + "The" → "The" as first word
  - Position 1 + "cat" → "cat" as second word
  - etc.

This tells GPT not just WHAT words, but WHERE they appear!
""")

# =============================================================================
# STEP 6: Combining Token + Position Embeddings
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Final Input to GPT")
print("="*70)

def create_gpt_input(token_ids, token_embedding, position_embedding):
    """
    Create the final input for GPT by combining token and position embeddings.
    
    REAL-WORLD EXAMPLE: Complete Sentence Representation
    -------------------------------------------------------
    For sentence "The cat sat":
    
    Token embeddings (WHAT words):
      "The" → [0.31, -0.15, 0.42, ...]
      "cat" → [0.22, 0.18, -0.33, ...]
      "sat" → [-0.12, 0.45, 0.28, ...]
    
    Position embeddings (WHERE words):
      Pos 0 → [0.10, -0.05, 0.15, ...]
      Pos 1 → [0.15, -0.08, 0.20, ...]
      Pos 2 → [0.20, -0.12, 0.25, ...]
    
    Combined (final input to GPT):
      Row 0: "The" + Pos 0 → [0.41, -0.20, 0.57, ...]
      Row 1: "cat" + Pos 1 → [0.37, 0.10, -0.13, ...]
      Row 2: "sat" + Pos 2 → [0.08, 0.33, 0.53, ...]
    
    This combined representation goes into transformer layers!
    
    Args:
        token_ids: Token IDs, shape (sequence_length,)
        token_embedding: TokenEmbedding layer
        position_embedding: PositionEmbedding layer
    
    Returns:
        Final input embeddings, shape (sequence_length, embedding_dim)
    """
    # Get token embeddings (WHAT words)
    token_embs = token_embedding.forward(token_ids)
    
    # Get position embeddings (WHERE words)
    seq_length = len(token_ids)
    pos_embs = position_embedding.forward(seq_length)
    
    # Combine: token + position (element-wise addition)
    final_input = token_embs + pos_embs
    
    return final_input

print("\n--- Combining Token + Position Embeddings ---")
print("="*50)
print("""
SCENARIO: Complete input preparation for GPT

Input text: "The cat sat on the mat"
↓ (tokenize)
Token IDs: [45, 12, 67, 23, 89]
↓ (embed)
Token embeddings: (5, 8) matrix
Position embeddings: (5, 8) matrix
↓ (add)
Final input: (5, 8) matrix ready for GPT!
""")

# Use same token IDs as before
token_ids = np.array([45, 12, 67, 23, 89])

# Create final input
final_input = create_gpt_input(token_ids, token_embedding, position_embedding)

print(f"\nToken IDs: {token_ids}")
print(f"Sequence length: {len(token_ids)} tokens")
print(f"Embedding dimension: {embedding_dim}")

print(f"\nFinal input shape: {final_input.shape}")
print(f"  → Ready for transformer layers!")

print(f"\nFinal input vectors (first 3 rows, first 4 dims):")
for i in range(3):
    print(f"  Row {i}: [{final_input[i, 0]:6.3f}, {final_input[i, 1]:6.3f}, {final_input[i, 2]:6.3f}, {final_input[i, 3]:6.3f}, ...]")
    print(f"         = token[{token_ids[i]}] + position[{i}]")

print("\n" + "="*70)
print("KEY INSIGHT:")
print("="*70)
print("""
FINAL INPUT FORMULA:
  final_input[i] = token_embedding[token_ids[i]] + position_embedding[i]

This combined embedding goes into the transformer layers!

TOKEN EMBEDDING  → Tells GPT "WHAT word is this?"
POSITION EMBEDDING → Tells GPT "WHERE is this word in sequence?"

Together: Complete representation for processing!

EXAMPLE:
  "The cat sat" vs "Sat the cat"
  
  Same token embeddings, DIFFERENT position embeddings
  → GPT can distinguish word order!
=============================================================================""")

# =============================================================================
# SUMMARY: Embeddings in GPT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Embeddings in GPT")
print("="*70)

print("""
REAL-WORLD NUMBERS:

1. TOKEN EMBEDDINGS (GPT-2 Small):
   - Vocabulary size: 50,257 tokens
   - Embedding dim: 768
   - Parameters: 50,257 × 768 = 38.6 million!
   - Stores: What each token means

2. POSITION EMBEDDINGS (GPT-2 Small):
   - Max positions: 1,024
   - Embedding dim: 768
   - Parameters: 1,024 × 768 = 786,432
   - Stores: Where each token appears

3. COMBINED INPUT:
   - Shape: (sequence_length, embedding_dim)
   - Example: (1024, 768) for full context
   - Each row: token_info + position_info

4. LEARNING:
   - Embeddings initialized randomly
   - Updated during training via backpropagation
   - After training: similar tokens → similar vectors
   - Captures semantic relationships!

5. WHY THIS MATTERS:
   - One-hot: [0, 1, 0, 0, ...] (sparse, no meaning)
   - Embeddings: [0.23, -0.15, 0.47, ...] (dense, meaningful)
   - "king" - "man" + "woman" ≈ "queen" (vector arithmetic!)

Next: In 03_attention.py, we'll learn the core mechanism of GPT -
      SELF-ATTENTION (how GPT understands relationships between words)!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Embeddings")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:

1. CHANGE EMBEDDING SIZE:
   embedding_dim = 16  # Larger embeddings (more features)
   embedding_dim = 3   # Smaller embeddings (less expressive)
   
   Question: How does size affect representation capacity?

2. CREATE LARGER VOCABULARY:
   vocab_size = 1000  # More tokens
   vocab_size = 10000  # Realistic mini-vocab
   
   Question: How many parameters for vocab_size=10000, dim=768?
   Answer: 10000 × 768 = 7.68 million parameters!

3. CHECK SHAPES:
   print(token_embedding.weights.shape)  # (vocab_size, embedding_dim)
   print(position_embedding.weights.shape)  # (max_position, embedding_dim)

4. COMPUTE SIMILARITY:
   For trained embeddings, try:
   similarity("cat", "dog")  # Should be high (both pets)
   similarity("cat", "airplane")  # Should be low (unrelated)

5. VISUALIZE (MENTALLY):
   Imagine each word as a point in 768-dimensional space.
   Similar words cluster together!
   - Animal words: one region
   - Food words: another region
   - Emotion words: another region

KEY TAKEAWAY:
- Embeddings convert discrete tokens to continuous vectors
- Token embedding = "what" the word is
- Position embedding = "where" the word appears
- Combined = Complete input for transformer
- Embeddings are LEARNED, not fixed!
""")
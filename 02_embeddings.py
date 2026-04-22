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

# Simple vocabulary
vocabulary = ["cat", "dog", "fish", "bird", "mouse"]

print("\n--- One-Hot Encoding Example ---")
print(f"Vocabulary: {vocabulary}")
print(f"Vocabulary size: {len(vocabulary)}")

# Encode some words
for word in ["cat", "dog", "fish"]:
    one_hot = create_one_hot_encoding(word, vocabulary)
    print(f"\n'{word}' → {one_hot}")
    print(f"  Index: {vocabulary.index(word)}")

print("\n" + "-"*70)
print("PROBLEMS WITH ONE-HOT ENCODING:")
print("-"*70)
print("""
1. SPARSE: Most values are 0, only one value is 1
2. NO SEMANTIC MEANING: 
   - 'cat' and 'dog' vectors are orthogonal (dot product = 0)
   - But cats and dogs are similar (both pets, animals)
3. HIGH DIMENSIONAL: Vector size = vocabulary size (can be 50,000+)
4. NO GENERALIZATION: Each word is independent

SOLUTION: Word Embeddings (dense vectors)!
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
    
    Instead of [0, 0, 1, 0, 0], we learn vectors like:
    cat → [0.25, -0.13, 0.47, 0.82, ...]
    dog → [0.23, -0.11, 0.51, 0.79, ...]
    
    Similar words have similar vectors!
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
        
        Cosine similarity = (A · B) / (||A|| * ||B||)
        Range: -1 (opposite) to 1 (identical)
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        similarity = dot_product / (norm1 * norm2)
        return similarity

# Create embeddings with 5 dimensions (real GPT uses 768+)
print("\n--- Word Embedding Example ---")
vocabulary = ["cat", "dog", "fish", "bird", "mouse", "animal", "pet", "house"]
embedding_dim = 5

embeddings = WordEmbedding(vocabulary, embedding_dim)

# Get embeddings for some words
print("\nEmbedding vectors:")
for word in ["cat", "dog", "fish", "bird"]:
    emb = embeddings.get_embedding(word)
    print(f"  {word:6} → {emb}")

# =============================================================================
# STEP 3: Understanding Embedding Similarity
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Embedding Similarity")
print("="*70)

print("\nCosine Similarity Between Words:")
print("-"*50)

# In real trained embeddings, similar words have similar vectors
# Here we'll demonstrate the concept

word_pairs = [
    ("cat", "dog"),
    ("cat", "fish"),
    ("cat", "house"),
    ("animal", "pet"),
]

for word1, word2 in word_pairs:
    sim = embeddings.similarity(word1, word2)
    print(f"  {word1:8} ↔ {word2:8}: {sim:.4f}")

print("\n" + "-"*70)
print("NOTE: These are random embeddings, so similarities are random.")
print("After training on text, similar words would have higher similarity!")
print("E.g., 'cat' and 'dog' would be more similar than 'cat' and 'house'")
print("="*70)

# =============================================================================
# STEP 4: Token Embeddings in GPT
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Token Embeddings in GPT")
print("="*70)

print("""
In GPT, we don't embed whole words - we embed TOKENS.

TOKENS:
- Subword units (e.g., "playing" → "play" + "ing")
- Can be as short as a character or as long as a word
- GPT-2 uses Byte Pair Encoding (BPE) with ~50,000 tokens

GPT EMBEDDING PROCESS:

1. Tokenize input text:
   "The cat sat" → ["The", " cat", " sat"]
   
2. Convert tokens to IDs:
   ["The", " cat", " sat"] → [464, 257, 1234]
   
3. Lookup embeddings:
   Each token ID → embedding vector (e.g., 768 dimensions)
   
4. Stack embeddings:
   Create matrix of shape (sequence_length, embedding_dim)

Let's implement this!
=============================================================================""")

class TokenEmbedding:
    """
    Token embedding layer like in GPT.
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
        np.random.seed(42)
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.02
        
    def forward(self, token_ids):
        """
        Get embeddings for a sequence of token IDs.
        
        Args:
            token_ids: Array of token IDs, shape (sequence_length,)
        
        Returns:
            Embeddings, shape (sequence_length, embedding_dim)
        """
        return self.weights[token_ids]

# Example: Mini vocabulary
print("\n--- Token Embedding Example ---")
vocab_size = 100  # Mini vocabulary
embedding_dim = 8  # Small for demonstration

token_embedding = TokenEmbedding(vocab_size, embedding_dim)

# Simulate tokenized input: "The cat sat" → token IDs
token_ids = np.array([45, 12, 67, 23, 89])  # 5 tokens

print(f"Token IDs: {token_ids}")
print(f"Token IDs shape: {token_ids.shape}")

# Get embeddings
embeddings_output = token_embedding.forward(token_ids)
print(f"Embeddings shape: {embeddings_output.shape}")
print(f"Each token → {embedding_dim}-dimensional vector")
print(f"\nEmbeddings:\n{embeddings_output}")

# =============================================================================
# STEP 5: Position Embeddings
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Position Embeddings")
print("="*70)

print("""
PROBLEM: Self-attention is position-agnostic!
"The cat sat" and "Sat the cat" would produce same attention.

SOLUTION: Add position embeddings to token embeddings!

POSITION EMBEDDINGS:
- Learned vector for each position in sequence
- Position 0: [0.1, -0.2, 0.3, ...]
- Position 1: [0.2, -0.1, 0.4, ...]
- etc.

FINAL INPUT = TOKEN EMBEDDING + POSITION EMBEDDING
""")

class PositionEmbedding:
    """
    Position embedding layer.
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
        np.random.seed(42)
        self.weights = np.random.randn(max_position, embedding_dim) * 0.02
        
    def forward(self, sequence_length):
        """
        Get position embeddings for positions 0 to sequence_length-1.
        
        Returns:
            Position embeddings, shape (sequence_length, embedding_dim)
        """
        return self.weights[:sequence_length]

print("\n--- Position Embedding Example ---")
max_position = 1024  # GPT supports sequences up to 1024 tokens
embedding_dim = 8

position_embedding = PositionEmbedding(max_position, embedding_dim)

# Get position embeddings for a 5-token sequence
position_embeddings = position_embedding.forward(5)
print(f"Position embeddings shape: {position_embeddings.shape}")
print(f"\nPosition embeddings:\n{position_embeddings}")

# =============================================================================
# STEP 6: Combining Token + Position Embeddings
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Final Input to GPT")
print("="*70)

def create_gpt_input(token_ids, token_embedding, position_embedding):
    """
    Create the final input for GPT by combining token and position embeddings.
    
    Args:
        token_ids: Token IDs, shape (sequence_length,)
        token_embedding: TokenEmbedding layer
        position_embedding: PositionEmbedding layer
    
    Returns:
        Final input embeddings, shape (sequence_length, embedding_dim)
    """
    # Get token embeddings
    token_embs = token_embedding.forward(token_ids)
    
    # Get position embeddings
    seq_length = len(token_ids)
    pos_embs = position_embedding.forward(seq_length)
    
    # Combine: token + position
    final_input = token_embs + pos_embs
    
    return final_input

print("\n--- Combining Token + Position Embeddings ---")

# Use same token IDs as before
token_ids = np.array([45, 12, 67, 23, 89])

# Create final input
final_input = create_gpt_input(token_ids, token_embedding, position_embedding)

print(f"Token IDs: {token_ids}")
print(f"Sequence length: {len(token_ids)}")
print(f"Embedding dimension: {embedding_dim}")
print(f"\nFinal input shape: {final_input.shape}")
print(f"\nFinal input (first 3 rows):\n{final_input[:3]}")

print("\n" + "-"*70)
print("KEY INSIGHT:")
print("-"*70)
print("""
Each row in final_input is:
  final_input[i] = token_embedding[token_ids[i]] + position_embedding[i]

This combined embedding goes into the transformer layers!

Token embedding tells GPT "WHAT word is this?"
Position embedding tells GPT "WHERE is this word in the sequence?"
=============================================================================""")

# =============================================================================
# SUMMARY: Embeddings in GPT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Embeddings in GPT")
print("="*70)

print("""
1. TOKEN EMBEDDINGS:
   - Vocabulary size: ~50,000 (GPT-2)
   - Embedding dim: 768 (GPT-2 small), 1600 (GPT-3 large)
   - Each token → dense vector

2. POSITION EMBEDDINGS:
   - Max positions: 1024 (GPT-2), 2048+ (GPT-3)
   - Same embedding dim as tokens
   - Each position → dense vector

3. COMBINED INPUT:
   - Shape: (sequence_length, embedding_dim)
   - Example: (1024, 768) for full GPT-2 context

4. LEARNING:
   - Embeddings are learned during training
   - Similar tokens get similar embeddings
   - Captures semantic relationships

Next: In 03_attention.py, we'll learn the core mechanism of GPT -
      SELF-ATTENTION!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Embeddings")
print("="*70)

print("""
Try these:

1. Change embedding dimension:
   embedding_dim = 16  # Larger embeddings

2. Create larger vocabulary:
   vocab_size = 1000

3. Check shapes:
   - What's the shape of token_embeddings.weights?
   - What's the shape of position_embeddings.weights?

4. Compute similarity:
   - Which words have highest similarity?
   - Why are random embeddings not meaningful yet?

Key Takeaway:
- Embeddings convert discrete tokens to continuous vectors
- Token + Position = Complete input representation
- Embeddings are learned, not fixed!
""")
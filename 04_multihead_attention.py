"""
=============================================================================
LESSON 4: Multi-Head Attention - Multiple Perspectives
=============================================================================

Single attention is powerful, but GPT uses MULTI-HEAD attention.
This is like having multiple attention mechanisms working in parallel!

KEY CONCEPTS:
1. Why Multi-Head? - Multiple representation subspaces
2. Multiple Heads - Parallel attention computations
3. Concatenation + Projection - Combining head outputs
4. Implementation - Building the complete layer

WHY MULTI-HEAD?

Think of it like reading a sentence with different focus:
- Head 1: Attends to syntactic relationships (grammar)
- Head 2: Attends to semantic relationships (meaning)
- Head 3: Attends to coreferences (pronoun references)
- Head 4: Attends to positional patterns

Each head learns DIFFERENT patterns!

Let's build multi-head attention!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Why Multi-Head Attention?")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Team of Experts Analyzing a Document
=========================================================

Imagine a company reviewing an important document. Instead of having
ONE person read it, they assemble a TEAM OF EXPERTS:

EXPERT 1 (Grammar Specialist):
  - Focuses on sentence structure
  - Notices: "The subject 'cat' connects to verb 'sat'"
  - Attends to: Subject-verb relationships

EXPERT 2 (Meaning Analyst):
  - Focuses on semantics and meaning
  - Notices: "comfortable describes the cat's state"
  - Attends to: Adjective-noun relationships

EXPERT 3 (Pronoun Detective):
  - Focuses on what pronouns refer to
  - Notices: "it" refers to "cat", not "mat"
  - Attends to: Coreference resolution

EXPERT 4 (Position Tracker):
  - Focuses on word order and position
  - Notices: "on the mat" comes after "sat"
  - Attends to: Positional patterns

SAME DOCUMENT, DIFFERENT PERSPECTIVES!

After each expert analyzes the document, they combine their findings
to get a COMPLETE understanding.

MULTI-HEAD ATTENTION WORKS THE SAME WAY:
- Multiple "experts" (heads) analyze the same input
- Each head focuses on different relationships
- Outputs are combined for final representation

SINGLE ATTENTION LIMITATION:
==============================

With one attention head, each token has only ONE way to attend 
to other tokens. But language is complex!

Example: "The cat sat on the mat because it was comfortable"

Different relationships to capture:
- "it" → "cat" (coreference - what does "it" refer to?)
- "comfortable" → "cat" (semantic - what is comfortable?)
- "on" → "mat" (syntactic - preposition-object)
- "sat" → "cat" (subject-verb relationship)

Multi-head captures ALL of these simultaneously!

=============================================================================""")

# =============================================================================
# STEP 2: Multi-Head Architecture
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Multi-Head Architecture")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Factory Assembly Line with Specialized Stations
===================================================================

Think of multi-head attention as a factory with multiple assembly lines:

INPUT: Raw materials (embeddings) enter the factory
       Shape: (seq_len, embedding_dim) = (5 tokens, 512 features)

STEP 1: DISTRIBUTION CENTER (Linear Projections)
       Raw materials are split and sent to parallel assembly lines
       - If 512 features and 8 heads, each head gets 512/8 = 64 features
       - Like sorting products by category for specialized processing

STEP 2: PARALLEL ASSEMBLY LINES (Attention per Head)
       Line 1: Processes grammar relationships
       Line 2: Processes meaning relationships  
       Line 3: Processes pronoun references
       Line 4: Processes positional patterns
       ...
       Line 8: Processes other patterns
       
       Each line produces: (seq_len, head_dim) = (5, 64)

STEP 3: QUALITY CONTROL (Concatenation)
       All assembly line outputs are gathered together
       Shape: (seq_len, num_heads × head_dim) = (5, 512)

STEP 4: FINAL PACKAGING (Output Projection)
       Combined products are packaged for shipping
       Final output: (seq_len, embedding_dim) = (5, 512)

FORMULA:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

=============================================================================""")

# =============================================================================
# STEP 3: Implementing Multi-Head Attention
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Implementing Multi-Head Attention")
print("="*70)

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
    """
    Multi-head attention layer.
    This is the exact implementation used in GPT!
    
    REAL-WORLD EXAMPLE: Complete Expert Team System
    ================================================
    
    This class implements the full multi-head attention mechanism.
    
    Think of it as managing a team of expert analysts:
    1. Each expert (head) has their own specialization
    2. Each expert analyzes the document independently
    3. All expert reports are combined
    4. A final summary is produced
    
    The key insight: Different experts notice different things!
    """
    
    def __init__(self, embedding_dim, num_heads):
        """
        Initialize multi-head attention.
        
        REAL-WORLD EXAMPLE: Setting Up the Expert Team
        -----------------------------------------------
        
        Imagine you're setting up a document analysis team:
        
        embedding_dim: How detailed is each document?
          - GPT-2: 768 features per token
          - Our example: 64 features (for learning)
        
        num_heads: How many experts on the team?
          - GPT-2 Small: 12 experts (heads)
          - GPT-3 Large: 96 experts (heads)
          - Our example: 4 experts (heads)
        
        head_dim: How specialized is each expert?
          - head_dim = embedding_dim / num_heads
          - Each expert focuses on a subset of features
          - Like each expert having a specific domain
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_heads: Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Each head works with a smaller dimension
        # Total dimension is split across heads
        assert embedding_dim % num_heads == 0, \
            "embedding_dim must be divisible by num_heads"
        
        self.head_dim = embedding_dim // num_heads
        
        print(f"Multi-Head Attention initialized")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  → Each head focuses on {self.head_dim} features")
        
        # Weight matrices for Q, K, V (project to all heads at once)
        # These are like the "expertise profiles" for each expert
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
        
        # Output projection matrix (concatenate heads → output dim)
        # This combines all expert reports into a final summary
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, head_dim).
        
        REAL-WORLD EXAMPLE: Sorting Mail into Departments
        -------------------------------------------------
        Imagine a company receiving mail. The mail needs to be
        sorted into different departments for specialized processing.
        
        Input: All mail in one pile (seq_len, embedding_dim)
               Example: (5 tokens, 64 features) = 5 envelopes with 64 pages each
        
        Process:
        1. Reshape: Organize pages into departments
           (5, 64) → (5, 4 departments, 16 pages per dept)
        
        2. Transpose: Put department piles together
           (5, 4, 16) → (4 departments, 5 envelopes, 16 pages)
        
        Output: Each department has its own pile to process
                (num_heads, seq_len, head_dim)
        
        Args:
            x: Input, shape (seq_len, embedding_dim)
        
        Returns:
            Split tensor, shape (num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[0]
        # Reshape: (seq_len, embedding_dim) → (seq_len, num_heads, head_dim)
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        # Transpose: (seq_len, num_heads, head_dim) → (num_heads, seq_len, head_dim)
        x = x.transpose(1, 0, 2)
        return x
    
    def _combine_heads(self, x):
        """
        Combine heads back together.
        
        REAL-WORLD EXAMPLE: Department Reports to CEO
        ----------------------------------------------
        After each department processes their mail, they send
        reports to the CEO who needs a combined view.
        
        Input: Reports from each department
               (num_heads, seq_len, head_dim)
               Example: (4 depts, 5 items, 16 features each)
        
        Process:
        1. Transpose: Reorder to (seq_len, num_heads, head_dim)
           Group by item, not by department
        
        2. Reshape: Flatten departments together
           (5, 4, 16) → (5, 64)
           Each item now has all 64 features from all departments
        
        Output: Combined report for the CEO
                (seq_len, embedding_dim)
        
        Args:
            x: Input, shape (num_heads, seq_len, head_dim)
        
        Returns:
            Combined tensor, shape (seq_len, embedding_dim)
        """
        # Transpose: (num_heads, seq_len, head_dim) → (seq_len, num_heads, head_dim)
        x = x.transpose(1, 0, 2)
        # Reshape: (seq_len, num_heads, head_dim) → (seq_len, embedding_dim)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.embedding_dim)
        return x
    
    def _single_head_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention for a single head.
        
        REAL-WORLD EXAMPLE: One Expert's Analysis
        ------------------------------------------
        This is what ONE expert (head) does:
        
        Q (Query): What this expert is looking for
        K (Key): What each token offers to this expert
        V (Value): The actual content this expert extracts
        
        The expert:
        1. Matches their queries to keys (finds relevant tokens)
        2. Computes attention weights (how interested am I?)
        3. Gathers values based on attention (learns from relevant tokens)
        
        Args:
            Q: Query, shape (seq_len, head_dim)
            K: Key, shape (seq_len, head_dim)
            V: Value, shape (seq_len, head_dim)
            mask: Optional causal mask
        
        Returns:
            attention_output: Shape (seq_len, head_dim)
            attention_weights: Shape (seq_len, seq_len)
        """
        d_k = K.shape[1]
        
        # Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax to get weights
        weights = softmax(scores)
        
        # Weighted sum of values
        output = np.dot(weights, V)
        
        return output, weights
    
    def forward(self, embeddings, use_causal_mask=True):
        """
        Forward pass of multi-head attention.
        
        REAL-WORLD EXAMPLE: Complete Team Analysis
        -------------------------------------------
        This is the full process of the expert team:
        
        1. PROJECTIONS: Create Q, K, V for all heads
           Like preparing documents for each expert
        
        2. SPLIT: Divide Q, K, V among heads
           Like distributing documents to departments
        
        3. ATTENTION: Each head computes attention
           Like each expert doing their analysis
        
        4. COMBINE: Concatenate all head outputs
           Like gathering all expert reports
        
        5. OUTPUT PROJECTION: Final mixing
           Like creating executive summary from all reports
        
        Args:
            embeddings: Input embeddings, shape (seq_len, embedding_dim)
            use_causal_mask: Whether to apply causal mask
        
        Returns:
            output: Shape (seq_len, embedding_dim)
            attention_weights: Dict of attention weights per head
        """
        seq_len = embeddings.shape[0]
        
        # Step 1: Linear projections
        Q = np.dot(embeddings, self.W_q)  # (seq_len, embedding_dim)
        K = np.dot(embeddings, self.W_k)  # (seq_len, embedding_dim)
        V = np.dot(embeddings, self.W_v)  # (seq_len, embedding_dim)
        
        # Step 2: Split into heads
        Q_heads = self._split_heads(Q)  # (num_heads, seq_len, head_dim)
        K_heads = self._split_heads(K)  # (num_heads, seq_len, head_dim)
        V_heads = self._split_heads(V)  # (num_heads, seq_len, head_dim)
        
        # Step 3: Create causal mask (same for all heads)
        mask = None
        if use_causal_mask:
            mask = create_causal_mask(seq_len)
        
        # Step 4: Compute attention for each head (in parallel conceptually)
        head_outputs = []
        attention_weights = {}
        
        for head_idx in range(self.num_heads):
            Q_head = Q_heads[head_idx]  # (seq_len, head_dim)
            K_head = K_heads[head_idx]  # (seq_len, head_dim)
            V_head = V_heads[head_idx]  # (seq_len, head_dim)
            
            # Compute attention
            output, weights = self._single_head_attention(Q_head, K_head, V_head, mask)
            head_outputs.append(output)
            attention_weights[f"head_{head_idx}"] = weights
        
        # Stack head outputs: (num_heads, seq_len, head_dim)
        head_outputs = np.stack(head_outputs, axis=0)
        
        # Step 5: Combine heads
        combined = self._combine_heads(head_outputs)  # (seq_len, embedding_dim)
        
        # Step 6: Output projection
        output = np.dot(combined, self.W_o)  # (seq_len, embedding_dim)
        
        return output, attention_weights

# =============================================================================
# STEP 4: Example Usage
# =============================================================================

print("\n--- Multi-Head Attention Example ---")
print("="*50)
print("""
REAL-WORLD SCENARIO: Document Analysis Team
============================================

You have a 6-paragraph document (6 tokens).
Your team has 4 experts (4 heads).
Each expert analyzes 16 features (head_dim = 16).

Let's see how the team processes this document!
""")

# Parameters (similar to GPT-2 small)
embedding_dim = 64  # Using smaller for demo (GPT-2 uses 768)
num_heads = 4       # GPT-2 small uses 12
seq_len = 6

print(f"\nConfiguration:")
print(f"  Sequence length: {seq_len} (6 tokens/paragraphs)")
print(f"  Embedding dimension: {embedding_dim} (64 features)")
print(f"  Number of heads: {num_heads} (4 experts)")
print(f"  Head dimension: {embedding_dim // num_heads} (16 features per expert)")

# Create multi-head attention layer
print("\n" + "-"*50)
print("Setting up the expert team...")
mha = MultiHeadAttention(embedding_dim, num_heads)

# Create sample embeddings (the document to analyze)
np.random.seed(42)
embeddings = np.random.randn(seq_len, embedding_dim)
print(f"\nDocument to analyze: {embeddings.shape}")
print(f"  → 6 paragraphs, each with 64 features")

# Forward pass - team analyzes the document
print(f"\n" + "-"*50)
print("Team analyzing document...")
output, attn_weights = mha.forward(embeddings)

print(f"\n" + "="*50)
print("ANALYSIS COMPLETE:")
print("="*50)
print(f"Output shape: {output.shape}")
print(f"  → 6 paragraphs, each now has 64 contextualized features")
print(f"Number of expert reports: {len(attn_weights)}")
print(f"  → Each head produced its own attention pattern")

# =============================================================================
# STEP 5: Visualizing Attention Patterns
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Attention Patterns Across Heads")
print("="*70)

print("\nEach head learns different attention patterns!")
print("Let's look at the attention weights for each head:\n")

for head_name, weights in attn_weights.items():
    print(f"{head_name} attention weights (shape {weights.shape}):")
    print(f"  Row sums (should be 1.0): {np.round(weights.sum(axis=1), 4)}")
    print(f"  Mean attention per position: {np.round(weights.mean(axis=0), 4)}")
    print()

print("-"*70)
print("INTERPRETING EXPERT REPORTS:")
print("-"*70)
print("""
Each head's attention pattern shows what that expert focuses on:

EXAMPLE READING (Head 0, Row 2):
  [0.35, 0.25, 0.20, 0.10, 0.05, 0.05]
  
  Token 2 (expert 0's analysis) attends to:
  - 35% to token 0 (first paragraph)
  - 25% to token 1 (second paragraph)
  - 20% to itself (current paragraph)
  - Less to later tokens (causal mask!)
  
  → This head focuses heavily on earlier context!

DIFFERENT HEADS, DIFFERENT PATTERNS:
- Head 0 might focus on: Recent context (nearby tokens)
- Head 1 might focus on: Beginning of sequence
- Head 2 might focus on: Self-attention (current token)
- Head 3 might focus on: Balanced distribution

This diversity is the POWER of multi-head attention!
=============================================================================""")

# =============================================================================
# STEP 6: Comparing Single vs Multi-Head
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Single vs Multi-Head Comparison")
print("="*70)

class SingleHeadAttention:
    """Single head attention for comparison."""
    
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.1
    
    def forward(self, embeddings):
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        d_k = K.shape[1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        seq_len = embeddings.shape[0]
        mask = create_causal_mask(seq_len)
        scores = scores + mask
        
        weights = softmax(scores)
        output = np.dot(weights, V)
        
        return output, weights

print("\nComparing single-head vs multi-head:")
print("="*50)
print("""
REAL-WORLD ANALOGY:
===================

SINGLE-HEAD: One person reading a document
- One perspective
- One set of patterns they notice
- Limited understanding

MULTI-HEAD: Team of 4 experts reading the same document
- Four different perspectives
- Four different pattern sets
- Rich, diverse understanding

Let's see the difference in outputs!
""")

# Single head
single_attn = SingleHeadAttention(embedding_dim)
single_output, single_weights = single_attn.forward(embeddings)

# Multi-head (already computed above)
multi_output = output

print(f"\nSingle-head output shape: {single_output.shape}")
print(f"Multi-head output shape: {multi_output.shape}")

print(f"\nSingle-head output (first token, first 5 dims):")
print(f"  {np.round(single_output[0, :5], 4)}")

print(f"\nMulti-head output (first token, first 5 dims):")
print(f"  {np.round(multi_output[0, :5], 4)}")

print("\n" + "-"*70)
print("KEY DIFFERENCE:")
print("-"*70)
print("""
SINGLE-HEAD:
  Output from ONE perspective
  → One way of understanding the text

MULTI-HEAD:
  Output combines FOUR perspectives
  → Grammar expert's understanding
  → Semantics expert's understanding
  → Coreference expert's understanding
  → Position expert's understanding
  → All combined into rich representation!

WHY MULTI-HEAD WINS:
1. Captures different relationship types simultaneously
2. Learns specialized attention patterns per head
3. More representational capacity
4. Better at handling ambiguous words (e.g., "bank")

EXAMPLE: "I went to the bank"
- Head 1 might attend to financial context → "bank" = money place
- Head 2 might attend to river context → "bank" = river edge
- Combined output captures both possibilities!

This is why ALL modern LLMs use multi-head attention!
=============================================================================""")

# =============================================================================
# STEP 7: Real GPT Numbers
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Multi-Head Attention in Real GPT Models")
print("="*70)

print("""
REAL GPT CONFIGURATIONS:
========================

GPT-2 Small (124M parameters):
  embedding_dim = 768
  num_heads = 12
  head_dim = 768 / 12 = 64
  Parameters in attention: 4 × 768² ≈ 2.4 million

GPT-2 Medium (355M parameters):
  embedding_dim = 1024
  num_heads = 16
  head_dim = 1024 / 16 = 64
  Parameters in attention: 4 × 1024² ≈ 4.2 million

GPT-2 Large (774M parameters):
  embedding_dim = 1280
  num_heads = 20
  head_dim = 1280 / 20 = 64
  Parameters in attention: 4 × 1280² ≈ 6.6 million

GPT-2 XL (1.5B parameters):
  embedding_dim = 1600
  num_heads = 25
  head_dim = 1600 / 25 = 64
  Parameters in attention: 4 × 1600² ≈ 10.2 million

GPT-3 (175B parameters):
  embedding_dim = 12288
  num_heads = 96
  head_dim = 12288 / 96 = 128
  Parameters in attention: 4 × 12288² ≈ 604 million!

PATTERN:
- head_dim stays around 64-128 across models
- More heads = more diverse attention patterns
- Larger models = more experts analyzing text!

=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Multi-Head Attention")
print("="*70)

print("""
REAL-WORLD ANALOGIES RECAP:
===========================

1. TEAM OF EXPERTS:
   - Grammar specialist → syntactic relationships
   - Meaning analyst → semantic relationships
   - Pronoun detective → coreference resolution
   - Position tracker → word order patterns

2. FACTORY ASSEMBLY:
   - Distribution center → split to heads
   - Parallel assembly lines → attention per head
   - Quality control → concatenate outputs
   - Final packaging → output projection

3. MAIL SORTING:
   - Split heads → sort mail to departments
   - Combine heads → department reports to CEO

MULTI-HEAD ATTENTION STEPS:
===========================

1. PROJECT: embeddings → Q, K, V using weight matrices
2. SPLIT: Divide Q, K, V into num_heads parts
3. ATTEND: Compute attention for each head independently
4. COMBINE: Concatenate all head outputs
5. PROJECT: Final linear layer to mix information

FORMULA:
========

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

PARAMETERS:
===========
- W_q, W_k, W_v: embedding_dim × embedding_dim each
- W_o: embedding_dim × embedding_dim
- Total: 4 × embedding_dim² parameters

WHY IT MATTERS:
===============
- Single attention = one perspective
- Multi-head = multiple perspectives
- Language is complex → needs multiple views
- This is the core of transformer success!

NEXT: We'll combine multi-head attention with feed-forward networks
      to build the complete Transformer Block!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Multi-Head Attention")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. CHANGE NUMBER OF HEADS:
   mha = MultiHeadAttention(embedding_dim=64, num_heads=8)
   
   Question: How does head_dim change?
   Answer: head_dim = 64/8 = 8 (smaller, more specialized heads)
   
   Try: num_heads=2 → head_dim=32 (larger, fewer heads)
   Try: num_heads=16 → head_dim=4 (tiny, many heads)

2. COMPARE ATTENTION PATTERNS:
   for head_name, weights in attn_weights.items():
       print(f"{head_name}: {weights[0]}")  # First token's attention
   
   Question: Do different heads show different patterns?
   Expectation: Yes! Each head learns different focus!

3. WITHOUT CAUSAL MASK:
   output, weights = mha.forward(embeddings, use_causal_mask=False)
   
   Question: How does attention change?
   Expectation: Tokens can attend to future tokens too!

4. SCALE UP:
   embedding_dim = 128, num_heads = 8
   head_dim = 128/8 = 16
   
   Question: How many parameters now?
   Answer: 4 × 128² = 65,536 parameters

5. VISUALIZE (MENTALLY):
   Imagine each head as a different colored heatmap:
   - Red head: Attends to nearby tokens
   - Blue head: Attends to sentence start
   - Green head: Attends to specific word types
   - Yellow head: Balanced attention
   
   Together: Full spectrum of attention patterns!

KEY TAKEAWAY:
=============
- Multi-head = multiple attention computations in parallel
- Each head learns different attention patterns
- Outputs are concatenated and projected
- This gives the model more representational power!
- Like having multiple experts instead of one person!

Next: 05_transformer_block.py - Complete transformer block!
=============================================================================""")
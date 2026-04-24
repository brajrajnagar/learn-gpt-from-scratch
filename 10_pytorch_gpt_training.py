"""
GPT from Scratch - Lesson 10: REAL GPT Training with PyTorch
=============================================================

WHAT WE'VE BUILT SO FAR (Lessons 1-9):
  1. Neural Network Basics (forward pass, activations)
  2. Embeddings (token + position)
  3. Self-Attention (Q, K, V, multi-head)
  4. Causal Masking (preventing future peeking)
  5. Transformer Block (attention + FFN + residuals)
  6. Complete GPT Model (stack of transformer blocks)
  7-9. Training concepts (loss, backprop, optimization)

TODAY'S PLACE IN PIPELINE:
  Lessons 1-9 (concepts with NumPy) → [THIS: Real Training with PyTorch]

WHY PYTORCH? (Why not keep using NumPy?)
  NumPy is great for learning, but for REAL training we need:
  - Automatic differentiation  (no manual backprop!)
  - GPU acceleration            (100x faster than CPU)
  - Production-ready code        (used by OpenAI, Meta, Google)

  Analogy: NumPy is like building a car engine by hand to learn how it works.
           PyTorch is like using a real engine to actually DRIVE somewhere.

WHAT WE'LL BUILD:
  A real GPT model (~10M parameters) trained on Shakespeare text.
  After training, it will generate Shakespeare-style text!

  "ROMEO: O, what light through yonder window breaks..."
                    ↑ Our model will learn to generate text like this!

HOW TO RUN:
  python 10_pytorch_gpt_training.py --mode train       # Train from scratch
  python 10_pytorch_gpt_training.py --mode resume      # Resume from checkpoint
  python 10_pytorch_gpt_training.py --mode inference    # Generate text only
  python 10_pytorch_gpt_training.py --mode inference --prompt "ROMEO:" --temperature 0.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os
import argparse
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)


# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
#
# This lets you control the script from the terminal:
#
#   --mode train       Train a new model from scratch
#   --mode resume      Continue training from a saved checkpoint
#   --mode inference   Just generate text (no training)
#
# Examples:
#   python 10_pytorch_gpt_training.py --mode train --max_iters 10000
#   python 10_pytorch_gpt_training.py --mode train --d_model 512 --n_heads 8 --n_blocks 12
#   python 10_pytorch_gpt_training.py --mode inference --prompt "To be or" --temperature 0.5
#   python 10_pytorch_gpt_training.py --mode resume --checkpoint data/checkpoint.pt
# =============================================================================

parser = argparse.ArgumentParser(description='GPT Training from Scratch')

# --- Mode ---
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'resume', 'inference'],
                    help='train: from scratch, resume: from checkpoint, inference: generate text')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt',
                    help='Path to checkpoint file (default: data/checkpoint.pt)')

# --- Inference ---
parser.add_argument('--prompt', type=str, default='ROMEO:',
                    help='Prompt for text generation in inference mode')
parser.add_argument('--max_tokens', type=int, default=300,
                    help='Max tokens to generate in inference mode')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for text generation (default: 0.7)')

# --- Data ---
parser.add_argument('--seq_length', type=int, default=256,
                    help='Sequence length — how many chars the model sees at once (default: 256)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size — sequences processed in parallel (default: 128)')

# --- Model Architecture ---
parser.add_argument('--d_model', type=int, default=384,
                    help='Embedding dimension (default: 384)')
parser.add_argument('--n_heads', type=int, default=6,
                    help='Number of attention heads (default: 6)')
parser.add_argument('--n_blocks', type=int, default=8,
                    help='Number of transformer blocks (default: 8)')
parser.add_argument('--d_ff', type=int, default=1536,
                    help='Feed-forward hidden dimension (default: 1536, typically 4x d_model)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (default: 0.2)')

# --- Training ---
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for AdamW (default: 3e-4)')
parser.add_argument('--max_iters', type=int, default=5000,
                    help='Total training steps (default: 5000)')
parser.add_argument('--eval_interval', type=int, default=500,
                    help='Evaluate every N steps (default: 500)')
parser.add_argument('--grad_clip', type=float, default=1.0,
                    help='Gradient clipping value (default: 1.0)')

args = parser.parse_args()


# =============================================================================
# STEP 1: Configuration
# =============================================================================
#
# Before building anything, we define our hyperparameters.
#
# Think of this like a recipe card:
#   - How much data per batch? (batch_size)
#   - How big is the model?    (d_model, n_heads, n_blocks)
#   - How fast do we learn?    (learning_rate)
#
# OUR MODEL vs GPT-2:
# ┌──────────────┬──────────┬──────────┐
# │ Parameter    │ Ours     │ GPT-2    │
# ├──────────────┼──────────┼──────────┤
# │ d_model      │ 384      │ 768      │
# │ n_heads      │ 6        │ 12       │
# │ n_blocks     │ 8        │ 12       │
# │ d_ff         │ 1536     │ 3072     │
# │ Total params │ ~10M     │ ~124M    │
# └──────────────┴──────────┴──────────┘
#
# Our model is ~12x smaller than GPT-2, but uses the SAME architecture.
# It's big enough to learn Shakespeare patterns well!
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Configuration")
print("="*70)

class Config:
    # Data
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    seq_length = args.seq_length        # How many characters the model sees at once
    batch_size = args.batch_size        # How many sequences to process in parallel

    # Model Architecture (GPT-2 style, ~10M params with defaults)
    vocab_size = 65                     # Will be set from data (number of unique characters)
    d_model = args.d_model              # Embedding dimension (each token becomes a vector)
    n_heads = args.n_heads              # Attention heads (different "perspectives")
    n_blocks = args.n_blocks            # Transformer blocks (layers of processing)
    d_ff = args.d_ff                    # Feed-forward hidden dim (typically 4x d_model)
    dropout = args.dropout              # Randomly drop connections (prevents overfitting)

    # Training
    learning_rate = args.learning_rate  # How big each learning step is
    max_iters = args.max_iters          # Total training steps
    eval_interval = args.eval_interval  # Evaluate every N steps
    eval_iters = 200                    # Average loss over N batches for stable evaluation
    grad_clip = args.grad_clip          # Clip gradients to prevent exploding gradients
    compile_model = True                # Use torch.compile for speed (CUDA only)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# ---- Device Detection ----
# PyTorch can run on different hardware:
#   CPU  - Works everywhere, but slow for training
#   CUDA - NVIDIA GPUs (fastest, used in data centers)
#   MPS  - Apple Silicon GPUs (M1/M2/M3/M4, great for local training)
if torch.backends.mps.is_available():
    config.device = 'mps'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    config.device = 'cuda'
    print("Using NVIDIA GPU (CUDA)")
else:
    config.device = 'cpu'
    print("Using CPU (no GPU available)")

print(f"Device: {config.device}")
print(f"Batch size: {config.batch_size}")
print(f"Sequence length: {config.seq_length}")


# =============================================================================
# STEP 2: Download and Prepare Data
# =============================================================================
#
# We'll train on Shakespeare's complete works (~1MB of text).
# This is a classic dataset for language model experiments!
#
# Sample from the dataset:
#   "ROMEO: But, soft! what light through yonder window breaks?
#    It is the east, and Juliet is the sun."
#
# WHY SHAKESPEARE?
#   - Clear patterns (character names, dialogue format, iambic pentameter)
#   - Small enough to train quickly
#   - Fun to see the model learn to "write" Shakespeare!
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Download Shakespeare Data")
print("="*70)

def download_data():
    """Download Shakespeare text if not already cached."""
    os.makedirs('data', exist_ok=True)
    data_path = 'data/shakespeare.txt'

    if not os.path.exists(data_path):
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(Config.data_url, data_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")
    print(f"\nFirst 200 characters:")
    print("-"*50)
    print(text[:200])
    print("-"*50)
    return text

text = download_data()


# =============================================================================
# STEP 3: Character Tokenizer
# =============================================================================
#
# THE PROBLEM: Neural networks work with NUMBERS, not text.
# THE SOLUTION: Convert each character to a unique number (token ID).
#
# Real GPT models use BPE (Byte-Pair Encoding) which groups characters
# into subwords like "play" or "ing". We use character-level tokenization
# for simplicity — each character is its own token.
#
# EXAMPLE:
#   Text:    "hello"
#   Tokens:  [46, 43, 50, 50, 53]    (each char → unique number)
#
#   encode("hello") → [46, 43, 50, 50, 53]
#   decode([46, 43, 50, 50, 53]) → "hello"
#
# VOCABULARY:
#   All unique characters in Shakespeare: \n, space, !, ', etc.
#   Total: 65 characters
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Build Character Tokenizer")
print("="*70)

# Step 3a: Find all unique characters and sort them
chars = sorted(list(set(text)))
vocab_size = len(chars)
config.vocab_size = vocab_size

print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Characters: {''.join(chars[:20])}...")

# Step 3b: Create bidirectional mappings
#   stoi: string → integer  ("h" → 46)
#   itos: integer → string  (46 → "h")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    """Convert string to list of token IDs."""
    return [stoi[c] for c in s]

def decode(l):
    """Convert list of token IDs back to string."""
    return ''.join([itos[i] for i in l])

# Demonstrate the tokenizer
print(f"\nTokenizer demo:")
print(f"  encode('hello')    → {encode('hello')}")
print(f"  decode([46,43...]) → '{decode(encode('hello'))}'")
print(f"  encode('ROMEO:')   → {encode('ROMEO:')}")
print(f"  Round-trip test: '{decode(encode('To be or not to be'))}'")


# =============================================================================
# STEP 4: Create Training Data
# =============================================================================
#
# HOW DOES A LANGUAGE MODEL LEARN?
#
# We give it sequences of characters and ask it to predict the NEXT one:
#
#   Input:  "To be or not to b"    → Target: "o be or not to be"
#   Input:  "ROMEO: O, what lig"   → Target: "OMEO: O, what ligh"
#
# Notice: target is just the input shifted by 1 position!
#
#   Position:  0  1  2  3  4  5  6  7
#   Input:     T  o     b  e     o  r
#   Target:    o     b  e     o  r     ← shifted right by 1
#
# At each position, the model predicts: "Given everything before me,
# what character comes next?"
#
# DATA SPLIT:
#   90% for training (model learns from this)
#   10% for validation (we check if model is actually learning, not memorizing)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Create Training Data")
print("="*70)

# Convert entire text to tensor of token IDs
data = torch.tensor(encode(text), dtype=torch.long)

# Split: first 90% for training, last 10% for validation
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(f"Total tokens:      {len(data):,}")
print(f"Training tokens:   {len(train_data):,} (90%)")
print(f"Validation tokens: {len(val_data):,} (10%)")

def get_batch(split, seq_length, batch_size, device):
    """
    Get a random batch of input-target sequence pairs.

    Example (seq_length=8, batch_size=2):
      Random positions: [1000, 5000]

      Batch[0] input:  data[1000:1008] = "To be or"
      Batch[0] target: data[1001:1009] = "o be or "  ← shifted by 1!

      Batch[1] input:  data[5000:5008] = "What is "
      Batch[1] target: data[5001:5009] = "hat is t"  ← shifted by 1!
    """
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - seq_length, (batch_size,))
    x = torch.stack([data_source[i:i+seq_length] for i in ix])
    y = torch.stack([data_source[i+1:i+seq_length+1] for i in ix])
    return x.to(device), y.to(device)

# Demonstrate a batch
demo_x, demo_y = get_batch('train', 8, 1, 'cpu')
print(f"\nBatch demo (seq_length=8):")
print(f"  Input:  {demo_x[0].tolist()} → '{decode(demo_x[0].tolist())}'")
print(f"  Target: {demo_y[0].tolist()} → '{decode(demo_y[0].tolist())}'")
print(f"  Notice: target is input shifted right by 1 character!")


# =============================================================================
# STEP 5: Multi-Head Attention (PyTorch version)
# =============================================================================
#
# Remember from Lesson 3? Multi-Head Attention lets each token ask:
#   "Which other tokens should I pay attention to?"
#
# CAUSAL (MASKED) ATTENTION:
#   In GPT, tokens can ONLY attend to tokens BEFORE them (not future ones).
#   This is enforced by a triangular mask:
#
#   Token 0: can see [0]
#   Token 1: can see [0, 1]
#   Token 2: can see [0, 1, 2]
#   Token 3: can see [0, 1, 2, 3]
#
#   Mask matrix (1=can see, 0=blocked):
#   ┌─────────────────┐
#   │ 1  0  0  0      │  ← Token 0 sees only itself
#   │ 1  1  0  0      │  ← Token 1 sees tokens 0,1
#   │ 1  1  1  0      │  ← Token 2 sees tokens 0,1,2
#   │ 1  1  1  1      │  ← Token 3 sees all past tokens
#   └─────────────────┘
#
# MULTI-HEAD: We run attention 6 times in parallel, each "head" learning
# to focus on different relationships:
#   Head 1 might learn: grammar patterns
#   Head 2 might learn: character name associations
#   Head 3 might learn: rhyme/rhythm patterns
#   ...etc.
#
# PyTorch version of what we built in Lesson 3, but with:
#   - nn.Linear instead of manual weight matrices
#   - nn.Dropout for regularization
#   - Efficient batched computation
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Multi-Head Attention (PyTorch)")
print("="*70)

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    FLOW:
      Input (B, T, d_model)
        ↓
      Project to Q, K, V                    ← "What am I looking for? What do I offer?"
        ↓
      Split into n_heads                     ← Multiple perspectives
        ↓
      Compute attention scores (Q @ K^T)     ← "How relevant is each token?"
        ↓
      Apply causal mask                      ← Block future tokens
        ↓
      Softmax → attention weights            ← Normalize to probabilities
        ↓
      Apply to Values (weights @ V)          ← Weighted combination
        ↓
      Concatenate heads + output projection  ← Combine perspectives
        ↓
      Output (B, T, d_model)
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # Each head works with this many dimensions

        # Linear projections for Q, K, V and output
        # Remember from Lesson 3:
        #   Q = "What am I looking for?"
        #   K = "What do I contain?"
        #   V = "What information do I provide?"
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Combines all heads

        # Dropout for regularization (randomly zeros some attention weights)
        self.attn_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of 1s
        # This prevents tokens from attending to future positions
        self.register_buffer('causal_mask',
            torch.tril(torch.ones(config.seq_length, config.seq_length))
                .view(1, 1, config.seq_length, config.seq_length))

    def forward(self, x):
        B, T, _ = x.shape  # B=batch, T=sequence length

        # Step 1: Project input to Q, K, V and reshape for multi-head
        #   (B, T, d_model) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 2: Compute attention scores
        #   score = Q @ K^T / sqrt(head_dim)
        #   Shape: (B, n_heads, T, T) — each token's score with every other token
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)

        # Step 3: Apply causal mask (set future positions to -infinity)
        #   After softmax, -inf becomes 0 → token can't attend to future
        scores = scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))

        # Step 4: Softmax → attention weights (probabilities that sum to 1)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Step 5: Apply attention weights to values
        #   out = attention_weights @ V → weighted combination of values
        out = attn @ V  # (B, n_heads, T, head_dim)

        # Step 6: Concatenate all heads and project
        #   (B, n_heads, T, head_dim) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)

        return out

print("MultiHeadAttention defined!")
print(f"  d_model={config.d_model}, n_heads={config.n_heads}, head_dim={config.d_model // config.n_heads}")
print(f"  Each of {config.n_heads} heads works with {config.d_model // config.n_heads} dimensions")


# =============================================================================
# STEP 6: Transformer Block (PyTorch version)
# =============================================================================
#
# Remember from Lesson 5? The Transformer Block combines:
#   1. Multi-Head Attention  (gather info from other tokens)
#   2. Feed-Forward Network  (process the gathered info)
#   3. Layer Normalization   (stabilize training)
#   4. Residual Connections  (help gradients flow in deep networks)
#
# ARCHITECTURE (Pre-LayerNorm — modern practice, used in GPT-2/3/4):
#
#   Input
#     ↓
#   ┌─────────────────────────────────┐
#   │  LayerNorm                      │
#   │  Multi-Head Attention           │
#   │  + Residual Connection          │  ← x = x + attention(LayerNorm(x))
#   └─────────────────────────────────┘
#     ↓
#   ┌─────────────────────────────────┐
#   │  LayerNorm                      │
#   │  Feed-Forward Network (GELU)    │
#   │  + Residual Connection          │  ← x = x + FFN(LayerNorm(x))
#   └─────────────────────────────────┘
#     ↓
#   Output (same shape as input!)
#
# WHY RESIDUAL CONNECTIONS?
#   Without them, gradients vanish in deep networks (8+ layers).
#   Residual = "add the input back to the output" so gradients
#   always have a direct path back through the network.
#
#   Analogy: It's like taking notes in a meeting. The residual connection
#   is your original knowledge, and the attention/FFN output is the new
#   insights. You don't forget what you knew — you ADD new info on top.
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Transformer Block (PyTorch)")
print("="*70)

class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention + FFN with LayerNorm and residuals.

    Input shape:  (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)  ← SAME shape! Blocks are stackable.
    """

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        # Sub-layer 1: Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

        # Sub-layer 2: Feed-Forward Network
        #   d_model → d_ff (expand) → d_model (compress back)
        #   GELU activation adds non-linearity (smoother version of ReLU)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)     # Expand:  384 → 1536
        self.ff2 = nn.Linear(d_ff, d_model)      # Compress: 1536 → 384

    def forward(self, x):
        # Sub-layer 1: Attention with residual
        # "What information do I need from other tokens?"
        x = x + self.attention(self.ln1(x))

        # Sub-layer 2: FFN with residual
        # "Now let me PROCESS that information."
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))

        return x

print("TransformerBlock defined!")
print(f"  Structure: Input → LN → Attention → +Residual → LN → FFN → +Residual → Output")
print(f"  FFN expansion: {config.d_model} → {config.d_ff} → {config.d_model}")


# =============================================================================
# STEP 7: Complete GPT Model
# =============================================================================
#
# Now we assemble ALL the pieces into a complete GPT model!
#
# FULL ARCHITECTURE:
#
#   Input token IDs: [52, 47, 43, 1, ...]     ("ROMEO:")
#     ↓
#   ┌────────────────────────────────────────┐
#   │  Token Embedding                       │  Each token → 384-dim vector
#   │  + Position Embedding                  │  Add position information
#   │  + Dropout                             │  Regularization
#   └────────────────────────────────────────┘
#     ↓
#   ┌────────────────────────────────────────┐
#   │  Transformer Block 1                   │  ← Basic patterns
#   │  Transformer Block 2                   │  ← Word-level patterns
#   │  Transformer Block 3                   │  ← Phrase patterns
#   │  Transformer Block 4                   │  ← Sentence structure
#   │  Transformer Block 5                   │  ← Dialogue patterns
#   │  Transformer Block 6                   │  ← Character behavior
#   │  Transformer Block 7                   │  ← Plot structure
#   │  Transformer Block 8                   │  ← High-level style
#   └────────────────────────────────────────┘
#     ↓
#   ┌────────────────────────────────────────┐
#   │  Final LayerNorm                       │  Stabilize final output
#   └────────────────────────────────────────┘
#     ↓
#   ┌────────────────────────────────────────┐
#   │  Output Projection (lm_head)           │  384-dim → 65 (vocab_size)
#   └────────────────────────────────────────┘
#     ↓
#   Logits: probability for each character    [0.01, 0.02, ..., 0.15, ...]
#     ↓
#   Predicted next character: "O"
#
# WEIGHT TYING:
#   The output projection (lm_head) shares weights with token_emb.
#   This means: the same matrix that converts tokens→vectors also
#   converts vectors→token probabilities. This reduces parameters
#   and improves performance!
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Complete GPT Model")
print("="*70)

class GPT(nn.Module):
    """
    Complete GPT language model.

    Input:  Token IDs  (batch, seq_len)       e.g., [[52, 47, 43, 1, ...]]
    Output: Logits     (batch, seq_len, vocab) e.g., probability of each next char
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Embedding layers ---
        # Token embedding: each of 65 characters → 384-dim vector
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Position embedding: each position (0-255) → 384-dim vector
        # This tells the model WHERE each token is in the sequence
        self.pos_emb = nn.Embedding(config.seq_length, config.d_model)

        # Embedding dropout (regularization)
        self.emb_dropout = nn.Dropout(config.dropout)

        # --- Transformer blocks (the "brain" of GPT) ---
        # Stack 8 identical blocks, each building deeper understanding
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_blocks)
        ])

        # --- Output layers ---
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output projection: 384-dim → 65 (one score per character)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Weight tying: share weights between token_emb and lm_head
        # Intuition: if "R" maps to vector [0.5, -0.3, ...] in embedding,
        # then a hidden state close to [0.5, -0.3, ...] should predict "R"
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        """
        Forward pass: token IDs → next-token predictions.

        Args:
            idx:     Token IDs, shape (batch, seq_len)
            targets: Target token IDs for loss computation (optional)

        Returns:
            logits: Raw predictions, shape (batch, seq_len, vocab_size)
            loss:   Cross-entropy loss (only if targets provided)
        """
        B, T = idx.shape

        # Step 1: Embeddings
        #   Token: "R" → [0.5, -0.3, 0.1, ...]  (what this character means)
        #   Position: pos=0 → [0.2, 0.1, -0.4, ...]  (where in the sequence)
        #   Combined: token + position (character identity + position info)
        tok_emb = self.token_emb(idx)                              # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device)) # (T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)                    # (B, T, d_model)

        # Step 2: Pass through transformer blocks
        #   Each block refines the representation:
        #   Block 1: "R is a letter, O follows R in ROMEO"
        #   Block 4: "This looks like a character name"
        #   Block 8: "A speech by Romeo is about to begin"
        x = self.blocks(x)       # (B, T, d_model)
        x = self.ln_final(x)     # (B, T, d_model)

        # Step 3: Project to vocabulary
        #   384-dim hidden state → 65 scores (one per character)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Step 4: Compute loss if targets provided
        #   Cross-entropy loss: "How wrong were our predictions?"
        #   Lower loss = better predictions
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  # (B*T, vocab_size)
                targets.view(-1),                          # (B*T,)
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively (one at a time).

        HOW GENERATION WORKS:
          1. Feed input tokens through the model
          2. Get probability distribution for next token
          3. Sample from that distribution
          4. Append sampled token to input
          5. Repeat!

        TEMPERATURE controls randomness:
          Low (0.3):  Conservative, repetitive, "safe" choices
          Medium (0.7): Creative but coherent (sweet spot!)
          High (1.3):  Wild, unpredictable, sometimes nonsensical

        Example:
          Input:  "ROMEO:"
          Step 1: Model predicts next char → " " (space)
          Step 2: "ROMEO: " → "O"
          Step 3: "ROMEO: O" → ","
          ...and so on, building text character by character!
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length (model can only see 256 chars at once)
            idx_cond = idx[:, -self.config.seq_length:]

            # Get predictions for the LAST position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Scale by temperature

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append new token
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

# ---- Create the model ----
model = GPT(config)

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel created!")
print(f"  Parameters:    {num_params:,} ({num_params/1e6:.2f}M)")
print(f"  Blocks:        {config.n_blocks}")
print(f"  Embedding dim: {config.d_model}")
print(f"  Attention heads: {config.n_heads} (each with {config.d_model // config.n_heads} dims)")
print(f"  FFN hidden dim:  {config.d_ff}")

# torch.compile: Fuses operations for speed (CUDA only, not supported on MPS/CPU)
if config.compile_model and hasattr(torch, 'compile') and config.device == 'cuda':
    try:
        model = torch.compile(model)
        print(f"  torch.compile: ENABLED (faster training)")
    except Exception as e:
        print(f"  torch.compile: SKIPPED ({e})")
else:
    print(f"  torch.compile: SKIPPED (not supported on {config.device})")


# =============================================================================
# STEP 8: Optimizer
# =============================================================================
#
# THE OPTIMIZER: How the model learns from its mistakes.
#
# TRAINING LOOP (simplified):
#   1. Model predicts next character → "z"
#   2. Actual next character was    → "o"
#   3. Loss function says: "You were wrong by THIS much"
#   4. Optimizer adjusts weights to be less wrong next time
#
# WHY AdamW?
#   - Adam: Adapts learning rate per-parameter (some need bigger steps)
#   - W (Weight Decay): Prevents weights from growing too large
#   - It's the standard optimizer for transformer training
#
# LEARNING RATE = 3e-4 (0.0003):
#   Too high → model learns too fast, becomes unstable (loss explodes!)
#   Too low  → model learns too slowly, wastes compute
#   3e-4     → Sweet spot for AdamW with transformers (Karpathy's recommendation)
# =============================================================================

print("\n" + "="*70)
print("STEP 8: Setup Optimizer")
print("="*70)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

print(f"Optimizer:     AdamW (industry standard for transformers)")
print(f"Learning rate: {config.learning_rate}")
print(f"Grad clipping: {config.grad_clip} (prevents exploding gradients)")


# =============================================================================
# STEP 9: Checkpoint Save/Load
# =============================================================================
#
# WHY CHECKPOINTS?
#   Training can take hours. If it crashes, you don't want to start over!
#   A checkpoint saves EVERYTHING needed to resume:
#
#   ┌─────────────────────────────────────────────┐
#   │ Checkpoint file (data/checkpoint.pt)         │
#   │                                              │
#   │  model_state_dict     ← All model weights    │
#   │  optimizer_state_dict ← Optimizer momentum    │
#   │  step                 ← Where we stopped     │
#   │  best_val_loss        ← Best score so far     │
#   │  config               ← Model architecture    │
#   │  chars                ← Vocabulary mapping     │
#   └─────────────────────────────────────────────┘
#
# USAGE:
#   Save: Automatically saved when validation loss improves
#   Load: python 10_pytorch_gpt_training.py --mode resume
#         python 10_pytorch_gpt_training.py --mode inference
# =============================================================================

print("\n" + "="*70)
print("STEP 9: Checkpoint Save/Load")
print("="*70)

def save_checkpoint(model, optimizer, step, best_val_loss, path):
    """Save full training checkpoint (model + optimizer + progress)."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_blocks': config.n_blocks,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'seq_length': config.seq_length,
            'learning_rate': config.learning_rate,
            'max_iters': config.max_iters,
        },
        'chars': chars,  # Save vocabulary for inference
    }
    torch.save(checkpoint, path)
    tqdm.write(f"Checkpoint saved at step {step} -> {path}")

def load_checkpoint(model, optimizer, path, device):
    """Load training checkpoint. Returns (start_step, best_val_loss)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Checkpoint loaded from {path}")
    print(f"  Resuming from step {start_step}, best val loss: {best_val_loss:.4f}")
    return start_step, best_val_loss

# Derive best checkpoint path from the main checkpoint path
# Example: data/checkpoint.pt -> data/checkpoint_best.pt
checkpoint_dir = os.path.dirname(args.checkpoint)
checkpoint_base = os.path.basename(args.checkpoint)
name, ext = os.path.splitext(checkpoint_base)
args.checkpoint_best = os.path.join(checkpoint_dir, f"{name}_best{ext}")

print("Checkpoint functions defined!")
print(f"  Latest checkpoint (resume):   {args.checkpoint}")
print(f"  Best checkpoint (inference):  {args.checkpoint_best}")


# =============================================================================
# STEP 10: Training Loop
# =============================================================================
#
# THE TRAINING LOOP: Where the model actually learns!
#
# Each training step:
#   ┌──────────────────────────────────────────────────────┐
#   │  1. GET BATCH        Random chunk of Shakespeare     │
#   │  2. FORWARD PASS     Model predicts next characters  │
#   │  3. COMPUTE LOSS     How wrong were the predictions? │
#   │  4. BACKWARD PASS    Compute gradients (auto!)       │
#   │  5. CLIP GRADIENTS   Prevent exploding gradients     │
#   │  6. UPDATE WEIGHTS   Optimizer adjusts the model     │
#   └──────────────────────────────────────────────────────┘
#
# WHAT THE METRICS MEAN:
#   Train Loss:  How well model fits training data (lower = better)
#   Val Loss:    How well model generalizes (lower = better, this is key!)
#   Perplexity:  "How confused is the model?" = e^(val_loss)
#                - Perplexity 65 = random guessing (65 chars in vocab)
#                - Perplexity 10 = model is narrowing down to ~10 options
#                - Perplexity 3  = model is quite confident (good!)
#
# EXPECTED PROGRESSION:
#   Step 0:    loss ~4.2, perplexity ~65  (random guessing)
#   Step 500:  loss ~2.0, perplexity ~7   (learning basic patterns)
#   Step 2000: loss ~1.5, perplexity ~4   (learning word patterns)
#   Step 5000: loss ~1.3, perplexity ~3.5 (generating coherent text)
# =============================================================================

print("\n" + "="*70)
print("STEP 10: Training")
print("="*70)

# Move model to GPU/MPS for faster training
model.to(config.device)

@torch.no_grad()
def estimate_loss():
    """
    Estimate validation loss by averaging over many batches.
    We average over eval_iters batches to get a stable estimate.
    (A single batch is too noisy to judge model quality.)
    """
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        X, Y = get_batch('val', config.seq_length, config.batch_size, config.device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def generate_text(prompt, max_length=200, temperature=1.0):
    """Generate text from a prompt string."""
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...]
    with torch.no_grad():
        generated = model.generate(x, max_length, temperature=temperature)
    return decode(generated[0].tolist())


# ---- INFERENCE MODE: Just generate text, no training ----
if args.mode == 'inference':
    # Prefer best checkpoint for inference (highest quality model)
    # Fall back to latest checkpoint if best doesn't exist
    if os.path.exists(args.checkpoint_best):
        checkpoint_path = args.checkpoint_best
        print(f"Using BEST checkpoint for inference: {checkpoint_path}")
    elif os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
        print(f"Best checkpoint not found, using latest: {checkpoint_path}")
    else:
        print(f"ERROR: No checkpoint found at {args.checkpoint} or {args.checkpoint_best}")
        print(f"  Train first: python 10_pytorch_gpt_training.py --mode train")
        exit(1)
    load_checkpoint(model, None, checkpoint_path, config.device)
    model.eval()

    print(f"\nGenerating with prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print("-" * 60)
    print(generate_text(args.prompt, max_length=args.max_tokens, temperature=args.temperature))
    print("-" * 60)
    exit(0)


# ---- TRAIN or RESUME MODE ----
start_step = 0
best_val_loss = float('inf')

if args.mode == 'resume':
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print(f"  Train first: python 10_pytorch_gpt_training.py --mode train")
        exit(1)
    start_step, best_val_loss = load_checkpoint(model, optimizer, args.checkpoint, config.device)
    model.to(config.device)
else:
    print("Training from scratch...")

print(f"\nTraining plan:")
print(f"  Steps: {start_step} → {config.max_iters}")
print(f"  Batch: {config.batch_size} sequences x {config.seq_length} chars = {config.batch_size * config.seq_length:,} chars/step")
print(f"  Evaluate every {config.eval_interval} steps")

pbar = tqdm(range(start_step, config.max_iters), desc="Training", unit="step")
for step in pbar:
    # 1. Get a random batch of training data
    X, Y = get_batch('train', config.seq_length, config.batch_size, config.device)

    # 2. Forward pass: model predicts next characters
    _, loss = model(X, Y)

    # 3. Backward pass: compute gradients (PyTorch does this automatically!)
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients

    # 4. Clip gradients to prevent training instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # 5. Update model weights
    optimizer.step()

    # Update progress bar
    pbar.set_postfix(train_loss=f"{loss.item():.4f}")

    # Periodic evaluation
    if step % config.eval_interval == 0 or step == config.max_iters - 1:
        val_loss = estimate_loss()
        perplexity = torch.exp(val_loss)
        pbar.set_postfix(
            train_loss=f"{loss.item():.4f}",
            val_loss=f"{val_loss.item():.4f}",
            ppl=f"{perplexity.item():.2f}"
        )
        tqdm.write(f"Step {step:<6} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Perplexity: {perplexity.item():.2f}")

        # Save latest checkpoint (always overwrite, for crash recovery/resume)
        save_checkpoint(model, optimizer, step, best_val_loss, args.checkpoint)

        # Save best checkpoint ONLY when validation loss improves (for inference)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, step, best_val_loss, args.checkpoint_best)
            tqdm.write(f"  *** New best val loss: {best_val_loss:.4f} ***")

# Save final checkpoint (always the latest state for resume)
save_checkpoint(model, optimizer, config.max_iters - 1, best_val_loss, args.checkpoint)
print("\nTraining complete!")


# =============================================================================
# STEP 11: Generate Shakespeare Text!
# =============================================================================
#
# Now the fun part! Let's see what our model learned.
#
# We'll generate text with different temperatures:
#   0.7 → Creative but coherent (good for readable text)
#   1.0 → Balanced (model's natural distribution)
#   1.3 → Chaotic (wild, sometimes surprising, sometimes nonsense)
#
# TEMPERATURE ANALOGY:
#   Imagine choosing your next word from a bag of tiles.
#   Low temp:  Only a few tiles in the bag (safe, predictable)
#   High temp: Many tiles in the bag (wild, unpredictable)
# =============================================================================

print("\n" + "="*70)
print("STEP 11: Generate Shakespeare Text!")
print("="*70)

# Load the best checkpoint for generation (highest quality model)
if os.path.exists(args.checkpoint_best):
    print(f"Loading BEST checkpoint for generation: {args.checkpoint_best}")
    load_checkpoint(model, None, args.checkpoint_best, config.device)
else:
    print(f"Best checkpoint not found, loading latest: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, config.device)
model.eval()

print("\n--- Temperature 0.7 (Creative but coherent) ---")
print(generate_text("ROMEO:", max_length=300, temperature=0.7))

print("\n--- Temperature 1.0 (Balanced) ---")
print(generate_text("JULIET:", max_length=300, temperature=1.0))

print("\n--- Temperature 1.3 (Wild and chaotic) ---")
print(generate_text("What is", max_length=300, temperature=1.3))


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY OF LESSON 10")
print("-"*70)
print("""
CONGRATULATIONS! You've trained a REAL GPT model from scratch!

WHAT YOU BUILT:
  1. Character Tokenizer     "hello" → [46, 43, 50, 50, 53]
  2. Multi-Head Attention     6 heads, each learning different patterns
  3. 8 Transformer Blocks     Progressive understanding, basic → abstract
  4. Complete GPT Model       ~10M parameters, GPT-2 architecture
  5. Training Loop            AdamW optimizer, gradient clipping
  6. Checkpoint System        Save/load/resume training
  7. Text Generation          Temperature-controlled sampling

THE FULL PIPELINE:
  Text → Tokenize → Embed → 8x[Attention + FFN] → Predict → Sample → Text

KEY TAKEAWAYS:
  - PyTorch handles backprop automatically (no manual gradients!)
  - GPU acceleration makes real training feasible
  - Same architecture as GPT-2, just smaller
  - Temperature controls creativity vs. coherence
  - Checkpoints let you pause and resume training

MODEL COMPARISON:
  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │ Ours     │ GPT-2    │ GPT-3    │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ Parameters   │ ~10M     │ 124M     │ 175B     │
  │ Blocks       │ 8        │ 12       │ 96       │
  │ d_model      │ 384      │ 768      │ 12288    │
  │ Training     │ Minutes  │ Days     │ Months   │
  │ Data         │ 1MB      │ 40GB     │ 570GB    │
  └──────────────┴──────────┴──────────┴──────────┘

HOW TO USE:
  Train:     python 10_pytorch_gpt_training.py --mode train
  Resume:    python 10_pytorch_gpt_training.py --mode resume
  Generate:  python 10_pytorch_gpt_training.py --mode inference --prompt "ROMEO:"

You now understand the COMPLETE GPT pipeline, from theory to practice!
""")
print("="*70)

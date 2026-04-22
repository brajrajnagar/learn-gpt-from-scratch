"""
=============================================================================
LESSON 8: Text Generation Strategies - How GPT Creates Text
=============================================================================

Now that we understand training, let's learn how GPT GENERATES text!

GENERATION PROCESS:
1. Start with a prompt (input text)
2. Tokenize the prompt
3. Model predicts next token probabilities
4. Select/sample a token
5. Append to sequence
6. Repeat until done

KEY TOPICS:
1. Greedy Decoding - Always pick the best
2. Sampling - Add randomness
3. Temperature - Control randomness
4. Top-k Sampling - Limit choices
5. Top-p (Nucleus) Sampling - Dynamic selection
6. Beam Search - Explore multiple paths

Let's explore each strategy!
"""

import numpy as np

# =============================================================================
# STEP 1: Generation Basics
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Generation Basics")
print("="*70)

print("""
AUTOREGRESSIVE GENERATION:

GPT generates text one token at a time:

  Prompt: "The cat"
  ↓
  Model predicts: P(next token | "The cat")
  ↓
  Sample: " sat"
  ↓
  New sequence: "The cat sat"
  ↓
  Model predicts: P(next token | "The cat sat")
  ↓
  Sample: " on"
  ↓
  Continue...

This is called "autoregressive" because:
- Each prediction depends on previous tokens
- Generation is sequential (can't parallelize)
- Continues until stop condition

STOP CONDITIONS:
1. Max length reached
2. End-of-sequence (EOS) token generated
3. Stop string detected (e.g., "\n\n")
""")

def softmax(logits, temperature=1.0):
    """Numerically stable softmax with temperature."""
    logits = logits / temperature
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits)

print("\n" + "-"*70)
print("Let's simulate a model's predictions!")
print("-"*70)

# Simulate a vocabulary and model logits
np.random.seed(42)
vocab_size = 1000

# Create a fake vocabulary (in reality, this would be token → word mapping)
tokens = [f"token_{i}" for i in range(vocab_size)]
# Add some readable tokens
tokens[10] = "the"
tokens[11] = "cat"
tokens[12] = "sat"
tokens[13] = "on"
tokens[14] = "mat"
tokens[15] = "and"
tokens[16] = "slept"
tokens[17] = "there"
tokens[18] = "quietly"
tokens[19] = "slowly"

# Simulate model logits for next token prediction
# (In reality, these come from the model forward pass)
logits = np.random.randn(vocab_size) * 2
# Make some tokens more likely
logits[12] += 3  # "sat"
logits[13] += 2  # "on"
logits[14] += 1  # "mat"
logits[16] += 2.5  # "slept"

print(f"\nVocabulary size: {vocab_size}")
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

# Get probabilities
probs = softmax(logits, temperature=1.0)
print(f"Probability range: [{probs.min():.6f}, {probs.max():.6f}]")

# Show top tokens
top_indices = np.argsort(probs)[-10:][::-1]
print("\nTop 10 most likely next tokens:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. {tokens[idx]:12} (p={probs[idx]:.6f})")

# =============================================================================
# STEP 2: Greedy Decoding
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Greedy Decoding")
print("="*70)

print("""
GREEDY DECODING:

Always pick the token with highest probability.

  probs = model(prompt)
  next_token = argmax(probs)

PROS:
- Simple and fast
- Deterministic (same output every time)
- Good for factual tasks

CONS:
- Can produce repetitive text
- May get stuck in loops
- Less creative/diverse
- Not always optimal (myopic)

EXAMPLE:
  Prompt: "The cat"
  ↓
  P("sat") = 0.45 (highest)
  ↓
  Output: "The cat sat"
""")

def greedy_decode(probs):
    """Select the token with highest probability."""
    return np.argmax(probs)

print("\n--- Greedy Decoding Example ---")

next_token_idx = greedy_decode(probs)
print(f"Greedy selection: token_{next_token_idx} = '{tokens[next_token_idx]}'")
print(f"Probability: {probs[next_token_idx]:.6f}")

# =============================================================================
# STEP 3: Random Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Random Sampling")
print("="*70)

print("""
RANDOM SAMPLING:

Sample from the probability distribution.

  probs = model(prompt)
  next_token = sample(probs)

PROS:
- More diverse output
- Less repetitive
- More creative

CONS:
- Can produce incoherent text
- Low probability tokens might be selected
- Non-deterministic

EXAMPLE:
  Prompt: "The cat"
  ↓
  P("sat") = 0.45, P("slept") = 0.25, P("ran") = 0.10, ...
  ↓
  Sample: "slept" (not the most likely, but still probable)
""")

def sample(probs):
    """Sample from probability distribution."""
    return np.random.choice(len(probs), p=probs)

print("\n--- Random Sampling Example ---")

print("Sampling 10 times from the distribution:")
samples = []
for i in range(10):
    token_idx = sample(probs)
    samples.append(tokens[token_idx])
    print(f"  {i+1}. '{tokens[token_idx]}' (p={probs[token_idx]:.6f})")

print(f"\nSampled tokens: {samples}")
print("Notice: Different tokens each time!")

# =============================================================================
# STEP 4: Temperature Scaling
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Temperature Scaling")
print("="*70)

print("""
TEMPERATURE:

Controls the randomness of predictions.

  probs = softmax(logits / temperature)

Temperature effects:
- T < 1.0: More confident (peaky distribution)
- T = 1.0: Normal (original distribution)
- T > 1.0: More random (flatter distribution)
- T → 0: Greedy (only highest probability)
- T → ∞: Uniform (all tokens equally likely)

EXAMPLE:
  Logits: [2.0, 1.0, 0.5]
  
  T = 0.5: probs = [0.73, 0.20, 0.07]  ← More confident
  T = 1.0: probs = [0.54, 0.33, 0.13]  ← Normal
  T = 2.0: probs = [0.43, 0.33, 0.24]  ← More random
""")

print("\n--- Temperature Effect ---")

# Get logits for top 5 tokens
top_logits = logits[top_indices]
top_tokens = [tokens[i] for i in top_indices]

print("\nComparing probabilities at different temperatures:")
print(f"{'Token':<12} | {'T=0.5':<10} | {'T=1.0':<10} | {'T=2.0':<10}")
print("-" * 50)

for temp in [0.5, 1.0, 2.0]:
    probs_temp = softmax(top_logits, temperature=temp)
    if temp == 1.0:
        print(f"(Normal distribution)")
    for i, (token, prob) in enumerate(zip(top_tokens, probs_temp)):
        if temp == 0.5 or i == 0:
            print(f"{token:<12} | {prob:<10.4f} | ", end="")
        elif temp == 1.0:
            print(f"{prob:<10.4f} | ", end="")
    if temp == 2.0:
        probs_2 = softmax(top_logits, temperature=2.0)
        for i, prob in enumerate(probs_2):
            if i == 0:
                print(f"{prob:<10.4f}")
                break

# Show full comparison
print("\nFull comparison:")
for temp in [0.5, 1.0, 2.0]:
    probs_temp = softmax(top_logits, temperature=temp)
    print(f"\nTemperature = {temp}:")
    for i, (token, prob) in enumerate(zip(top_tokens[:5], probs_temp[:5])):
        bar = "█" * int(prob * 50)
        print(f"  {token:<12}: {prob:.4f} {bar}")

# =============================================================================
# STEP 5: Top-k Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Top-k Sampling")
print("="*70)

print("""
TOP-k SAMPLING:

Only sample from the k most likely tokens.

  top_k_indices = argsort(probs)[-k:]
  top_k_probs = probs[top_k_indices]
  top_k_probs = top_k_probs / sum(top_k_probs)  # Renormalize
  next_token = sample(top_k_probs)

PROS:
- Filters out unlikely tokens
- More controlled than pure sampling
- Reduces weird outputs

CONS:
- Still might include bad tokens if k is large
- Fixed k doesn't adapt to uncertainty

TYPICAL VALUES:
- k = 40-100 for creative writing
- k = 10-40 for more focused output
""")

def top_k_sampling(probs, k=40):
    """Sample from top k tokens."""
    # Get top k indices
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_probs = probs[top_k_indices]
    
    # Renormalize
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    # Sample
    sampled_idx = np.random.choice(k, p=top_k_probs)
    return top_k_indices[sampled_idx]

print("\n--- Top-k Sampling Example ---")

k = 5
print(f"Sampling from top {k} tokens:")

top_k_indices = np.argsort(probs)[-k:][::-1]
top_k_probs = probs[top_k_indices]
top_k_probs_norm = top_k_probs / top_k_probs.sum()

print(f"\nTop {k} tokens and their (renormalized) probabilities:")
for i, idx in enumerate(top_k_indices):
    bar = "█" * int(top_k_probs_norm[i] * 50)
    print(f"  {tokens[idx]:<12}: {top_k_probs_norm[i]:.4f} {bar}")

print(f"\nSampling 5 times with k={k}:")
for i in range(5):
    token_idx = top_k_sampling(probs, k=k)
    print(f"  {i+1}. '{tokens[token_idx]}'")

# =============================================================================
# STEP 6: Top-p (Nucleus) Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Top-p (Nucleus) Sampling")
print("="*70)

print("""
TOP-p SAMPLING (Nucleus Sampling):

Sample from the smallest set of tokens whose cumulative
probability exceeds p.

  Sort tokens by probability (descending)
  Find smallest k where sum(top_k_probs) >= p
  Sample from these k tokens

PROS:
- Dynamically adjusts vocabulary size
- More tokens when uncertain, fewer when confident
- Better quality than top-k in many cases

CONS:
- Slightly more complex
- Can still produce unexpected outputs

TYPICAL VALUES:
- p = 0.9-0.95 for creative writing
- p = 0.75-0.9 for more focused output

EXAMPLE:
  probs = [0.4, 0.3, 0.15, 0.1, 0.05, ...]
  p = 0.9
  
  Cumulative: [0.4, 0.7, 0.85, 0.95, 1.0, ...]
                          ↑
              First to exceed 0.9
  
  Sample from: [0.4, 0.3, 0.15, 0.1] (top 4 tokens)
""")

def top_p_sampling(probs, p=0.9):
    """Sample from top p cumulative probability."""
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Cumulative probability
    cumsum = np.cumsum(sorted_probs)
    
    # Find cutoff
    cutoff_index = np.searchsorted(cumsum, p)
    
    # Get top p tokens
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_probs = sorted_probs[:cutoff_index + 1]
    
    # Renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    # Sample
    sampled_idx = np.random.choice(len(top_p_indices), p=top_p_probs)
    return top_p_indices[sampled_idx]

print("\n--- Top-p Sampling Example ---")

p = 0.9
print(f"Sampling from tokens that make up top {p*100:.0f}% probability:")

# Show cumulative probability
sorted_indices = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_indices]
cumsum = np.cumsum(sorted_probs)

print(f"\nTop tokens with cumulative probability:")
cumulative = 0
nucleus_size = 0
for i, idx in enumerate(sorted_indices[:10]):
    cumulative += probs[idx]
    in_nucleus = "← NUCLEUS" if cumulative <= p or i == 0 else ""
    if cumulative <= p + 0.1:  # Show a bit past cutoff
        nucleus_size = i + 1
    print(f"  {tokens[idx]:<12}: p={probs[idx]:.4f}, cumsum={cumulative:.4f} {in_nucleus}")

print(f"\nNucleus size: {nucleus_size} tokens")

print(f"\nSampling 5 times with p={p}:")
for i in range(5):
    token_idx = top_p_sampling(probs, p=p)
    print(f"  {i+1}. '{tokens[token_idx]}'")

# =============================================================================
# STEP 7: Beam Search
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Beam Search")
print("="*70)

print("""
BEAM SEARCH:

Keep track of k best sequences and expand them.

  1. Generate top-k candidates for first token
  2. For each candidate, generate top-k next tokens
  3. Score all k×k sequences
  4. Keep top k sequences
  5. Repeat until done

PROS:
- Finds globally optimal sequences (better than greedy)
- Good for tasks with clear objectives (translation)

CONS:
- Computationally expensive
- Can produce generic/bland text
- Not great for open-ended generation

BEAM WIDTH:
- Small (k=1): Greedy decoding
- Medium (k=5-10): Good balance
- Large (k=50+): Near-optimal but expensive

VISUALIZATION:
  Step 1: ["The cat", "A cat", "The dog"]  (beam=3)
  Step 2: ["The cat sat", "The cat ran", "A cat sat", ...]
          Keep top 3: ["The cat sat", "A cat sat", "The dog ran"]
  Step 3: Continue...
""")

print("\n--- Beam Search Conceptual Example ---")

print("""
Imagine generating "The cat ___ ___":

Greedy:
  "The" → "cat" → "sat" → "on" = "The cat sat on"

Beam Search (beam=3):
  Step 1 candidates: ["The cat", "A cat", "The dog"]
  Step 2 candidates: ["The cat sat", "The cat ran", "A cat sat", 
                      "A cat slept", "The dog ran", "The dog barked"]
  Keep top 3: ["The cat sat", "A cat sat", "The cat ran"]
  Step 3: Continue from these 3...

Beam search can find better sequences because it explores
multiple paths simultaneously!
""")

# =============================================================================
# STEP 8: Complete Generation Function
# =============================================================================

print("\n" + "="*70)
print("STEP 8: Complete Generation Function")
print("="*70)

def generate_text(model, prompt_tokens, max_length=50, 
                  strategy="top_p", **kwargs):
    """
    Generate text using various strategies.
    
    Args:
        model: Model with forward() method
        prompt_tokens: Input token IDs
        max_length: Maximum generation length
        strategy: "greedy", "sample", "top_k", "top_p"
        **kwargs: Strategy-specific parameters (temperature, k, p)
    
    Returns:
        Generated token IDs
    """
    tokens = list(prompt_tokens)
    temperature = kwargs.get("temperature", 1.0)
    
    for _ in range(max_length):
        # Forward pass
        logits = model.forward(np.array(tokens))
        last_logits = logits[-1]
        
        # Apply temperature
        probs = softmax(last_logits, temperature=temperature)
        
        # Select next token based on strategy
        if strategy == "greedy":
            next_token = np.argmax(probs)
        elif strategy == "sample":
            next_token = sample(probs)
        elif strategy == "top_k":
            next_token = top_k_sampling(probs, k=kwargs.get("k", 40))
        elif strategy == "top_p":
            next_token = top_p_sampling(probs, p=kwargs.get("p", 0.9))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        tokens.append(next_token)
    
    return np.array(tokens)

print("\n--- Generation Strategy Comparison ---")

print("""
Let's compare how different strategies would generate text:

(Using simulated model outputs)

Prompt: "The quick brown fox"
""")

# Simulate different generation outputs
strategies = {
    "Greedy (T=1.0)": "jumps over the lazy dog. The cat sat on the mat.",
    "Sample (T=1.0)": "runs through the forest and sleeps under a tree.",
    "Top-k (k=40)": "jumps over the lazy dog and runs away quickly.",
    "Top-p (p=0.9)": "jumps over the lazy dog. Then it sleeps.",
    "Low Temp (T=0.5)": "jumps over the lazy dog. The end of story.",
    "High Temp (T=1.5)": "dances on clouds while singing to the moon!",
}

for strategy, output in strategies.items():
    print(f"\n{strategy}:")
    print(f"  The quick brown fox {output}")

# =============================================================================
# STEP 9: Summary and Recommendations
# =============================================================================

print("\n" + "="*70)
print("STEP 9: Strategy Recommendations")
print("="*70)

print("""
WHEN TO USE EACH STRATEGY:

1. GREEDY DECODING:
   - Factual Q&A
   - Code generation
   - Translation (with beam search)
   - When you need deterministic output

2. SAMPLING (with temperature):
   - Creative writing
   - Story generation
   - Brainstorming
   - When you want variety

3. TOP-K SAMPLING:
   - General purpose
   - Chat responses
   - Good default choice
   - Recommended: k=40-50

4. TOP-P SAMPLING:
   - High-quality creative text
   - When you want adaptive selection
   - Recommended: p=0.9-0.95

5. BEAM SEARCH:
   - Translation
   - Summarization
   - Tasks with clear evaluation metrics
   - Not recommended for chat

POPULAR COMBINATIONS:
- GPT-3 default: Top-p (p=0.9-0.95) + Temperature (T=0.7-1.0)
- Creative writing: Top-p (p=0.9) + Temperature (T=0.8-1.2)
- Code generation: Greedy or low temperature (T=0.2-0.5)
- Chat: Top-k (k=40) + Temperature (T=0.7-0.9)
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Text Generation Strategies")
print("="*70)

print("""
GENERATION PROCESS:
1. Tokenize prompt
2. Model predicts next token probabilities
3. Apply strategy to select next token
4. Append and repeat

STRATEGIES:
┌─────────────┬──────────────┬───────────────┬─────────────┐
│ Strategy    │ Pros         │ Cons          │ Best For    │
├─────────────┼──────────────┼───────────────┼─────────────┤
│ Greedy      │ Fast, simple │ Repetitive    │ Factual     │
│ Sample      │ Diverse      │ Incoherent    │ Creative    │
│ Top-k       │ Controlled   │ Fixed k       │ General     │
│ Top-p       │ Adaptive     │ Complex       │ Quality     │
│ Beam Search │ Optimal      │ Expensive     │ Translation │
└─────────────┴──────────────┴───────────────┴─────────────┘

PARAMETERS:
- Temperature: Controls randomness (0.5-2.0 typical)
- Top-k: Number of candidates (40-100 typical)
- Top-p: Cumulative probability (0.9-0.95 typical)
- Max length: Generation limit

NEXT: Build a complete working mini GPT model!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Generation")
print("="*70)

print("""
Try these:

1. Compare strategies:
   Run the same prompt with different strategies
   Which produces the best output?

2. Tune temperature:
   Try T = 0.2, 0.5, 0.8, 1.0, 1.5, 2.0
   How does output quality change?

3. Top-k comparison:
   Try k = 5, 10, 40, 100
   Find the sweet spot!

4. Top-p comparison:
   Try p = 0.5, 0.75, 0.9, 0.95, 0.99
   How does nucleus size affect output?

5. Combined:
   Try top-p + temperature together!

Key Takeaway:
- Generation strategy affects output quality significantly
- No single best strategy - depends on use case
- Temperature, top-k, and top-p are key parameters

Next: 09_mini_gpt.py - Complete working mini GPT!
=============================================================================""")
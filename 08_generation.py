"""
=============================================================================
LESSON 8: Text Generation - Making GPT Write
=============================================================================

Now we learn how to GENERATE text with our trained GPT!

REAL-WORLD ANALOGY: Creative Writing Class
==========================================

Imagine teaching a student to write stories:

1. GREEDY WRITING (Greedy Decoding)
   - Student always picks most obvious next word
   - "The cat ___" -> "sat" (most common)
   - Result: Boring but coherent

2. RANDOM WRITING (Pure Sampling)
   - Student picks any word randomly
   - "The cat ___" -> could be "purple" or "flew"
   - Result: Creative but nonsensical

3. TOP-K WRITING (Top-k Sampling)
   - Student picks from k best options
   - "The cat ___" -> picks from {sat, slept, ate, jumped}
   - Result: Balanced - sensible but varied

4. TEMPERATURE ADJUSTMENT (Temperature Scaling)
   - Cold (low temp): Very predictable
   - Hot (high temp): Very creative
   - Result: Controls creativity level

Let's learn each strategy!
"""

import numpy as np

# =============================================================================
# STEP 1: Greedy Decoding
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Greedy Decoding - Always Pick the Best")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Following GPS Directions
============================================

Greedy decoding is like always taking the fastest route:

AT EACH INTERSECTION:
- Route A: 5 minutes
- Route B: 8 minutes
- Route C: 12 minutes

GREEDY CHOICE: Always pick Route A (fastest now)

PROBLEM: Might miss scenic route or end up in traffic!

GREEDY DECODING FOR TEXT:
=========================

AT EACH STEP:
- "the": 45% probability
- "cat": 30% probability
- "dog": 15% probability

GREEDY CHOICE: Always pick "the"

PROBLEM: Text becomes repetitive and boring!

EXAMPLE:
Prompt: "The weather today is"
Greedy: "The weather today is the weather today is the..."

ADVANTAGES:
- Deterministic (same input = same output)
- Fast (no sampling needed)
- Coherent (picks likely words)

DISADVANTAGES:
- Repetitive (gets stuck in loops)
- Less creative (always safe choice)
- Can be boring (no surprises)
=============================================================================""")

def greedy_decode(probs):
    """
    Select the token with highest probability.
    
    REAL-WORLD EXAMPLE: Multiple Choice Test Strategy
    
    Student taking test:
    Question: "What is the capital of France?"
    Options: A) London  B) Paris  C) Berlin  D) Madrid
    
    GREEDY STRATEGY: Always pick highest probability
    - Student thinks: "Paris seems most likely"
    - Picks: B) Paris
    
    This works well when you're confident!
    
    Args:
        probs: Probability distribution over vocabulary
    
    Returns:
        token_id: Index of highest probability token
    """
    return np.argmax(probs)

print("\n--- Greedy Decoding Demo ---")
print("="*50)

# Simulated model predictions
vocab = ["the", "cat", "sat", "on", "mat", "slept", "ate", "jumped"]
probs = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03])

print(f"Vocabulary: {vocab}")
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum():.2f}")

# Greedy choice
chosen_idx = greedy_decode(probs)
print(f"\nGreedy choice: '{vocab[chosen_idx]}' (p={probs[chosen_idx]*100:.1f}%)")
print(f"-> Always picks the most likely option!")

# Simulate greedy generation
print("\n" + "-"*50)
print("Greedy Generation Example:")
print("-"*50)

# Simulated probabilities for next tokens
generations = [
    np.array([0.40, 0.30, 0.15, 0.15]),  # Step 1
    np.array([0.50, 0.25, 0.15, 0.10]),  # Step 2
    np.array([0.45, 0.35, 0.10, 0.10]),  # Step 3
    np.array([0.55, 0.20, 0.15, 0.10]),  # Step 4
]
words = [["The", "A", "My", "That"],
         ["cat", "dog", "bird", "fish"],
         ["sat", "slept", "ate", "ran"],
         ["on", "under", "near", "by"]]

greedy_output = []
for i, p in enumerate(generations):
    idx = greedy_decode(p)
    greedy_output.append(words[i][idx])
    print(f"Step {i+1}: Chose '{words[i][idx]}' (p={p[idx]*100:.1f}%)")

print(f"\nGreedy output: '{' '.join(greedy_output)}'")
print("-> Coherent but predictable!")

# =============================================================================
# STEP 2: Random Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Random Sampling - Embrace Creativity")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Choosing Lunch Randomly
===========================================

Pure sampling is like spinning a wheel to choose lunch:

LUNCH WHEEL:
- Pizza: 40% (big slice)
- Salad: 30% (medium slice)
- Burger: 20% (small slice)
- Soup: 10% (tiny slice)

SPIN THE WHEEL:
- Might get Pizza (most likely)
- Might get Soup (surprise!)

RANDOM SAMPLING FOR TEXT:
==========================

AT EACH STEP, sample from distribution:
- "the": 45% chance
- "cat": 30% chance
- "dog": 15% chance
- "bird": 10% chance

SAMPLING: Could pick any of them!

ADVANTAGES:
- Creative and diverse output
- Less repetitive than greedy
- Can produce surprising combinations

DISADVANTAGES:
- Can produce nonsense
- Less coherent
- Unpredictable quality

EXAMPLE:
Prompt: "The cat"
Sampling: "The cat purple elephant flying yesterday..."
=============================================================================""")

def sample_decode(probs):
    """
    Sample from the probability distribution.
    
    REAL-WORLD EXAMPLE: Spinning a Prize Wheel
    
    Wheel sections proportional to probabilities:
    - Section A: 40% of wheel
    - Section B: 30% of wheel
    - Section C: 20% of wheel
    - Section D: 10% of wheel
    
    SPIN: Random point on wheel determines winner!
    - 1000 spins: ~400 A, ~300 B, ~200 C, ~100 D
    
    This is exactly how sampling works!
    
    Args:
        probs: Probability distribution (sums to 1)
    
    Returns:
        token_id: Sampled token index
    """
    # Cumulative probabilities
    cumulative_probs = np.cumsum(probs)
    
    # Generate random number between 0 and 1
    r = np.random.random()
    
    # Find first cumulative prob > r
    for i, cum_prob in enumerate(cumulative_probs):
        if cum_prob > r:
            return i
    
    # Fallback (shouldn't happen if probs sum to 1)
    return len(probs) - 1

print("\n--- Random Sampling Demo ---")
print("="*50)

# Same probabilities as greedy demo
probs = np.array([0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.03])

print(f"Vocabulary: {vocab}")
print(f"Probabilities: {probs}")

# Sample multiple times to show distribution
print("\nSampling 20 times:")
samples = []
for i in range(20):
    idx = sample_decode(probs)
    samples.append(vocab[idx])

print(f"Samples: {samples}")

# Count frequencies
from collections import Counter
counts = Counter(samples)
print(f"\nFrequency count:")
for word in vocab:
    count = counts.get(word, 0)
    bar = "#" * count
    expected = probs[vocab.index(word)] * 20
    print(f"  {word}: {count:2d} times (expected ~{expected:.1f}) {bar}")

print("\n-> Higher probability words appear more often!")
print("-> But lower probability words still get picked sometimes!")

# =============================================================================
# STEP 3: Temperature Scaling
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Temperature Scaling - Control Creativity")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Weather and Mood
====================================

Temperature affects behavior:

COLD DAY (Low Temperature):
- People stay indoors
- Predictable behavior
- Same routines

HOT DAY (High Temperature):
- People more active
- Unpredictable behavior
- Try new things

TEMPERATURE IN GPT:
===================

LOW TEMPERATURE (T = 0.2):
- "Cold" - very confident
- Sharp probability distribution
- Almost always picks top choice
- Like greedy decoding

NORMAL TEMPERATURE (T = 1.0):
- Original distribution
- Balanced creativity/coherence

HIGH TEMPERATURE (T = 2.0):
- "Hot" - more random
- Flatter probability distribution
- More diverse word choices
- More creative but less coherent

MATHEMATICALLY:
- Divide logits by temperature before softmax
- Low T: Amplifies differences (confident)
- High T: Reduces differences (uncertain)
=============================================================================""")

def apply_temperature(logits, temperature):
    """
    Apply temperature scaling to logits.
    
    REAL-WORLD EXAMPLE: Adjusting Contrast on TV
    
    ORIGINAL IMAGE (T=1.0):
    - Normal contrast
    - Balanced colors
    
    LOW TEMPERATURE (T=0.5):
    - High contrast
    - Bright areas brighter, dark areas darker
    - More extreme differences
    
    HIGH TEMPERATURE (T=2.0):
    - Low contrast
    - Everything looks similar
    - Less extreme differences
    
    Args:
        logits: Raw model scores (before softmax)
        temperature: Temperature value
                   - <1.0: More confident/predictable
                   - =1.0: Normal
                   - >1.0: More creative/random
    
    Returns:
        scaled_logits: Temperature-adjusted logits
    """
    return logits / temperature

def softmax_with_temperature(logits, temperature):
    """Softmax with temperature scaling."""
    scaled_logits = apply_temperature(logits, temperature)
    return softmax(scaled_logits)

def softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("\n--- Temperature Scaling Demo ---")
print("="*50)

# Raw model outputs (logits)
logits = np.array([3.0, 2.0, 1.5, 1.0, 0.5])
words = ["the", "cat", "sat", "on", "mat"]

print(f"Original logits: {logits}")
print(f"Words: {words}")

temperatures = [0.2, 0.5, 1.0, 2.0, 5.0]

print("\nProbability distributions at different temperatures:")
print("-"*60)

for temp in temperatures:
    probs = softmax_with_temperature(logits, temp)
    
    # Find most likely word
    top_idx = np.argmax(probs)
    top_word = words[top_idx]
    top_prob = probs[top_idx]
    
    print(f"\nTemperature = {temp}:")
    print(f"  Top choice: '{top_word}' at {top_prob*100:.1f}%")
    
    # Visual bar chart
    for word, prob in zip(words, probs):
        bar = "#" * int(prob * 30)
        print(f"    {word}: {prob*100:5.1f}% {bar}")

print("\n" + "-"*50)
print("KEY OBSERVATIONS:")
print("  Low T (0.2): Almost all probability on top choice")
print("  Normal T (1.0): Balanced distribution")
print("  High T (5.0): Nearly uniform - very random!")

# =============================================================================
# STEP 4: Top-k Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Top-k Sampling - Best of Both Worlds")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Restaurant Menu Selection
=============================================

Top-k sampling is like choosing from a curated menu:

FULL MENU (All 100 dishes):
- Some dishes are amazing
- Some are mediocre
- Some you'd never order

TOP-K MENU (Best 5 dishes):
- Only the most popular items
- All good choices
- Still variety within quality

TOP-K SAMPLING:
===============

STEP 1: Rank all words by probability
STEP 2: Keep only top k words
STEP 3: Renormalize probabilities (must sum to 1)
STEP 4: Sample from these k words

EXAMPLE (k=3):
Original: the(40%), cat(25%), sat(15%), on(10%), mat(5%), ...
Top-3: the, cat, sat
Renormalized: the(50%), cat(31%), sat(19%)
Sample from these three!

ADVANTAGES:
- Avoids very unlikely words
- More coherent than pure sampling
- More creative than greedy
- Prevents weird word choices

DISADVANTAGES:
- Fixed k might be too restrictive or too loose
- Might cut off good words just below threshold
=============================================================================""")

def topk_decode(probs, k):
    """
    Sample from top-k tokens.
    
    REAL-WORLD EXAMPLE: Award Ceremony
    
    NOMINATIONS (All tokens):
    - 100 nominees for Best Actor
    
    TOP-K (Finalists):
    - Keep only top 5 nominees
    - Renormalize votes among these 5
    
    WINNER: Chosen from top 5 only!
    
    Args:
        probs: Probability distribution
        k: Number of top tokens to keep
    
    Returns:
        token_id: Sampled token from top-k
    """
    # Get indices of top-k tokens
    top_k_indices = np.argsort(probs)[-k:]
    
    # Get probabilities of top-k tokens
    top_k_probs = probs[top_k_indices]
    
    # Renormalize (must sum to 1)
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    # Sample from top-k
    chosen_idx = sample_decode(top_k_probs)
    
    # Map back to original index
    return top_k_indices[chosen_idx]

print("\n--- Top-k Sampling Demo ---")
print("="*50)

# Probability distribution
probs = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
words = ["the", "cat", "sat", "on", "mat", "slept", "ate", "jumped", "ran", "hid"]

print(f"Full vocabulary: {words}")
print(f"Original probs: {probs}")
print(f"Sum: {probs.sum():.2f}")

for k in [1, 3, 5]:
    print(f"\n" + "-"*40)
    print(f"TOP-{k} SAMPLING:")
    print("-"*40)
    
    # Show top-k words
    top_k_indices = np.argsort(probs)[-k:]
    top_k_words = [words[i] for i in top_k_indices]
    top_k_probs = probs[top_k_indices]
    renorm_probs = top_k_probs / top_k_probs.sum()
    
    print(f"Top-{k} words: {top_k_words}")
    print(f"Renormalized probs: {renorm_probs}")
    print(f"Sum: {renorm_probs.sum():.2f}")
    
    # Sample several times
    samples = []
    for _ in range(20):
        idx = topk_decode(probs, k)
        samples.append(words[idx])
    
    counts = Counter(samples)
    print(f"Samples (20x): {samples}")

# =============================================================================
# STEP 5: Top-p (Nucleus) Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Top-p (Nucleus) Sampling - Dynamic Top-k")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Budget Shopping
===================================

Top-p sampling is like shopping with a budget:

SHOPPING LIST (All items ranked by desire):
- Laptop: $800 (must have!)
- Phone: $600 (really want)
- Headphones: $200 (nice to have)
- Mouse: $50 (optional)
- Keyboard: $40 (optional)

BUDGET (p = 0.95 = 95% of desire):
- Keep buying until we've covered 95% of desire
- Stop when cumulative "desire" reaches threshold

TOP-P SAMPLING:
===============

STEP 1: Sort words by probability (descending)
STEP 2: Compute cumulative probability
STEP 3: Keep adding words until cumulative > p
STEP 4: Renormalize and sample

EXAMPLE (p=0.9):
Original: the(40%), cat(25%), sat(15%), on(10%), mat(5%), ...
Cumulative: the(40%), cat(65%), sat(80%), on(90%), mat(95%), ...
Keep until 90%: the, cat, sat, on
Sample from these!

ADVANTAGE over Top-k:
- Adapts to distribution shape
- More tokens when uncertain
- Fewer tokens when confident
=============================================================================""")

def topp_decode(probs, p):
    """
    Sample from top-p (nucleus) tokens.
    
    REAL-WORLD EXAMPLE: Covering 90% of Customer Requests
    
    CUSTOMER REQUESTS (sorted by frequency):
    - "Password reset": 35%
    - "Login issue": 25%
    - "Account locked": 15%
    - "Forgot username": 10%
    - "Billing question": 8%
    - "Other": 7%
    
    COVER 90% of requests:
    - Keep categories until cumulative >= 90%
    - Result: Password, Login, Account, Forgot, Billing (93%)
    - Sample from these for training focus!
    
    Args:
        probs: Probability distribution
        p: Cumulative probability threshold (0 to 1)
    
    Returns:
        token_id: Sampled token from top-p
    """
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probability
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find cutoff where cumulative > p
    cutoff = np.searchsorted(cumulative_probs, p)
    
    # Keep tokens up to cutoff
    top_p_indices = sorted_indices[:cutoff + 1]
    top_p_probs = probs[top_p_indices]
    
    # Renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    # Sample
    chosen_idx = sample_decode(top_p_probs)
    
    return top_p_indices[chosen_idx]

print("\n--- Top-p Sampling Demo ---")
print("="*50)

# Same distribution
probs = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
words = ["the", "cat", "sat", "on", "mat", "slept", "ate", "jumped", "ran", "hid"]

print(f"Original probs: {probs}")

# Show cumulative
sorted_indices = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_indices]
cumulative = np.cumsum(sorted_probs)

print("\nSorted (descending) with cumulative:")
for i, idx in enumerate(sorted_indices):
    print(f"  {words[idx]}: {probs[idx]*100:5.1f}% (cumulative: {cumulative[i]*100:.1f}%)")

for p in [0.5, 0.8, 0.95]:
    print(f"\n" + "-"*40)
    print(f"TOP-p (p={p}) SAMPLING:")
    print("-"*40)
    
    # Find which tokens are kept
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p)
    
    kept_indices = sorted_indices[:cutoff + 1]
    kept_words = [words[i] for i in kept_indices]
    kept_probs = probs[kept_indices]
    renorm = kept_probs / kept_probs.sum()
    
    print(f"Kept words: {kept_words}")
    print(f"Coverage: {cumulative[cutoff]*100:.1f}%")
    print(f"Renormalized: {renorm}")
    
    # Sample several times
    samples = []
    for _ in range(20):
        idx = topp_decode(probs, p)
        samples.append(words[idx])
    
    print(f"Samples (20x): {samples}")

# =============================================================================
# STEP 6: Complete Generation Example
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete Generation Example")
print("="*70)

print("""
Let's generate text using different strategies!

PROMPT: "Once upon a time"

We'll use a simulated model to generate continuations.
""")

class TextGenerator:
    """Simple text generator with multiple decoding strategies."""
    
    def __init__(self, vocab, word_to_idx, idx_to_word):
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = len(vocab)
    
    def get_simulated_probs(self, context, seed=42):
        """
        Simulate model predictions based on context.
        
        This is a DEMO - real model would use neural network!
        """
        np.random.seed(seed + len(context))
        
        # Create context-dependent probabilities
        base_probs = np.random.dirichlet(np.ones(self.vocab_size))
        
        # Add some structure based on context
        if len(context) > 0:
            last_word_idx = context[-1]
            base_probs[last_word_idx] *= 0.5  # Less likely to repeat
        
        # Normalize
        probs = base_probs / base_probs.sum()
        
        return probs
    
    def generate(self, prompt_tokens, max_length, strategy='greedy', 
                 temperature=1.0, top_k=None, top_p=None):
        """
        Generate text with specified strategy.
        
        Args:
            prompt_tokens: Starting token IDs
            max_length: Maximum tokens to generate
            strategy: 'greedy', 'sample', 'topk', 'topp'
            temperature: Temperature scaling
            top_k: For top-k sampling
            top_p: For top-p sampling
        
        Returns:
            generated_tokens: Full sequence including prompt
        """
        tokens = list(prompt_tokens)
        
        for _ in range(max_length):
            # Get model predictions
            logits = np.random.randn(self.vocab_size)  # Simulated
            probs = softmax(logits / temperature)
            
            # Apply decoding strategy
            if strategy == 'greedy':
                next_token = greedy_decode(probs)
            elif strategy == 'sample':
                next_token = sample_decode(probs)
            elif strategy == 'topk' and top_k is not None:
                next_token = topk_decode(probs, top_k)
            elif strategy == 'topp' and top_p is not None:
                next_token = topp_decode(probs, top_p)
            else:
                next_token = sample_decode(probs)
            
            tokens.append(next_token)
        
        return tokens

# Create vocabulary for demo
demo_vocab = 100  # Small vocab for demo
idx_to_word = {i: f"word{i}" for i in range(demo_vocab)}
word_to_idx = {v: k for k, v in idx_to_word.items()}

# Add some common words for readability
common_words = ["the", "cat", "sat", "on", "mat", "once", "upon", "time", "there", "was"]
for i, word in enumerate(common_words):
    idx_to_word[i] = word
    word_to_idx[word] = i

# Create generator
generator = TextGenerator(demo_vocab, word_to_idx, idx_to_word)

# Prompt: "once upon a time" (using indices 5, 6, 7)
prompt = [5, 6, 7]

print("\n" + "="*50)
print("GENERATION COMPARISON")
print("="*50)

strategies = [
    ('greedy', {'temperature': 0.5}),
    ('sample', {'temperature': 1.0}),
    ('topk', {'top_k': 10, 'temperature': 1.0}),
    ('topp', {'top_p': 0.9, 'temperature': 1.0}),
]

for strategy, kwargs in strategies:
    tokens = generator.generate(prompt, max_length=10, strategy=strategy, **kwargs)
    words = [idx_to_word[t] for t in tokens]
    print(f"\n{strategy.upper()}:")
    print(f"  {' '.join(words)}")

# =============================================================================
# SUMMARY: Text Generation Strategies
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Text Generation Strategies")
print("="*70)

print("""
WHAT WE LEARNED:
================
1. Greedy Decoding - Always pick most likely
2. Random Sampling - Sample from full distribution
3. Temperature - Control creativity level
4. Top-k Sampling - Sample from k best options
5. Top-p Sampling - Sample until probability threshold

STRATEGY COMPARISON:
====================

+-------------+-------------+-------------+-------------+
| Strategy    | Coherence   | Creativity  | Diversity   |
+-------------+-------------+-------------+-------------+
| Greedy      | High        | Low         | Low         |
| Sample      | Medium      | High        | High        |
| Top-k       | High        | Medium      | Medium      |
| Top-p       | High        | Medium      | Medium      |
+-------------+-------------+-------------+-------------+

BEST PRACTICES:
===============

FOR FACTUAL CONTENT:
- Use greedy or low temperature (0.2-0.5)
- Accuracy over creativity

FOR CREATIVE WRITING:
- Use top-k or top-p with temperature 0.8-1.2
- Balance coherence and creativity

FOR BRAINSTORMING:
- Use high temperature (1.5-2.0)
- Embrace randomness

FOR CONVERSATION:
- Use top-p (0.9) with temperature 0.7-1.0
- Natural and varied responses

RECOMMENDED DEFAULT:
- Top-p (p=0.9) with temperature 1.0
- Good balance for most tasks!

NEXT: Complete Mini GPT
=======================
Now we have all components!
Next, we combine everything into a working Mini GPT:
- Full model architecture
- Training loop
- Text generation
- Train on sample data

Next: 09_mini_gpt.py
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Experiment with Generation")
print("="*70)

print("""
Try these experiments:

1. CHANGE TEMPERATURE:
   generate(prompt, temperature=0.1)  # Very focused
   generate(prompt, temperature=2.0)  # Very creative
   
   Question: How does temperature affect output?
   Answer: Low = repetitive, High = random

2. CHANGE TOP-K:
   generate(prompt, strategy='topk', top_k=5)   # Very focused
   generate(prompt, strategy='topk', top_k=50)  # More diverse
   
   Question: How does k affect quality?
   Answer: Small k = safer, Large k = more varied

3. CHANGE TOP-P:
   generate(prompt, strategy='topp', top_p=0.5)  # Very focused
   generate(prompt, strategy='topp', top_p=0.95) # More diverse
   
   Question: How does p affect output?
   Answer: Low p = few options, High p = many options

4. COMBINE STRATEGIES:
   generate(prompt, temperature=0.8, top_p=0.9)
   
   Question: What's the best combination?
   Answer: Depends on task - experiment!

KEY TAKEAWAY:
=============
Generation strategy controls output style!
- Greedy: Safe, coherent, boring
- Sampling: Creative, diverse, sometimes nonsense
- Top-k/Top-p: Best balance
- Temperature: Fine-tune creativity

Choose strategy based on your needs!
=============================================================================""")
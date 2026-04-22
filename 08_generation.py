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
REAL-WORLD EXAMPLE: Choose Your Own Adventure Book
===================================================

GPT generation is like reading a "Choose Your Own Adventure" book:

BOOK PAGE (Current Context):
  "You enter a dark cave. You see two paths:
   Path A goes left into darkness.
   Path B goes right toward light."

YOUR CHOICE (Token Selection):
  - Turn to page 45 (go left)
  - Turn to page 72 (go right)

GPT DOES THE SAME THING:

CURRENT TEXT: "The cat sat on the"
MODEL PREDICTIONS:
  - "mat" (page 45) - 45% chance
  - "couch" (page 72) - 25% chance
  - "floor" (page 23) - 15% chance
  - "bed" (page 89) - 10% chance
  - Other pages - 5% chance

DECISION TIME:
  - Greedy: Always pick page 45 ("mat")
  - Sampling: Roll dice, might pick any page
  - Top-k: Only consider pages 45, 72, 23, 89
  - Top-p: Consider pages that make up 90% probability

NEXT PAGE (New Context):
  "The cat sat on the mat."
  → Now predict the NEXT word!
  → Continue until story ends

AUTOREGRESSIVE = Building One Word at a Time
=============================================

Like building a tower with blocks:
1. Place first block (prompt)
2. Predict where next block goes
3. Place next block
4. Repeat until tower is complete

Each block placement affects where future blocks can go!

STOP CONDITIONS:
================
1. MAX LENGTH: "I've written 100 words, time to stop"
2. END TOKEN: "I generated [END], story complete!"
3. STOP STRING: "I see '\n\n', that means done"
=============================================================================""")

def softmax(logits, temperature=1.0):
    """
    Numerically stable softmax with temperature.
    
    REAL-WORLD EXAMPLE: Converting Scores to Percentages
    =====================================================
    
    Imagine converting test scores to class rankings:
    
    RAW SCORES (logits): [85, 72, 91, 68, 77]
    
    SOFTMAX converts to percentages:
    - Student 1: 23% (highest scorer)
    - Student 2: 15%
    - Student 3: 31% (best!)
    - Student 4: 12%
    - Student 5: 19%
    
    All percentages add to 100% (probability distribution)
    
    TEMPERATURE adjusts confidence:
    - Low temp: "Student 3 is THE BEST at 60%!"
    - High temp: "Everyone has a fair chance!"
    
    Args:
        logits: Raw prediction scores
        temperature: Controls confidence (lower = more confident)
    
    Returns:
        Probability distribution (sums to 1.0)
    """
    logits = logits / temperature
    logits_max = np.max(logits)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits)

print("\n" + "-"*70)
print("Let's simulate a model's predictions!")
print("-"*70)
print("""
SCENARIO: Model predicting next word after "The cat"

We'll simulate what the model "thinks" should come next
""")

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
# Make some tokens more likely (model "prefers" these)
logits[12] += 3  # "sat" - most likely
logits[13] += 2  # "on" - pretty likely
logits[14] += 1  # "mat" - somewhat likely
logits[16] += 2.5  # "slept" - very likely

print(f"\n📊 Model Predictions:")
print(f"  Vocabulary size: {vocab_size:,} possible words")
print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

# Get probabilities
probs = softmax(logits, temperature=1.0)
print(f"  Probability range: [{probs.min():.6f}, {probs.max():.6f}]")

# Show top tokens
top_indices = np.argsort(probs)[-10:][::-1]
print(f"\n🏆 Top 10 most likely next words:")
for i, idx in enumerate(top_indices):
    confidence = "⭐⭐⭐" if probs[idx] > 0.1 else "⭐⭐" if probs[idx] > 0.05 else "⭐"
    print(f"  {i+1}. {tokens[idx]:12} (p={probs[idx]*100:.2f}%) {confidence}")

# =============================================================================
# STEP 2: Greedy Decoding
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Greedy Decoding - Always Pick the Best")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Always Ordering the Most Popular Dish
==========================================================

Imagine you're at a restaurant and ALWAYS order the most popular item:

GREEDY CUSTOMER:
  Waiter: "What would you like?"
  Menu probabilities: 
    - Pasta: 35% (most popular)
    - Pizza: 25%
    - Salad: 20%
    - Soup: 15%
    - Steak: 5%
  
  Greedy Customer: "I'll have the Pasta!" (always)

EVERY VISIT: Same order, no variety!

PROS OF GREEDY:
✅ Consistent - you know what you're getting
✅ Safe - always getting the "best" option
✅ Fast - no deliberation needed

CONS OF GREEDY:
❌ Boring - same thing every time
❌ Missing out - never try new things
❌ Can get stuck - pasta, pasta, pasta...

GREEDY DECODING IN GPT:
======================

Prompt: "The cat"
Model predicts: P("sat")=45%, P("slept")=25%, P("ran")=10%...
Greedy picks: "sat" (highest probability)

Result: "The cat sat"

Next prediction: P("on")=50%, P("and")=20%...
Greedy picks: "on"

Result: "The cat sat on"

GREEDY IS DETERMINISTIC:
- Same prompt = same output every time
- Good for facts, code, math
- Bad for creativity, stories, chat
=============================================================================""")

def greedy_decode(probs):
    """
    Select the token with highest probability.
    
    REAL-WORLD EXAMPLE: Valedictorian Selection
    ===========================================
    
    In a class of 1000 students, who has the highest GPA?
    
    GPAs (probs): [3.2, 3.8, 4.0, 3.5, ...]
                  ↑
              Student 3 wins!
    
    Greedy decode = argmax(probs)
    = "Pick the champion"
    
    Args:
        probs: Probability distribution
    
    Returns:
        Index of highest probability token
    """
    return np.argmax(probs)

print("\n--- Greedy Decoding Example ---")
print("="*50)
print("""
SCENARIO: Picking the single most likely word

Like choosing the championship winner
""")

next_token_idx = greedy_decode(probs)
print(f"🎯 Greedy selection: '{tokens[next_token_idx]}'")
print(f"   Probability: {probs[next_token_idx]*100:.4f}%")
print(f"   → This word will ALWAYS be picked (deterministic)")

# =============================================================================
# STEP 3: Random Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Random Sampling - Roll the Dice")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Spinning a Prize Wheel
==========================================

Imagine a prize wheel where each slice size = probability:

PRIZE WHEEL (Vocabulary):
  ┌─────────────────────────────────┐
  │  🎁 PASTA (35%) - Biggest slice │
  │  🍕 PIZZA (25%) - Large slice   │
  │  🥗 SALAD (20%) - Medium slice  │
  │  🍲 SOUP (15%) - Small slice    │
  │  🥩 STEAK (5%) - Tiny slice     │
  └─────────────────────────────────┘

SPIN THE WHEEL (Sample):
  - Pasta has biggest slice (35% chance)
  - But ANY prize could win!
  - Even steak has 5% chance!

SAMPLING IN GPT:
===============

Prompt: "The cat"
Model wheel:
  - "sat" (45% slice)
  - "slept" (25% slice)
  - "ran" (10% slice)
  - Others (20% combined)

Spin 1: "slept" wins! (not the most likely, but possible)
Spin 2: "sat" wins! (most likely wins this time)
Spin 3: "ran" wins! (surprise!)

PROS OF SAMPLING:
✅ Creative - unexpected combinations
✅ Diverse - different outputs each time
✅ Natural - humans aren't deterministic either

CONS OF SAMPLING:
❌ Can be incoherent - might pick weird words
❌ Low probability tokens can win - "The cat quantum"
❌ Unpredictable - can't reproduce outputs

SAMPLING = np.random.choice()
- Python rolls the dice for you
- Each spin is independent
- Over many spins, distribution matches probabilities
=============================================================================""")

def sample(probs):
    """
    Sample from probability distribution.
    
    REAL-WORLD EXAMPLE: Lottery Drawing
    ====================================
    
    Imagine a lottery with 1000 tickets:
    - Ticket #12 has 450 tickets (45% chance)
    - Ticket #16 has 250 tickets (25% chance)
    - Others share remaining 300 tickets
    
    Drawing: Pick one ticket at random
    - More tickets = more likely to win
    - But ANY ticket could win!
    
    Args:
        probs: Probability distribution (must sum to 1.0)
    
    Returns:
        Selected token index
    """
    return np.random.choice(len(probs), p=probs)

print("\n--- Random Sampling Example ---")
print("="*50)
print("""
SCENARIO: Spinning the word wheel 10 times

Each spin is independent - anything can happen!
""")

print("🎡 Sampling 10 times from the distribution:")
samples = []
for i in range(10):
    token_idx = sample(probs)
    samples.append(tokens[token_idx])
    rarity = "🏆 Common" if probs[token_idx] > 0.1 else "⭐ Uncommon" if probs[token_idx] > 0.05 else "💎 Rare"
    print(f"  {i+1}. '{tokens[token_idx]}' (p={probs[token_idx]*100:.4f}%) {rarity}")

print(f"\n📊 Sampled words: {samples}")
unique_samples = len(set(samples))
print(f"  → {unique_samples} unique words out of 10 samples")
print(f"  → Notice: Different tokens each time!")

# =============================================================================
# STEP 4: Temperature Scaling
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Temperature Scaling - Control the Randomness")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Thermostat for Creativity
=============================================

Temperature is like a thermostat that controls creativity:

🥶 COLD (T=0.1) - "Follow the Recipe Exactly"
   - Chef is VERY confident
   - "This is THE way to make it"
   - Distribution is PEAKY (one clear winner)
   - Good for: Facts, code, math
   
   "The capital of France is ___"
   → "Paris" (99.9% confidence)

😐 WARM (T=1.0) - "Cook Naturally"
   - Chef is normally confident
   - Distribution is BALANCED
   - Good for: General conversation
   
   "Once upon a time"
   → "there" (40%), "in" (25%), "a" (20%)...

🔥 HOT (T=2.0) - "Experimental Fusion Cuisine"
   - Chef is feeling adventurous
   - "Let's try something new!"
   - Distribution is FLAT (everything's possible)
   - Good for: Poetry, brainstorming
   
   "The moonlight danced"
   → "gracefully" (15%), "purple" (12%), "quantum" (8%)...

MATHEMATICALLY:
===============

Temperature DIVIDES the logits before softmax:

  probs = softmax(logits / temperature)

LOW TEMP (0.5):
  logits = [2, 1, 0.5]
  logits / 0.5 = [4, 2, 1]  ← Differences amplified!
  probs = [0.73, 0.20, 0.07]  ← More confident!

HIGH TEMP (2.0):
  logits = [2, 1, 0.5]
  logits / 2.0 = [1, 0.5, 0.25]  ← Differences reduced!
  probs = [0.43, 0.33, 0.24]  ← More uniform!

EXTREME CASES:
==============

T → 0: GREEDY (only the top token matters)
T = 1: NORMAL (original distribution)
T → ∞: UNIFORM (all tokens equally likely)
=============================================================================""")

print("\n--- Temperature Effect Demonstration ---")
print("="*50)
print("""
SCENARIO: Same model, different temperature settings

Watch how the probability distribution changes!
""")

# Get logits for top 5 tokens
top_logits = logits[top_indices[:5]]
top_tokens = [tokens[i] for i in top_indices[:5]]

print(f"\n🌡️ Comparing probabilities at different temperatures:")
print(f"   Top 5 tokens: {top_tokens}")

print("\n📊 Probability Distribution by Temperature:")
print("="*70)

for temp in [0.2, 0.5, 1.0, 1.5, 2.0]:
    probs_temp = softmax(top_logits, temperature=temp)
    
    # Create visual bar
    emoji = "🥶" if temp < 0.5 else "😐" if temp < 1.2 else "🔥"
    print(f"\n{emoji} Temperature = {temp}:")
    
    for token, prob in zip(top_tokens, probs_temp):
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"   {token:<12}: {prob:6.2%} {bar}")

print("\n" + "-"*70)
print("KEY OBSERVATION:")
print("-"*70)
print("""
🥶 COLD (T=0.2):
   → One token dominates (peaky)
   → Almost greedy behavior
   → "I'm VERY sure about this"

😐 NORMAL (T=1.0):
   → Natural distribution
   → Top tokens favored, others possible
   → "Here's my best guess"

🔥 HOT (T=2.0):
   → Flatter distribution
   → More tokens are viable
   → "Let's keep our options open"
=============================================================================""")

# =============================================================================
# STEP 5: Top-k Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Top-k Sampling - Limit Your Options")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Netflix Top 10 List
=======================================

Imagine Netflix only shows you the TOP 10 movies:

WITHOUT TOP-K (Full Sampling):
  - 50,000 movies to choose from
  - Might accidentally pick "Room 731" (1993 Polish documentary)
  - Overwhelming!

WITH TOP-K (k=10):
  - Only 10 movies to consider
  - All are popular/watched
  - Much easier decision!

TOP-K SAMPLING WORKS THE SAME WAY:

STEP 1: Get all probabilities
  ["sat": 45%, "slept": 25%, "ran": 10%, "ate": 5%, ... "xyz": 0.001%]

STEP 2: Keep only top k
  k=3: ["sat": 45%, "slept": 25%, "ran": 10%]
  Rest are discarded!

STEP 3: Renormalize (make them sum to 100%)
  Total: 45% + 25% + 10% = 80%
  New: ["sat": 56%, "slept": 31%, "ran": 13%]

STEP 4: Sample from these k tokens
  - Still random, but controlled
  - Can't pick weird low-probability tokens

WHY TOP-K HELPS:
================

✅ Filters out garbage tokens
✅ More coherent than pure sampling
✅ Still creative (multiple options)
✅ Simple to understand

❌ Fixed k doesn't adapt
   - When model is confident: k=40 might include bad tokens
   - When model is uncertain: k=40 might exclude good tokens

TYPICAL VALUES:
===============
- k=1: Greedy decoding
- k=5-10: Focused, constrained
- k=40-50: Good balance (GPT-3 default range)
- k=100+: Very diverse, risk of incoherence
=============================================================================""")

def top_k_sampling(probs, k=40):
    """
    Sample from top k tokens.
    
    REAL-WORLD EXAMPLE: Job Interview Shortlist
    ===========================================
    
    HR receives 1000 applications (vocab_size):
    
    STEP 1: Rank by qualifications
    STEP 2: Shortlist top k=10 candidates
    STEP 3: Renormalize (all 10 are now "viable")
    STEP 4: Make final selection from shortlist
    
    Result: Can't hire the unqualified candidate
            (but also might not hire THE BEST)
    
    Args:
        probs: Probability distribution
        k: Number of top tokens to consider
    
    Returns:
        Selected token index from top k
    """
    # Get top k indices
    top_k_indices = np.argsort(probs)[-k:][::-1]
    top_k_probs = probs[top_k_indices]
    
    # Renormalize (make them sum to 1.0)
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    # Sample from shortlist
    sampled_idx = np.random.choice(k, p=top_k_probs)
    return top_k_indices[sampled_idx]

print("\n--- Top-k Sampling Example ---")
print("="*50)
print("""
SCENARIO: Only considering the top 5 words

Like choosing from a curated shortlist
""")

k = 5
print(f"📋 Sampling from top {k} tokens:")

top_k_indices = np.argsort(probs)[-k:][::-1]
top_k_probs = probs[top_k_indices]
top_k_probs_norm = top_k_probs / top_k_probs.sum()

print(f"\n🏆 Top {k} tokens (renormalized):")
for i, idx in enumerate(top_k_indices):
    bar = "█" * int(top_k_probs_norm[i] * 50)
    original_pct = probs[idx] * 100
    new_pct = top_k_probs_norm[i] * 100
    print(f"   {tokens[idx]:<12}: {new_pct:5.1f}% (was {original_pct:.1f}%) {bar}")

print(f"\n🎲 Sampling 5 times with k={k}:")
for i in range(5):
    token_idx = top_k_sampling(probs, k=k)
    print(f"   {i+1}. '{tokens[token_idx]}'")

print(f"\n   → All samples are from the top {k}!")
print(f"   → Can never pick weird low-probability tokens")

# =============================================================================
# STEP 6: Top-p (Nucleus) Sampling
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Top-p (Nucleus) Sampling - Dynamic Options")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Budget-Based Shopping
=========================================

Imagine shopping with a $100 budget:

TOP-K APPROACH (Fixed number of items):
  "I'll buy exactly 5 items"
  → Might buy 5 cheap items ($20 total) - wasteful!
  → Might buy 5 expensive items ($500) - over budget!

TOP-P APPROACH (Budget-based):
  "I'll buy items until I reach $100"
  → Adapts to prices!
  → Expensive items? Buy fewer.
  → Cheap items? Buy more.

TOP-P SAMPLING WORKS THE SAME WAY:

BUDGET = Cumulative Probability (e.g., p=0.9 = 90%)

STEP 1: Sort by probability (descending)
  ["sat": 45%, "slept": 25%, "ran": 10%, "ate": 8%, "left": 5%, "other": 7%]

STEP 2: Add up probabilities until we hit budget
  45% = 45% (keep going, under 90%)
  45% + 25% = 70% (keep going)
  70% + 10% = 80% (keep going)
  80% + 8% = 88% (keep going)
  88% + 5% = 93% ✓ STOP! (exceeded 90%)

STEP 3: Nucleus = ["sat", "slept", "ran", "ate", "left"]
  5 tokens that make up 93% of probability

STEP 4: Renormalize and sample

WHY TOP-P IS SMART:
===================

When model is CONFIDENT:
  ["sat": 80%, "slept": 15%, "ran": 5%]
  → Only 2-3 tokens needed to reach 90%
  → Small nucleus, focused sampling

When model is UNCERTAIN:
  ["sat": 20%, "slept": 18%, "ran": 15%, ... many tokens ...]
  → Need 20+ tokens to reach 90%
  → Large nucleus, diverse sampling

TOP-P ADAPTS TO UNCERTAINTY!

TYPICAL VALUES:
===============
- p=0.75: Very focused (75% of probability mass)
- p=0.9: Good balance (GPT-3 default)
- p=0.95: More diverse
- p=0.99: Almost full sampling
=============================================================================""")

def top_p_sampling(probs, p=0.9):
    """
    Sample from smallest set of tokens whose cumulative probability >= p.
    
    REAL-WORLD EXAMPLE: Pizza Budget
    =================================
    
    You have a "90% probability budget":
    
    PIZZA SLICES (tokens) sorted by deliciousness (prob):
    - Pepperoni: 35%
    - Cheese: 25%
    - Veggie: 15%
    - Hawaiian: 10%
    - BBQ: 8%
    - Anchovy: 4%
    - Spinach: 3%
    
    BUDGET TRACKING:
    - After Pepperoni: 35% (keep eating!)
    - After Cheese: 60% (keep eating!)
    - After Veggie: 75% (keep eating!)
    - After Hawaiian: 85% (keep eating!)
    - After BBQ: 93% ✓ STOP! (exceeded 90%)
    
    NUCLEUS = [Pepperoni, Cheese, Veggie, Hawaiian, BBQ]
    → These 5 slices make up 93% of deliciousness
    → Sample from these!
    
    Args:
        probs: Probability distribution
        p: Cumulative probability threshold (0.9 = 90%)
    
    Returns:
        Selected token index from nucleus
    """
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Cumulative probability (running total)
    cumsum = np.cumsum(sorted_probs)
    
    # Find where we exceed the budget
    cutoff_index = np.searchsorted(cumsum, p)
    
    # Get nucleus tokens (those within budget)
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_probs = sorted_probs[:cutoff_index + 1]
    
    # Renormalize
    top_p_probs = top_p_probs / top_p_probs.sum()
    
    # Sample from nucleus
    sampled_idx = np.random.choice(len(top_p_indices), p=top_p_probs)
    return top_p_indices[sampled_idx]

print("\n--- Top-p Sampling Example ---")
print("="*50)
print("""
SCENARIO: Sampling from tokens that make up 90% probability

The nucleus grows/shrinks based on model confidence!
""")

p = 0.9
print(f"🎯 Sampling from tokens that make up top {p*100:.0f}% probability:")

# Show cumulative probability
sorted_indices = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_indices]
cumsum = np.cumsum(sorted_probs)

print(f"\n📊 Top tokens with cumulative probability:")
nucleus_end = 0
for i, idx in enumerate(sorted_indices[:15]):
    cumulative = cumsum[i]
    in_nucleus = cumulative <= p
    if in_nucleus:
        nucleus_end = i + 1
    status = "✅ NUCLEUS" if in_nucleus else "⛔ EXCLUDED"
    print(f"   {tokens[idx]:<12}: p={probs[idx]*100:5.1f}%, cumsum={cumulative*100:5.1f}% {status}")

print(f"\n🌰 Nucleus size: {nucleus_end} tokens")
print(f"   → These {nucleus_end} tokens contain {p*100:.0f}% of probability")

print(f"\n🎲 Sampling 5 times with p={p}:")
for i in range(5):
    token_idx = top_p_sampling(probs, p=p)
    print(f"   {i+1}. '{tokens[token_idx]}' (p={probs[token_idx]*100:.2f}%)")

# =============================================================================
# STEP 7: Beam Search
# =============================================================================

print("\n" + "="*70)
print("STEP 7: Beam Search - Explore Multiple Paths")
print("="*70)

print("""
REAL-WORLD EXAMPLE: GPS Route Planning
======================================

Imagine GPS finding the best route:

GREEDY APPROACH:
  - At each intersection, pick the road that looks best
  - Might miss better routes!
  - Fast but potentially suboptimal

BEAM SEARCH (beam=3):
  - Track 3 best routes simultaneously
  - At each intersection, expand all 3
  - Keep the 3 best new routes
  - Continue until destination

VISUALIZATION:
==============

Step 1: Generate initial candidates
  ["The cat", "A cat", "The dog"]  ← beam=3

Step 2: Expand each candidate
  From "The cat": ["The cat sat", "The cat ran", "The cat slept"]
  From "A cat": ["A cat sat", "A cat ran", "A cat slept"]
  From "The dog": ["The dog ran", "The dog barked", "The dog ate"]
  
  All 9 candidates scored!
  Keep top 3: ["The cat sat", "A cat sat", "The cat ran"]

Step 3: Continue from these 3...

PROS OF BEAM SEARCH:
====================
✅ Finds globally optimal sequences
✅ Better than greedy for translation
✅ Explores multiple possibilities

CONS OF BEAM SEARCH:
====================
❌ Computationally expensive (kx more work)
❌ Can produce generic/bland text
❌ Not great for open-ended generation

BEAM WIDTH TRADEOFF:
====================
- beam=1: Greedy decoding (fast, myopic)
- beam=3-5: Good balance
- beam=10-20: Near-optimal but expensive
- beam=50+: Usually overkill

WHEN TO USE:
============
✅ Translation (clear "correct" answer)
✅ Summarization (optimize for ROUGE score)
✅ Tasks with evaluation metrics

❌ Creative writing (too rigid)
❌ Chat (produces generic responses)
❌ Open-ended generation
=============================================================================""")

print("\n--- Beam Search Conceptual Example ---")
print("="*50)
print("""
SCENARIO: Generating "The cat ___ ___" with beam=3

Watch how beam search explores multiple paths!
""")

print("""
📍 START: Empty sequence

STEP 1: First word candidates
   1. "The" (score: 0.95)
   2. "A" (score: 0.85)
   3. "My" (score: 0.75)
   Keep all 3 (beam=3)

STEP 2: Expand each, score combinations
   From "The":
     → "The cat" (0.95 × 0.90 = 0.855) ← Best!
     → "The dog" (0.95 × 0.70 = 0.665)
     → "The bird" (0.95 × 0.40 = 0.380)
   
   From "A":
     → "A cat" (0.85 × 0.80 = 0.680)
     → "A dog" (0.85 × 0.60 = 0.510)
   
   From "My":
     → "My cat" (0.75 × 0.75 = 0.562)
   
   Keep top 3:
   1. "The cat" (0.855) ←
   2. "A cat" (0.680) ←
   3. "The dog" (0.665) ←

STEP 3: Expand these 3, keep top 3...
   → "The cat sat" (best so far!)
   → "A cat sat" 
   → "The cat ran"

RESULT: Beam search found "The cat sat" 
        (which greedy also would have found)
        BUT it explored alternatives!
=============================================================================""")

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
    
    REAL-WORLD EXAMPLE: Ice Cream Customization
    ===========================================
    
    model = Ice cream machine
    prompt_tokens = Your starting flavor
    max_length = How many scoops
    strategy = How you choose flavors
    
    OPTIONS:
    - "greedy": Always get the most popular flavor
    - "sample": Random flavor each time
    - "top_k": Choose from top 40 flavors
    - "top_p": Choose from flavors that make up 90% of orders
    
    temperature = How adventurous you feel
    - Low (0.2): Stick to classics
    - Normal (1.0): Usual preferences
    - High (2.0): Try anything!
    
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
print("="*50)
print("""
SCENARIO: Different strategies generating from same prompt

Watch how each strategy produces different outputs!
""")

print("""
📝 Prompt: "The quick brown fox"

📊 Strategy Comparison:
""")

# Simulate different generation outputs
strategies = {
    "🎯 Greedy (T=1.0)": "jumps over the lazy dog. The cat sat on the mat.",
    "🎲 Sample (T=1.0)": "runs through the forest and sleeps under a tree.",
    "📦 Top-k (k=40)": "jumps over the lazy dog and runs away quickly.",
    "🌰 Top-p (p=0.9)": "jumps over the lazy dog. Then it sleeps.",
    "🥶 Low Temp (T=0.5)": "jumps over the lazy dog. The end of story.",
    "🔥 High Temp (T=1.5)": "dances on clouds while singing to the moon!",
}

for strategy, output in strategies.items():
    emoji = strategy.split()[0]
    name = strategy.split()[1]
    print(f"\n{emoji} {name}:")
    print(f"   The quick brown fox {output}")

# =============================================================================
# STEP 9: Summary and Recommendations
# =============================================================================

print("\n" + "="*70)
print("STEP 9: Strategy Recommendations")
print("="*70)

print("""
REAL-WORLD GUIDE: Choosing the Right Strategy
==============================================

Think of generation strategies like cooking styles:

📐 GREEDY = Follow Recipe Exactly
   Best for: Baking (precision matters)
   Use when: Code generation, factual Q&A, math
   Avoid when: Creative writing, chat, stories
   
   Example: "2+2=" → "4" (always, no variation needed)

🎲 SAMPLING = Improvise Cooking
   Best for: Experimental cuisine
   Use when: Brainstorming, poetry, unique content
   Avoid when: Consistency needed, factual accuracy
   
   Example: "Write a poem about" → Different each time!

📦 TOP-K = Choose from Top Recipes
   Best for: Everyday cooking
   Use when: Chat, general writing, balanced output
   Avoid when: Need maximum creativity or precision
   
   Example: "The weather today is" → Coherent but varied

🌰 TOP-P = Adapt to Ingredients Available
   Best for: Chef's choice menu
   Use when: High-quality creative, adaptive generation
   Avoid when: Need deterministic output
   
   Example: "Once upon a time" → Adapts to context

🔍 BEAM SEARCH = Test All Recipe Variations
   Best for: Competition cooking
   Use when: Translation, summarization, optimization
   Avoid when: Chat, creative writing (too rigid)
   
   Example: Translate "Hello" to French → "Bonjour" (optimal)

POPULAR PRESETS:
================

🤖 GPT-3 Default:
   Top-p (p=0.9-0.95) + Temperature (T=0.7-1.0)
   → Good for most tasks

✍️ Creative Writing:
   Top-p (p=0.9) + Temperature (T=0.8-1.2)
   → Balanced creativity and coherence

💻 Code Generation:
   Greedy or Low Temperature (T=0.2-0.5)
   → Precision is critical

💬 Chat/Conversation:
   Top-k (k=40-50) + Temperature (T=0.7-0.9)
   → Natural, engaging responses

📚 Summarization:
   Beam Search (beam=4-6) or Top-p (p=0.95)
   → Capture key information

=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Text Generation Strategies")
print("="*70)

print("""
GENERATION PROCESS:
===================
1. Tokenize prompt → Convert text to numbers
2. Model forward pass → Get logits
3. Apply temperature → Adjust confidence
4. Softmax → Convert to probabilities
5. Apply strategy → Select next token
6. Append → Add to sequence
7. Repeat → Until done!

STRATEGY COMPARISON TABLE:
==========================
┌─────────────┬──────────────┬───────────────┬─────────────────┐
│ Strategy    │ Pros         │ Cons          │ Best For        │
├─────────────┼──────────────┼───────────────┼─────────────────┤
│ Greedy      │ Fast, simple │ Repetitive    │ Factual, code   │
│ Sample      │ Diverse      │ Incoherent    │ Creative        │
│ Top-k       │ Controlled   │ Fixed k       │ General purpose │
│ Top-p       │ Adaptive     │ Complex       │ Quality text    │
│ Beam Search │ Optimal      │ Expensive     │ Translation     │
└─────────────┴──────────────┴───────────────┴─────────────────┘

PARAMETER GUIDE:
================

Temperature (T):
  0.1-0.3 → Very focused, almost greedy
  0.5-0.7 → Slightly creative
  0.8-1.0 → Balanced (default)
  1.2-1.5 → Creative
  1.5-2.0 → Wild, unpredictable

Top-k (k):
  1 → Greedy decoding
  5-10 → Very focused
  40-50 → Good balance (recommended)
  100+ → Very diverse

Top-p (p):
  0.5-0.75 → Very focused
  0.8-0.9 → Balanced (recommended)
  0.9-0.95 → Creative
  0.95-0.99 → Almost full sampling

KEY INSIGHTS:
=============
1. No single "best" strategy - depends on task!
2. Temperature affects ALL strategies
3. Top-p is more adaptive than top-k
4. Greedy is deterministic, others aren't
5. Beam search is for optimization, not creativity

NEXT: Build a complete working mini GPT model!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Generation")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. COMPARE STRATEGIES:
   Run the same prompt with different strategies
   
   Prompt: "Once upon a time"
   - Greedy: Most likely continuation
   - Sample: Unexpected direction
   - Top-p: Balanced creative
   
   Question: Which output do you prefer?

2. TEMPERATURE SWEEP:
   Try T = 0.2, 0.5, 0.8, 1.0, 1.5, 2.0
   
   Prompt: "The meaning of life is"
   
   Question: How does output change?
   Expectation: Low = confident, High = varied

3. TOP-K COMPARISON:
   Try k = 5, 10, 40, 100
   
   Question: What's the sweet spot?
   Expectation: k=40-50 is usually good

4. TOP-P COMPARISON:
   Try p = 0.5, 0.75, 0.9, 0.95, 0.99
   
   Question: How does nucleus size affect output?
   Expectation: Higher p = more diverse

5. COMBINED TUNING:
   Try top-p (p=0.9) + temperature (T=0.8)
   
   Question: Better than either alone?
   Expectation: Often yes! Best of both worlds.

6. VISUALIZE DISTRIBUTIONS:
   Plot probability distributions at different temps
   
   Question: How does shape change?
   Expectation: Low = peaky, High = flat

KEY TAKEAWAY:
=============
- Generation strategy dramatically affects output
- Temperature is the "creativity knob"
- Top-p adapts to model confidence
- No single best setting - experiment!
- Different tasks need different strategies

Next: 09_mini_gpt.py - Complete working mini GPT!
=============================================================================""")